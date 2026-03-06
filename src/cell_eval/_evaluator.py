import logging
import multiprocessing as mp
import os
import hashlib
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
from pdex import parallel_differential_expression
from scipy.sparse import issparse
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr

from cell_eval.utils import guess_is_lognorm

from ._pipeline import MetricPipeline
from ._types import DEComparison, PerturbationAnndataPair, initialize_de_comparison

logger = logging.getLogger(__name__)

BASELINE_FC_METRICS = [
    "overlap_at_N",
    "overlap_at_50",
    "overlap_at_100",
    "overlap_at_200",
    "overlap_at_500",
    "precision_at_N",
    "precision_at_50",
    "precision_at_100",
    "precision_at_200",
    "precision_at_500",
    "de_spearman_lfc_sig",
    "de_direction_match",
]
BASELINE_CACHE_VERSION = "v1"


class MetricsEvaluator:
    """
    Evaluates benchmarking metrics of a predicted and real anndata object.

    Arguments
    =========

    adata_pred: ad.AnnData | str
        Predicted anndata object or path to anndata object.
    adata_real: ad.AnnData | str
        Real anndata object or path to anndata object.
    de_pred: pl.DataFrame | str | None = None
        Predicted differential expression results or path to differential expression results.
        If `None`, differential expression will be computed using parallel_differential_expression
    de_real: pl.DataFrame | str | None = None
        Real differential expression results or path to differential expression results.
        If `None`, differential expression will be computed using parallel_differential_expression
    control_pert: str = "non-targeting"
        Control perturbation name.
    pert_col: str = "target"
        Perturbation column name.
    celltype_col: str | None = None
        Optional cell-type column in `adata.obs` used to split evaluation context.
    de_method: str = "wilcoxon"
        Differential expression method.
    num_threads: int = -1
        Number of threads for parallel differential expression.
    batch_size: int = 100
        Batch size for parallel differential expression.
    outdir: str = "./cell-eval-outdir"
        Output directory.
    allow_discrete: bool = False
        Allow discrete data.
    prefix: str | None = None
        Prefix for output files.
    pdex_kwargs: dict[str, Any] | None = None
        Keyword arguments for parallel_differential_expression.
        These will overwrite arguments passed to MetricsEvaluator.__init__ if they conflict.
    """

    def __init__(
        self,
        adata_pred: ad.AnnData | str,
        adata_real: ad.AnnData | str,
        de_pred: pl.DataFrame | str | None = None,
        de_real: pl.DataFrame | str | None = None,
        control_pert: str = "non-targeting",
        pert_col: str = "target",
        de_method: str = "wilcoxon",
        num_threads: int = -1,
        batch_size: int = 100,
        outdir: str = "./cell-eval-outdir",
        allow_discrete: bool = False,
        prefix: str | None = None,
        celltype_col: str | None = None,
        pdex_kwargs: dict[str, Any] | None = None,
        skip_de: bool = False,
    ):
        # Enable a global string cache for categorical columns
        pl.enable_string_cache()

        if os.path.exists(outdir):
            logger.warning(
                f"Output directory {outdir} already exists, potential overwrite occurring"
            )
        os.makedirs(outdir, exist_ok=True)

        self.anndata_pair = _build_anndata_pair(
            real=adata_real,
            pred=adata_pred,
            control_pert=control_pert,
            pert_col=pert_col,
            allow_discrete=allow_discrete,
        )

        if celltype_col is not None and (de_pred is not None or de_real is not None):
            raise ValueError(
                "de_pred/de_real are not supported when using celltype_col splitting; "
                "run with raw anndata inputs so DE can be computed per cell type."
            )

        if celltype_col is None:
            if skip_de:
                self.de_comparison = None
            else:
                self.de_comparison = _build_de_comparison(
                    anndata_pair=self.anndata_pair,
                    de_pred=de_pred,
                    de_real=de_real,
                    de_method=de_method,
                    num_threads=num_threads if num_threads != -1 else mp.cpu_count(),
                    batch_size=batch_size,
                    allow_discrete=allow_discrete,
                    outdir=outdir,
                    prefix=prefix,
                    pdex_kwargs=pdex_kwargs or {},
                )
        else:
            self.de_comparison = None

        self.outdir = outdir
        self.prefix = prefix
        self.control_pert = control_pert
        self.pert_col = pert_col
        self.de_method = de_method
        self.num_threads = num_threads if num_threads != -1 else mp.cpu_count()
        self.batch_size = batch_size
        self.allow_discrete = allow_discrete
        self.pdex_kwargs = pdex_kwargs or {}
        self.model_comparison: pl.DataFrame | None = None
        self.celltype_col = celltype_col
        self.skip_de = skip_de
        self._contexts: list[tuple[str | None, PerturbationAnndataPair, DEComparison | None]] = []

        if celltype_col is None:
            self._contexts.append((None, self.anndata_pair, self.de_comparison))
        else:
            self._contexts = self._build_contexts(
                celltype_col=celltype_col,
                de_method=de_method,
            )

    def _build_contexts(
        self,
        celltype_col: str,
        de_method: str,
    ) -> list[tuple[str | None, PerturbationAnndataPair, DEComparison | None]]:
        if celltype_col not in self.anndata_pair.real.obs.columns:
            raise ValueError(
                f"Celltype column '{celltype_col}' missing in adata_real.obs"
            )
        if celltype_col not in self.anndata_pair.pred.obs.columns:
            raise ValueError(
                f"Celltype column '{celltype_col}' missing in adata_pred.obs"
            )

        real_celltypes = set(self.anndata_pair.real.obs[celltype_col].to_numpy(str))
        pred_celltypes = set(self.anndata_pair.pred.obs[celltype_col].to_numpy(str))
        missing_in_real = sorted(pred_celltypes - real_celltypes)
        if missing_in_real:
            raise ValueError(
                "adata_pred contains cell types that are missing in adata_real for "
                f"'{celltype_col}': {missing_in_real}"
            )

        extra_in_real = sorted(real_celltypes - pred_celltypes)
        if extra_in_real:
            logger.warning(
                "Ignoring cell types present only in adata_real for '%s': %s",
                celltype_col,
                extra_in_real,
            )

        contexts: list[tuple[str | None, PerturbationAnndataPair, DEComparison | None]] = []
        for ct in sorted(pred_celltypes):
            real_ct = self.anndata_pair.real[
                self.anndata_pair.real.obs[celltype_col].to_numpy(str) == ct
            ].copy()
            pred_ct = self.anndata_pair.pred[
                self.anndata_pair.pred.obs[celltype_col].to_numpy(str) == ct
            ].copy()
            pair = PerturbationAnndataPair(
                real=real_ct,
                pred=pred_ct,
                control_pert=self.control_pert,
                pert_col=self.pert_col,
            )
            de_comparison = (
                None
                if self.skip_de
                else _build_de_comparison(
                    anndata_pair=pair,
                    de_method=de_method,
                    num_threads=self.num_threads,
                    batch_size=self.batch_size,
                    allow_discrete=self.allow_discrete,
                    outdir=self.outdir,
                    prefix=self._build_context_prefix(ct),
                    pdex_kwargs=self.pdex_kwargs,
                )
            )
            contexts.append((ct, pair, de_comparison))

        return contexts

    def _build_context_prefix(self, context: str | None) -> str | None:
        if context is None:
            return self.prefix
        if self.prefix:
            return f"{self.prefix}_{context}"
        return context

    @staticmethod
    def _sanitize_name(value: str | None) -> str | None:
        if value is None:
            return None
        return value.replace("/", "-")

    def _build_output_paths(
        self,
        basename: str,
        prefix: str | None,
    ) -> tuple[str, str]:
        sanitized_basename = self._sanitize_name(basename) or "results.csv"
        sanitized_prefix = self._sanitize_name(prefix)
        outpath = os.path.join(
            self.outdir,
            f"{sanitized_prefix}_{sanitized_basename}"
            if sanitized_prefix
            else sanitized_basename,
        )
        agg_outpath = os.path.join(
            self.outdir,
            f"{sanitized_prefix}_agg_{sanitized_basename}"
            if sanitized_prefix
            else f"agg_{sanitized_basename}",
        )
        return outpath, agg_outpath

    @staticmethod
    def _compute_merged_aggregate(
        merged_results: pl.DataFrame,
        celltype_col: str,
    ) -> pl.DataFrame:
        metric_columns = [
            col
            for col in merged_results.columns
            if col not in {"perturbation", celltype_col}
        ]
        if not metric_columns:
            return pl.DataFrame()
        return merged_results.select(metric_columns).describe()

    def compute(
        self,
        profile: Literal["full", "vcc", "minimal", "de", "anndata"] = "full",
        metric_configs: dict[str, dict[str, Any]] | None = None,
        skip_metrics: list[str] | None = None,
        basename: str = "results.csv",
        write_csv: bool = True,
        break_on_error: bool = False,
        include_baselines: list[
            Literal["pert-mean", "cell-type-mean", "nearest-cell-type-transfer"]
        ]
        | None = None,
        baseline_celltype_col: str = "celltype",
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        if self.celltype_col is None:
            return self.compute_runner(
                profile=profile,
                metric_configs=metric_configs,
                skip_metrics=skip_metrics,
                basename=basename,
                write_csv=write_csv,
                break_on_error=break_on_error,
                include_baselines=include_baselines,
                baseline_celltype_col=baseline_celltype_col,
                anndata_pair=self.anndata_pair,
                de_comparison=self.de_comparison,
                prefix=self.prefix,
            )

        merged_results_by_context: list[pl.DataFrame] = []
        for context, pair, de_comparison in self._contexts:
            context_prefix = self._build_context_prefix(context)
            context_results, _ = self.compute_runner(
                profile=profile,
                metric_configs=metric_configs,
                skip_metrics=skip_metrics,
                basename=basename,
                write_csv=write_csv,
                break_on_error=break_on_error,
                include_baselines=include_baselines,
                baseline_celltype_col=baseline_celltype_col,
                anndata_pair=pair,
                de_comparison=de_comparison,
                prefix=context_prefix,
            )
            merged_results_by_context.append(
                context_results.with_columns(
                    pl.lit(context).alias(self.celltype_col)
                )
            )

        merged_results = pl.concat(merged_results_by_context, how="diagonal_relaxed")
        merged_agg = self._compute_merged_aggregate(merged_results, self.celltype_col)
        if write_csv:
            outpath, agg_outpath = self._build_output_paths(
                basename=basename,
                prefix=self.prefix,
            )
            logger.info(f"Writing merged perturbation level metrics to {outpath}")
            merged_results.write_csv(outpath)
            logger.info(f"Writing merged aggregate metrics to {agg_outpath}")
            merged_agg.write_csv(agg_outpath)

        return merged_results, merged_agg

    def compute_runner(
        self,
        profile: Literal["full", "vcc", "minimal", "de", "anndata"] = "full",
        metric_configs: dict[str, dict[str, Any]] | None = None,
        skip_metrics: list[str] | None = None,
        basename: str = "results.csv",
        write_csv: bool = True,
        break_on_error: bool = False,
        include_baselines: list[
            Literal["pert-mean", "cell-type-mean", "nearest-cell-type-transfer"]
        ]
        | None = None,
        baseline_celltype_col: str = "celltype",
        anndata_pair: PerturbationAnndataPair | None = None,
        de_comparison: DEComparison | None = None,
        prefix: str | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        anndata_pair = anndata_pair or self.anndata_pair
        de_comparison = self.de_comparison if de_comparison is None else de_comparison
        pipeline = MetricPipeline(
            profile=profile,
            metric_configs=metric_configs,
            break_on_error=break_on_error,
        )
        if skip_metrics is not None:
            pipeline.skip_metrics(skip_metrics)
        pipeline.compute_de_metrics(de_comparison)
        pipeline.compute_anndata_metrics(anndata_pair)
        results = pipeline.get_results()
        agg_results = pipeline.get_agg_results()

        if write_csv:
            outpath, agg_outpath = self._build_output_paths(
                basename=basename,
                prefix=prefix,
            )
            logger.info(f"Writing perturbation level metrics to {outpath}")
            results.write_csv(outpath)

            logger.info(f"Writing aggregate metrics to {agg_outpath}")
            agg_results.write_csv(agg_outpath)

        if include_baselines:
            prev_pair = self.anndata_pair
            prev_de = self.de_comparison
            prev_prefix = self.prefix
            try:
                self.anndata_pair = anndata_pair
                self.de_comparison = de_comparison
                self.prefix = prefix
                baseline_agg_results = self._compute_baselines(
                    profile=profile,
                    metric_configs=metric_configs,
                    skip_metrics=skip_metrics,
                    basename=basename,
                    write_csv=write_csv,
                    break_on_error=break_on_error,
                    include_baselines=include_baselines,
                    baseline_celltype_col=baseline_celltype_col,
                )
            finally:
                self.anndata_pair = prev_pair
                self.de_comparison = prev_de
                self.prefix = prev_prefix
            self.model_comparison = _build_model_comparison_table(
                pred_agg=agg_results,
                baseline_aggs=baseline_agg_results,
            )
            if write_csv and self.model_comparison is not None:
                model_prefix = self._sanitize_name(prefix)
                model_basename = self._sanitize_name(basename) or "results.csv"
                model_outpath = os.path.join(
                    self.outdir,
                    (
                        f"{model_prefix}_model_comparison_{model_basename}"
                        if model_prefix
                        else f"model_comparison_{model_basename}"
                    ),
                )
                logger.info(f"Writing model comparison metrics to {model_outpath}")
                self.model_comparison.write_csv(model_outpath)

        return results, agg_results

    def _compute_baselines(
        self,
        profile: Literal["full", "vcc", "minimal", "de", "anndata"],
        metric_configs: dict[str, dict[str, Any]] | None,
        skip_metrics: list[str] | None,
        basename: str,
        write_csv: bool,
        break_on_error: bool,
        include_baselines: list[
            Literal["pert-mean", "cell-type-mean", "nearest-cell-type-transfer"]
        ],
        baseline_celltype_col: str = "celltype",
    ) -> dict[str, pl.DataFrame]:
        if self.de_comparison is None:
            logger.warning("Skipping baselines because DE comparison is unavailable")
            return {}

        de_real = self.de_comparison.real.data
        genes = np.asarray(self.anndata_pair.real.var_names.to_numpy(str))
        baseline_de_tables: dict[str, pl.DataFrame] = {}
        cache_dir = os.path.join(self.outdir, ".baseline_cache")
        os.makedirs(cache_dir, exist_ok=True)

        for baseline_name in include_baselines:
            if baseline_name == "pert-mean":
                baseline_de_tables[baseline_name] = _load_or_build_baseline_de(
                    baseline_name=baseline_name,
                    adata_real=self.anndata_pair.real,
                    pert_col=self.pert_col,
                    control_pert=self.control_pert,
                    genes=genes,
                    cache_dir=cache_dir,
                    celltype_col=(
                        baseline_celltype_col
                        if baseline_celltype_col in self.anndata_pair.real.obs.columns
                        else None
                    ),
                )
            elif baseline_name == "cell-type-mean":
                if baseline_celltype_col not in self.anndata_pair.real.obs.columns:
                    logger.warning(
                        "Skipping cell-type-mean baseline: "
                        f"column '{baseline_celltype_col}' not found in adata_real.obs"
                    )
                    continue
                baseline_de_tables[baseline_name] = _load_or_build_baseline_de(
                    baseline_name=baseline_name,
                    adata_real=self.anndata_pair.real,
                    pert_col=self.pert_col,
                    control_pert=self.control_pert,
                    genes=genes,
                    cache_dir=cache_dir,
                    celltype_col=baseline_celltype_col,
                )
            elif baseline_name == "nearest-cell-type-transfer":
                if baseline_celltype_col not in self.anndata_pair.real.obs.columns:
                    logger.warning(
                        "Skipping nearest-cell-type-transfer baseline: "
                        f"column '{baseline_celltype_col}' not found in adata_real.obs"
                    )
                    continue
                baseline_de_tables[baseline_name] = _load_or_build_baseline_de(
                    baseline_name=baseline_name,
                    adata_real=self.anndata_pair.real,
                    pert_col=self.pert_col,
                    control_pert=self.control_pert,
                    genes=genes,
                    cache_dir=cache_dir,
                    celltype_col=baseline_celltype_col,
                )
            else:
                logger.warning(f"Unrecognized baseline requested: {baseline_name}")

        baseline_agg_results: dict[str, pl.DataFrame] = {}
        baseline_metric_configs = _build_baseline_metric_configs(metric_configs)
        for baseline_name, baseline_de in baseline_de_tables.items():
            baseline_prefix = (
                f"{self.prefix}_{baseline_name}" if self.prefix else baseline_name
            )
            baseline_perts = baseline_de["target"].unique().to_list()
            if len(baseline_perts) == 0:
                logger.warning(
                    f"Skipping baseline '{baseline_name}' because no valid perturbations were produced"
                )
                continue
            real_for_baseline = de_real.filter(pl.col("target").is_in(baseline_perts))
            baseline_comparison = initialize_de_comparison(
                real=real_for_baseline,
                pred=baseline_de,
            )
            pipeline = MetricPipeline(
                profile=None,
                metric_configs=baseline_metric_configs,
                break_on_error=break_on_error,
            )
            pipeline.add_metrics(BASELINE_FC_METRICS)
            if skip_metrics is not None:
                pipeline.skip_metrics(skip_metrics)
            pipeline.compute_de_metrics(baseline_comparison)
            baseline_results = pipeline.get_results()
            baseline_results = _append_baseline_effect_metrics(
                baseline_results=baseline_results,
                de_real=real_for_baseline,
                de_pred=baseline_de,
                genes=genes,
            )
            baseline_agg = baseline_results.drop("perturbation").describe()

            if write_csv:
                outpath = os.path.join(self.outdir, f"{baseline_prefix}_{basename}")
                agg_outpath = os.path.join(
                    self.outdir, f"{baseline_prefix}_agg_{basename}"
                )
                logger.info(f"Writing baseline metrics to {outpath}")
                baseline_results.write_csv(outpath)
                logger.info(f"Writing baseline aggregate metrics to {agg_outpath}")
                baseline_agg.write_csv(agg_outpath)

            baseline_agg_results[baseline_name] = baseline_agg

        return baseline_agg_results


def _build_anndata_pair(
    real: ad.AnnData | str,
    pred: ad.AnnData | str,
    control_pert: str,
    pert_col: str,
    allow_discrete: bool = False,
):
    if isinstance(real, str):
        logger.info(f"Reading real anndata from {real}")
        real = ad.read_h5ad(real)
    if isinstance(pred, str):
        logger.info(f"Reading pred anndata from {pred}")
        pred = ad.read_h5ad(pred)

    # Validate that the input is normalized and log-transformed
    _convert_to_normlog(real, which="real", allow_discrete=allow_discrete)
    _convert_to_normlog(pred, which="pred", allow_discrete=allow_discrete)

    # Build the anndata pair
    return PerturbationAnndataPair(
        real=real, pred=pred, control_pert=control_pert, pert_col=pert_col
    )


def _convert_to_normlog(
    adata: ad.AnnData,
    which: str | None = None,
    allow_discrete: bool = False,
):
    """Performs a norm-log conversion if the input is integer data (inplace).

    Will skip if the input is not integer data.
    """
    if guess_is_lognorm(adata=adata, validate=not allow_discrete):
        logger.info(
            "Input is found to be log-normalized already - skipping transformation."
        )
        return  # Input is already log-normalized

    # User specified that they want to allow discrete data
    if allow_discrete:
        if which:
            logger.info(
                f"Discovered integer data for {which}. Configuration set to allow discrete. "
                "Make sure this is intentional."
            )
        else:
            logger.info(
                "Discovered integer data. Configuration set to allow discrete. "
                "Make sure this is intentional."
            )
        return  # proceed without conversion

    # Convert the data to norm-log
    if which:
        logger.info(f"Discovered integer data for {which}. Converting to norm-log.")
    sc.pp.normalize_total(adata=adata, inplace=True)  # normalize to median
    sc.pp.log1p(adata)  # log-transform (log1p)


def _build_de_comparison(
    anndata_pair: PerturbationAnndataPair | None = None,
    de_pred: pl.DataFrame | str | None = None,
    de_real: pl.DataFrame | str | None = None,
    de_method: str = "wilcoxon",
    num_threads: int = 1,
    batch_size: int = 100,
    allow_discrete: bool = False,
    outdir: str | None = None,
    prefix: str | None = None,
    pdex_kwargs: dict[str, Any] | None = None,
):
    return initialize_de_comparison(
        real=_load_or_build_de(
            mode="real",
            de_path=de_real,
            anndata_pair=anndata_pair,
            de_method=de_method,
            num_threads=num_threads,
            batch_size=batch_size,
            allow_discrete=allow_discrete,
            outdir=outdir,
            prefix=prefix,
            pdex_kwargs=pdex_kwargs or {},
        ),
        pred=_load_or_build_de(
            mode="pred",
            de_path=de_pred,
            anndata_pair=anndata_pair,
            de_method=de_method,
            num_threads=num_threads,
            batch_size=batch_size,
            allow_discrete=allow_discrete,
            outdir=outdir,
            prefix=prefix,
            pdex_kwargs=pdex_kwargs or {},
        ),
    )


def _build_pdex_kwargs(
    reference: str,
    groupby_key: str,
    num_workers: int,
    batch_size: int,
    metric: str,
    allow_discrete: bool,
    pdex_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pdex_kwargs = pdex_kwargs or {}
    if "reference" not in pdex_kwargs:
        pdex_kwargs["reference"] = reference
    if "groupby_key" not in pdex_kwargs:
        pdex_kwargs["groupby_key"] = groupby_key
    if "num_workers" not in pdex_kwargs:
        pdex_kwargs["num_workers"] = num_workers
    if "batch_size" not in pdex_kwargs:
        pdex_kwargs["batch_size"] = batch_size
    if "metric" not in pdex_kwargs:
        pdex_kwargs["metric"] = metric
    if "is_log1p" not in pdex_kwargs:
        if allow_discrete:
            pdex_kwargs["is_log1p"] = False
        else:
            pdex_kwargs["is_log1p"] = True

    # always return polars DataFrames
    pdex_kwargs["as_polars"] = True
    return pdex_kwargs


def _load_or_build_de(
    mode: Literal["pred", "real"],
    de_path: pl.DataFrame | str | None = None,
    anndata_pair: PerturbationAnndataPair | None = None,
    de_method: str = "wilcoxon",
    num_threads: int = 1,
    batch_size: int = 100,
    outdir: str | None = None,
    prefix: str | None = None,
    allow_discrete: bool = False,
    pdex_kwargs: dict[str, Any] | None = None,
) -> pl.DataFrame:
    if de_path is None:
        if anndata_pair is None:
            raise ValueError("anndata_pair must be provided if de_path is not provided")
        logger.info(f"Computing DE for {mode} data")
        pdex_kwargs = _build_pdex_kwargs(
            reference=anndata_pair.control_pert,
            groupby_key=anndata_pair.pert_col,
            num_workers=num_threads,
            metric=de_method,
            batch_size=batch_size,
            allow_discrete=allow_discrete,
            pdex_kwargs=pdex_kwargs or {},
        )
        logger.info(f"Using the following pdex kwargs: {pdex_kwargs}")
        frame = parallel_differential_expression(
            adata=anndata_pair.real if mode == "real" else anndata_pair.pred,
            **pdex_kwargs,
        )
        if outdir is not None:
            if prefix is not None:
                prefix = prefix.replace(
                    "/", "-"
                )  # some prefixes (e.g. HepG2/C3A) may have slashes in them
            pathname = f"{mode}_de.csv" if not prefix else f"{prefix}_{mode}_de.csv"
            logger.info(f"Writing {mode} DE results to: {pathname}")
            frame.write_csv(os.path.join(outdir, pathname))

        return frame  # type: ignore
    elif isinstance(de_path, str):
        logger.info(f"Reading {mode} DE results from {de_path}")
        if pdex_kwargs:
            logger.warning("pdex_kwargs are ignored when reading from a CSV file")
        return pl.read_csv(
            de_path,
            schema_overrides={
                "target": pl.Utf8,
                "feature": pl.Utf8,
            },
        )
    elif isinstance(de_path, pl.DataFrame):
        if pdex_kwargs:
            logger.warning("pdex_kwargs are ignored when reading from a CSV file")
        return de_path
    elif isinstance(de_path, pd.DataFrame):
        if pdex_kwargs:
            logger.warning("pdex_kwargs are ignored when reading from a CSV file")
        return pl.from_pandas(de_path)
    else:
        raise TypeError(f"Unexpected type for de_path: {type(de_path)}")


def _load_or_build_baseline_de(
    baseline_name: Literal[
        "pert-mean", "cell-type-mean", "nearest-cell-type-transfer"
    ],
    adata_real: ad.AnnData,
    pert_col: str,
    control_pert: str,
    genes: np.ndarray,
    cache_dir: str,
    celltype_col: str | None = None,
) -> pl.DataFrame:
    cache_key = _baseline_cache_key(
        baseline_name=baseline_name,
        adata_real=adata_real,
        pert_col=pert_col,
        control_pert=control_pert,
        celltype_col=celltype_col,
    )
    cache_path = os.path.join(cache_dir, f"{baseline_name}_{cache_key}.csv")
    if os.path.exists(cache_path):
        logger.info(f"Loading cached baseline DE for '{baseline_name}' from {cache_path}")
        try:
            return pl.read_csv(
                cache_path,
                schema_overrides={
                    "target": pl.Utf8,
                    "feature": pl.Utf8,
                    "fold_change": pl.Float64,
                    "p_value": pl.Float64,
                    "fdr": pl.Float64,
                },
            )
        except Exception as err:
            logger.warning(
                f"Failed to read cached baseline DE ({cache_path}): {err}. Recomputing."
            )

    if baseline_name == "pert-mean":
        deltas = _build_pert_mean_deltas(
            adata_real=adata_real,
            pert_col=pert_col,
            control_pert=control_pert,
            celltype_col=celltype_col,
        )
    elif baseline_name == "cell-type-mean":
        if celltype_col is None:
            raise ValueError("celltype_col is required for 'cell-type-mean' baseline")
        deltas = _build_celltype_mean_deltas(
            adata_real=adata_real,
            pert_col=pert_col,
            celltype_col=celltype_col,
            control_pert=control_pert,
        )
    elif baseline_name == "nearest-cell-type-transfer":
        if celltype_col is None:
            raise ValueError(
                "celltype_col is required for 'nearest-cell-type-transfer' baseline"
            )
        deltas = _build_nearest_celltype_transfer_deltas(
            adata_real=adata_real,
            pert_col=pert_col,
            celltype_col=celltype_col,
            control_pert=control_pert,
        )
    else:
        raise ValueError(f"Unsupported baseline: {baseline_name}")

    baseline_de = _deltas_to_de_frame(deltas=deltas, genes=genes)
    baseline_de.write_csv(cache_path)
    logger.info(f"Cached baseline DE for '{baseline_name}' at {cache_path}")
    return baseline_de


def _baseline_cache_key(
    baseline_name: str,
    adata_real: ad.AnnData,
    pert_col: str,
    control_pert: str,
    celltype_col: str | None = None,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(BASELINE_CACHE_VERSION.encode("utf-8"))
    hasher.update(baseline_name.encode("utf-8"))
    hasher.update(pert_col.encode("utf-8"))
    hasher.update(control_pert.encode("utf-8"))
    hasher.update((celltype_col or "<none>").encode("utf-8"))
    hasher.update(f"{adata_real.n_obs}:{adata_real.n_vars}".encode("utf-8"))
    _hash_str_array(hasher, np.asarray(adata_real.var_names.to_numpy(str)))

    source_path = getattr(adata_real, "filename", None)
    if isinstance(source_path, str) and source_path and os.path.exists(source_path):
        st = os.stat(source_path)
        hasher.update(os.path.abspath(source_path).encode("utf-8"))
        hasher.update(str(st.st_size).encode("utf-8"))
        hasher.update(str(st.st_mtime_ns).encode("utf-8"))
        return hasher.hexdigest()

    if pert_col not in adata_real.obs.columns:
        raise ValueError(f"Perturbation column '{pert_col}' missing in adata_real.obs")
    _hash_str_array(hasher, np.asarray(adata_real.obs[pert_col].to_numpy(str)))
    if celltype_col is not None:
        if celltype_col not in adata_real.obs.columns:
            raise ValueError(f"Celltype column '{celltype_col}' missing in adata_real.obs")
        _hash_str_array(hasher, np.asarray(adata_real.obs[celltype_col].to_numpy(str)))

    _hash_matrix(hasher, adata_real.X)
    return hasher.hexdigest()


def _hash_str_array(hasher: Any, arr: np.ndarray) -> None:
    hasher.update(str(arr.shape).encode("utf-8"))
    for value in arr.astype(str):
        hasher.update(value.encode("utf-8"))
        hasher.update(b"\0")


def _hash_matrix(hasher: Any, matrix: Any) -> None:
    if issparse(matrix):
        hasher.update(str(matrix.shape).encode("utf-8"))
        hasher.update(matrix.data.tobytes())
        hasher.update(matrix.indices.tobytes())
        hasher.update(matrix.indptr.tobytes())
        return

    dense = np.asarray(matrix)
    hasher.update(str(dense.shape).encode("utf-8"))
    hasher.update(str(dense.dtype).encode("utf-8"))
    hasher.update(np.ascontiguousarray(dense).tobytes())


def _mean_profile_for_mask(adata: ad.AnnData, mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        raise ValueError("Cannot compute mean profile for an empty mask")
    matrix = adata[mask].X
    if issparse(matrix):
        return np.asarray(matrix.mean(axis=0)).ravel().astype(np.float64)
    return np.asarray(matrix, dtype=np.float64).mean(axis=0)


def _build_pert_mean_deltas(
    adata_real: ad.AnnData,
    pert_col: str,
    control_pert: str,
    celltype_col: str | None = None,
) -> dict[str, np.ndarray]:
    perts_all = adata_real.obs[pert_col].to_numpy(str)
    ctrl_mask = perts_all == control_pert
    if not np.any(ctrl_mask):
        raise ValueError(f"Control perturbation '{control_pert}' was not found")

    perts = np.unique(perts_all)
    perts = perts[perts != control_pert]

    # If no celltype information is provided, return global per-pert effects.
    if celltype_col is None:
        global_ctrl = _mean_profile_for_mask(adata_real, ctrl_mask)
        out: dict[str, np.ndarray] = {}
        for pert in perts:
            pert_mask = perts_all == pert
            if not np.any(pert_mask):
                continue
            out[str(pert)] = _mean_profile_for_mask(adata_real, pert_mask) - global_ctrl
        return out

    celltypes_all = adata_real.obs[celltype_col].to_numpy(str)
    celltypes = np.unique(celltypes_all)
    deltas_by_pert: dict[str, np.ndarray] = {}

    for pert in perts:
        donor_deltas: list[np.ndarray] = []
        pert_name = str(pert)

        for ct in celltypes:
            target_mask = (celltypes_all == ct) & (perts_all == pert_name)
            if not np.any(target_mask):
                continue

            donor_pert_mask = (celltypes_all != ct) & (perts_all == pert_name)
            donor_ctrl_mask = (celltypes_all != ct) & (perts_all == control_pert)

            if np.any(donor_pert_mask) and np.any(donor_ctrl_mask):
                donor_deltas.append(
                    _mean_profile_for_mask(adata_real, donor_pert_mask)
                    - _mean_profile_for_mask(adata_real, donor_ctrl_mask)
                )
            # If donor set is missing, skip this (celltype, pert) combination.

        if donor_deltas:
            deltas_by_pert[pert_name] = np.mean(np.vstack(donor_deltas), axis=0)

    return deltas_by_pert


def _build_celltype_mean_deltas(
    adata_real: ad.AnnData,
    pert_col: str,
    celltype_col: str,
    control_pert: str,
) -> dict[str, np.ndarray]:
    perts_all = adata_real.obs[pert_col].to_numpy(str)
    celltypes_all = adata_real.obs[celltype_col].to_numpy(str)
    perts = np.unique(perts_all)
    perts = perts[perts != control_pert]
    celltypes = np.unique(celltypes_all)

    ctrl_mask = perts_all == control_pert
    if not np.any(ctrl_mask):
        raise ValueError(f"Control perturbation '{control_pert}' was not found")
    global_ctrl = _mean_profile_for_mask(adata_real, ctrl_mask)

    mean_delta_by_ct: dict[str, np.ndarray] = {}
    for ct in celltypes:
        ct_ctrl_mask = (celltypes_all == ct) & ctrl_mask
        ct_ctrl = (
            _mean_profile_for_mask(adata_real, ct_ctrl_mask)
            if np.any(ct_ctrl_mask)
            else global_ctrl
        )
        ct_deltas: list[np.ndarray] = []
        for pert in perts:
            mask = (celltypes_all == ct) & (perts_all == pert)
            if np.any(mask):
                ct_deltas.append(_mean_profile_for_mask(adata_real, mask) - ct_ctrl)
        mean_delta_by_ct[ct] = (
            np.mean(np.vstack(ct_deltas), axis=0)
            if ct_deltas
            else np.zeros(adata_real.shape[1], dtype=np.float64)
        )

    deltas_by_pert: dict[str, np.ndarray] = {}
    for pert in perts:
        ct_deltas_for_pert: list[np.ndarray] = []
        for ct in celltypes:
            if not np.any((perts_all == pert) & (celltypes_all == ct)):
                continue
            ct_deltas_for_pert.append(mean_delta_by_ct[ct])
        deltas_by_pert[str(pert)] = (
            np.mean(np.vstack(ct_deltas_for_pert), axis=0)
            if ct_deltas_for_pert
            else np.zeros(adata_real.shape[1], dtype=np.float64)
        )
    return deltas_by_pert


def _build_nearest_celltype_transfer_deltas(
    adata_real: ad.AnnData,
    pert_col: str,
    celltype_col: str,
    control_pert: str,
) -> dict[str, np.ndarray]:
    perts_all = adata_real.obs[pert_col].to_numpy(str)
    celltypes_all = adata_real.obs[celltype_col].to_numpy(str)
    perts = np.unique(perts_all)
    perts = perts[perts != control_pert]

    ctrl_mask = perts_all == control_pert
    if not np.any(ctrl_mask):
        raise ValueError(f"Control perturbation '{control_pert}' was not found")

    # Build control-state profiles from the control subset only.
    ctrl_subset = adata_real[ctrl_mask]
    ctrl_celltypes = np.unique(ctrl_subset.obs[celltype_col].to_numpy(str))
    ctrl_profiles: dict[str, np.ndarray] = {}
    for ct in ctrl_celltypes:
        ct_mask = ctrl_subset.obs[celltype_col].to_numpy(str) == ct
        if np.any(ct_mask):
            ctrl_profiles[ct] = _mean_profile_for_mask(ctrl_subset, ct_mask)
    if not ctrl_profiles:
        raise ValueError("No control cells were found to build transfer baseline")

    # Find nearest donor cell type via pairwise distances on control-state profiles.
    ordered_ct = sorted(ctrl_profiles.keys())
    donor_by_ct: dict[str, str] = {}
    if len(ordered_ct) == 1:
        donor_by_ct[ordered_ct[0]] = ordered_ct[0]
    else:
        profile_mat = np.vstack([ctrl_profiles[ct] for ct in ordered_ct])
        dist_mat = squareform(pdist(profile_mat, metric="euclidean"))
        np.fill_diagonal(dist_mat, np.inf)
        nearest_idx = np.argmin(dist_mat, axis=1)
        donor_by_ct = {
            ct: ordered_ct[int(nearest_idx[i])] for i, ct in enumerate(ordered_ct)
        }

    delta_by_ct_pert: dict[tuple[str, str], np.ndarray] = {}
    mean_delta_by_ct: dict[str, np.ndarray] = {}
    for ct, ctrl_profile in ctrl_profiles.items():
        ct_deltas: list[np.ndarray] = []
        for pert in perts:
            mask = (celltypes_all == ct) & (perts_all == pert)
            if not np.any(mask):
                continue
            delta = _mean_profile_for_mask(adata_real, mask) - ctrl_profile
            delta_by_ct_pert[(ct, str(pert))] = delta
            ct_deltas.append(delta)
        mean_delta_by_ct[ct] = (
            np.mean(np.vstack(ct_deltas), axis=0)
            if ct_deltas
            else np.zeros(adata_real.shape[1], dtype=np.float64)
        )

    deltas_by_pert: dict[str, np.ndarray] = {}
    for pert in perts:
        ct_deltas_for_pert: list[np.ndarray] = []
        for ct in np.unique(celltypes_all):
            if not np.any((celltypes_all == ct) & (perts_all == pert)):
                continue
            if ct not in donor_by_ct:
                continue
            donor_ct = donor_by_ct[ct]
            donor_delta = delta_by_ct_pert.get((donor_ct, str(pert)))
            if donor_delta is None:
                donor_delta = mean_delta_by_ct[donor_ct]
            ct_deltas_for_pert.append(donor_delta)
        deltas_by_pert[str(pert)] = (
            np.mean(np.vstack(ct_deltas_for_pert), axis=0)
            if ct_deltas_for_pert
            else np.zeros(adata_real.shape[1], dtype=np.float64)
        )
    return deltas_by_pert


def _deltas_to_de_frame(
    deltas: dict[str, np.ndarray],
    genes: np.ndarray,
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    n_genes = genes.shape[0]
    for pert, delta in deltas.items():
        delta = np.asarray(delta, dtype=np.float64)
        if delta.shape[0] != n_genes:
            raise ValueError(
                f"Delta length mismatch for {pert}: {delta.shape[0]} != {n_genes}"
            )
        fold_change = np.exp(np.clip(delta, -20.0, 20.0))
        frames.append(
            pl.DataFrame(
                {
                    "target": np.repeat(pert, n_genes),
                    "feature": genes,
                    "fold_change": fold_change,
                    "p_value": np.repeat(0.5, n_genes),
                    "fdr": np.repeat(0.5, n_genes),
                }
            )
        )
    if not frames:
        return pl.DataFrame(
            schema={
                "target": pl.Utf8,
                "feature": pl.Utf8,
                "fold_change": pl.Float64,
                "p_value": pl.Float64,
                "fdr": pl.Float64,
            }
        )
    return pl.concat(frames, how="vertical")


def _build_baseline_metric_configs(
    user_metric_configs: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    baseline_metric_configs = {
        metric: {"fdr_threshold": 1.0} for metric in BASELINE_FC_METRICS
    }
    if user_metric_configs:
        for metric, cfg in user_metric_configs.items():
            if metric in baseline_metric_configs:
                baseline_metric_configs[metric].update(cfg)
    return baseline_metric_configs


def _append_baseline_effect_metrics(
    baseline_results: pl.DataFrame,
    de_real: pl.DataFrame,
    de_pred: pl.DataFrame,
    genes: np.ndarray,
) -> pl.DataFrame:
    real_delta = _de_to_delta_map(de_real, genes)
    pred_delta = _de_to_delta_map(de_pred, genes)

    rows: list[dict[str, float | str]] = []
    for pert in sorted(set(real_delta) & set(pred_delta)):
        x = pred_delta[pert]
        y = real_delta[pert]
        rows.append(
            {
                "perturbation": pert,
                "pearson_delta": _safe_corr(pearsonr, x, y),
                "de_spearman_lfc": _safe_corr(spearmanr, x, y),
                "spearman_effect_size": _safe_corr(
                    spearmanr, np.abs(x), np.abs(y)
                ),
            }
        )

    if not rows:
        return baseline_results

    extra = pl.DataFrame(rows)
    if baseline_results.height == 0:
        return extra
    return baseline_results.join(extra, on="perturbation", how="left")


def _de_to_delta_map(de_frame: pl.DataFrame, genes: np.ndarray) -> dict[str, np.ndarray]:
    features = genes.astype(str).tolist()
    gene_index = {g: i for i, g in enumerate(features)}
    out: dict[str, np.ndarray] = {}

    for pert in de_frame["target"].unique().to_list():
        vec = np.zeros(len(features), dtype=np.float64)
        sub = de_frame.filter(pl.col("target") == pert).select(["feature", "fold_change"])
        for feature, fold_change in sub.iter_rows():
            idx = gene_index.get(str(feature))
            if idx is None:
                continue
            fc = max(float(fold_change), 1e-12)
            vec[idx] = np.log2(fc)
        out[str(pert)] = vec
    return out


def _safe_corr(
    func: Any,
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    try:
        value = float(func(x, y).correlation)
        if np.isnan(value):
            return 0.0
        return value
    except Exception:
        return 0.0


def _extract_mean_row(agg: pl.DataFrame, label: str) -> pl.DataFrame:
    if "statistic" not in agg.columns:
        raise ValueError("Aggregate table missing 'statistic' column")
    mean_row = agg.filter(pl.col("statistic") == "mean")
    if mean_row.height == 0:
        mean_row = agg.head(1)
    metric_cols = [c for c in mean_row.columns if c != "statistic"]
    return mean_row.select(metric_cols).with_columns(pl.lit(label).alias("model")).select(
        ["model", *metric_cols]
    )


def _build_model_comparison_table(
    pred_agg: pl.DataFrame,
    baseline_aggs: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    label_map = {
        "pert-mean": "pertmean",
        "cell-type-mean": "contextmean",
        "nearest-cell-type-transfer": "celltype_transfer",
    }
    rows = [_extract_mean_row(pred_agg, label="pred")]
    for baseline_key, agg in baseline_aggs.items():
        label = label_map.get(baseline_key, baseline_key)
        rows.append(_extract_mean_row(agg, label=label))
    return pl.concat(rows, how="diagonal_relaxed")
