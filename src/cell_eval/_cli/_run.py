import argparse as ap
import importlib.metadata
import logging
import os

import polars as pl

from .. import KNOWN_PROFILES
from ._const import (
    DEFAULT_CELLTYPE_COL,
    DEFAULT_CTRL,
    DEFAULT_OUTDIR,
    DEFAULT_PERT_COL,
)

logger = logging.getLogger(__name__)


def parse_args_run(parser: ap.ArgumentParser):
    """
    CLI for evaluation
    """
    parser.add_argument(
        "-ap",
        "--adata-pred",
        type=str,
        help="Path to the predicted adata object to evaluate",
        required=True,
    )
    parser.add_argument(
        "-ar",
        "--adata-real",
        type=str,
        help="Path to the real adata object to evaluate against",
        required=True,
    )
    parser.add_argument(
        "-dp",
        "--de-pred",
        type=str,
        help="Path to the predicted DE results "
        f"(computed with pdex from adata-pred if not provided and saved to {DEFAULT_OUTDIR}/pred_de.csv)",
        required=False,
    )
    parser.add_argument(
        "-dr",
        "--de-real",
        type=str,
        help="Path to the real DE results "
        f"(computed with pdex from adata-real if not provided and saved to {DEFAULT_OUTDIR}/real_de.csv)",
        required=False,
    )
    parser.add_argument(
        "--control-pert",
        type=str,
        default=DEFAULT_CTRL,
        help="Name of the control perturbation [default: %(default)s]",
    )
    parser.add_argument(
        "--pert-col",
        type=str,
        default=DEFAULT_PERT_COL,
        help="Name of the column designated perturbations [default: %(default)s]",
    )
    parser.add_argument(
        "--celltype-col",
        "--cell-type-key",
        dest="celltype_col",
        type=str,
        help=(
            "Name of the cell-type column in adata.obs to split evaluation by "
            "(optional). When provided, per-celltype outputs are written and merged "
            "results/agg_results are also written."
        ),
    )
    parser.add_argument(
        "--embed-key",
        type=str,
        default=None,
        help="Key for embedded data (.obsm) in the AnnData object used in some metrics (evaluated over .X otherwise)",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default=DEFAULT_OUTDIR,
        help="Output directory to write to [default: %(default)s]",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads to use for parallel processing [default: %(default)s]",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for parallel processing [default: %(default)s]",
    )
    parser.add_argument(
        "--de-method",
        type=str,
        default="wilcoxon",
        help="Method to use for differential expression analysis [default: %(default)s]",
    )
    parser.add_argument(
        "--allow-discrete",
        action="store_true",
        help="Allow discrete data to be evaluated (usually expected to be norm-logged inputs)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="full",
        help="Profile of metrics to compute [default: %(default)s]",
        choices=KNOWN_PROFILES,
    )
    parser.add_argument(
        "--skip-metrics",
        type=str,
        help="Metrics to skip (comma-separated for multiple) (see docs for more details)",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        default=None,
        help=(
            "Optional baselines to evaluate in addition to adata-pred "
            "(comma-separated): pert-mean,cell-type-mean,nearest-cell-type-transfer"
        ),
    )
    parser.add_argument(
        "--baseline-celltype-col",
        type=str,
        default=DEFAULT_CELLTYPE_COL,
        help=(
            "Cell type column for cell-type-mean and nearest-cell-type-transfer baselines "
            "[default: %(default)s]"
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s {version}".format(
            version=importlib.metadata.version("cell_eval")
        ),
    )


def build_outdir(outdir: str):
    if os.path.exists(outdir):
        logger.warning(
            f"Output directory {outdir} already exists, potential overwrite occurring"
        )
    os.makedirs(outdir, exist_ok=True)


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


def run_evaluation(args: ap.Namespace):
    import anndata as ad

    from cell_eval import MetricsEvaluator
    from cell_eval.utils import split_anndata_on_celltype

    # Set metric config for embed key if provided
    metric_kwargs = (
        {
            "discrimination_score_l2": {"embed_key": args.embed_key},
            "discrimination_score_cosine": {"embed_key": args.embed_key},
            "pearson_edistance": {"n_jobs": args.num_threads},
        }
        if args.embed_key is not None
        else {}
    )

    skip_metrics = args.skip_metrics.split(",") if args.skip_metrics else None
    baselines = [b.strip() for b in args.baselines.split(",")] if args.baselines else []
    valid_baselines = {
        "pert-mean",
        "cell-type-mean",
        "nearest-cell-type-transfer",
    }
    baselines = [b for b in baselines if b in valid_baselines]

    if args.celltype_col is not None:
        real = ad.read_h5ad(args.adata_real)
        pred = ad.read_h5ad(args.adata_pred)

        real_split = split_anndata_on_celltype(real, args.celltype_col)
        pred_split = split_anndata_on_celltype(pred, args.celltype_col)

        real_celltypes = set(real_split.keys())
        pred_celltypes = set(pred_split.keys())
        missing_in_real = sorted(pred_celltypes - real_celltypes)
        if missing_in_real:
            raise ValueError(
                "adata_pred contains cell types that are missing in adata_real for "
                f"'{args.celltype_col}': {missing_in_real}"
            )
        extra_in_real = sorted(real_celltypes - pred_celltypes)
        if extra_in_real:
            logger.warning(
                "Ignoring cell types present only in adata_real for '%s': %s",
                args.celltype_col,
                extra_in_real,
            )

        merged_results_by_celltype: list[pl.DataFrame] = []
        for ct in sorted(pred_celltypes):
            real_ct = real_split[ct]
            pred_ct = pred_split[ct]

            evaluator = MetricsEvaluator(
                adata_pred=pred_ct,
                adata_real=real_ct,
                de_pred=args.de_pred,
                de_real=args.de_real,
                control_pert=args.control_pert,
                pert_col=args.pert_col,
                de_method=args.de_method,
                num_threads=args.num_threads,
                batch_size=args.batch_size,
                outdir=args.outdir,
                allow_discrete=args.allow_discrete,
                prefix=ct,
                skip_de=args.profile == "pds",
            )
            results, _ = evaluator.compute(
                profile=args.profile,
                metric_configs=metric_kwargs,
                skip_metrics=skip_metrics,
                basename="results.csv",
                include_baselines=baselines or None,
                baseline_celltype_col=args.baseline_celltype_col,
            )
            merged_results_by_celltype.append(
                results.with_columns(pl.lit(ct).alias(args.celltype_col))
            )

        merged_results = pl.concat(merged_results_by_celltype, how="diagonal_relaxed")
        merged_agg = _compute_merged_aggregate(merged_results, args.celltype_col)
        merged_results_outpath = os.path.join(args.outdir, "results.csv")
        merged_agg_outpath = os.path.join(args.outdir, "agg_results.csv")
        logger.info(f"Writing merged perturbation level metrics to {merged_results_outpath}")
        merged_results.write_csv(merged_results_outpath)
        logger.info(f"Writing merged aggregate metrics to {merged_agg_outpath}")
        merged_agg.write_csv(merged_agg_outpath)

    else:
        evaluator = MetricsEvaluator(
            adata_pred=args.adata_pred,
            adata_real=args.adata_real,
            de_pred=args.de_pred,
            de_real=args.de_real,
            control_pert=args.control_pert,
            pert_col=args.pert_col,
            de_method=args.de_method,
            num_threads=args.num_threads,
            batch_size=args.batch_size,
            outdir=args.outdir,
            allow_discrete=args.allow_discrete,
            skip_de=args.profile == "pds",
        )
        evaluator.compute(
            profile=args.profile,
            metric_configs=metric_kwargs,
            skip_metrics=skip_metrics,
            basename="results.csv",
            include_baselines=baselines or None,
            baseline_celltype_col=args.baseline_celltype_col,
        )
