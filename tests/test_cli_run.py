import argparse as ap
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl

import cell_eval
from cell_eval._cli._run import run_evaluation


def _build_multicelltype_adata(
    n_genes: int = 12,
    n_cells_per_combo: int = 6,
) -> ad.AnnData:
    cell_types = ["cell_type_a", "cell_type_b"]
    perturbations = ["control", "pert_1", "pert_2"]

    obs_records: list[dict[str, str]] = []
    for cell_type in cell_types:
        for perturbation in perturbations:
            for _ in range(n_cells_per_combo):
                obs_records.append(
                    {
                        "cell_type": cell_type,
                        "perturbation": perturbation,
                    }
                )

    n_cells = len(obs_records)
    matrix = np.random.uniform(0.0, 5.0, size=(n_cells, n_genes))
    obs = pd.DataFrame(obs_records)
    obs.index = obs.index.astype(str)
    return ad.AnnData(X=matrix, obs=obs)


def _build_run_args(adata_pred: Path, adata_real: Path, outdir: Path) -> ap.Namespace:
    return ap.Namespace(
        adata_pred=str(adata_pred),
        adata_real=str(adata_real),
        de_pred=None,
        de_real=None,
        control_pert="control",
        pert_col="perturbation",
        celltype_col="cell_type",
        embed_key=None,
        outdir=str(outdir),
        num_threads=1,
        batch_size=50,
        de_method="wilcoxon",
        allow_discrete=False,
        profile="minimal",
        skip_metrics=None,
        baselines=None,
        baseline_celltype_col="celltype",
    )


def test_run_evaluation_celltype_split_writes_merged_outputs(tmp_path: Path):
    adata_real = _build_multicelltype_adata()
    adata_pred = adata_real.copy()

    adata_real_path = tmp_path / "adata_real.h5ad"
    adata_pred_path = tmp_path / "adata_pred.h5ad"
    outdir = tmp_path / "out"

    adata_real.write_h5ad(adata_real_path)
    adata_pred.write_h5ad(adata_pred_path)

    class DummyMetricsEvaluator:
        def __init__(
            self,
            adata_pred: ad.AnnData,
            adata_real: ad.AnnData,
            outdir: str,
            prefix: str | None = None,
            **_: object,
        ) -> None:
            self.adata_pred = adata_pred
            self.adata_real = adata_real
            self.outdir = outdir
            self.prefix = prefix
            os.makedirs(self.outdir, exist_ok=True)

        def compute(
            self,
            basename: str = "results.csv",
            **_: object,
        ) -> tuple[pl.DataFrame, pl.DataFrame]:
            metric_value = float(self.adata_real.n_obs)
            results = pl.DataFrame(
                {
                    "perturbation": ["pert_1"],
                    "dummy_metric": [metric_value],
                }
            )
            agg_results = results.drop("perturbation").describe()

            outpath = (
                outdir / f"{self.prefix}_{basename}"
                if self.prefix
                else outdir / basename
            )
            agg_outpath = (
                outdir / f"{self.prefix}_agg_{basename}"
                if self.prefix
                else outdir / f"agg_{basename}"
            )
            results.write_csv(outpath)
            agg_results.write_csv(agg_outpath)
            return results, agg_results

    original_evaluator = cell_eval.MetricsEvaluator
    cell_eval.MetricsEvaluator = DummyMetricsEvaluator
    try:
        run_evaluation(_build_run_args(adata_pred_path, adata_real_path, outdir))
    finally:
        cell_eval.MetricsEvaluator = original_evaluator

    merged_results_path = outdir / "results.csv"
    merged_agg_path = outdir / "agg_results.csv"
    assert merged_results_path.exists()
    assert merged_agg_path.exists()

    merged_results = pl.read_csv(merged_results_path)
    merged_agg = pl.read_csv(merged_agg_path)

    assert "cell_type" in merged_results.columns
    assert merged_results.height > 0
    assert merged_agg.height > 0

    for cell_type in ("cell_type_a", "cell_type_b"):
        assert (outdir / f"{cell_type}_results.csv").exists()
        assert (outdir / f"{cell_type}_agg_results.csv").exists()


def test_run_evaluation_allows_extra_real_celltypes(tmp_path: Path):
    adata_real = _build_multicelltype_adata()
    adata_pred = adata_real[adata_real.obs["cell_type"] != "cell_type_b"].copy()

    adata_real_path = tmp_path / "adata_real.h5ad"
    adata_pred_path = tmp_path / "adata_pred.h5ad"
    outdir = tmp_path / "out"

    adata_real.write_h5ad(adata_real_path)
    adata_pred.write_h5ad(adata_pred_path)

    class DummyMetricsEvaluator:
        def __init__(
            self,
            adata_pred: ad.AnnData,
            adata_real: ad.AnnData,
            outdir: str,
            prefix: str | None = None,
            **_: object,
        ) -> None:
            self.adata_pred = adata_pred
            self.adata_real = adata_real
            self.outdir = outdir
            self.prefix = prefix
            os.makedirs(self.outdir, exist_ok=True)

        def compute(
            self,
            basename: str = "results.csv",
            **_: object,
        ) -> tuple[pl.DataFrame, pl.DataFrame]:
            results = pl.DataFrame(
                {
                    "perturbation": ["pert_1"],
                    "dummy_metric": [float(self.adata_pred.n_obs)],
                }
            )
            agg_results = results.drop("perturbation").describe()
            outpath = (
                outdir / f"{self.prefix}_{basename}"
                if self.prefix
                else outdir / basename
            )
            agg_outpath = (
                outdir / f"{self.prefix}_agg_{basename}"
                if self.prefix
                else outdir / f"agg_{basename}"
            )
            results.write_csv(outpath)
            agg_results.write_csv(agg_outpath)
            return results, agg_results

    original_evaluator = cell_eval.MetricsEvaluator
    cell_eval.MetricsEvaluator = DummyMetricsEvaluator
    try:
        run_evaluation(_build_run_args(adata_pred_path, adata_real_path, outdir))
    finally:
        cell_eval.MetricsEvaluator = original_evaluator

    merged_results = pl.read_csv(outdir / "results.csv")
    assert set(merged_results["cell_type"].to_list()) == {"cell_type_a"}
