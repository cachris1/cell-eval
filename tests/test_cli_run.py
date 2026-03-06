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


class DummyMetricsEvaluator:
    def __init__(
        self,
        adata_pred: str,
        adata_real: str,
        outdir: str,
        celltype_col: str | None = None,
        prefix: str | None = None,
        **_: object,
    ) -> None:
        self.adata_pred = ad.read_h5ad(adata_pred)
        self.adata_real = ad.read_h5ad(adata_real)
        self.outdir = Path(outdir)
        self.celltype_col = celltype_col
        self.prefix = prefix
        os.makedirs(self.outdir, exist_ok=True)

    def compute(
        self,
        basename: str = "results.csv",
        **_: object,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        if self.celltype_col is None:
            results = pl.DataFrame({"perturbation": ["pert_1"], "dummy_metric": [1.0]})
            agg_results = results.drop("perturbation").describe()
            results.write_csv(self.outdir / basename)
            agg_results.write_csv(self.outdir / f"agg_{basename}")
            return results, agg_results

        real_ct = set(self.adata_real.obs[self.celltype_col].to_numpy(str))
        pred_ct = set(self.adata_pred.obs[self.celltype_col].to_numpy(str))
        assert pred_ct.issubset(real_ct)

        frames: list[pl.DataFrame] = []
        for ct in sorted(pred_ct):
            ct_frame = pl.DataFrame(
                {
                    "perturbation": ["pert_1"],
                    "dummy_metric": [float(self.adata_pred.n_obs)],
                    self.celltype_col: [ct],
                }
            )
            frames.append(ct_frame)
            ct_no_context = ct_frame.drop(self.celltype_col)
            ct_no_context.write_csv(self.outdir / f"{ct}_{basename}")
            ct_no_context.drop("perturbation").describe().write_csv(
                self.outdir / f"{ct}_agg_{basename}"
            )

        merged_results = pl.concat(frames, how="diagonal_relaxed")
        merged_agg = merged_results.drop("perturbation", self.celltype_col).describe()
        merged_results.write_csv(self.outdir / basename)
        merged_agg.write_csv(self.outdir / f"agg_{basename}")
        return merged_results, merged_agg


def test_run_evaluation_celltype_split_writes_merged_outputs(tmp_path: Path):
    adata_real = _build_multicelltype_adata()
    adata_pred = adata_real.copy()

    adata_real_path = tmp_path / "adata_real.h5ad"
    adata_pred_path = tmp_path / "adata_pred.h5ad"
    outdir = tmp_path / "out"

    adata_real.write_h5ad(adata_real_path)
    adata_pred.write_h5ad(adata_pred_path)

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

    original_evaluator = cell_eval.MetricsEvaluator
    cell_eval.MetricsEvaluator = DummyMetricsEvaluator
    try:
        run_evaluation(_build_run_args(adata_pred_path, adata_real_path, outdir))
    finally:
        cell_eval.MetricsEvaluator = original_evaluator

    merged_results = pl.read_csv(outdir / "results.csv")
    assert set(merged_results["cell_type"].to_list()) == {"cell_type_a"}
