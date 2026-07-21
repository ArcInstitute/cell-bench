import shutil

import numpy as np
import polars as pl
import pytest

from cell_eval import MetricsEvaluator
from cell_eval._evaluator import SB_METRICS, _spearman_brown_correct
from cell_eval.data import CONTROL_VAR, PERT_COL, build_random_anndata

OUTDIR = "TEST_OUTPUT_CEILING"


def test_spearman_brown_doubling_on_reliability_metrics():
    """SB doubling r' = 2r/(1+r) is applied to reliability (best_value == ONE)
    metric columns."""
    df = pl.DataFrame(
        {
            "perturbation": ["a", "b"],
            "pearson_delta": [0.5, 1.0 / 3.0],
            "overlap_at_N": [0.6, 0.2],
        }
    )
    out = _spearman_brown_correct(df)
    # 2*0.5/(1+0.5)=2/3 ; 2*(1/3)/(1+1/3)=0.5
    assert out["pearson_delta"].to_list() == pytest.approx([2 / 3, 0.5], abs=1e-6)
    # 2*0.6/1.6=0.75 ; 2*0.2/1.2=1/3
    assert out["overlap_at_N"].to_list() == pytest.approx([0.75, 1 / 3], abs=1e-6)


def test_non_reliability_and_excluded_metrics_are_nan():
    """Error metrics, unbounded counts, and the excluded reliabilities
    (clustering_agreement, pearson_edistance) are emitted as NaN."""
    df = pl.DataFrame(
        {
            "perturbation": ["a"],
            "pearson_delta": [0.5],  # reliability -> SB
            "mse": [0.01],  # error metric -> NaN
            "mae": [0.02],  # error metric -> NaN
            "de_nsig_counts_real": [10.0],  # unbounded count -> NaN
            "clustering_agreement": [0.4],  # excluded reliability -> NaN
            "pearson_edistance": [0.9],  # excluded reliability -> NaN
        }
    )
    out = _spearman_brown_correct(df)
    assert out["pearson_delta"][0] == pytest.approx(2 / 3, abs=1e-6)
    for col in (
        "mse",
        "mae",
        "de_nsig_counts_real",
        "clustering_agreement",
        "pearson_edistance",
    ):
        assert np.isnan(out[col][0]), col
    # SB_METRICS is the explicit inclusion list; the excluded ones are absent
    assert "pearson_delta" in SB_METRICS
    assert "clustering_agreement" not in SB_METRICS
    assert "pearson_edistance" not in SB_METRICS


def test_disjoint_halves_share_no_cells():
    adata_real = build_random_anndata()
    evaluator = MetricsEvaluator(
        adata_pred=adata_real.copy(),
        adata_real=adata_real,
        control_pert=CONTROL_VAR,
        pert_col=PERT_COL,
        outdir=OUTDIR,
        skip_de=True,
    )
    half_real, half_pred = evaluator._disjoint_halves(seed=0)

    # disjoint: no original cell (by name) appears in both halves
    assert set(half_real.obs_names).isdisjoint(set(half_pred.obs_names))
    # both halves carry the control + every (splittable) perturbation
    assert CONTROL_VAR in set(half_real.obs[PERT_COL].astype(str))
    assert set(half_real.obs[PERT_COL].astype(str)) == set(
        half_pred.obs[PERT_COL].astype(str)
    )
    shutil.rmtree(OUTDIR)


def test_compute_ceiling_end_to_end():
    """compute_ceiling returns (results, agg): reliability metrics are SB-corrected
    and bounded in [0, 1]; error metrics come back as NaN."""
    adata_real = build_random_anndata()
    evaluator = MetricsEvaluator(
        adata_pred=adata_real.copy(),
        adata_real=adata_real,
        control_pert=CONTROL_VAR,
        pert_col=PERT_COL,
        outdir=OUTDIR,
        skip_de=True,
    )
    results, agg = evaluator.compute_ceiling(
        profile="anndata", write_csv=False, break_on_error=True
    )
    assert results.height > 0
    assert "perturbation" in results.columns

    assert "pearson_delta" in results.columns
    pv = results["pearson_delta"].drop_nulls().to_numpy()
    assert np.all(pv <= 1.0 + 1e-9)  # SB doubling can never exceed 1

    # error metrics are emitted as NaN (no defensible SB ceiling)
    for col in ("mse", "mae"):
        if col in results.columns:
            assert np.all(np.isnan(results[col].to_numpy()))
    # excluded reliability metrics are NaN too
    for col in ("clustering_agreement", "pearson_edistance"):
        if col in results.columns:
            assert np.all(np.isnan(results[col].to_numpy()))
    shutil.rmtree(OUTDIR)
