import shutil

import numpy as np
import polars as pl
import pytest

from cell_eval import MetricsEvaluator
from cell_eval._evaluator import SB_METRICS, _spearman_brown_correct
from cell_eval.data import CONTROL_VAR, PERT_COL, build_random_anndata

OUTDIR = "TEST_OUTPUT_CEILING"


def test_spearman_brown_doubling_on_reliability_metrics():
    """SB doubling r' = 2r/(1+r) is applied to the reliability metrics named in the
    explicit SB_METRICS list."""
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


def test_non_positive_reliability_is_nan():
    """2r/(1+r) is a pole for r <= 0, not a correction: r = -0.9 gives -18.0 and
    r = -1 divides by zero (a silent -inf in polars). A non-positive split-half
    reliability means the halves do not agree, so there is no defensible ceiling
    -> NaN, and never a negative "ceiling" or an inf."""
    df = pl.DataFrame(
        {
            "perturbation": ["a", "b", "c", "d", "e"],
            # r = -1 would divide by zero; -0.9 -> -18.0; -0.5 -> -2.0;
            # 0.0 is the boundary (the guard is > 0, not >= 0); 0.5 still works.
            "pearson_delta": [-1.0, -0.9, -0.5, 0.0, 0.5],
        }
    )
    out = _spearman_brown_correct(df)["pearson_delta"].to_list()
    assert all(np.isnan(v) for v in out[:4]), out
    assert out[4] == pytest.approx(2 / 3, abs=1e-6)  # r > 0 still corrected

    # a null mean (metric produced no value) also falls through to NaN
    null_df = pl.DataFrame(
        {"perturbation": ["a"], "pearson_delta": [None]},
        schema={"perturbation": pl.Utf8, "pearson_delta": pl.Float64},
    )
    assert np.isnan(_spearman_brown_correct(null_df)["pearson_delta"][0])


def test_disjoint_halves_requires_two_cells_and_a_control():
    """The split fails loudly rather than with a bare numpy/pipeline error when
    nothing can be split, or when the control specifically cannot be."""
    adata_real = build_random_anndata()

    def _evaluator(adata):
        return MetricsEvaluator(
            adata_pred=adata.copy(),
            adata_real=adata,
            control_pert=CONTROL_VAR,
            pert_col=PERT_COL,
            outdir=OUTDIR,
            skip_de=True,
        )

    # one cell per perturbation -> nothing is splittable
    obs = adata_real.obs
    first_of_each = [
        int(np.flatnonzero((obs[PERT_COL].astype(str) == p).to_numpy())[0])
        for p in obs[PERT_COL].astype(str).unique()
    ]
    with pytest.raises(ValueError, match="no perturbation has >= 2 cells"):
        _evaluator(adata_real[first_of_each].copy())._disjoint_halves(seed=0)

    # control has a single cell, other perturbations are fine
    is_ctrl = (obs[PERT_COL].astype(str) == CONTROL_VAR).to_numpy()
    keep = np.concatenate([np.flatnonzero(is_ctrl)[:1], np.flatnonzero(~is_ctrl)])
    with pytest.raises(ValueError, match="cannot compute a disjoint-split"):
        _evaluator(adata_real[np.sort(keep)].copy())._disjoint_halves(seed=0)

    shutil.rmtree(OUTDIR, ignore_errors=True)


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
    """compute_ceiling returns (results, agg): `results` is the raw per-perturbation
    self-split; `agg` is the SB-corrected per-context ceiling (one row) - reliability
    metrics bounded in [0, 1], error / excluded metrics NaN."""
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
    # per-perturbation self-split measurements
    assert results.height > 0
    assert "perturbation" in results.columns

    # the ceiling is the SB-corrected aggregate (mean over perturbations) - one row
    assert agg.height == 1
    assert "pearson_delta" in agg.columns
    # Two-sided: after the r > 0 guard an SB ceiling is either NaN (non-positive
    # reliability, no defensible ceiling) or in (0, 1] - doubling an r in (0, 1] can
    # neither exceed 1 nor come out negative. A one-sided `<= 1.0` would silently
    # admit a blown-up pole value (r = -0.9 -> -18.0, r = -1 -> -inf).
    for col in agg.columns:
        if col in SB_METRICS:
            v = agg[col].to_numpy()
            assert np.all(np.isnan(v) | ((v > 0.0) & (v <= 1.0 + 1e-9))), col

    # error metrics and excluded reliabilities have no ceiling -> NaN in the aggregate
    for col in ("mse", "mae", "clustering_agreement", "pearson_edistance"):
        if col in agg.columns:
            assert np.all(np.isnan(agg[col].to_numpy()))
    shutil.rmtree(OUTDIR)


def _ceiling_only_evaluator(adata_real, **kwargs):
    return MetricsEvaluator(
        adata_pred=None,
        adata_real=adata_real,
        control_pert=CONTROL_VAR,
        pert_col=PERT_COL,
        outdir=OUTDIR,
        **kwargs,
    )


def test_ceiling_only_mode_shape():
    """adata_pred=None is ceiling-only: the main DE comparison is skipped, the pair's
    pred side is the real object itself (a placeholder that is never scored), and
    compute() refuses rather than silently scoring real against itself."""
    adata_real = build_random_anndata()
    evaluator = _ceiling_only_evaluator(adata_real)

    assert evaluator.ceiling_only
    assert evaluator.de_comparison is None  # main comparison skipped
    # documented invariant: the placeholder aliases real, it is not a copy
    assert evaluator.anndata_pair.real is evaluator.anndata_pair.pred

    with pytest.raises(ValueError, match="ceiling-only mode"):
        evaluator.compute(profile="anndata", write_csv=False)

    shutil.rmtree(OUTDIR, ignore_errors=True)


def test_ceiling_only_matches_ceiling_with_a_prediction():
    """The ceiling is a property of the real data alone, so supplying a prediction
    must not change it: the same seed yields the same ceiling either way."""
    adata_real = build_random_anndata()

    with_pred = MetricsEvaluator(
        adata_pred=adata_real.copy(),
        adata_real=adata_real,
        control_pert=CONTROL_VAR,
        pert_col=PERT_COL,
        outdir=OUTDIR,
        skip_de=True,
    )
    _, agg_pred = with_pred.compute_ceiling(
        profile="anndata", write_csv=False, break_on_error=True, seed=0
    )

    ceiling_only = _ceiling_only_evaluator(adata_real, skip_de=True)
    _, agg_only = ceiling_only.compute_ceiling(
        profile="anndata", write_csv=False, break_on_error=True, seed=0
    )

    assert agg_only.columns == agg_pred.columns
    for col in agg_pred.columns:
        a, b = agg_pred[col].to_numpy(), agg_only[col].to_numpy()
        assert np.all(np.isnan(a) == np.isnan(b)), col
        mask = ~np.isnan(a)
        assert np.allclose(a[mask], b[mask], rtol=1e-12, atol=0.0), col

    shutil.rmtree(OUTDIR, ignore_errors=True)


def test_cli_rejects_missing_prediction_without_ceiling():
    """Omitting --adata-pred without --ceiling leaves nothing to compute. That is a
    usage error, so it exits 2 with a message rather than raising a traceback."""
    import argparse

    from cell_eval._cli._run import run_evaluation

    args = argparse.Namespace(
        adata_pred=None,
        ceiling=False,
        embed_key=None,
        skip_metrics=None,
        num_threads=1,
    )
    with pytest.raises(SystemExit) as exc:
        run_evaluation(args)
    assert exc.value.code == 2


def test_ceiling_only_warns_that_precomputed_de_is_unused(caplog):
    """Precomputed DE cannot be reused in ceiling-only mode (the ceiling runs DE on
    its own halves), so it warns for both sides rather than failing or going quiet."""
    adata_real = build_random_anndata()
    de = pl.DataFrame({"target": ["a"], "feature": ["g"], "p_value": [0.5]})

    with caplog.at_level("WARNING"):
        _ceiling_only_evaluator(adata_real, de_pred=de, de_real=de, skip_de=True)

    warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any("de_pred is ignored in ceiling-only mode" in m for m in warnings)
    assert any("de_real is ignored in ceiling-only mode" in m for m in warnings)

    shutil.rmtree(OUTDIR, ignore_errors=True)
