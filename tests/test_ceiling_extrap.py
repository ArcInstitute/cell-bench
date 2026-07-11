import shutil

import numpy as np
import pytest

from cell_eval import MetricsEvaluator
from cell_eval._evaluator import _extrapolate_ceiling_curve, _extrapolate_metric
from cell_eval.data import CONTROL_VAR, PERT_COL, build_random_anndata

OUTDIR = "TEST_OUTPUT_EXTRAP"


def test_extrapolation_on_attenuation_curve():
    """A true reliability curve m(f)=f/(f+k) must extrapolate to its full-depth
    value 2/(2+k) (frac=2) with a clean attenuation fit."""
    k = 1.0
    fracs = np.array([1.0, 0.5, 0.25])
    values = fracs / (fracs + k)  # attenuation / reliability form

    out = _extrapolate_metric(fracs, values, target=2.0)
    expected = 2.0 / (2.0 + k)  # value at frac=2 (full depth)

    assert out["model"] == "attenuation"
    assert out["extrap"] == pytest.approx(expected, abs=1e-6)
    assert out["resid"] == pytest.approx(0.0, abs=1e-9)


def test_extrapolation_flat_curve_stays_flat():
    """A depth-independent (flat) metric must extrapolate to the same value."""
    fracs = np.array([1.0, 0.5, 0.25])
    values = np.array([0.5, 0.5, 0.5])

    out = _extrapolate_metric(fracs, values, target=2.0)
    extrap = out["extrap"]
    assert isinstance(extrap, float)
    assert extrap == pytest.approx(0.5, abs=1e-6)


def test_extrapolation_linear_fallback_outside_unit_range():
    """Metrics not in (0,1) (e.g. counts) can't use the attenuation form and
    fall back to a linear fit."""
    fracs = np.array([1.0, 0.5, 0.25])
    values = np.array([1000.0, 500.0, 250.0])  # count-like, linear in depth

    out = _extrapolate_metric(fracs, values, target=2.0)
    assert out["model"] == "linear"
    # linear through (0.25,250),(0.5,500),(1,1000) -> slope 1000 -> f=2 : 2000
    assert out["extrap"] == pytest.approx(2000.0, rel=1e-3)


def test_extrapolate_ceiling_curve_table_shape():
    curve = {
        1.0: {"pert_r": 0.5, "de_spearman_sig": 0.5},
        0.5: {"pert_r": 1 / 3, "de_spearman_sig": 0.5},
        0.25: {"pert_r": 0.2, "de_spearman_sig": 0.5},
    }
    table = _extrapolate_ceiling_curve(curve, target_frac=2.0)

    assert set(table["metric"]) == {"pert_r", "de_spearman_sig"}
    for col in ("metric", "m@1", "m@0.5", "m@0.25", "extrap", "resid", "model"):
        assert col in table.columns
    assert "sb" not in table.columns  # SB is a validation tool, not shipped output


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
    counts = evaluator.anndata_pair.real.obs[PERT_COL].value_counts()
    eligible = {str(p) for p in counts.index}
    half_real, half_pred = evaluator._disjoint_halves(
        seed=0, frac=1.0, eligible=eligible
    )

    # disjoint: no original cell (by name) appears in both halves
    assert set(half_real.obs_names).isdisjoint(set(half_pred.obs_names))
    # both halves carry the control + every eligible perturbation
    assert CONTROL_VAR in set(half_real.obs[PERT_COL].astype(str))
    assert set(half_real.obs[PERT_COL].astype(str)) == set(
        half_pred.obs[PERT_COL].astype(str)
    )
    shutil.rmtree(OUTDIR)
