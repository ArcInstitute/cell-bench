"""Bit-exact equivalence for the per-perturbation-scan DE metrics.

`DENsigCounts` and `compute_generic_auc` (pr/roc) used to loop over every
perturbation doing a full-table `.filter(target == pert)` (or per-pert
`get_significant_genes`). They now slice the table once -- a grouped count for
`DENsigCounts`, a single `partition_by` for the AUC metrics. These are pure
performance changes: the numeric output must be identical to the pre-optimization
implementations, which are reproduced verbatim below as references.
"""

import math

import numpy as np
import polars as pl
from sklearn.metrics import auc, average_precision_score, roc_curve

from cell_eval._types import DEComparison, DEResults
from cell_eval.metrics._de import DENsigCounts, compute_pr_auc, compute_roc_auc

_FDR = 0.05

# Genes g0..g5 for every pert. real fdr is chosen so P0/P1 have a label mix
# (real AUC), P2 is all-significant and P3 all-non-significant (both -> nan);
# pred fdr varies the scores and the per-side significant counts.
_GENES = [f"g{i}" for i in range(6)]
_REAL_FDR = {
    "P0": [0.01, 0.02, 0.60, 0.70, 0.80, 0.90],
    "P1": [0.01, 0.01, 0.01, 0.60, 0.70, 0.80],
    "P2": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    "P3": [0.50, 0.60, 0.70, 0.80, 0.90, 0.95],
}
_PRED_FDR = {
    "P0": [0.02, 0.50, 0.04, 0.90, 0.10, 0.80],
    "P1": [0.01, 0.30, 0.02, 0.40, 0.20, 0.60],
    "P2": [0.30, 0.01, 0.50, 0.02, 0.70, 0.03],
    "P3": [0.04, 0.50, 0.60, 0.01, 0.70, 0.80],
}


def _frame(fdr_by_pert: dict[str, list[float]]) -> pl.DataFrame:
    target, feature, lfc, fdr = [], [], [], []
    for pert, fdrs in fdr_by_pert.items():
        for gi, (gene, f) in enumerate(zip(_GENES, fdrs)):
            target.append(pert)
            feature.append(gene)
            # Deterministic nonzero lfc; unused by these metrics but required.
            lfc.append((gi + 1) * (1.0 if gi % 2 == 0 else -1.0))
            fdr.append(f)
    return pl.DataFrame(
        {
            "target": target,
            "feature": feature,
            "log2_fold_change": lfc,
            "p_value": fdr,
            "fdr": fdr,
        }
    )


def _make_comparison() -> DEComparison:
    return DEComparison(
        real=DEResults(_frame(_REAL_FDR), name="real"),
        pred=DEResults(_frame(_PRED_FDR), name="pred"),
    )


def _reference_densig_counts(
    data: DEComparison, fdr_threshold: float
) -> dict[str, dict[str, int]]:
    """Verbatim pre-optimization DENsigCounts.__call__."""
    counts = {}
    for pert in data.iter_perturbations():
        real_sig = data.real.get_significant_genes(pert, fdr_threshold)
        pred_sig = data.pred.get_significant_genes(pert, fdr_threshold)
        counts[pert] = {"real": int(real_sig.size), "pred": int(pred_sig.size)}
    return counts


def _reference_generic_auc(data: DEComparison, method: str) -> dict[str, float]:
    """Verbatim pre-optimization compute_generic_auc (per-pert .filter loop)."""
    target_col = data.real.target_col
    feature_col = data.real.feature_col
    real_fdr_col = data.real.fdr_col
    pred_fdr_col = data.pred.fdr_col

    labeled_real = data.real.data.with_columns(
        (pl.col(real_fdr_col) < 0.05).cast(pl.Float32).alias("label")
    ).select([target_col, feature_col, "label"])

    pred_q = pl.col(pred_fdr_col).fill_null(1.0).clip(1e-10, 1.0)
    merged = (
        labeled_real.join(
            data.pred.data.select([target_col, feature_col, pred_fdr_col]),
            on=[target_col, feature_col],
            how="left",
            coalesce=True,
        )
        .drop_nulls(["label"])
        .with_columns(
            pred_q.alias(pred_fdr_col),
            (-pred_q.log10()).alias("nlp"),
        )
    )

    results: dict[str, float] = {}
    for pert in data.iter_perturbations():
        pert_data = merged.filter(pl.col(target_col) == pert)
        if pert_data.shape[0] == 0:
            results[pert] = float("nan")
            continue
        labels = pert_data["label"].to_numpy()
        scores = pert_data["nlp"].to_numpy()
        if not (0 < labels.sum() < len(labels)):
            results[pert] = float("nan")
            continue
        match method:
            case "pr":
                results[pert] = float(average_precision_score(labels, scores))
            case "roc":
                fpr, tpr, _ = roc_curve(labels, scores)
                results[pert] = float(auc(fpr, tpr))
            case _:
                raise ValueError(f"Invalid AUC method: {method}")
    return results


def _assert_auc_equal(got: dict, expected: dict) -> None:
    assert list(got.keys()) == list(expected.keys())
    for k in expected:
        gv, ev = got[k], expected[k]
        if isinstance(ev, float) and math.isnan(ev):
            assert isinstance(gv, float) and math.isnan(gv), (k, gv)
        else:
            assert gv == ev, (k, gv, ev)


def test_densig_counts_matches_reference() -> None:
    comparison = _make_comparison()
    got = DENsigCounts(fdr_threshold=_FDR)(comparison)
    expected = _reference_densig_counts(comparison, _FDR)
    assert got == expected
    # Sanity: P3 has no significant real genes -> count 0 (reindex fill path).
    assert got[np.str_("P3")]["real"] == 0
    assert got[np.str_("P2")]["real"] == 6


def test_pr_auc_matches_reference() -> None:
    comparison = _make_comparison()
    got = compute_pr_auc(comparison)
    expected = _reference_generic_auc(comparison, "pr")
    _assert_auc_equal(got, expected)
    # Sanity: degenerate label sets -> nan; mixed -> finite.
    assert math.isnan(got[np.str_("P2")])
    assert math.isnan(got[np.str_("P3")])
    assert math.isfinite(got[np.str_("P0")])


def test_roc_auc_matches_reference() -> None:
    comparison = _make_comparison()
    got = compute_roc_auc(comparison)
    expected = _reference_generic_auc(comparison, "roc")
    _assert_auc_equal(got, expected)
    assert math.isfinite(got[np.str_("P1")])
