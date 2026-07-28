"""Equivalence + memoization guards for the DE overlap metric.

`DEComparison.compute_overlap` memoizes the per-side rank-matrix pivot keyed by
`(sort_by, fdr_threshold)` and tests perturbation membership against precomputed
column sets. These are pure performance changes: the overlap/precision values
must stay bit-identical to a from-scratch reference, and the memoization must
collapse the repeated `get_top_genes` calls (one per registered overlap variant)
to a single pivot per side.
"""

import polars as pl

from cell_eval._types import DEComparison, DEResults
from cell_eval._types._enums import DESortBy

# (target, feature, log2_fold_change, p_value, fdr). Within each perturbation the
# significant genes have distinct |log2_fold_change|, so the descending sort is
# unambiguous and the reference does not depend on polars' tie handling.
_REAL_ROWS = [
    ("A", "g1", 3.0, 0.001, 0.01),
    ("A", "g2", 2.0, 0.002, 0.02),
    ("A", "g3", 1.0, 0.004, 0.04),
    ("A", "g4", 0.5, 0.090, 0.10),  # not significant
    ("B", "g1", -2.5, 0.001, 0.01),
    ("B", "g2", 1.5, 0.003, 0.03),
    ("B", "g5", 0.8, 0.150, 0.20),  # not significant
    ("C", "g3", 2.2, 0.250, 0.30),  # not significant
    ("C", "g4", 1.1, 0.350, 0.40),  # not significant
]

_PRED_ROWS = [
    ("A", "g1", 2.8, 0.001, 0.01),
    ("A", "g3", 2.5, 0.002, 0.02),
    ("A", "g2", 1.0, 0.004, 0.04),
    ("A", "g4", 0.9, 0.003, 0.03),  # significant in pred only
    ("B", "g1", -2.0, 0.002, 0.02),
    ("B", "g2", 0.5, 0.400, 0.50),  # not significant
    ("B", "g5", 1.2, 0.001, 0.01),  # significant in pred only
    ("C", "g3", 1.9, 0.002, 0.02),  # significant in pred only
    ("C", "g4", 0.3, 0.400, 0.50),  # not significant
]

_FDR = 0.05


def _rows_to_df(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "target": [r[0] for r in rows],
            "feature": [r[1] for r in rows],
            "log2_fold_change": [r[2] for r in rows],
            "p_value": [r[3] for r in rows],
            "fdr": [r[4] for r in rows],
        }
    )


def _ordered_sig_genes(rows: list[tuple], fdr_threshold: float) -> dict[str, list[str]]:
    """Reference: per-pert features, FDR-filtered, sorted by |lfc| descending."""
    by_pert: dict[str, list[tuple[float, str]]] = {}
    for target, feature, lfc, _p, fdr in rows:
        by_pert.setdefault(target, [])
        if fdr < fdr_threshold:
            by_pert[target].append((abs(lfc), feature))
    return {
        pert: [g for _, g in sorted(items, key=lambda t: t[0], reverse=True)]
        for pert, items in by_pert.items()
    }


def _ref_overlap(
    real_order: dict[str, list[str]],
    pred_order: dict[str, list[str]],
    perts: list[str],
    k: int | None,
    metric: str,
) -> dict[str, float]:
    """Reference overlap/precision mirroring compute_overlap's exact k_eff math."""
    out: dict[str, float] = {}
    for pert in perts:
        real_genes = real_order.get(pert, [])
        pred_genes = pred_order.get(pert, [])
        if metric == "overlap":
            k_eff = len(real_genes) if not k else k
            k_eff = min(k_eff, len(real_genes))
        else:  # precision
            k_eff = len(pred_genes) if not k else k
            k_eff = min(k_eff, len(pred_genes))
        if k_eff == 0:
            out[pert] = 0.0
        else:
            inter = set(real_genes[:k_eff]) & set(pred_genes[:k_eff])
            out[pert] = len(inter) / k_eff
    return out


def _make_comparison() -> DEComparison:
    return DEComparison(
        real=DEResults(_rows_to_df(_REAL_ROWS), name="real"),
        pred=DEResults(_rows_to_df(_PRED_ROWS), name="pred"),
    )


def test_compute_overlap_matches_reference() -> None:
    """Optimized compute_overlap is bit-identical to a from-scratch reference."""
    comparison = _make_comparison()
    perts = list(comparison.get_perts())

    real_order = _ordered_sig_genes(_REAL_ROWS, _FDR)
    pred_order = _ordered_sig_genes(_PRED_ROWS, _FDR)

    for metric in ("overlap", "precision"):
        for k in (None, 1, 2, 50, 500):
            got = comparison.compute_overlap(k=k, metric=metric, fdr_threshold=_FDR)
            expected = _ref_overlap(real_order, pred_order, perts, k, metric)
            assert got == expected, f"metric={metric} k={k}: {got} != {expected}"


def test_get_top_genes_memoized_across_variants() -> None:
    """The 10 overlap variants must reuse one pivot per side, not rebuild each."""
    comparison = _make_comparison()

    # Mirror metrics/_impl.py: {overlap, precision} x {None, 50, 100, 200, 500}
    # all hit the same default (sort_by, fdr_threshold).
    for metric in ("overlap", "precision"):
        for k in (None, 50, 100, 200, 500):
            comparison.compute_overlap(k=k, metric=metric, fdr_threshold=_FDR)

    assert len(comparison.real._top_genes_cache) == 1
    assert len(comparison.pred._top_genes_cache) == 1


def test_get_top_genes_cache_distinguishes_keys() -> None:
    """Different (sort_by, fdr_threshold) keys must produce separate entries."""
    comparison = _make_comparison()

    comparison.compute_overlap(
        k=None,
        metric="overlap",
        fdr_threshold=_FDR,
        sort_by=DESortBy.ABS_LOG2_FOLD_CHANGE,
    )
    comparison.compute_overlap(
        k=None,
        metric="overlap",
        fdr_threshold=0.10,
        sort_by=DESortBy.ABS_LOG2_FOLD_CHANGE,
    )
    comparison.compute_overlap(
        k=None, metric="overlap", fdr_threshold=_FDR, sort_by=DESortBy.PVALUE
    )

    # 0.05/abs, 0.10/abs, 0.05/pvalue -> three distinct cache keys.
    assert len(comparison.real._top_genes_cache) == 3


def test_compute_overlap_no_significant_genes_all_zero() -> None:
    """Early-return branch: when one side has no significant genes, all 0.0."""
    rows = [
        ("A", "g1", 3.0, 0.5, 0.5),
        ("A", "g2", 2.0, 0.6, 0.6),
        ("B", "g1", 1.0, 0.7, 0.7),
    ]
    comparison = DEComparison(
        real=DEResults(_rows_to_df(rows), name="real"),
        pred=DEResults(_rows_to_df(rows), name="pred"),
    )
    got = comparison.compute_overlap(k=None, metric="overlap", fdr_threshold=_FDR)
    assert got == {"A": 0.0, "B": 0.0}
