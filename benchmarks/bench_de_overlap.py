"""Microbenchmark for the memoized DE overlap metric.

The DE ``full``/``de`` profile registers 10 overlap variants
(``{overlap, precision} x k in {None, 50, 100, 200, 500}``), and every one of
them calls ``DEComparison.compute_overlap`` -> ``DEResults.get_top_genes`` on the
same real/pred pair with the same default ``sort_by`` / ``fdr_threshold``. Before
the optimization each call rebuilt a polars ``.pivot()`` with one column per
perturbation, and the per-perturbation loop tested membership against
``matrix.columns`` (which rebuilds a fresh list on every access). At thousands of
perturbations that is 10 redundant wide pivots plus an O(n_pert^2) membership
loop.

This script runs the full 10-variant pattern with the pre-optimization
implementation (verbatim baseline below) and with the current memoized one,
across a range of perturbation counts, and confirms the two produce identical
results.

Run with the package importable, e.g.::

    python benchmarks/bench_de_overlap.py
    python benchmarks/bench_de_overlap.py --n-pert 1000 4000 8000

``_reference_get_top_genes`` / ``_reference_compute_overlap`` are verbatim copies
of the pre-optimization implementation, kept here only as a benchmark baseline;
they are not used by the package.
"""

from __future__ import annotations

import argparse
import gc
import logging
import time
from typing import Literal

import numpy as np
import polars as pl

from cell_eval._types import DEComparison, DEResults
from cell_eval._types._enums import DESortBy

# The pre-optimization metric logged INFO lines per DEResults construction; keep
# the benchmark output clean.
logging.getLogger("cell_eval._types._de").setLevel(logging.WARNING)

# Mirrors metrics/_impl.py registration.
VARIANTS: list[tuple[Literal["overlap", "precision"], int | None]] = [
    (metric, k)
    for metric in ("overlap", "precision")
    for k in (None, 50, 100, 200, 500)
]


def _reference_get_top_genes(
    de: DEResults,
    sort_by: DESortBy,
    fdr_threshold: float | None = None,
) -> pl.DataFrame:
    """Verbatim pre-optimization DEResults.get_top_genes (no memoization)."""
    fdr_threshold = fdr_threshold if fdr_threshold is not None else 0.05
    descending = sort_by in {
        DESortBy.LOG2_FOLD_CHANGE,
        DESortBy.ABS_LOG2_FOLD_CHANGE,
    }
    rank_matrix = (
        de.data.filter(pl.col(de.fdr_col) < fdr_threshold)
        .with_columns(
            rank=pl.struct(sort_by.value)
            .rank("ordinal", descending=descending)
            .over("target")
            - 1
        )
        .pivot(index="rank", on="target", values="feature")
        .sort("rank")
    )
    missing_perts = set(de.get_perts()) - set(rank_matrix.columns)
    if missing_perts:
        rank_matrix = rank_matrix.with_columns(
            [pl.lit(None).alias(p) for p in missing_perts]
        )
    return rank_matrix


def _reference_compute_overlap(
    comparison: DEComparison,
    k: int | None,
    metric: Literal["overlap", "precision"] = "overlap",
    fdr_threshold: float | None = None,
    sort_by: DESortBy = DESortBy.ABS_LOG2_FOLD_CHANGE,
) -> dict[str, float]:
    """Verbatim pre-optimization DEComparison.compute_overlap.

    Rebuilds both rank matrices on every call and tests membership against the
    polars ``.columns`` list (one fresh list per access).
    """
    real_sig_rank_matrix = _reference_get_top_genes(
        comparison.real, sort_by=sort_by, fdr_threshold=fdr_threshold
    )
    pred_sig_rank_matrix = _reference_get_top_genes(
        comparison.pred, sort_by=sort_by, fdr_threshold=fdr_threshold
    )

    if real_sig_rank_matrix.shape[0] == 0 or pred_sig_rank_matrix.shape[0] == 0:
        return {pert: 0.0 for pert in comparison.iter_perturbations()}

    overlaps = {}
    for pert in comparison.iter_perturbations():
        if (
            pert not in real_sig_rank_matrix.columns
            or pert not in pred_sig_rank_matrix.columns
        ):
            overlaps[pert] = 0.0
            continue

        real_genes = real_sig_rank_matrix[pert].drop_nulls().to_numpy()
        pred_genes = pred_sig_rank_matrix[pert].drop_nulls().to_numpy()

        if metric == "overlap":
            k_eff = real_genes.size if not k else k
            k_eff = min(k_eff, real_genes.size)
        elif metric == "precision":
            k_eff = pred_genes.size if not k else k
            k_eff = min(k_eff, pred_genes.size)
        else:
            raise ValueError(f"Invalid metric: {metric}")

        if k_eff == 0:
            overlaps[pert] = 0.0
        else:
            real_subset = real_genes[:k_eff]
            pred_subset = pred_genes[:k_eff]
            overlaps[pert] = np.intersect1d(real_subset, pred_subset).size / k_eff

    return overlaps


def _make_side(
    perts: np.ndarray, genes: np.ndarray, n_sig: int, rng: np.random.Generator
) -> pl.DataFrame:
    n_pert = perts.size
    targets = np.repeat(perts, n_sig)
    feats = np.empty(n_pert * n_sig, dtype=object)
    for i in range(n_pert):
        feats[i * n_sig : (i + 1) * n_sig] = rng.choice(
            genes, size=n_sig, replace=False
        )
    lfc = rng.normal(0.0, 2.0, size=n_pert * n_sig)
    fdr = np.full(n_pert * n_sig, 0.01)
    return pl.DataFrame(
        {
            "target": targets,
            "feature": feats,
            "log2_fold_change": lfc,
            "p_value": fdr,
            "fdr": fdr,
        }
    )


def make_frames(
    n_pert: int, n_genes: int = 2000, n_sig: int = 100, seed: int = 0
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Synthetic real/pred DE tables: every pert has `n_sig` significant genes."""
    rng = np.random.default_rng(seed)
    perts = np.array([f"P{i}" for i in range(n_pert)])
    genes = np.array([f"g{j}" for j in range(n_genes)])
    real = _make_side(perts, genes, n_sig, rng)
    pred = _make_side(perts, genes, n_sig, rng)
    return real, pred


def _run(comparison: DEComparison, fn) -> dict:
    return {(m, k): fn(comparison, k=k, metric=m) for (m, k) in VARIANTS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pert", type=int, nargs="+", default=[1000, 2000, 4000])
    parser.add_argument("--n-genes", type=int, default=2000)
    parser.add_argument("--n-sig", type=int, default=100)
    args = parser.parse_args()

    print(
        f"{'n_pert':>8} | {'old (s)':>10} | {'new (s)':>10} | "
        f"{'speedup':>8} | identical"
    )
    print("-" * 60)

    for n_pert in args.n_pert:
        real_df, pred_df = make_frames(n_pert, args.n_genes, args.n_sig)

        comp_old = DEComparison(
            real=DEResults(real_df, name="real"),
            pred=DEResults(pred_df, name="pred"),
        )
        comp_new = DEComparison(
            real=DEResults(real_df, name="real"),
            pred=DEResults(pred_df, name="pred"),
        )

        gc.collect()
        t0 = time.perf_counter()
        old_results = _run(comp_old, _reference_compute_overlap)
        t_old = time.perf_counter() - t0

        gc.collect()
        t0 = time.perf_counter()
        new_results = {
            (m, k): comp_new.compute_overlap(k=k, metric=m) for (m, k) in VARIANTS
        }
        t_new = time.perf_counter() - t0

        identical = all(old_results[v] == new_results[v] for v in VARIANTS)
        speedup = t_old / t_new if t_new else float("inf")
        print(
            f"{n_pert:>8} | {t_old:>10.3f} | {t_new:>10.3f} | "
            f"{speedup:>7.1f}x | {identical}"
        )


if __name__ == "__main__":
    main()
