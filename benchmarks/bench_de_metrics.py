"""Microbenchmark for the per-perturbation-scan DE metrics.

`DENsigCounts` and `compute_generic_auc` (pr/roc) used to loop over every
perturbation doing a full-table `.filter(target == pert)` (or per-pert
`get_significant_genes`). They now slice the table once -- a grouped count for
`DENsigCounts`, a single `partition_by` for the AUC metrics. This script times
the pre-optimization implementations (verbatim baselines below) against the
current ones across a range of perturbation counts and confirms identical
output.

Run with the package importable, e.g.::

    python benchmarks/bench_de_metrics.py
    python benchmarks/bench_de_metrics.py --n-pert 1000 4000 8000 --n-genes 50

`_reference_densig_counts` / `_reference_generic_auc` are verbatim copies of the
pre-optimization implementations, kept here only as benchmark baselines; they
are not used by the package.
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import time

import numpy as np
import polars as pl
from sklearn.metrics import auc, average_precision_score, roc_curve

from cell_eval._types import DEComparison, DEResults
from cell_eval.metrics._de import DENsigCounts, compute_pr_auc, compute_roc_auc

logging.getLogger("cell_eval._types._de").setLevel(logging.WARNING)

_FDR = 0.05


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
        .with_columns(pred_q.alias(pred_fdr_col), (-pred_q.log10()).alias("nlp"))
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


def _make_side(n_pert: int, n_genes: int, sig_p: float, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_pert * n_genes
    target = np.repeat([f"P{i}" for i in range(n_pert)], n_genes)
    feature = np.tile([f"g{j}" for j in range(n_genes)], n_pert)
    is_sig = rng.random(n) < sig_p
    # Distinct fdr values -> distinct AUC scores; sig genes below threshold.
    fdr = np.where(is_sig, rng.uniform(1e-6, 0.049, n), rng.uniform(0.051, 1.0, n))
    lfc = rng.normal(0.0, 2.0, n)
    return pl.DataFrame(
        {
            "target": target,
            "feature": feature,
            "log2_fold_change": lfc,
            "p_value": fdr,
            "fdr": fdr,
        }
    )


def make_comparison(n_pert: int, n_genes: int, seed: int = 0) -> DEComparison:
    return DEComparison(
        real=DEResults(_make_side(n_pert, n_genes, 0.3, seed + 1), name="real"),
        pred=DEResults(_make_side(n_pert, n_genes, 0.3, seed + 2), name="pred"),
    )


def _auc_equal(a: dict, b: dict) -> bool:
    if list(a.keys()) != list(b.keys()):
        return False
    for k in a:
        av, bv = a[k], b[k]
        if isinstance(av, float) and math.isnan(av):
            if not (isinstance(bv, float) and math.isnan(bv)):
                return False
        elif av != bv:
            return False
    return True


def _timed(fn):
    gc.collect()
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-pert", type=int, nargs="+", default=[1000, 2000, 4000, 8000]
    )
    parser.add_argument("--n-genes", type=int, default=50)
    args = parser.parse_args()

    hdr = (
        f"{'n_pert':>7} | {'rows':>8} | "
        f"{'nsig old':>9} {'nsig new':>9} {'x':>5} | "
        f"{'pr old':>8} {'pr new':>8} {'x':>5} | "
        f"{'roc old':>8} {'roc new':>8} {'x':>5} | ok"
    )
    print(hdr)
    print("-" * len(hdr))

    for n_pert in args.n_pert:
        comp = make_comparison(n_pert, args.n_genes)
        rows = n_pert * args.n_genes

        old_nsig, t_on = _timed(lambda: _reference_densig_counts(comp, _FDR))
        new_nsig, t_nn = _timed(lambda: DENsigCounts(fdr_threshold=_FDR)(comp))
        old_pr, t_op = _timed(lambda: _reference_generic_auc(comp, "pr"))
        new_pr, t_np = _timed(lambda: compute_pr_auc(comp))
        old_roc, t_or = _timed(lambda: _reference_generic_auc(comp, "roc"))
        new_roc, t_nr = _timed(lambda: compute_roc_auc(comp))

        ok = (
            old_nsig == new_nsig
            and _auc_equal(old_pr, new_pr)
            and _auc_equal(old_roc, new_roc)
        )

        def sp(o: float, n: float) -> float:
            return o / n if n else float("inf")

        print(
            f"{n_pert:>7} | {rows:>8} | "
            f"{t_on:>9.3f} {t_nn:>9.3f} {sp(t_on, t_nn):>4.1f}x | "
            f"{t_op:>8.3f} {t_np:>8.3f} {sp(t_op, t_np):>4.1f}x | "
            f"{t_or:>8.3f} {t_nr:>8.3f} {sp(t_or, t_nr):>4.1f}x | {ok}"
        )


if __name__ == "__main__":
    main()
