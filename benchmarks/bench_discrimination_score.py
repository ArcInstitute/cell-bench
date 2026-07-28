"""Microbenchmark for the vectorized ``discrimination_score``.

Compares the original per-perturbation loop against the vectorized
implementation across a range of perturbation counts, and confirms the two
produce identical normalized ranks.

Run with the package importable, e.g.::

    python benchmarks/bench_discrimination_score.py
    python benchmarks/bench_discrimination_score.py --n-pert 100 1000 10000

The ``_reference_discrimination_score`` function below is a verbatim copy of
the pre-optimization implementation, kept here only as a benchmark baseline;
it is not used by the package.
"""

from __future__ import annotations

import argparse
import gc
import time

import anndata as ad
import numpy as np
import pandas as pd
import sklearn.metrics as skm

from cell_eval._types import PerturbationAnndataPair
from cell_eval.metrics._anndata import discrimination_score

CONTROL = "non-targeting"
PERT_COL = "perturbation"


def make_pair(
    n_pert: int,
    n_genes: int = 2000,
    n_cells: int = 5000,
    seed: int = 0,
    frac_targeting: float = 0.8,
) -> PerturbationAnndataPair:
    """Synthetic pair where `frac_targeting` of perts are named after genes."""
    rng = np.random.default_rng(seed)
    var_names = np.array([f"gene_{i}" for i in range(n_genes)])
    n_targeting = min(int(round(n_pert * frac_targeting)), n_genes)
    pert_names = list(var_names[:n_targeting]) + [
        f"drug_{k}" for k in range(n_pert - n_targeting)
    ]
    all_labels = np.concatenate([pert_names, [CONTROL]])
    labels = rng.choice(all_labels, size=max(n_cells, all_labels.size))
    labels[: all_labels.size] = all_labels

    def build(off: int) -> ad.AnnData:
        r = np.random.default_rng(seed + 1000 * off)
        a = ad.AnnData(X=r.standard_normal((labels.size, n_genes)))
        a.obs[PERT_COL] = pd.Categorical(labels)
        a.var_names = var_names
        a.obs_names = [f"cell_{i}" for i in range(labels.size)]
        return a

    return PerturbationAnndataPair(
        real=build(1), pred=build(2), pert_col=PERT_COL, control_pert=CONTROL
    )


def _reference_discrimination_score(
    data, metric="l1", embed_key=None, exclude_target_gene=True
):
    """Verbatim pre-optimization per-perturbation loop (benchmark baseline)."""
    if metric in ("l1", "manhattan", "cityblock"):
        embed_key = None
    real_effects = np.vstack(
        [
            d.perturbation_effect("real", abs=False)
            for d in data.iter_bulk_arrays(embed_key=embed_key)
        ]
    )
    pred_effects = np.vstack(
        [
            d.perturbation_effect("pred", abs=False)
            for d in data.iter_bulk_arrays(embed_key=embed_key)
        ]
    )
    norm_ranks = {}
    for p_idx, p in enumerate(data.perts):
        if exclude_target_gene and not embed_key:
            include_mask = np.flatnonzero(data.genes != p)
        else:
            include_mask = np.ones(real_effects.shape[1], dtype=bool)
        distances = skm.pairwise_distances(
            real_effects[:, include_mask],
            pred_effects[p_idx, include_mask].reshape(1, -1),
            metric=metric,
        ).flatten()
        sorted_indices = np.argsort(distances)
        p_index = np.flatnonzero(data.perts == p)[0]
        rank = np.flatnonzero(sorted_indices == p_index)[0]
        norm_ranks[str(p)] = 1 - rank / data.perts.size
    return norm_ranks


def _time(fn, *args, repeats=1, **kwargs):
    best = float("inf")
    out = None
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        best = min(best, time.perf_counter() - t0)
    return best, out


def bench_discrimination(sizes, metrics, n_genes, repeats, run_old):
    print("\n## discrimination_score: old (loop) vs new (vectorized)\n")
    # Warm up numba/sklearn/BLAS on a tiny input so the first measured call is
    # not penalized by one-time import/JIT cost.
    warm = make_pair(n_pert=50, n_genes=n_genes, n_cells=200, seed=99)
    for metric in metrics:
        discrimination_score(warm, metric=metric)
        if run_old:
            _reference_discrimination_score(warm, metric=metric)
    header = "| n_pert | metric | old (s) | new (s) | speedup | ranks identical |"
    print(header)
    print("|---|---|---|---|---|---|")
    for n_pert in sizes:
        data = make_pair(n_pert=n_pert, n_genes=n_genes, n_cells=3 * n_pert, seed=0)
        for metric in metrics:
            t_new, new_out = _time(
                discrimination_score, data, metric=metric, repeats=repeats
            )
            if run_old:
                t_old, old_out = _time(
                    _reference_discrimination_score,
                    data,
                    metric=metric,
                    repeats=repeats,
                )
                keys = sorted(old_out)
                identical = np.array_equal(
                    np.array([old_out[k] for k in keys]),
                    np.array([new_out[k] for k in keys]),
                )
                speed = f"{t_old / t_new:.1f}x"
                old_s = f"{t_old:.3f}"
            else:
                old_s, speed, identical = "(skipped)", "-", "-"
            print(
                f"| {n_pert} | {metric} | {old_s} | {t_new:.3f} | {speed} "
                f"| {identical} |"
            )
            del new_out
            gc.collect()


def print_env():
    """Print the machine + library versions so the run is reproducible."""
    import platform
    from importlib.metadata import version

    cpu = platform.processor() or "unknown"
    if platform.system() == "Darwin":
        import subprocess

        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            cpu = out or cpu
        except Exception:
            pass
    libs = ", ".join(f"{p} {version(p)}" for p in ("numpy", "scipy", "scikit-learn"))
    print("## environment\n")
    print(f"- platform: {platform.platform()}")
    print(f"- cpu: {cpu}")
    print(f"- python: {platform.python_version()}")
    print(f"- libs: {libs}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pert", type=int, nargs="+", default=[100, 1000, 10000])
    ap.add_argument("--metrics", nargs="+", default=["l1", "l2", "cosine"])
    ap.add_argument("--n-genes", type=int, default=2000)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument(
        "--no-old",
        action="store_true",
        help="skip the slow reference loop (new impl only)",
    )
    args = ap.parse_args()

    print_env()
    bench_discrimination(
        args.n_pert, args.metrics, args.n_genes, args.repeats, run_old=not args.no_old
    )


if __name__ == "__main__":
    main()
