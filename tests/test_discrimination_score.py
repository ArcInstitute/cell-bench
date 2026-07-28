"""Equivalence tests for the vectorized ``discrimination_score``.

The vectorized implementation must reproduce the original per-perturbation
loop exactly, including the target-gene-exclusion path where each perturbation
drops a different feature column. Because the output values are discrete ranks
(spaced ``1 / n_pert`` apart), identical rankings are asserted with
``array_equal``, not just ``allclose``.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import sklearn.metrics as skm

from cell_eval._types import PerturbationAnndataPair
from cell_eval.metrics._anndata import discrimination_score

CONTROL = "non-targeting"
PERT_COL = "perturbation"


def _reference_discrimination_score(
    data, metric="l1", embed_key=None, exclude_target_gene=True
):
    """Verbatim copy of the original per-perturbation loop implementation."""
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


def _make_pair(
    n_pert=60,
    n_genes=400,
    n_cells=1500,
    seed=0,
    embed_dim=16,
    frac_targeting=0.8,
    var_names=None,
):
    """A real/pred pair where a fraction of perts are named after gene columns."""
    rng = np.random.default_rng(seed)
    if var_names is None:
        var_names = np.array([f"gene_{i}" for i in range(n_genes)])
    n_genes = var_names.size
    n_targeting = min(int(round(n_pert * frac_targeting)), n_genes)
    pert_names = list(var_names[:n_targeting]) + [
        f"drug_{k}" for k in range(n_pert - n_targeting)
    ]
    all_labels = np.concatenate([np.asarray(pert_names), [CONTROL]])
    labels = rng.choice(all_labels, size=max(n_cells, all_labels.size))
    labels[: all_labels.size] = all_labels

    def build(off):
        r = np.random.default_rng(seed + 1000 * off)
        a = ad.AnnData(X=r.standard_normal((labels.size, n_genes)))
        a.obs[PERT_COL] = pd.Categorical(labels)
        a.var_names = var_names
        a.obs_names = [f"cell_{i}" for i in range(labels.size)]
        if embed_dim:
            a.obsm["X_emb"] = r.standard_normal((labels.size, embed_dim))
        return a

    return PerturbationAnndataPair(
        real=build(1), pred=build(2), pert_col=PERT_COL, control_pert=CONTROL
    )


def _ranks(out):
    return np.array([out[k] for k in sorted(out)])


@pytest.mark.parametrize("metric", ["l1", "l2", "cosine"])
@pytest.mark.parametrize("exclude_target_gene", [True, False])
@pytest.mark.parametrize("embed_key", [None, "X_emb"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_reference(metric, exclude_target_gene, embed_key, seed):
    data = _make_pair(seed=seed)
    ref = _reference_discrimination_score(
        data,
        metric=metric,
        embed_key=embed_key,
        exclude_target_gene=exclude_target_gene,
    )
    new = discrimination_score(
        data,
        metric=metric,
        embed_key=embed_key,
        exclude_target_gene=exclude_target_gene,
    )
    assert set(ref) == set(new)
    # Discrete ranks: identical ordering must yield identical values.
    np.testing.assert_array_equal(_ranks(ref), _ranks(new))


@pytest.mark.parametrize("metric", ["l1", "l2", "cosine"])
@pytest.mark.parametrize("frac_targeting", [0.0, 1.0])
def test_matches_reference_extreme_targeting(metric, frac_targeting):
    """No perts named after genes, and every pert named after a gene."""
    data = _make_pair(seed=3, frac_targeting=frac_targeting)
    ref = _reference_discrimination_score(data, metric=metric)
    new = discrimination_score(data, metric=metric)
    np.testing.assert_array_equal(_ranks(ref), _ranks(new))


@pytest.mark.parametrize("metric", ["chebyshev", "correlation"])
def test_exotic_metric_fallback(metric):
    """Metrics without a closed-form column correction take the exact fallback."""
    data = _make_pair(seed=4, frac_targeting=0.9)
    ref = _reference_discrimination_score(data, metric=metric)
    new = discrimination_score(data, metric=metric)
    np.testing.assert_array_equal(_ranks(ref), _ranks(new))


@pytest.mark.parametrize("metric", ["l1", "l2", "cosine"])
def test_duplicate_gene_name_safety_net(metric):
    """A perturbation matching two gene columns must drop both (multi-col net)."""
    var_names = np.array([f"gene_{i}" for i in range(40)])
    var_names[10] = "gene_5"  # "gene_5" now matches two columns
    data = _make_pair(
        n_pert=20,
        n_genes=40,
        n_cells=600,
        seed=5,
        embed_dim=0,
        frac_targeting=1.0,
        var_names=var_names,
    )
    ref = _reference_discrimination_score(data, metric=metric)
    new = discrimination_score(data, metric=metric)
    np.testing.assert_array_equal(_ranks(ref), _ranks(new))


def _make_target_dominated_pair(
    n_pert=30, n_genes=200, n_cells=1200, seed=7, spike=30.0
):
    """Pair where each perturbation's effect is concentrated in its own target
    gene, so the target-excluded (masked) vector is near-zero -- the degenerate
    case for cosine, where a masked squared norm can round negative."""
    rng = np.random.default_rng(seed)
    var_names = np.array([f"gene_{i}" for i in range(n_genes)])
    pert_names = list(var_names[:n_pert])  # every pert is named after a gene
    all_labels = np.concatenate([np.asarray(pert_names), [CONTROL]])
    labels = rng.choice(all_labels, size=max(n_cells, all_labels.size))
    labels[: all_labels.size] = all_labels
    col_of = {name: i for i, name in enumerate(var_names)}

    def build(off):
        r = np.random.default_rng(seed + 1000 * off)
        x = r.standard_normal((labels.size, n_genes))
        for j, lab in enumerate(labels):
            if lab in col_of:  # perturbed cell: spike its target gene column
                x[j, col_of[lab]] += spike
        a = ad.AnnData(X=x)
        a.obs[PERT_COL] = pd.Categorical(labels)
        a.var_names = var_names
        a.obs_names = [f"cell_{i}" for i in range(labels.size)]
        return a

    return PerturbationAnndataPair(
        real=build(1), pred=build(2), pert_col=PERT_COL, control_pert=CONTROL
    )


@pytest.mark.parametrize("metric", ["l1", "l2", "cosine"])
def test_target_gene_dominated_effects(metric):
    """Near-zero masked vectors must match the loop and never produce NaN."""
    data = _make_target_dominated_pair()
    ref = _reference_discrimination_score(data, metric=metric)
    new = discrimination_score(data, metric=metric)
    assert not np.isnan(_ranks(new)).any()
    np.testing.assert_array_equal(_ranks(ref), _ranks(new))
