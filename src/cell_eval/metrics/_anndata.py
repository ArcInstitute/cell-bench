"""Array metrics module."""

from logging import getLogger
from typing import Any, Callable, Literal, Sequence, cast

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import sklearn.metrics as skm
from scipy.sparse import issparse
from scipy.stats import pearsonr
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from .._types import PerturbationAnndataPair

logger = getLogger(__name__)


def pearson_delta(
    data: PerturbationAnndataPair, embed_key: str | None = None
) -> dict[str, float]:
    """Compute Pearson correlation between mean differences from control."""
    return _generic_evaluation(
        data,
        pearsonr,
        use_delta=True,
        embed_key=embed_key,
    )


def mse(
    data: PerturbationAnndataPair, embed_key: str | None = None
) -> dict[str, float]:
    """Compute mean squared error of each perturbation from control."""
    return _generic_evaluation(
        data, skm.mean_squared_error, use_delta=False, embed_key=embed_key
    )


def mae(
    data: PerturbationAnndataPair, embed_key: str | None = None
) -> dict[str, float]:
    """Compute mean absolute error of each perturbation from control."""
    return _generic_evaluation(
        data, skm.mean_absolute_error, use_delta=False, embed_key=embed_key
    )


def mse_delta(
    data: PerturbationAnndataPair, embed_key: str | None = None
) -> dict[str, float]:
    """Compute mean squared error of each perturbation-control delta."""
    return _generic_evaluation(
        data, skm.mean_squared_error, use_delta=True, embed_key=embed_key
    )


def mae_delta(
    data: PerturbationAnndataPair, embed_key: str | None = None
) -> dict[str, float]:
    """Compute mean absolute error of each perturbation-control delta."""
    return _generic_evaluation(
        data, skm.mean_absolute_error, use_delta=True, embed_key=embed_key
    )


def edistance(
    data: PerturbationAnndataPair,
    embed_key: str | None = None,
    metric: str = "euclidean",
    **kwargs,
) -> float:
    """Compute Euclidean distance of each perturbation-control delta."""

    def _edistance(
        x: np.ndarray,
        y: np.ndarray,
        metric: str = "euclidean",
        precomp_sigma_y: float | None = None,
        **kwargs,
    ) -> float:
        sigma_x = skm.pairwise_distances(x, metric=metric, **kwargs).mean()
        sigma_y = (
            precomp_sigma_y
            if precomp_sigma_y is not None
            else skm.pairwise_distances(y, metric=metric, **kwargs).mean()
        )
        delta = skm.pairwise_distances(x, y, metric=metric, **kwargs).mean()
        return 2 * delta - sigma_x - sigma_y

    d_real = np.zeros(data.perts.size)
    d_pred = np.zeros(data.perts.size)

    # Precompute sigma for control data (reused by all perturbations)
    logger.info("Precomputing sigma for control data (real)")
    precomp_sigma_real = skm.pairwise_distances(
        data.ctrl_matrix(which="real", embed_key=embed_key), metric=metric, **kwargs
    ).mean()

    logger.info("Precomputing sigma for control data (pred)")
    precomp_sigma_pred = skm.pairwise_distances(
        data.ctrl_matrix(which="pred", embed_key=embed_key), metric=metric, **kwargs
    ).mean()

    for idx, delta in enumerate(data.iter_cell_arrays(embed_key=embed_key)):
        d_real[idx] = _edistance(
            delta.pert_real,
            delta.ctrl_real,
            precomp_sigma_y=precomp_sigma_real,
            metric=metric,
            **kwargs,
        )
        d_pred[idx] = _edistance(
            delta.pert_pred,
            delta.ctrl_pred,
            precomp_sigma_y=precomp_sigma_pred,
            metric=metric,
            **kwargs,
        )

    return pearsonr(d_real, d_pred).correlation


def discrimination_score(
    data: PerturbationAnndataPair,
    metric: str = "l1",
    embed_key: str | None = None,
    exclude_target_gene: bool = True,
) -> dict[str, float]:
    """Base implementation for discrimination score computation.

    Best score is 1.0 - worst score is 0.0.

    Args:
        data: PerturbationAnndataPair containing real and predicted data
        embed_key: Key for embedding data in obsm, None for expression data
        metric: Metric for distance calculation (e.g., "l1", "l2", see `scipy.metrics.pairwise.distance_metrics`)
        exclude_target_gene: Whether to exclude target gene from calculation

    Returns:
        Dictionary mapping perturbation names to normalized ranks
    """
    if metric == "l1" or metric == "manhattan" or metric == "cityblock":
        # Ignore the embedding key for L1
        embed_key = None

    # Compute perturbation effects for all perturbations. The underlying
    # pseudobulk is memoized on the pair, so this is shared across metrics.
    real_effects = np.vstack(
        [
            d.perturbation_effect(which="real", abs=False)
            for d in data.iter_bulk_arrays(embed_key=embed_key)
        ]
    )
    pred_effects = np.vstack(
        [
            d.perturbation_effect(which="pred", abs=False)
            for d in data.iter_bulk_arrays(embed_key=embed_key)
        ]
    )

    # dist_matrix[i, j] = distance(pred_effect[i], real_effect[j]); each row is
    # the vector of distances from one predicted effect to all real effects.
    #
    # When excluding the target gene on expression data, perturbation i drops a
    # *different* feature column (the gene named like perturbation i), so a
    # single unmasked pairwise call would not reproduce the per-perturbation
    # masked distances. We instead compute the full matrix once and apply an
    # exact, vectorized rank-1 correction that removes the target gene's
    # contribution from each row. For metrics without a closed-form column
    # correction we fall back to exact per-row masked distances.
    do_exclude = exclude_target_gene and not embed_key
    family = _distance_family(metric)

    if not do_exclude:
        dist_matrix = skm.pairwise_distances(pred_effects, real_effects, metric=metric)
    elif family is None:
        dist_matrix = _masked_distance_matrix(
            real_effects, pred_effects, data.genes, data.perts, metric
        )
    else:
        # `family` is narrowed to Literal["l1", "l2", "cosine"] here.
        dist_matrix = _excluded_distance_matrix(
            real_effects,
            pred_effects,
            data.genes,
            data.perts,
            family,
        )

    # Rank of the matching perturbation within each row, by ascending distance.
    # order[i] lists columns by increasing distance, so the rank of perturbation
    # i is the position of column i within row i. A boolean match locates that
    # position directly -- equivalent to argsort(argsort(.)) but without a
    # second full-matrix sort (cheaper, and the mask is bool rather than int).
    n_pert = data.perts.size
    order = np.argsort(dist_matrix, axis=1)
    ranks = np.where(order == np.arange(n_pert)[:, None])[1]

    return {str(p): float(1 - ranks[i] / n_pert) for i, p in enumerate(data.perts)}


def _distance_family(metric: str) -> Literal["l1", "l2", "cosine"] | None:
    """Map a pairwise metric name onto a family with a closed-form column drop."""
    match metric.lower():
        case "l1" | "manhattan" | "cityblock":
            return "l1"
        case "l2" | "euclidean":
            return "l2"
        case "cosine":
            return "cosine"
        case _:
            return None


def _target_gene_columns(genes: np.ndarray, perts: np.ndarray) -> list[list[int]]:
    """Feature columns whose gene name matches each perturbation (usually 0 or 1)."""
    gene_to_cols: dict[str, list[int]] = {}
    for col, g in enumerate(genes):
        gene_to_cols.setdefault(str(g), []).append(col)
    return [gene_to_cols.get(str(p), []) for p in perts]


def _masked_row(
    real_effects: np.ndarray,
    pred_row: np.ndarray,
    excluded_cols: Sequence[int],
    metric: str,
) -> np.ndarray:
    """Distances from one predicted effect to all real effects, dropping columns."""
    if excluded_cols:
        mask = np.ones(real_effects.shape[1], dtype=bool)
        mask[list(excluded_cols)] = False
        return skm.pairwise_distances(
            real_effects[:, mask], pred_row[mask].reshape(1, -1), metric=metric
        ).flatten()
    return skm.pairwise_distances(
        real_effects, pred_row.reshape(1, -1), metric=metric
    ).flatten()


def _masked_distance_matrix(
    real_effects: np.ndarray,
    pred_effects: np.ndarray,
    genes: np.ndarray,
    perts: np.ndarray,
    metric: str,
) -> np.ndarray:
    """Per-row masked distance matrix for metrics without a column correction."""
    excluded = _target_gene_columns(genes, perts)
    return np.vstack(
        [
            _masked_row(real_effects, pred_effects[i], excluded[i], metric)
            for i in range(perts.size)
        ]
    )


def _excluded_distance_matrix(
    real_effects: np.ndarray,
    pred_effects: np.ndarray,
    genes: np.ndarray,
    perts: np.ndarray,
    family: Literal["l1", "l2", "cosine"],
) -> np.ndarray:
    """Full distance matrix with each row's target gene contribution removed.

    Row i corresponds to perturbation i; the feature column named like
    perturbation i is dropped from that row's distances only. The result is
    numerically equivalent (up to floating-point summation order) to computing
    `pairwise_distances` on the masked columns per perturbation.
    """
    n_pert = perts.size
    excluded = _target_gene_columns(genes, perts)

    has_target = np.zeros(n_pert, dtype=bool)
    tcol = np.zeros(n_pert, dtype=np.intp)
    multi: list[int] = []
    for i, cols in enumerate(excluded):
        if len(cols) == 1:
            has_target[i] = True
            tcol[i] = cols[0]
        elif len(cols) > 1:
            multi.append(i)

    rows = np.arange(n_pert)
    mask2d = has_target[:, None]
    # pred_at[i]    = pred_effects[i, tcol[i]]
    # real_at[i, j] = real_effects[j, tcol[i]]
    pred_at = pred_effects[rows, tcol]
    real_at = real_effects[:, tcol].T

    match family:
        case "l1":
            out = skm.pairwise_distances(pred_effects, real_effects, metric="l1")
            out -= np.where(mask2d, np.abs(pred_at[:, None] - real_at), 0.0)
        case "l2":
            out = skm.pairwise_distances(pred_effects, real_effects, metric="l2")
            corr = np.where(mask2d, (pred_at[:, None] - real_at) ** 2, 0.0)
            out = np.sqrt(np.maximum(out**2 - corr, 0.0))
        case _:  # cosine: drop the column from the dot product and both norms
            dot = pred_effects @ real_effects.T
            pred_sq = np.einsum("ij,ij->i", pred_effects, pred_effects)
            real_sq = np.einsum("ij,ij->i", real_effects, real_effects)
            dot -= np.where(mask2d, pred_at[:, None] * real_at, 0.0)
            pred_sq_m = pred_sq - np.where(has_target, pred_at**2, 0.0)
            real_sq_m = real_sq[None, :] - np.where(mask2d, real_at**2, 0.0)
            # An effect dominated by its target gene can leave a masked squared
            # norm at a tiny negative value from float rounding; clip to 0 so the
            # norm is real (not NaN). The resulting zero norm is then handled
            # like sklearn below (cosine similarity 0 -> distance 1).
            denom = np.sqrt(np.maximum(pred_sq_m, 0.0))[:, None] * np.sqrt(
                np.maximum(real_sq_m, 0.0)
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                cos = dot / denom
            cos = np.where(denom == 0.0, 0.0, cos)
            out = np.clip(1 - cos, 0.0, 2.0)

    # Safety net for the rare case of duplicate gene names matching a single
    # perturbation (more than one column to drop): recompute those rows exactly.
    for i in multi:
        out[i] = _masked_row(real_effects, pred_effects[i], excluded[i], family)

    return out


def _generic_evaluation(
    data: PerturbationAnndataPair,
    func: Callable[[np.ndarray, np.ndarray], float],
    use_delta: bool = False,
    embed_key: str | None = None,
) -> dict[str, float]:
    """Generic evaluation function for anndata pair."""
    res = {}
    for bulk_array in data.iter_bulk_arrays(embed_key=embed_key):
        if use_delta:
            x = bulk_array.perturbation_effect(which="pred", abs=False)
            y = bulk_array.perturbation_effect(which="real", abs=False)
        else:
            x = bulk_array.pert_pred
            y = bulk_array.pert_real

        result = func(x, y)
        if isinstance(result, tuple):
            result = result[0]

        res[bulk_array.key] = float(result)

    return res


# TODO: clean up this implementation
class ClusteringAgreement:
    """Compute clustering agreement between real and predicted perturbation centroids."""

    def __init__(
        self,
        embed_key: str | None = None,
        real_resolution: float = 1.0,
        pred_resolutions: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0),
        metric: Literal["ami", "nmi", "ari"] = "ami",
        n_neighbors: int = 15,
    ) -> None:
        self.embed_key = embed_key
        self.real_resolution = real_resolution
        self.pred_resolutions = pred_resolutions
        self.metric = metric
        self.n_neighbors = n_neighbors

    @staticmethod
    def _score(
        labels_real: Sequence[int],
        labels_pred: Sequence[int],
        metric: Literal["ami", "nmi", "ari"],
    ) -> float:
        if metric == "ami":
            return adjusted_mutual_info_score(labels_real, labels_pred)
        if metric == "nmi":
            return normalized_mutual_info_score(labels_real, labels_pred)
        if metric == "ari":
            return (adjusted_rand_score(labels_real, labels_pred) + 1) / 2
        raise ValueError(f"Unknown metric: {metric}")

    @staticmethod
    def _cluster_leiden(
        adata: ad.AnnData,
        resolution: float,
        key_added: str,
        n_neighbors: int = 15,
    ) -> None:
        if key_added in adata.obs:
            return
        if "neighbors" not in adata.uns:
            sc.pp.neighbors(
                adata, n_neighbors=min(n_neighbors, adata.n_obs - 1), use_rep="X"
            )
        sc.tl.leiden(
            adata,
            resolution=resolution,
            key_added=key_added,
            flavor="igraph",
            n_iterations=2,
        )

    @staticmethod
    def _centroid_ann(
        adata: ad.AnnData,
        category_key: str,
        control_pert: str,
        embed_key: str | None = None,
    ) -> ad.AnnData:
        # Isolate the features
        # embed_key may be None; narrow it before .get (whose key must be str) and
        # keep feats untyped since it may be a dense array or a sparse matrix.
        feats: Any = (
            adata.X if embed_key is None else adata.obsm.get(embed_key, adata.X)
        )

        # Convert to float if not already
        if feats.dtype != np.dtype("float64"):
            feats = feats.astype(np.float64)

        # Densify if required
        if issparse(feats):
            feats = feats.toarray()

        cats = cast(pd.Series, adata.obs[category_key]).values
        uniq, inv = np.unique(cats, return_inverse=True)
        centroids = np.zeros((uniq.size, feats.shape[1]), dtype=feats.dtype)

        for i, cat in enumerate(uniq):
            mask = cats == cat
            if np.any(mask):
                centroids[i] = feats[mask].mean(axis=0)

        adc = ad.AnnData(X=centroids)
        adc.obs[category_key] = uniq
        return adc[adc.obs[category_key] != control_pert]

    def __call__(self, data: PerturbationAnndataPair) -> float:
        cats_sorted = sorted([c for c in data.perts if c != data.control_pert])

        # 2. build centroids
        ad_real_cent = self._centroid_ann(
            adata=data.real,
            category_key=data.pert_col,
            control_pert=data.control_pert,
            embed_key=self.embed_key,
        )
        ad_pred_cent = self._centroid_ann(
            adata=data.pred,
            category_key=data.pert_col,
            control_pert=data.control_pert,
            embed_key=self.embed_key,
        )

        # 3. cluster real once
        real_key = "real_clusters"
        self._cluster_leiden(
            ad_real_cent, self.real_resolution, real_key, self.n_neighbors
        )
        ad_real_cent.obs = (
            cast(pd.DataFrame, ad_real_cent.obs)
            .set_index(data.pert_col)
            .loc[cats_sorted]
        )
        real_labels = pd.Categorical(ad_real_cent.obs[real_key])

        # 4. sweep predicted resolutions
        best_score = 0.0
        ad_pred_cent.obs = (
            cast(pd.DataFrame, ad_pred_cent.obs)
            .set_index(data.pert_col)
            .loc[cats_sorted]
        )
        for r in self.pred_resolutions:
            pred_key = f"pred_clusters_{r}"
            self._cluster_leiden(ad_pred_cent, r, pred_key, self.n_neighbors)
            pred_labels = pd.Categorical(ad_pred_cent.obs[pred_key])
            score = self._score(real_labels, pred_labels, self.metric)  # type: ignore
            best_score = max(best_score, score)

        return float(best_score)
