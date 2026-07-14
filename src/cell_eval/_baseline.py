import logging
from typing import Any, cast

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
from numpy.typing import NDArray
from pdex import pdex
from scipy.sparse import issparse

from ._evaluator import _build_pdex_kwargs, _convert_to_normlog

logger = logging.getLogger(__name__)


def build_base_mean_adata(
    adata: ad.AnnData | str,
    counts_df: pl.DataFrame | str | None = None,
    pert_col: str = "target_gene",
    control_pert: str = "non-targeting",
    counts_col: str = "n_cells",
    as_delta: bool = False,
    allow_discrete: bool = False,
    output_path: str | None = None,
    output_de_path: str | None = None,
    num_threads: int = 1,
    pdex_kwargs: dict[str, Any] = {},
) -> ad.AnnData:
    if isinstance(adata, str):
        logger.info(f"Reading adata from path: {adata}")
        adata = ad.read_h5ad(adata)

    # Convert to normalized log space if necessary
    _convert_to_normlog(adata=adata, allow_discrete=allow_discrete)

    counts = (
        _load_counts_df(
            counts_df=counts_df,
            pert_col=pert_col,
            control_pert=control_pert,
            counts_col=counts_col,
        )
        if counts_df is not None
        else _build_counts_df_from_adata(
            adata=adata,
            pert_col=pert_col,
            control_pert=control_pert,
            counts_col=counts_col,
        )
    )
    baseline = _build_pert_baseline(
        adata=adata, pert_col=pert_col, control_pert=control_pert, as_delta=as_delta
    )

    obs = (
        counts.select([pl.col(pert_col).repeat_by(counts_col)])
        .explode(pert_col)
        .to_pandas()
    )
    obs.index = obs.index.astype(str).str.replace("^", "p.", regex=True)

    logger.info("Assembling baseline adata from perturbation mean")
    baseline_adata = ad.AnnData(
        X=np.full(
            (int(counts[counts_col].sum()), baseline.size),
            baseline,
        ),
        var=cast(pd.DataFrame, adata.var),
        obs=obs,
    )

    logger.info("Concatenating baseline adata with controls from original adata")
    baseline_adata = ad.concat(
        [baseline_adata, adata[adata.obs[pert_col] == control_pert]]
    )

    if output_path is not None:
        logger.info(f"Saving baseline data to {output_path}")
        baseline_adata.write_h5ad(output_path)

    if output_de_path is not None:
        logger.info("Calculating differential expression")
        pdex_kwargs = _build_pdex_kwargs(
            groupby=pert_col,
            reference=control_pert,
            threads=num_threads,
            allow_discrete=allow_discrete,
            pdex_kwargs=pdex_kwargs,
        )
        frame = pdex(
            adata=baseline_adata,
            mode="ref",
            **pdex_kwargs,
        )
        logger.info(f"Saving differential expression results to {output_de_path}")
        frame.write_csv(output_de_path)

    return baseline_adata


def _load_counts_df(
    counts_df: pl.DataFrame | str,
    pert_col: str = "target_gene",
    control_pert: str = "non-targeting",
    counts_col: str = "n_cells",
) -> pl.DataFrame:
    if isinstance(counts_df, str):
        logger.info(f"Loading counts from {counts_df}")
        counts_df = pl.read_csv(counts_df)

    if pert_col not in counts_df.columns:
        raise ValueError(
            f"Column '{pert_col}' not found in counts_df: {counts_df.columns}"
        )

    if counts_col not in counts_df.columns:
        raise ValueError(
            f"Column '{counts_col}' not found in counts_df: {counts_df.columns}"
        )

    logger.info(f"Filtering out counts from {control_pert}")
    return counts_df.filter(
        pl.col(pert_col) != control_pert  # drop control pert
    )


def _build_counts_df_from_adata(
    adata: ad.AnnData,
    pert_col: str = "target_gene",
    control_pert: str = "non-targeting",
    counts_col: str = "n_cells",
) -> pl.DataFrame:
    if pert_col not in adata.obs.columns:
        raise ValueError(
            f"Column '{pert_col}' not found in adata.obs: {adata.obs.columns}"
        )
    if control_pert not in cast(pd.Series, adata.obs[pert_col]).unique():
        raise ValueError(
            f"Control pert '{control_pert}' not found in adata.obs[{pert_col}]: {cast(pd.Series, adata.obs[pert_col]).unique()}"
        )
    logger.info("Building counts DataFrame from adata")
    return (
        pl.DataFrame(adata.obs)
        .group_by(pert_col)
        .len()
        .rename({"len": counts_col})
        .filter(pl.col(pert_col) != control_pert)
    )


def _build_pert_baseline(
    adata: ad.AnnData,
    pert_col: str = "target_gene",
    control_pert: str = "non-targeting",
    as_delta: bool = False,
) -> NDArray[np.float64]:
    if pert_col not in adata.obs.columns:
        raise ValueError(
            f"Column '{pert_col}' not found in adata.obs: {adata.obs.columns}"
        )
    unique_perts = cast(pd.Series, adata.obs[pert_col]).unique()
    if control_pert not in unique_perts:
        raise ValueError(
            f"Control pert '{control_pert}' not found in unique_perts: {unique_perts}"
        )

    logger.info("Building perturbation-level means")
    pert_means = (
        pl.DataFrame(
            adata.X if not issparse(adata.X) else adata.X.toarray()  # type: ignore
        )
        .with_columns(pl.Series(pert_col, adata.obs[pert_col]))
        .group_by(pert_col)
        .mean()
    )

    names = pert_means.drop_in_place(pert_col).to_numpy()
    pert_mask = names != control_pert
    pert_matrix = pert_means.to_numpy()

    if as_delta:
        logger.info("Calculating delta from control means")
        delta = pert_matrix[pert_mask] - pert_matrix[~pert_mask]

        logger.info("Calculating mean delta")
        mean_delta = delta.mean(axis=0)

        return mean_delta
    else:
        logger.info("Calculating mean of perturbation-level means")
        mean_pert = pert_matrix.mean(axis=0)

        return mean_pert
