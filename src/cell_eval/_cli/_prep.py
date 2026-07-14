import argparse as ap
import importlib.metadata
import logging
import os
import shutil
import subprocess
from tempfile import TemporaryDirectory
from typing import cast

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
from scipy.sparse import csr_matrix, issparse

from .._evaluator import _convert_to_normlog
from ._const import DEFAULT_CELLTYPE_COL, DEFAULT_NTC_NAME, DEFAULT_PERT_COL

logger = logging.getLogger(__name__)

VALID_ENCODINGS = [64, 32]
EXPECTED_GENE_DIM = 18080
MAX_CELL_DIM = 200000


def parse_args_prep(parser: ap.ArgumentParser):
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to h5ad to read",
        required=True,
    )
    parser.add_argument(
        "-g",
        "--genes",
        type=str,
        required=True,
        help="CSV file containing expected gene names and order",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to vcc to write [default: <input>.prep.vcc]",
    )
    parser.add_argument(
        "-p",
        "--pert-col",
        type=str,
        default=DEFAULT_PERT_COL,
        help="Name of the column designated perturbations [default: %(default)s]",
    )
    parser.add_argument(
        "-c",
        "--celltype-col",
        type=str,
        help="Name of the column designated celltype (optional)",
    )
    parser.add_argument(
        "-n",
        "--ntc-name",
        type=str,
        default=DEFAULT_NTC_NAME,
        help="Name of the column designated negative control (optional) [default: %(default)s]",
    )
    parser.add_argument(
        "-P",
        "--output-pert-col",
        type=str,
        default=DEFAULT_PERT_COL,
        help="Name of the column designated perturbations in the output [default: %(default)s]",
    )
    parser.add_argument(
        "-C",
        "--output-celltype-col",
        type=str,
        default=DEFAULT_CELLTYPE_COL,
        help="Name of the column designated celltype in the output [default: %(default)s]",
    )
    parser.add_argument(
        "-e",
        "--encoding",
        type=int,
        help=f"Bit size to encode ({VALID_ENCODINGS}) [default: %(default)s]",
        default=32,
    )
    parser.add_argument(
        "--allow-discrete",
        action="store_true",
        help="Bypass log normalization in case we incorrectly guess the data is discrete",
    )
    parser.add_argument(
        "--expected-gene-dim",
        type=int,
        help=f"Expected gene dimension ({EXPECTED_GENE_DIM}) [default: %(default)s]. Set to -1 to disable.",
        default=EXPECTED_GENE_DIM,
    )
    parser.add_argument(
        "--max-cell-dim",
        type=int,
        help=f"Maximum cell dimension ({MAX_CELL_DIM}) [default: %(default)s]. Set to -1 to disable.",
        default=MAX_CELL_DIM,
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s {version}".format(
            version=importlib.metadata.version("cell_eval")
        ),
    )


def strip_anndata(
    adata: ad.AnnData,
    output_path: str,
    genelist: list[str],
    pert_col: str = "target_gene",
    celltype_col: str | None = None,
    output_pert_col: str = DEFAULT_PERT_COL,
    output_celltype_col: str = DEFAULT_CELLTYPE_COL,
    ntc_name: str = DEFAULT_NTC_NAME,
    encoding: int = 64,
    allow_discrete: bool = False,
    max_cell_dim: int | None = MAX_CELL_DIM,
    exp_gene_dim: int | None = EXPECTED_GENE_DIM,
):
    # Force anndata var to string
    adata.var.index = adata.var.index.astype(str)

    if pert_col not in adata.obs:
        raise ValueError(
            f"Provided perturbation column: '{pert_col}' missing from anndata: {adata.obs.columns}"
        )
    if celltype_col:
        if celltype_col not in adata.obs:
            raise ValueError(
                f"Provided celltype column: '{celltype_col}' missing from anndata: {adata.obs.columns}"
            )
    if ntc_name not in cast(pd.Series, adata.obs[pert_col]).unique():
        raise ValueError(
            f"Provided negative control name: '{ntc_name}' missing from anndata: {cast(pd.Series, adata.obs[pert_col]).unique()}"
        )

    # Check if expected dimension is provided and matches the length of the genelist
    if exp_gene_dim and len(genelist) != exp_gene_dim:
        logger.warning(
            f"Provided gene dimension: {len(genelist)} does not match expected gene dimension: {exp_gene_dim}."
        )
        logger.info(f"Setting expected gene dimension to {len(genelist)}")
        exp_gene_dim = len(genelist)

    if adata.var_names.tolist() != genelist:
        missing_genes = set(genelist) - set(adata.var_names.tolist())
        extra_genes = set(adata.var_names.tolist()) - set(genelist)
        if len(missing_genes) == 0 and len(extra_genes) == 0:
            logger.warning(
                "Provided anndata contains all expected genes but they are out of order."
            )
            logger.info("Reordering genes...")
            adata = adata[:, np.array(genelist)]
        else:
            raise ValueError(
                "Provided gene list does not match anndata gene names:\n"
                f"Missing genes: {missing_genes}\n"
                f"Extra genes: {extra_genes}"
            )

    if exp_gene_dim and adata.shape[1] != exp_gene_dim:
        raise ValueError(
            f"Provided gene dimension: {adata.shape[1]} does not match expected gene dimension: {exp_gene_dim}"
        )

    if max_cell_dim and adata.shape[0] > max_cell_dim:
        raise ValueError(
            f"Provided cell dimension: {adata.shape[0]} exceeds maximum cell dimension: {max_cell_dim}"
        )

    if encoding not in VALID_ENCODINGS:
        raise ValueError(f"Encoding must be in {VALID_ENCODINGS}")

    dtype = np.dtype(np.float64)  # force bound
    match encoding:
        case 64:
            logger.info("Using 64-bit float encoding")
            dtype = np.dtype(np.float64)
        case 32:
            logger.info("Using 32-bit float encoding")
            dtype = np.dtype(np.float32)

    logger.info("Setting data to sparse if not already")
    new_x = (
        adata.X.astype(dtype)  # type: ignore
        if issparse(adata.X)
        else csr_matrix(adata.X.astype(dtype))  # type: ignore
    )

    logger.info("Simplifying obs dataframe")
    new_obs = pd.DataFrame(
        {output_pert_col: cast(pd.Series, adata.obs[pert_col]).values},
        index=np.arange(adata.shape[0]).astype(str),
    )
    if celltype_col:
        new_obs[output_celltype_col] = cast(pd.Series, adata.obs[celltype_col]).values

    logger.info("Simplifying var dataframe")
    new_var = pd.DataFrame(
        index=adata.var.index.values,
    )

    logger.info("Creating final minimal AnnData object")
    minimal = ad.AnnData(
        X=new_x,
        obs=new_obs,
        var=new_var,
    )

    logger.info("Applying normlog transformation if required")
    _convert_to_normlog(minimal, allow_discrete=allow_discrete)

    # Create a temporary directory to work in
    with TemporaryDirectory() as temp_dir:
        # Create temp files with specific names
        tmp_h5ad = os.path.join(temp_dir, "pred.h5ad")
        tmp_watermark = os.path.join(temp_dir, "watermark.txt")

        # Write the h5ad file
        logger.info(f"Writing h5ad output to {tmp_h5ad}")
        minimal.write_h5ad(tmp_h5ad)

        # Zstd compress the h5ad file (will create pred.h5ad.zst)
        logger.info(f"Zstd compressing {tmp_h5ad}")
        subprocess.run(["zstd", "-T0", "-f", "--rm", tmp_h5ad])

        # Write the watermark file
        with open(tmp_watermark, "w") as f:
            f.write("vcc-prep")

        # Pack the files into a tarball
        logger.info(f"Packing files into {output_path}")
        subprocess.run(
            [
                "tar",
                "-cf",
                output_path,
                "-C",
                temp_dir,
                "pred.h5ad.zst",
                "watermark.txt",
            ]
        )

        logger.info("Done")


def _validate_tools_in_path():
    if shutil.which("tar") is None:
        raise ValueError("tar is not installed")
    if shutil.which("zstd") is None:
        raise ValueError("zstd is not installed")
    return True


def run_prep(args: ap.Namespace):
    _validate_tools_in_path()

    logger.info("Reading input anndata")
    adata = ad.read_h5ad(args.input)

    logger.info("Reading gene list")
    genelist = (
        pl.read_csv(args.genes, has_header=False).to_series(0).cast(str).to_list()
    )

    logger.info("Preparing anndata")
    strip_anndata(
        adata,
        genelist=genelist,
        output_path=args.output
        if args.output
        else args.input.replace(".h5ad", ".prep.vcc"),
        pert_col=args.pert_col,
        celltype_col=args.celltype_col,
        output_pert_col=args.output_pert_col,
        output_celltype_col=args.output_celltype_col,
        ntc_name=args.ntc_name,
        encoding=args.encoding,
        allow_discrete=args.allow_discrete,
        exp_gene_dim=args.expected_gene_dim if args.expected_gene_dim != -1 else None,
        max_cell_dim=args.max_cell_dim if args.max_cell_dim != -1 else None,
    )
