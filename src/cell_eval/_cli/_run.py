import argparse as ap
import importlib.metadata
import logging
import os

from .. import KNOWN_PROFILES
from ._const import DEFAULT_CTRL, DEFAULT_OUTDIR, DEFAULT_PERT_COL

logger = logging.getLogger(__name__)


def parse_args_run(parser: ap.ArgumentParser):
    """
    CLI for evaluation
    """
    parser.add_argument(
        "-ap",
        "--adata-pred",
        type=str,
        help="Path to the predicted adata object to evaluate",
        required=True,
    )
    parser.add_argument(
        "-ar",
        "--adata-real",
        type=str,
        help="Path to the real adata object to evaluate against",
        required=True,
    )
    parser.add_argument(
        "-dp",
        "--de-pred",
        type=str,
        help="Path to the predicted DE results "
        f"(computed with pdex from adata-pred if not provided and saved to {DEFAULT_OUTDIR}/pred_de.csv)",
        required=False,
    )
    parser.add_argument(
        "-dr",
        "--de-real",
        type=str,
        help="Path to the real DE results "
        f"(computed with pdex from adata-real if not provided and saved to {DEFAULT_OUTDIR}/real_de.csv)",
        required=False,
    )
    parser.add_argument(
        "--control-pert",
        type=str,
        default=DEFAULT_CTRL,
        help="Name of the control perturbation [default: %(default)s]",
    )
    parser.add_argument(
        "--pert-col",
        type=str,
        default=DEFAULT_PERT_COL,
        help="Name of the column designated perturbations [default: %(default)s]",
    )
    parser.add_argument(
        "--celltype-col",
        type=str,
        help="Name of the column designated celltype to split results by (optional)",
    )
    parser.add_argument(
        "--embed-key",
        type=str,
        default=None,
        help="Key for embedded data (.obsm) in the AnnData object used in some metrics (evaluated over .X otherwise)",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default=DEFAULT_OUTDIR,
        help="Output directory to write to [default: %(default)s]",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads to use for parallel processing [default: %(default)s]",
    )
    parser.add_argument(
        "--allow-discrete",
        action="store_true",
        help="Allow discrete data to be evaluated (usually expected to be norm-logged inputs)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="full",
        help="Profile of metrics to compute [default: %(default)s]",
        choices=KNOWN_PROFILES,
    )
    parser.add_argument(
        "--skip-metrics",
        type=str,
        help="Metrics to skip (comma-separated for multiple) (see docs for more details)",
    )
    parser.add_argument(
        "--ceiling",
        action="store_true",
        help="Additionally compute a data ceiling: a real-data-only upper bound on "
        "each metric, estimated by splitting the real data into two disjoint halves "
        "of n/2 cells and applying the Spearman-Brown correction (2r/(1+r)) to map "
        "each reliability metric back to full depth. Writes ceiling_results.csv / "
        "agg_ceiling_results.csv alongside the normal results.",
    )
    parser.add_argument(
        "--ceiling-seed",
        type=int,
        default=0,
        help="Random seed for the data ceiling disjoint split [default: %(default)s]",
    )
    parser.add_argument(
        "--cpm-filter",
        type=float,
        default=None,
        help="Optional pooled-CPM floor threshold (T) forwarded to pdex: a (target, "
        "gene) row is dropped when the gene's bulk CPM is <= T in BOTH the target and "
        "the reference, removing the below-detection floor. OFF by default (default DE "
        "behavior unchanged); pass e.g. 5 to enable (T is dataset-dependent).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s {version}".format(
            version=importlib.metadata.version("cell_eval")
        ),
    )


def build_outdir(outdir: str):
    if os.path.exists(outdir):
        logger.warning(
            f"Output directory {outdir} already exists, potential overwrite occurring"
        )
    os.makedirs(outdir, exist_ok=True)


def run_evaluation(args: ap.Namespace):
    import anndata as ad

    from cell_eval import MetricsEvaluator
    from cell_eval.utils import split_anndata_on_celltype

    # Set metric config for embed key if provided
    metric_kwargs = (
        {
            "discrimination_score_l2": {"embed_key": args.embed_key},
            "discrimination_score_cosine": {"embed_key": args.embed_key},
            "pearson_edistance": {"n_jobs": args.num_threads},
        }
        if args.embed_key is not None
        else {}
    )

    skip_metrics = args.skip_metrics.split(",") if args.skip_metrics else None

    if args.celltype_col is not None:
        real = ad.read_h5ad(args.adata_real)
        pred = ad.read_h5ad(args.adata_pred)

        real_split = split_anndata_on_celltype(real, args.celltype_col)
        pred_split = split_anndata_on_celltype(pred, args.celltype_col)

        assert len(real_split) == len(pred_split), (
            f"Number of celltypes in real and pred anndata must match: {len(real_split)} != {len(pred_split)}"
        )

        for ct in real_split.keys():
            real_ct = real_split[ct]
            pred_ct = pred_split[ct]

            evaluator = MetricsEvaluator(
                adata_pred=pred_ct,
                adata_real=real_ct,
                de_pred=args.de_pred,
                de_real=args.de_real,
                control_pert=args.control_pert,
                pert_col=args.pert_col,
                num_threads=args.num_threads,
                outdir=args.outdir,
                allow_discrete=args.allow_discrete,
                prefix=ct,
                skip_de=args.profile == "pds",
                pdex_kwargs={"cpm_filter": args.cpm_filter},
            )
            evaluator.compute(
                profile=args.profile,
                metric_configs=metric_kwargs,
                skip_metrics=skip_metrics,
                basename="results.csv",
            )
            if args.ceiling:
                evaluator.compute_ceiling(
                    profile=args.profile,
                    metric_configs=metric_kwargs,
                    skip_metrics=skip_metrics,
                    basename="ceiling_results.csv",
                    seed=args.ceiling_seed,
                )

    else:
        evaluator = MetricsEvaluator(
            adata_pred=args.adata_pred,
            adata_real=args.adata_real,
            de_pred=args.de_pred,
            de_real=args.de_real,
            control_pert=args.control_pert,
            pert_col=args.pert_col,
            num_threads=args.num_threads,
            outdir=args.outdir,
            allow_discrete=args.allow_discrete,
            skip_de=args.profile == "pds",
            pdex_kwargs={"cpm_filter": args.cpm_filter},
        )
        evaluator.compute(
            profile=args.profile,
            metric_configs=metric_kwargs,
            skip_metrics=skip_metrics,
            basename="results.csv",
        )
        if args.ceiling:
            evaluator.compute_ceiling(
                profile=args.profile,
                metric_configs=metric_kwargs,
                skip_metrics=skip_metrics,
                basename="ceiling_results.csv",
                seed=args.ceiling_seed,
            )
