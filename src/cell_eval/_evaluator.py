import logging
import multiprocessing as mp
import os
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
from pdex import pdex

from cell_eval.utils import guess_is_lognorm

from ._pipeline import MetricPipeline
from ._types import PerturbationAnndataPair, initialize_de_comparison
from .utils import _cast_float16_to_float32

logger = logging.getLogger(__name__)


def _available_cpus() -> int:
    """Return CPUs the current process is allowed to use.

    Uses ``os.sched_getaffinity`` on Linux so SLURM/cgroup/taskset limits are
    respected; falls back to ``mp.cpu_count`` on macOS/Windows where that API
    is unavailable (those platforms typically run locally without cgroup caps).
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return mp.cpu_count()


class MetricsEvaluator:
    """
    Evaluates benchmarking metrics of a predicted and real anndata object.

    Arguments
    =========

    adata_pred: ad.AnnData | str
        Predicted anndata object or path to anndata object.
    adata_real: ad.AnnData | str
        Real anndata object or path to anndata object.
    de_pred: pl.DataFrame | str | None = None
        Predicted differential expression results or path to differential expression results.
        If `None`, differential expression will be computed using parallel_differential_expression
    de_real: pl.DataFrame | str | None = None
        Real differential expression results or path to differential expression results.
        If `None`, differential expression will be computed using parallel_differential_expression
    control_pert: str = "non-targeting"
        Control perturbation name.
    pert_col: str = "target"
        Perturbation column name.
    num_threads: int = -1
        Number of threads for parallel differential expression.
    outdir: str = "./cell-eval-outdir"
        Output directory.
    allow_discrete: bool = False
        Allow discrete data.
    prefix: str | None = None
        Prefix for output files.
    pdex_kwargs: dict[str, Any] | None = None
        Keyword arguments for parallel_differential_expression.
        These will overwrite arguments passed to MetricsEvaluator.__init__ if they conflict.
    """

    def __init__(
        self,
        adata_pred: ad.AnnData | str,
        adata_real: ad.AnnData | str,
        de_pred: pl.DataFrame | str | None = None,
        de_real: pl.DataFrame | str | None = None,
        control_pert: str = "non-targeting",
        pert_col: str = "target",
        num_threads: int = -1,
        outdir: str = "./cell-eval-outdir",
        allow_discrete: bool = False,
        prefix: str | None = None,
        pdex_kwargs: dict[str, Any] | None = None,
        skip_de: bool = False,
    ):
        # Enable a global string cache for categorical columns
        pl.enable_string_cache()

        if num_threads == -1:
            num_threads = _available_cpus()

        if os.path.exists(outdir):
            logger.warning(
                f"Output directory {outdir} already exists, potential overwrite occurring"
            )
        os.makedirs(outdir, exist_ok=True)

        # Stored so the data ceiling (compute_ceiling) can reuse the exact same
        # DE / pdex configuration as the main evaluation for comparability.
        self._num_threads = num_threads
        self._allow_discrete = allow_discrete
        self._skip_de = skip_de
        self._pdex_kwargs = pdex_kwargs or {}

        self.anndata_pair = _build_anndata_pair(
            real=adata_real,
            pred=adata_pred,
            control_pert=control_pert,
            pert_col=pert_col,
            allow_discrete=allow_discrete,
        )

        if skip_de:
            self.de_comparison = None
        else:
            self.de_comparison = _build_de_comparison(
                anndata_pair=self.anndata_pair,
                de_pred=de_pred,
                de_real=de_real,
                num_threads=num_threads,
                allow_discrete=allow_discrete,
                outdir=outdir,
                prefix=prefix,
                pdex_kwargs=self._pdex_kwargs,
            )

        self.outdir = outdir
        self.prefix = prefix

    def compute(
        self,
        profile: Literal["full", "vcc", "minimal", "de", "anndata"] = "full",
        metric_configs: dict[str, dict[str, Any]] | None = None,
        skip_metrics: list[str] | None = None,
        basename: str = "results.csv",
        write_csv: bool = True,
        break_on_error: bool = False,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        pipeline = MetricPipeline(
            profile=profile,
            metric_configs=metric_configs,
            break_on_error=break_on_error,
        )
        if skip_metrics is not None:
            pipeline.skip_metrics(skip_metrics)
        pipeline.compute_de_metrics(self.de_comparison)
        pipeline.compute_anndata_metrics(self.anndata_pair)
        results = pipeline.get_results()
        agg_results = pipeline.get_agg_results()

        if write_csv:
            self._write_results(results, agg_results, basename)

        return results, agg_results

    def compute_ceiling(
        self,
        profile: Literal["full", "vcc", "minimal", "de", "anndata", "pds"] = "full",
        metric_configs: dict[str, dict[str, Any]] | None = None,
        skip_metrics: list[str] | None = None,
        fracs: tuple[float, ...] = (1.0, 0.5, 0.25),
        seed: int = 0,
        agg: Literal["mean", "median"] = "mean",
        basename: str = "ceiling_results.csv",
        write_csv: bool = True,
        break_on_error: bool = False,
    ) -> pl.DataFrame:
        """Estimate a data ceiling: the maximum achievable score per metric.

        Uses the real data only. The real data is split into two *disjoint* halves
        (no cell in both) at several depths, each metric is measured on that
        self-split at each depth, and the metric-vs-depth curve is extrapolated to
        full depth - an unbiased estimate of the ceiling any model could reach
        given the noise inherent in the real data. For reliability-like
        (correlation) metrics this reduces to the analytical Spearman-Brown
        correction; it generalizes the same idea to metrics with no closed form.

        A disjoint split is used (rather than a bootstrap self-split) because a
        bootstrap draws the two halves from the same cells, so they are not
        independent - which biases the ceiling in both directions (not a reliable
        upper bound): the shared cells make the halves over-agree (inflating it),
        while the duplicate cells over-call the FDR-gated DE metrics and drag the
        recovery metrics down. The cost of a disjoint split is depth (each half is
        at most ``n/2``), which the depth extrapolation corrects for.

        For each ``frac`` in ``fracs`` every perturbation's cells (and the
        control's) are shuffled and split without replacement into two halves of
        ``floor(frac * n/2)`` cells. ``frac=1`` uses all cells (each half ``n/2``);
        the full-depth target (each half ``n``) is ``frac=2``, where the curve is
        extrapolated to. The same ``pdex_kwargs`` / ``allow_discrete`` / ``skip_de``
        as the main evaluation are reused so the ceiling is directly comparable,
        and the sweep DE is computed in-memory (never written to disk).

        Returns a per-metric table with the measured depth curve (``m@<frac>``)
        and the extrapolation to full depth (``extrap``, alongside an
        ``extrap_linear`` companion, the fit residual and the model used).
        """
        fracs = tuple(sorted(set(fracs), reverse=True))
        if any(f <= 0.0 or f > 1.0 for f in fracs):
            raise ValueError(f"fracs must be in (0, 1]; got {fracs}")

        # Fix the perturbation set across depths so the curve isn't confounded by
        # small perts dropping out at shallow depths: keep only perts (incl. the
        # control) with enough cells to split at the shallowest requested depth.
        pert_col = self.anndata_pair.pert_col
        control = self.anndata_pair.control_pert
        counts = self.anndata_pair.real.obs[pert_col].value_counts()
        min_cells = int(np.ceil(2.0 / min(fracs)))
        eligible = {str(p) for p, c in counts.items() if c >= min_cells}
        if control not in eligible:
            raise ValueError(
                f"Control '{control}' has too few cells to split at "
                f"frac={min(fracs)} (needs >= {min_cells})."
            )
        dropped = {str(p) for p in counts.index} - eligible
        if dropped:
            logger.warning(
                f"Depth-extrapolation ceiling: dropping {len(dropped)} perturbation(s) "
                f"with < {min_cells} cells (too few to split at frac={min(fracs)})."
            )

        curve: dict[float, dict[str, float]] = {}
        for frac in fracs:
            logger.info(f"Ceiling depth sweep: frac={frac:g} (seed={seed})")
            half_real, half_pred = self._disjoint_halves(seed, frac, eligible)
            pair = PerturbationAnndataPair(
                real=half_real,
                pred=half_pred,
                control_pert=control,
                pert_col=pert_col,
                embed_key=self.anndata_pair.embed_key,
            )
            de = None
            if not self._skip_de:
                de = _build_de_comparison(
                    anndata_pair=pair,
                    num_threads=self._num_threads,
                    allow_discrete=self._allow_discrete,
                    outdir=None,  # keep the sweep DE in-memory; never persisted
                    prefix=None,
                    pdex_kwargs=dict(self._pdex_kwargs),
                )
            pipeline = MetricPipeline(
                profile=profile,
                metric_configs=metric_configs,
                break_on_error=break_on_error,
            )
            if skip_metrics is not None:
                pipeline.skip_metrics(skip_metrics)
            pipeline.compute_de_metrics(de)
            pipeline.compute_anndata_metrics(pair)
            curve[frac] = _aggregate_metric_values(pipeline.get_results(), agg)

        table = _extrapolate_ceiling_curve(curve, target_frac=2.0)

        if write_csv:
            prefix = self.prefix.replace("/", "-") if self.prefix is not None else None
            outname = basename.replace("/", "-")
            outpath = os.path.join(
                self.outdir, f"{prefix}_{outname}" if prefix else outname
            )
            logger.info(f"Writing depth-extrapolated ceiling to {outpath}")
            table.write_csv(outpath)

        return table

    def _disjoint_halves(
        self, seed: int, frac: float, eligible: set[str] | None = None
    ) -> tuple[ad.AnnData, ad.AnnData]:
        """Split the real data into two *disjoint* halves at depth ``frac``.

        Each perturbation's cells are shuffled and split without replacement into
        two halves of ``floor(frac * n/2)`` cells each, so no cell appears in both
        halves - the independence a bootstrap self-split lacks. Because the shuffle
        is seeded per call and the group order is stable, shallower depths are
        nested prefixes of deeper ones (monotone subsampling). ``eligible`` (if
        given) restricts to a fixed perturbation set so every depth uses the same
        perts.
        """
        real = self.anndata_pair.real
        pert_col = self.anndata_pair.pert_col
        rng = np.random.default_rng(seed)

        a_idx: list[np.ndarray] = []
        b_idx: list[np.ndarray] = []
        for pert, idx in real.obs.groupby(pert_col, observed=True).indices.items():
            if eligible is not None and str(pert) not in eligible:
                continue
            idx = np.array(idx)
            rng.shuffle(idx)
            h = int(frac * (idx.size // 2))
            if h < 1:
                continue
            a_idx.append(idx[:h])
            b_idx.append(idx[h : 2 * h])

        # Disjoint split has no duplicate rows, so obs names stay unique.
        half_real = real[np.concatenate(a_idx)].copy()
        half_pred = real[np.concatenate(b_idx)].copy()
        return half_real, half_pred

    def _write_results(
        self,
        results: pl.DataFrame,
        agg_results: pl.DataFrame,
        basename: str,
    ) -> None:
        # some prefixes/basenames (e.g. HepG2/C3A) may have slashes in them
        prefix = self.prefix.replace("/", "-") if self.prefix is not None else None
        basename = basename.replace("/", "-")

        outpath = os.path.join(
            self.outdir,
            f"{prefix}_{basename}" if prefix else basename,
        )
        agg_outpath = os.path.join(
            self.outdir,
            f"{prefix}_agg_{basename}" if prefix else f"agg_{basename}",
        )

        logger.info(f"Writing perturbation level metrics to {outpath}")
        results.write_csv(outpath)

        logger.info(f"Writing aggregate metrics to {agg_outpath}")
        agg_results.write_csv(agg_outpath)


def _aggregate_metric_values(results: pl.DataFrame, agg: str) -> dict[str, float]:
    """Collapse the per-perturbation results to one scalar per metric."""
    out: dict[str, float] = {}
    if results.is_empty():
        return out
    for col in results.columns:
        if col == "perturbation" or not results[col].dtype.is_numeric():
            continue
        arr = results[col].drop_nulls().to_numpy()
        if arr.size == 0:
            continue
        out[col] = float(np.median(arr) if agg == "median" else np.mean(arr))
    return out


def _extrapolate_metric(
    fracs: np.ndarray, values: np.ndarray, target: float
) -> dict[str, float | str | None]:
    """Extrapolate one metric's depth curve to ``target`` (full depth = 2).

    Primary model is the reliability/attenuation form ``1/m = a + b/frac`` (the
    standard measurement-error model, in which reliability grows with depth); it is
    only well-defined for reliability-like metrics (``0 < m < 1``). Otherwise we
    fall back to a linear fit (the downsampling curves were observed to be ~linear).
    """
    mask = np.isfinite(values)
    f, m = fracs[mask], values[mask]
    out: dict[str, float | str | None] = {
        "extrap": None,
        "extrap_linear": None,
        "resid": None,
        "model": None,
    }
    if f.size < 2:
        return out
    # Linear fit m = a + b*frac (matches the ~linear downsampling behaviour).
    a_lin, b_lin = np.linalg.lstsq(np.vstack([np.ones_like(f), f]).T, m, rcond=None)[0]
    out["extrap_linear"] = float(a_lin + b_lin * target)
    # Attenuation fit 1/m = a + b/frac; reduces to SB for a single deepest point.
    if np.all((m > 0.0) & (m < 1.0)):
        design = np.vstack([np.ones_like(f), 1.0 / f]).T
        a_att, b_att = np.linalg.lstsq(design, 1.0 / m, rcond=None)[0]
        inv = a_att + b_att / target
        out["extrap"] = float(1.0 / inv) if inv > 0 else None
        out["resid"] = float(np.sqrt(np.mean((design @ [a_att, b_att] - 1.0 / m) ** 2)))
        out["model"] = "attenuation"
    else:
        out["extrap"] = out["extrap_linear"]
        out["model"] = "linear"
    return out


def _extrapolate_ceiling_curve(
    curve: dict[float, dict[str, float]], target_frac: float = 2.0
) -> pl.DataFrame:
    """Build the per-metric ceiling table from the measured depth curve.

    Columns: ``metric``, the measured value at each depth (``m@<frac>``), the
    extrapolation to full depth (``extrap``) plus its ``extrap_linear`` companion,
    the fit residual (``resid``) and the model used (``model``).
    """
    fracs = sorted(curve.keys(), reverse=True)
    metrics = sorted({m for depth in curve.values() for m in depth})
    f_arr = np.array(fracs, dtype=float)

    rows: list[dict[str, Any]] = []
    for metric in metrics:
        vals = np.array([curve[fr].get(metric, np.nan) for fr in fracs], dtype=float)
        row: dict[str, Any] = {"metric": metric}
        for fr in fracs:
            row[f"m@{fr:g}"] = curve[fr].get(metric)
        row.update(_extrapolate_metric(f_arr, vals, target_frac))
        rows.append(row)
    return pl.DataFrame(rows)


def _build_anndata_pair(
    real: ad.AnnData | str,
    pred: ad.AnnData | str,
    control_pert: str,
    pert_col: str,
    allow_discrete: bool = False,
):
    if isinstance(real, str):
        logger.info(f"Reading real anndata from {real}")
        real = ad.read_h5ad(real)
    if isinstance(pred, str):
        logger.info(f"Reading pred anndata from {pred}")
        pred = ad.read_h5ad(pred)

    # Cast float16 to float32 since NUMBA (used by pdex) does not support float16
    _cast_float16_to_float32(real, which="real")
    _cast_float16_to_float32(pred, which="pred")

    # Validate that the input is normalized and log-transformed
    _convert_to_normlog(real, which="real", allow_discrete=allow_discrete)
    _convert_to_normlog(pred, which="pred", allow_discrete=allow_discrete)

    # Build the anndata pair
    return PerturbationAnndataPair(
        real=real, pred=pred, control_pert=control_pert, pert_col=pert_col
    )


def _convert_to_normlog(
    adata: ad.AnnData,
    which: str | None = None,
    allow_discrete: bool = False,
):
    """Performs a norm-log conversion if the input is integer data (inplace).

    Will skip if the input is not integer data.
    """
    if guess_is_lognorm(adata=adata, validate=not allow_discrete):
        logger.info(
            "Input is found to be log-normalized already - skipping transformation."
        )
        return  # Input is already log-normalized

    # User specified that they want to allow discrete data
    if allow_discrete:
        if which:
            logger.info(
                f"Discovered integer data for {which}. Configuration set to allow discrete. "
                "Make sure this is intentional."
            )
        else:
            logger.info(
                "Discovered integer data. Configuration set to allow discrete. "
                "Make sure this is intentional."
            )
        return  # proceed without conversion

    # Convert the data to norm-log
    if which:
        logger.info(f"Discovered integer data for {which}. Converting to norm-log.")
    sc.pp.normalize_total(adata=adata, inplace=True)  # normalize to median
    sc.pp.log1p(adata)  # log-transform (log1p)


def _build_de_comparison(
    anndata_pair: PerturbationAnndataPair | None = None,
    de_pred: pl.DataFrame | str | None = None,
    de_real: pl.DataFrame | str | None = None,
    num_threads: int = 1,
    allow_discrete: bool = False,
    outdir: str | None = None,
    prefix: str | None = None,
    pdex_kwargs: dict[str, Any] | None = None,
):
    return initialize_de_comparison(
        real=_load_or_build_de(
            mode="real",
            de_path=de_real,
            anndata_pair=anndata_pair,
            num_threads=num_threads,
            allow_discrete=allow_discrete,
            outdir=outdir,
            prefix=prefix,
            pdex_kwargs=pdex_kwargs or {},
        ),
        pred=_load_or_build_de(
            mode="pred",
            de_path=de_pred,
            anndata_pair=anndata_pair,
            num_threads=num_threads,
            allow_discrete=allow_discrete,
            outdir=outdir,
            prefix=prefix,
            pdex_kwargs=pdex_kwargs or {},
        ),
    )


def _build_pdex_kwargs(
    reference: str,
    groupby: str,
    threads: int,
    allow_discrete: bool,
    pdex_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pdex_kwargs = pdex_kwargs or {}
    if "reference" not in pdex_kwargs:
        pdex_kwargs["reference"] = reference
    if "groupby" not in pdex_kwargs:
        pdex_kwargs["groupby"] = groupby
    if "threads" not in pdex_kwargs:
        pdex_kwargs["threads"] = threads
    if "is_log1p" not in pdex_kwargs:
        if allow_discrete:
            pdex_kwargs["is_log1p"] = False
        else:
            pdex_kwargs["is_log1p"] = True
    # Keep cell-eval's default DE behavior unchanged from pdex<0.2.5: pin epsilon=0
    # (pdex>=0.2.5 defaults it to 1e-9) and leave the pooled-CPM floor filter OFF.
    # Both are opt-in — enable the filter via --cpm-filter / pdex_kwargs["cpm_filter"].
    if "epsilon" not in pdex_kwargs:
        pdex_kwargs["epsilon"] = 0.0
    return pdex_kwargs


def _load_or_build_de(
    mode: Literal["pred", "real"],
    de_path: pl.DataFrame | str | None = None,
    anndata_pair: PerturbationAnndataPair | None = None,
    num_threads: int = 1,
    outdir: str | None = None,
    prefix: str | None = None,
    allow_discrete: bool = False,
    pdex_kwargs: dict[str, Any] | None = None,
) -> pl.DataFrame:
    if de_path is None:
        if anndata_pair is None:
            raise ValueError("anndata_pair must be provided if de_path is not provided")
        logger.info(f"Computing DE for {mode} data")
        pdex_kwargs = _build_pdex_kwargs(
            reference=anndata_pair.control_pert,
            groupby=anndata_pair.pert_col,
            threads=num_threads,
            allow_discrete=allow_discrete,
            pdex_kwargs=pdex_kwargs or {},
        )
        logger.info(f"Using the following pdex kwargs: {pdex_kwargs}")
        frame = pdex(
            adata=anndata_pair.real if mode == "real" else anndata_pair.pred,
            mode="ref",
            **pdex_kwargs,
        )
        if outdir is not None:
            if prefix is not None:
                prefix = prefix.replace(
                    "/", "-"
                )  # some prefixes (e.g. HepG2/C3A) may have slashes in them
            pathname = f"{mode}_de.csv" if not prefix else f"{prefix}_{mode}_de.csv"
            logger.info(f"Writing {mode} DE results to: {pathname}")
            frame.write_csv(os.path.join(outdir, pathname))

        return frame  # type: ignore
    elif isinstance(de_path, str):
        logger.info(f"Reading {mode} DE results from {de_path}")
        if pdex_kwargs:
            logger.warning("pdex_kwargs are ignored when reading from a CSV file")
        return pl.read_csv(
            de_path,
            schema_overrides={
                "target": pl.Utf8,
                "feature": pl.Utf8,
            },
        )
    elif isinstance(de_path, pl.DataFrame):
        if pdex_kwargs:
            logger.warning("pdex_kwargs are ignored when reading from a CSV file")
        return de_path
    elif isinstance(de_path, pd.DataFrame):
        if pdex_kwargs:
            logger.warning("pdex_kwargs are ignored when reading from a CSV file")
        return pl.from_pandas(de_path)
    else:
        raise TypeError(f"Unexpected type for de_path: {type(de_path)}")
