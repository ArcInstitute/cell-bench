# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**cell-eval** is a Python package and CLI tool for evaluating the performance of models that predict cellular responses to perturbations at the single-cell level. Developed by the Arc Research Institute.

It generally revolves around a *real* anndata and a *predicted* anndata where it measures the general differences between the two across a variety of metrics.

- Python 3.11–3.12, managed with **UV** and built with **hatchling**
- CLI entry point: `cell-eval` (defined in `src/cell_eval/__main__.py`)

## Common Commands

```bash
# Install dependencies
uv sync --all-extras --dev

# Run all tests
uv run pytest -v

# Run a single test
uv run pytest tests/test_eval.py::test_broken_adata_not_normlog -v

# Formatting (check / fix)
uv run ruff format --check
uv run ruff format

# Type checking
uv run ty check

# Verify CLI works
uv run cell-eval --help
```

CI runs: formatting, typing, pytest, and cli-test (see `.github/workflows/CI.yml`).

## Architecture

### Core Data Flow

```
AnnData inputs (predicted + real)
  → MetricsEvaluator (validation, normalization, DE computation)
    → MetricPipeline (profile-based metric selection + execution)
      → metrics_registry (global MetricRegistry instance)
        → individual metric functions
  → polars DataFrames (per-perturbation + aggregated results)
```

### Key Abstractions

- **`MetricsEvaluator`** (`src/cell_eval/_evaluator.py`) — Main programmatic entry point. Validates input AnnData objects, computes differential expression via `pdex`, and orchestrates the metric pipeline. `compute_ceiling()` estimates a per-metric data ceiling (upper bound) from the real data alone: it splits the data into two *disjoint* halves of `n/2` cells (no cell in both), runs the full pipeline (DE computed in-memory) on that self-split, and averages each metric over perturbations and maps the reliability metrics in the explicit `SB_METRICS` list from half depth to full depth with the Spearman-Brown correction `r'=2r/(1+r)` (all other metrics — error, counts, `clustering_agreement`, `pearson_edistance` — emitted as NaN). Returns `(results, agg_results)`: the raw per-perturbation self-split and the SB-corrected per-metric ceiling. Exposed via the `run --ceiling` / `--ceiling-seed` CLI flags (additive: writes `ceiling_results.csv` / `agg_ceiling_results.csv`).

- **`MetricRegistry`** (`src/cell_eval/metrics/_registry.py`) — Global singleton `metrics_registry`. Metrics are registered with a name, type (`DE` or `ANNDATA_PAIR`), compute function, and best-value indicator. Supports both plain functions and class-based metrics requiring instantiation.

- **`MetricPipeline`** (`src/cell_eval/_pipeline/_runner.py`) — Selects and runs metrics based on a profile (`full`, `minimal`, `vcc`, `de`, `anndata`, `pds`). Collects per-perturbation results and aggregates them.

- **`Metric` protocol** (`src/cell_eval/metrics/base.py`) — All metric functions take either a `PerturbationAnndataPair` or `DEComparison` and return `float | dict[str, float]`.

- **Type system** (`src/cell_eval/_types/`) — Immutable dataclasses: `PerturbationAnndataPair`, `DEComparison`, plus enums `MetricType`, `MetricBestValue`, `DESortBy`.

### Metrics

Metrics are split into two categories registered in `src/cell_eval/metrics/_impl.py`:

- **AnnData metrics** (`_anndata.py`): pearson_delta, mse, mae, mse_delta, mae_delta, discrimination_score, clustering_agreement, edistance
- **DE metrics** (`_de.py`): overlap/precision at N, spearman correlations, direction match, significant gene recall, ROC/PR AUC

### CLI

Subcommands in `src/cell_eval/_cli/`: `prep` (data preparation for VCC), `run` (evaluation; `--ceiling`/`--ceiling-seed` additionally emit a real-data ceiling), `baseline` (create baseline), `score` (normalize against baseline). CLI defaults are in `_cli/_const.py`.

### Test Data Utilities

`cell_eval.data` provides `build_random_anndata()` and `downsample_cells()` for generating synthetic AnnData objects in tests.

## Conventions

- Uses `polars` (not pandas) for DataFrames
- Uses `match`/`case` statements (Python 3.10+ syntax)
- Type hints throughout; PEP 561 `py.typed` marker present
- Private modules prefixed with `_` (public API is re-exported from `__init__.py`)
