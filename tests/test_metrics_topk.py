import numpy as np
import pytest
import anndata as ad

from cell_eval._types import PerturbationAnndataPair
from cell_eval.metrics import top_k_accuracy


def _make_anndata(matrix: np.ndarray, perts: list[str], genes: list[str]) -> ad.AnnData:
    adata = ad.AnnData(X=matrix.astype(np.float64))
    adata.obs["pert"] = perts
    adata.var_names = genes
    return adata


def _build_pair(real_matrix: np.ndarray, pred_matrix: np.ndarray) -> PerturbationAnndataPair:
    genes = ["g1", "g2"]
    perts = ["ctrl", "ctrl", "A", "A", "B", "B"]

    adata_real = _make_anndata(real_matrix, perts, genes)
    adata_pred = _make_anndata(pred_matrix, perts, genes)

    return PerturbationAnndataPair(
        real=adata_real,
        pred=adata_pred,
        pert_col="pert",
        control_pert="ctrl",
    )


def test_topk_accuracy_perfect_match() -> None:
    real_matrix = np.array(
        [
            [0.0, 0.0],
            [0.1, -0.1],
            [1.0, 0.0],
            [1.0, 0.1],
            [-1.0, 0.0],
            [-1.0, -0.1],
        ]
    )

    pred_matrix = np.array(
        [
            [0.0, 0.05],
            [0.05, -0.05],
            [1.05, 0.05],
            [0.95, -0.05],
            [-1.05, 0.0],
            [-0.95, 0.05],
        ]
    )

    pair = _build_pair(real_matrix, pred_matrix)

    scores = top_k_accuracy(pair, k=1)

    assert scores["A"] == pytest.approx(1.0)
    assert scores["B"] == pytest.approx(1.0)


def test_topk_accuracy_mismatch() -> None:
    real_matrix = np.array(
        [
            [0.0, 0.0],
            [0.1, -0.1],
            [1.0, 0.0],
            [1.0, 0.1],
            [-1.0, 0.0],
            [-1.0, -0.1],
        ]
    )

    pred_matrix = np.array(
        [
            [0.0, 0.05],
            [0.05, -0.05],
            [1.05, 0.05],
            [0.95, -0.05],
            [2.0, 2.0],
            [2.2, 1.8],
        ]
    )

    pair = _build_pair(real_matrix, pred_matrix)

    scores = top_k_accuracy(pair, k=1)

    assert scores["A"] == pytest.approx(1.0)
    assert scores["B"] == pytest.approx(0.0)
