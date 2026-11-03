from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from geomloss import SamplesLoss
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier


def _sinkhorn_loss() -> SamplesLoss:
    return SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.5, debias=True, backend="tensorized")


def sinkhorn_ot(true_data: np.ndarray, pred_data: np.ndarray) -> float:
    true_data = np.asarray(true_data)
    pred_data = np.asarray(pred_data)
    true_tensor = torch.as_tensor(true_data, dtype=torch.float64)
    pred_tensor = torch.as_tensor(pred_data, dtype=torch.float64)
    return float(_sinkhorn_loss()(true_tensor, pred_tensor).item())


def weighted_sinkhorn_ot(true_data: np.ndarray, pred_data: np.ndarray, pred_weights: np.ndarray) -> float:
    true_tensor = torch.as_tensor(np.asarray(true_data), dtype=torch.float64)
    pred_tensor = torch.as_tensor(np.asarray(pred_data), dtype=torch.float64)
    true_mass = torch.full((len(true_tensor),), 1.0 / len(true_tensor), dtype=torch.float64)
    pred_mass = torch.as_tensor(pred_weights / pred_weights.sum(), dtype=torch.float64)
    return float(_sinkhorn_loss()(true_mass, true_tensor, pred_mass, pred_tensor).item())


def gene_time_correlation_median(
    true_trajectory: Sequence[np.ndarray],
    pred_trajectory: Sequence[np.ndarray],
    pred_weights: Sequence[np.ndarray] | None = None,
    top_k: int = 200,
) -> float:
    true_means = np.stack([cells.mean(0) for cells in true_trajectory])
    if pred_weights is None:
        pred_means = np.stack([cells.mean(0) for cells in pred_trajectory])
    else:
        pred_means = np.stack(
            [(cells * (w / w.sum())[:, None]).sum(0) for cells, w in zip(pred_trajectory, pred_weights)]
        )
    centered_true = true_means - true_means.mean(0)
    centered_pred = pred_means - pred_means.mean(0)
    denominator = np.sqrt((centered_true**2).sum(0) * (centered_pred**2).sum(0))
    correlation = np.where(
        denominator > 1e-12, (centered_true * centered_pred).sum(0) / np.maximum(denominator, 1e-12), np.nan
    )
    top_genes = np.argsort(true_means.var(0))[::-1][:top_k]
    return float(np.nanmedian(correlation[top_genes]))


class CellTypeProportions:
    def __init__(self, train_cells: np.ndarray, train_labels: np.ndarray, n_components: int = 50, n_neighbors: int = 15):
        self.pca = PCA(n_components, random_state=0).fit(train_cells)
        self.knn = KNeighborsClassifier(n_neighbors).fit(self.pca.transform(train_cells), train_labels)
        self.classes = list(self.knn.classes_)

    def __call__(self, cells: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        labels = self.knn.predict(self.pca.transform(cells))
        weights = np.ones(len(cells)) if weights is None else weights
        return np.array([weights[labels == cell_type].sum() for cell_type in self.classes]) / weights.sum()


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())
