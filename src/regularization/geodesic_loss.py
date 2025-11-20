# src/regularization/geodesic_loss.py
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import numpy as np
from src.metrics.geodesic import GeodesicDistanceComputer


class GeodesicRatioRegularizer(nn.Module):
    """Регуляризатор для контроля геодезического расстояния."""

    def __init__(self, n_neighbors=15, target_ratio=1.8, lambda_reg=0.1):
        """
        Args:
            n_neighbors: количество соседей (минимум 10!)
            target_ratio: целевое отношение geo/euc (>1.0!)
            lambda_reg: вес регуляризации
        """
        super().__init__()
        self.n_neighbors = max(int(n_neighbors), 15)  # Гарантируем минимум 15
        self.target_ratio = float(target_ratio)
        self.lambda_reg = float(lambda_reg)
        self.computer = GeodesicDistanceComputer(n_neighbors=self.n_neighbors)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch_size, embedding_dim)

        Returns:
            torch.Tensor: скаляр loss
        """
        X_np = embeddings.detach().cpu().numpy()
        n_samples = X_np.shape[0]

        if n_samples < self.n_neighbors + 2:
            return torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)

        try:
            ratios = self.computer.compute_ratios_efficient(X_np)

            if len(ratios) == 0 or np.all(ratios == 1.0):
                return torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)

            # MSE между actual и target ratios
            loss = np.mean((ratios - self.target_ratio) ** 2)
            loss_tensor = torch.tensor(loss, device=embeddings.device, dtype=embeddings.dtype)

            print(
                f"[DEBUG loss] Geodesic: {(loss_tensor * self.lambda_reg).item():.6f} (mean_ratio={np.mean(ratios):.4f}, target={self.target_ratio})")

            return loss_tensor * self.lambda_reg

        except Exception as e:
            print(f"[ERROR] GeodesicRatioRegularizer: {e}")
            import traceback
            traceback.print_exc()
            return torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)


def _geodesic_ratio_loss(
        embeddings: torch.Tensor,
        n_neighbors: int = 15,
        target_ratio: float = 1.8,
        lambda_reg: float = 0.1,
) -> torch.Tensor:
    """
    Вычисляет геодезический ratio loss.

    Args:
        embeddings: (batch_size, embedding_dim)
        n_neighbors: количество соседей
        target_ratio: целевое отношение
        lambda_reg: вес регуляризации

    Returns:
        torch.Tensor: скаляр loss
    """
    regularizer = GeodesicRatioRegularizer(
        n_neighbors=n_neighbors,
        target_ratio=target_ratio,
        lambda_reg=lambda_reg
    )
    return regularizer(embeddings)


def compute_geodesic_loss(
        embeddings: torch.Tensor,
        labels: torch.Tensor = None,
        cfg: Optional[object] = None,
) -> torch.Tensor:
    """
    Диспетчер для геодезических регуляризаторов.

    Args:
        embeddings: (batch_size, embedding_dim)
        labels: не используется
        cfg: конфиг с параметрами

    Returns:
        torch.Tensor: скаляр loss
    """
    if cfg is None:
        return torch.zeros(1, device=embeddings.device, dtype=embeddings.dtype)

    metric = getattr(cfg, "metric", None)
    if metric is None or metric == "none" or metric == "":
        return torch.zeros(1, device=embeddings.device, dtype=embeddings.dtype)

    weight = getattr(cfg, "weight", 0.1)
    if weight == 0:
        return torch.zeros(1, device=embeddings.device, dtype=embeddings.dtype)

    if metric in {"geodesic", "geodesic_ratio"}:
        geo_cfg = getattr(cfg, "geodesic_ratio", {})
        n_neighbors = max(int(getattr(geo_cfg, "n_neighbors", 15)), 15)
        target_ratio = float(getattr(geo_cfg, "target_ratio", 1.8))
        lambda_reg = float(getattr(geo_cfg, "lambda_reg", 0.1))

        return _geodesic_ratio_loss(
            embeddings,
            n_neighbors=n_neighbors,
            target_ratio=target_ratio,
            lambda_reg=lambda_reg
        )

    return torch.zeros(1, device=embeddings.device, dtype=embeddings.dtype)
