# src/regularization/geodesic_loss.py
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from src.metrics.geodesic import GeodesicDistanceComputer


class GeodesicRatioRegularizer(nn.Module):
    """
    Регуляризатор, который контролирует отношение геодезического
    к евклидову расстоянию в пространстве представлений.

    Использует GeodesicDistanceComputer из метрик для общего кода.
    """

    def __init__(self, n_neighbors=10, target_ratio=1.0, lambda_reg=0.1):
        """
        Args:
            n_neighbors: количество соседей для графа
            target_ratio: целевое отношение (1.0 = плоское многообразие)
            lambda_reg: вес регуляризации
        """
        super().__init__()
        self.n_neighbors = n_neighbors
        self.target_ratio = target_ratio
        self.lambda_reg = lambda_reg
        self.computer = GeodesicDistanceComputer(n_neighbors=n_neighbors)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет штраф регуляризации за отклонение от целевого отношения.

        Args:
            embeddings: torch.Tensor (batch_size, embedding_dim)

        Returns:
            torch.Tensor: скаляр - значение регуляризации
        """
        # Конвертируем в numpy для вычисления расстояний
        X_np = embeddings.detach().cpu().numpy()
        n_samples = X_np.shape[0]

        if n_samples < self.n_neighbors + 2:
            return torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)

        try:
            ratios, mask = self.computer.compute_ratios(X_np)

            # Извлекаем только верхний треугольник (уникальные пары)
            upper_triangle = np.triu_indices(n_samples, k=1)
            valid_ratios = ratios[upper_triangle][mask[upper_triangle]]

            if len(valid_ratios) == 0:
                return torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)

            loss = np.mean((valid_ratios - self.target_ratio) ** 2)

            # Возвращаем взвешенный штраф как torch.Tensor
            return torch.tensor(loss, device=embeddings.device, dtype=embeddings.dtype) * self.lambda_reg

        except Exception as e:
            print(f"[WARNING] GeodesicRatioRegularizer error: {e}")
            return torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)


def _geodesic_ratio_loss(
    embeddings: torch.Tensor,
    n_neighbors: int = 10,
    target_ratio: float = 1.0,
    lambda_reg: float = 0.1,
) -> torch.Tensor:
    """
    Вычисляет геодезический ratio loss напрямую (функциональный стиль).

    Args:
        embeddings: torch.Tensor (batch_size, embedding_dim)
        n_neighbors: количество соседей для графа
        target_ratio: целевое отношение расстояний
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
    По аналогии с compute_geometric_loss() из geometric_loss.py

    Args:
        embeddings: torch.Tensor представления (batch_size, embedding_dim)
        labels: torch.Tensor метки класса (не используются в геодезических loss'ах)
        cfg: конфиг объект с параметрами регуляризации

    Returns:
        torch.Tensor: значение loss'а

    Поддерживаемые метрики (из cfg.metric):
        - "geodesic" - основной геодезический регуляризатор
        - "geodesic_ratio" - синоним для "geodesic"
        - "none" - отключить (возвращает 0)

    Параметры из cfg:
        - weight: вес регуляризации (если 0, возвращает 0)
        - geodesic_ratio.n_neighbors: количество соседей (default 10)
        - geodesic_ratio.target_ratio: целевое отношение (default 1.0)
        - geodesic_ratio.lambda_reg: вес loss (default 0.1)
    """

    # Обработка None конфига
    if cfg is None:
        return torch.zeros(1, device=embeddings.device, dtype=embeddings.dtype)

    # Получаем метрику
    metric = getattr(cfg, "metric", None)

    # Если метрика "none" или пусто, возвращаем нулевой loss
    if metric is None or metric == "none" or metric == "":
        return torch.zeros(1, device=embeddings.device, dtype=embeddings.dtype)

    # Проверяем вес - если 0, то не нужно считать loss
    weight = getattr(cfg, "weight", 0.1)
    if weight == 0:
        return torch.zeros(1, device=embeddings.device, dtype=embeddings.dtype)

    # Диспетчер по типам геодезических регуляризаторов
    if metric in {"geodesic", "geodesic_ratio"}:
        geo_cfg = getattr(cfg, "geodesic_ratio", {})
        n_neighbors = getattr(geo_cfg, "n_neighbors", 10)
        target_ratio = getattr(geo_cfg, "target_ratio", 1.0)
        lambda_reg = getattr(geo_cfg, "lambda_reg", 0.1)

        return _geodesic_ratio_loss(
            embeddings,
            n_neighbors=n_neighbors,
            target_ratio=target_ratio,
            lambda_reg=lambda_reg
        )

    # Неизвестная метрика
    print(f"[WARNING] Unknown geodesic regularization metric: {metric}, returning zero loss")
    return torch.zeros(1, device=embeddings.device, dtype=embeddings.dtype)
