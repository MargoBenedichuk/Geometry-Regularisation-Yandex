# src/regularizers/geodesic_loss.py

import torch
import torch.nn as nn
import numpy as np
from src.metrics.geodesic_metrics import GeodesicDistanceComputer


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

    def forward(self, embeddings):
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
            print(f"Warning in GeodesicRatioRegularizer: {e}")
            return torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)
