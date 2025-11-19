# src/regularization/geometric_mark_loss.py
"""
Геометрический регуляризатор, основанный на контроле локальной размерности.
Дифференцируемый регуляризатор для латентного пространства.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class LocalDimensionComputer:
    """
    Вычислитель локальной внутренней размерности через k-NN граф.
    Использует дифференцируемые операции PyTorch.
    """

    def __init__(self, n_neighbors=10, eps=1e-6):
        """
        Args:
            n_neighbors: количество соседей для оценки
            eps: epsilon для избежания деления на ноль
        """
        self.n_neighbors = n_neighbors
        self.eps = eps

    def compute_local_dimension(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Вычисление локальной размерности через корреляционную размерность.

        Метод: используем k-NN расстояния для оценки размерности
        d̂ = -1 / mean(log(r_k / r_N))

        Args:
            embeddings: torch.Tensor (batch_size, embedding_dim)

        Returns:
            torch.Tensor (batch_size,) - оценка локальной размерности
        """
        device = embeddings.device
        n_samples = embeddings.shape[0]

        # Вычисляем матрицу расстояний
        # embeddings: [N, D] → distances: [N, N]
        distances = torch.cdist(embeddings, embeddings)

        # k = min(n_neighbors, n_samples - 1)
        k = min(self.n_neighbors, n_samples - 1)

        if k < 2:
            # Слишком мало точек, возвращаем константу
            return torch.ones(n_samples, device=device)

        # Получаем k+1 ближайших расстояний (включая саму точку)
        # sorted_distances: [N, N]
        sorted_distances = torch.sort(distances, dim=1)[0]

        # Берем расстояния до k-го и N-го соседей
        # r_k: расстояние до k-го соседа [N]
        # r_N: расстояние до самого дальнего соседа [N]
        r_k = sorted_distances[:, k]
        r_N = sorted_distances[:, -1]  # Последнее расстояние (максимальное)

        # Предотвращаем деление на ноль и логарифм от нуля
        r_N = torch.clamp(r_N, min=self.eps)
        r_k = torch.clamp(r_k, min=self.eps)

        # Вычисляем логарифмические отношения
        # log_ratios: [N]
        log_ratios = torch.log(r_k / r_N)

        # Избегаем NaN и inf
        valid_mask = ~(torch.isnan(log_ratios) | torch.isinf(log_ratios))

        # Оценка размерности
        local_dims = torch.ones(n_samples, device=device)

        if valid_mask.any():
            # d_hat = -1 / mean(log_ratios)
            valid_log_ratios = log_ratios[valid_mask]

            # Избегаем деления на очень маленькие числа
            mean_log_ratio = torch.mean(valid_log_ratios)
            mean_log_ratio = torch.clamp(mean_log_ratio, min=-1e6, max=-self.eps)

            # Вычисляем размерность
            d_hat = -1.0 / mean_log_ratio

            # Зажимаем в разумные границы [0.1, embedding_dim]
            d_hat = torch.clamp(d_hat, min=0.1, max=embeddings.shape[1] * 2.0)

            # Устанавливаем значения для валидных точек
            local_dims[valid_mask] = d_hat

        return local_dims


class LocalDimensionRegularizer(nn.Module):
    """
    Регуляризатор, контролирующий локальную размерность латентного пространства.
    Цель: сделать пространство более однородным по размерности.
    """

    def __init__(self, target_dimension=2.0, n_neighbors=10, lambda_reg=0.1):
        """
        Args:
            target_dimension: целевая локальная размерность (обычно 2-4)
            n_neighbors: количество соседей для оценки размерности
            lambda_reg: вес регуляризации
        """
        super().__init__()
        self.target_dimension = target_dimension
        self.n_neighbors = n_neighbors
        self.lambda_reg = lambda_reg
        self.computer = LocalDimensionComputer(n_neighbors=n_neighbors)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет штраф за отклонение локальной размерности от целевой.

        Args:
            embeddings: torch.Tensor (batch_size, embedding_dim)

        Returns:
            torch.Tensor: скаляр - значение регуляризации
        """
        # Вычисляем локальные размерности
        local_dims = self.computer.compute_local_dimension(embeddings)

        # MSE штраф за отклонение от целевой размерности
        dimension_penalty = torch.mean((local_dims - self.target_dimension) ** 2)

        # Штраф за высокую дисперсию размерностей (хотим однородность)
        dimension_variance = torch.var(local_dims)
        variance_penalty = dimension_variance

        # Комбинированный штраф
        total_penalty = dimension_penalty + 0.1 * variance_penalty

        return self.lambda_reg * total_penalty


class DimensionalityUniformityRegularizer(nn.Module):
    """
    Регуляризатор, контролирующий однородность размерности по всему пространству.
    Цель: минимизировать дисперсию локальной размерности.
    """

    def __init__(self, n_neighbors=10, lambda_reg=0.1, target_uniformity=0.5):
        """
        Args:
            n_neighbors: количество соседей
            lambda_reg: вес регуляризации
            target_uniformity: целевая однородность (меньше = более однородно)
        """
        super().__init__()
        self.n_neighbors = n_neighbors
        self.lambda_reg = lambda_reg
        self.target_uniformity = target_uniformity
        self.computer = LocalDimensionComputer(n_neighbors=n_neighbors)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Штраф за неоднородность размерности."""
        local_dims = self.computer.compute_local_dimension(embeddings)

        # Вычисляем коэффициент вариации
        mean_dim = torch.mean(local_dims)
        var_dim = torch.var(local_dims)

        # Коэффициент вариации = std / mean
        std_dim = torch.sqrt(var_dim + 1e-6)
        uniformity_score = std_dim / (torch.abs(mean_dim) + 1e-6)

        # Штраф за высокую неоднородность
        penalty = torch.relu(uniformity_score - self.target_uniformity)

        return self.lambda_reg * penalty


class CombinedGeometricRegularizer(nn.Module):
    """
    Комбинированный регуляризатор, объединяющий несколько геометрических констреинтов.
    """

    def __init__(
            self,
            target_dimension=2.0,
            target_uniformity=0.5,
            n_neighbors=10,
            lambda_dimension=0.1,
            lambda_uniformity=0.05,
    ):
        """
        Args:
            target_dimension: целевая размерность
            target_uniformity: целевая однородность
            n_neighbors: количество соседей
            lambda_dimension: вес штрафа за размерность
            lambda_uniformity: вес штрафа за однородность
        """
        super().__init__()

        self.dimension_regularizer = LocalDimensionRegularizer(
            target_dimension=target_dimension,
            n_neighbors=n_neighbors,
            lambda_reg=lambda_dimension
        )

        self.uniformity_regularizer = DimensionalityUniformityRegularizer(
            n_neighbors=n_neighbors,
            lambda_reg=lambda_uniformity,
            target_uniformity=target_uniformity
        )

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Вычисляет комбинированный штраф.

        Returns:
            Tuple[loss, metrics_dict]
        """
        dimension_loss = self.dimension_regularizer(embeddings)
        uniformity_loss = self.uniformity_regularizer(embeddings)

        total_loss = dimension_loss + uniformity_loss

        metrics = {
            'dimension_loss': dimension_loss.item(),
            'uniformity_loss': uniformity_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, metrics


def compute_geometric_loss(
        embeddings: torch.Tensor,
        labels: torch.Tensor = None,
        cfg: Optional[object] = None,
) -> torch.Tensor:
    """
    Диспетчер для геометрических регуляризаторов.
    По аналогии с compute_geodesic_loss().

    Args:
        embeddings: torch.Tensor (batch_size, embedding_dim)
        labels: torch.Tensor (batch_size,) - опционально
        cfg: OmegaConf config с параметрами

    Returns:
        torch.Tensor: скаляр - значение регуляризации
    """

    if cfg is None:
        return torch.zeros(1, device=embeddings.device, dtype=embeddings.dtype)

    # Получаем параметры из конфига
    metric = getattr(cfg, "metric", None)
    weight = getattr(cfg, "weight", 0.1)

    # Если отключено
    if metric is None or metric == "none" or metric == "" or weight == 0:
        return torch.zeros(1, device=embeddings.device, dtype=embeddings.dtype)

    # Параметры регуляризаторов
    n_neighbors = getattr(cfg, "n_neighbors", 10)
    target_dim = getattr(cfg, "target_dimension", 2.0)
    target_uniformity = getattr(cfg, "target_uniformity", 0.5)

    try:
        if metric in {"local_dimension", "dimension", "uniformity", "geometric_mark"}:
            regularizer = LocalDimensionRegularizer(
                target_dimension=target_dim,
                n_neighbors=n_neighbors,
                lambda_reg=weight
            )
            loss = regularizer(embeddings)
            return loss

        elif metric == "combined":
            regularizer = CombinedGeometricRegularizer(
                target_dimension=target_dim,
                target_uniformity=target_uniformity,
                n_neighbors=n_neighbors,
                lambda_dimension=weight,
                lambda_uniformity=weight * 0.5
            )
            loss, _ = regularizer(embeddings)
            return loss

        else:
            return torch.zeros(1, device=embeddings.device, dtype=embeddings.dtype)

    except Exception as e:
        print(f"[WARNING] Geometric regularization failed: {e}, returning zero loss")
        return torch.zeros(1, device=embeddings.device, dtype=embeddings.dtype)
