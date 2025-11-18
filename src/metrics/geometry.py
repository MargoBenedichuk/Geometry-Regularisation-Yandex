# src/metrics/geometry.py

import torch
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.manifold import trustworthiness

from src.metrics.geodesic_metrics import GeodesicMetric


class LocalIntrinsicDimension:
    """Вычисление локальной внутренней размерности."""

    @staticmethod
    def compute(embeddings, k=10):
        """
        Вычисляет локальную размерность для каждой точки.

        Args:
            embeddings: np.ndarray (n_samples, embedding_dim)
            k: количество соседей

        Returns:
            np.ndarray: локальная размерность для каждой точки
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        n_points = embeddings.shape[0]

        if n_points < k + 2:
            return np.ones(n_points)

        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=min(n_points, k + 2)).fit(embeddings)
        dists_nn, _ = nbrs.kneighbors(embeddings)

        local_dims = []

        for i in range(n_points):
            dists = dists_nn[i][1:k + 2]  # Исключаем саму точку

            if len(dists) < 3 or np.any(dists < 1e-8):
                local_dims.append(1.0)
                continue

            log_ratios = np.log(dists[:-1] / dists[-1])

            if np.any(np.isnan(log_ratios)) or np.any(np.isinf(log_ratios)):
                local_dims.append(1.0)
                continue

            d_hat = -1 / np.mean(log_ratios)
            d_hat = np.clip(d_hat, 0.1, embeddings.shape[1] * 2.0)
            local_dims.append(float(d_hat))

        return np.array(local_dims)


def compute_geometry_summary(embeddings, labels):
    """
    Вычисляет полную геометрическую статистику пространства представлений.

    Args:
        embeddings: np.ndarray (n_samples, embedding_dim)
        labels: np.ndarray (n_samples,)

    Returns:
        dict: полная статистика
    """
    summary = {}

    # === 1. Локальная размерность ===
    local_dims = LocalIntrinsicDimension.compute(embeddings, k=10)
    summary['local_dimension'] = {
        'mean': float(np.mean(local_dims)),
        'std': float(np.std(local_dims)),
        'min': float(np.min(local_dims)),
        'max': float(np.max(local_dims))
    }

    # === 2. Геодезические расстояния ===
    geo_metric = GeodesicMetric(n_neighbors=10)

    # Глобальная статистика
    geo_global = geo_metric.compute_global_stats(embeddings)
    summary['geodesic_global'] = geo_global

    # Поклассовая статистика
    geo_class = geo_metric.compute_class_wise_stats(embeddings, labels)
    summary['geodesic_class_wise'] = geo_class

    # Межклассовая статистика
    geo_inter = geo_metric.compute_inter_class_stats(embeddings, labels)
    summary['geodesic_inter_class'] = geo_inter

    # === 3. Качество кластеризации ===
    if len(np.unique(labels)) > 1 and len(embeddings) > 1:
        try:
            silhouette = silhouette_score(embeddings, labels)
            summary['silhouette_score'] = float(silhouette)
        except:
            summary['silhouette_score'] = 0.0
    else:
        summary['silhouette_score'] = 0.0

    # === 4. Trustworthiness (сохранение локальной структуры) ===
    try:
        # Trustworthiness измеряет, насколько хорошо сохранена локальная структура
        trust = trustworthiness(embeddings, embeddings, n_neighbors=min(10, len(embeddings) - 1))
        summary['trustworthiness'] = float(trust)
    except:
        summary['trustworthiness'] = 0.0

    # === 5. Общая информация ===
    summary['num_samples'] = int(len(embeddings))
    summary['embedding_dim'] = int(embeddings.shape[1])
    summary['num_classes'] = int(len(np.unique(labels)))

    return summary
