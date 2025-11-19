# src/metrics/geometry.py

import torch
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.manifold import trustworthiness


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


def compute_geometry_summary(embeddings, labels, k=10):
    """
    Вычисляет геометрическую статистику пространства представлений.
    ТОЛЬКО локальная размерность, silhouette и trustworthiness.

    Args:
        embeddings: np.ndarray (n_samples, embedding_dim) или torch.Tensor
        labels: np.ndarray (n_samples,) или torch.Tensor
        k: количество соседей для локальной размерности

    Returns:
        dict: геометрическая статистика
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    summary = {}

    # === 1. Локальная размерность ===
    try:
        local_dims = LocalIntrinsicDimension.compute(embeddings, k=k)
        summary['local_dimension'] = {
            'mean': float(np.mean(local_dims)),
            'std': float(np.std(local_dims)),
            'min': float(np.min(local_dims)),
            'max': float(np.max(local_dims)),
            'median': float(np.median(local_dims))
        }
    except Exception as e:
        print(f"Warning: Failed to compute local dimension: {e}")
        summary['local_dimension'] = {
            'mean': 1.0,
            'std': 0.0,
            'min': 1.0,
            'max': 1.0,
            'median': 1.0
        }

    # === 2. Silhouette Score (качество кластеризации) ===
    if len(np.unique(labels)) > 1 and len(embeddings) > 1:
        try:
            silhouette = silhouette_score(embeddings, labels)
            summary['silhouette_score'] = float(silhouette)
        except Exception as e:
            print(f"Warning: Failed to compute silhouette score: {e}")
            summary['silhouette_score'] = 0.0
    else:
        summary['silhouette_score'] = 0.0

    # === 3. Trustworthiness (сохранение локальной структуры) ===
    try:
        k_trust = min(k, len(embeddings) - 1)
        if k_trust > 0:
            trust = trustworthiness(embeddings, embeddings, n_neighbors=k_trust)
            summary['trustworthiness'] = float(trust)
        else:
            summary['trustworthiness'] = 0.0
    except Exception as e:
        print(f"Warning: Failed to compute trustworthiness: {e}")
        summary['trustworthiness'] = 0.0

    # === 4. Общая информация ===
    summary['num_samples'] = int(len(embeddings))
    summary['embedding_dim'] = int(embeddings.shape[1])
    summary['num_classes'] = int(len(np.unique(labels)))

    return summary
