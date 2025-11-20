import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import shortest_path


class GeodesicEuclideanRatioMetric:
    """
    Метрика отношения геодезического расстояния к евклидову расстоянию.
    Используется для оценки нелинейности многообразия данных.
    """

    def __init__(self, n_neighbors=5):
        """
        Args:
            n_neighbors: количество ближайших соседей для построения графа
        """
        self.n_neighbors = n_neighbors

    def compute_euclidean_distances(self, X):
        """
        Вычисление попарных евклидовых расстояний.

        Args:
            X: np.ndarray или torch.Tensor, размер (n_samples, n_features)

        Returns:
            np.ndarray: матрица расстояний (n_samples, n_samples)
        """
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()

        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        euclidean_dist = np.sqrt(np.sum(diff ** 2, axis=2))

        return euclidean_dist

    def compute_geodesic_distances(self, X):
        """
        Вычисление геодезических расстояний через граф k-ближайших соседей.

        Args:
            X: np.ndarray или torch.Tensor, размер (n_samples, n_features)

        Returns:
            np.ndarray: матрица геодезических расстояний (n_samples, n_samples)
        """
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()

        n_samples = X.shape[0]

        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors + 1,
                                algorithm='auto',
                                metric='euclidean').fit(X)

        distances, indices = nbrs.kneighbors(X)

        graph = np.full((n_samples, n_samples), np.inf)

        for i in range(n_samples):
            for j, neighbor_idx in enumerate(indices[i]):
                if neighbor_idx != i:
                    graph[i, neighbor_idx] = distances[i, j]
                    graph[neighbor_idx, i] = distances[i, j]

        # Вычисление кратчайших путей (алгоритм Floyd-Warshall или Dijkstra)
        ### Работает крайне долго
        geodesic_dist = shortest_path(graph, method='auto', directed=False)
        return geodesic_dist

    def compute_ratio(self, X, sample_pairs=None):
        """
        Вычисление отношения геодезического расстояния к евклидову.

        Args:
            X: np.ndarray или torch.Tensor, размер (n_samples, n_features)
            sample_pairs: list of tuples или None - пары индексов для вычисления,
                         если None, вычисляет для всех пар

        Returns:
            float: среднее отношение geodesic/euclidean
            dict: статистика (mean, std, min, max)
        """
        euclidean_dist = self.compute_euclidean_distances(X)
        geodesic_dist = self.compute_geodesic_distances(X)

        # Избегаем деления на ноль и бесконечных расстояний
        mask = (euclidean_dist > 1e-10) & (geodesic_dist < np.inf)

        if sample_pairs is None:
            # Вычисляем для всех пар (исключая диагональ)
            n = X.shape[0] if isinstance(X, np.ndarray) else X.shape[0]
            mask = mask & (np.eye(n) == 0)

            ratios = geodesic_dist[mask] / euclidean_dist[mask]
        else:
            # Вычисляем только для заданных пар
            ratios = []
            for i, j in sample_pairs:
                if mask[i, j]:
                    ratios.append(geodesic_dist[i, j] / euclidean_dist[i, j])
            ratios = np.array(ratios)

        stats = {
            'mean': np.mean(ratios),
            'std': np.std(ratios),
            'min': np.min(ratios),
            'max': np.max(ratios)
        }

        return stats['mean'], stats


def example():
    from sklearn.datasets import make_swiss_roll

    X, _ = make_swiss_roll(n_samples=500, noise=0.1, random_state=42)

    metric = GeodesicEuclideanRatioMetric(n_neighbors=10)
    mean_ratio, stats = metric.compute_ratio(X)

    print(f"Среднее отношение Geodesic/Euclidean: {mean_ratio:.4f}")
    print(f"Статистика: {stats}")

    X_torch = torch.tensor(X, dtype=torch.float32)
    mean_ratio_torch, stats_torch = metric.compute_ratio(X_torch)
    print(f"\nС PyTorch тензором: {mean_ratio_torch:.4f}")
