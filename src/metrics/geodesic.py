# src/metrics/geodesic.py

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
import warnings

warnings.filterwarnings('ignore')


class GeodesicDistanceComputer:
    """Исправленный вычислитель геодезических расстояний."""

    def __init__(self, n_neighbors=15, eps=1e-8, max_geodesic=1e6, subsample_size=2000):
        """
        Args:
            n_neighbors: количество соседей для графа
            subsample_size: размер подвыборки для больших датасетов
        """
        self.n_neighbors = max(int(n_neighbors), 15)
        self.eps = eps
        self.max_geodesic = max_geodesic
        self.subsample_size = subsample_size

    def _subsample_if_needed(self, X):
        """Субсэмплировать большие датасеты."""
        n_samples = X.shape[0]

        if n_samples > self.subsample_size:
            np.random.seed(42)
            indices = np.random.choice(n_samples, size=self.subsample_size, replace=False)
            return X[indices], indices

        return X, np.arange(n_samples)

    def compute_euclidean_neighbors(self, X):
        """Получить k ближайших соседей."""
        n_samples = X.shape[0]
        k = min(self.n_neighbors, n_samples - 1)

        if k < 2:
            return np.array([]), np.array([])

        nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=-1).fit(X)
        distances, indices = nbrs.kneighbors(X)

        return distances, indices

    def compute_geodesic_distances_fast(self, X):
        """Вычислить shortest paths через граф соседей."""
        n_samples = X.shape[0]

        distances, indices = self.compute_euclidean_neighbors(X)

        if distances.size == 0:
            print("[ERROR] No neighbors found")
            return None, None, None

        # Построить граф из соседей
        row, col, data = [], [], []
        for i in range(n_samples):
            for j, neighbor_idx in enumerate(indices[i]):
                row.append(i)
                col.append(neighbor_idx)
                data.append(distances[i, j])

        if not row:
            print("[ERROR] Empty graph")
            return None, None, None

        graph_sparse = csr_matrix((data, (row, col)), shape=(n_samples, n_samples))

        try:
            geodesic_dist = shortest_path(
                graph_sparse,
                method='D',
                directed=False,
                return_predecessors=False
            )

            if hasattr(geodesic_dist, 'toarray'):
                geodesic_dist = geodesic_dist.toarray()

            geodesic_dist[geodesic_dist == np.inf] = self.max_geodesic
            return distances, indices, geodesic_dist

        except Exception as e:
            print(f"[ERROR] Geodesic computation: {e}")
            return None, None, None

    def compute_ratios_efficient(self, X):
        """
        Вычисляем ratios ТОЛЬКО для НЕ-соседей!

        Потому что:
        - Для соседей: geo_dist = euc_dist (они уже в графе) → ratio = 1.0
        - Для НЕ-соседей: geo_dist > euc_dist (окольный путь) → ratio > 1.0
        """
        n_samples = X.shape[0]

        if n_samples < self.n_neighbors + 2:
            return np.array([1.0])

        X_work, _ = self._subsample_if_needed(X)
        n_work = X_work.shape[0]

        distances, indices, geodesic_dist = self.compute_geodesic_distances_fast(X_work)

        if distances is None or geodesic_dist is None:
            return np.array([1.0])

        # Нужны ВСЕ попарные евклидовы расстояния для поиска НЕ-соседей
        print(f"[DEBUG geo] Computing all pairwise euclidean distances...")
        euc_all = squareform(pdist(X_work, metric='euclidean'))

        # Набор соседей (для быстрого поиска)
        neighbor_set = set()
        for i in range(n_work):
            for neighbor_idx in indices[i]:
                neighbor_set.add((i, neighbor_idx))
                neighbor_set.add((neighbor_idx, i))

        ratios_list = []

        for i in range(n_work):
            for j in range(i + 1, n_work):
                if (i, j) not in neighbor_set:
                    euc_dist = euc_all[i, j]
                    geo_dist = geodesic_dist[i, j]

                    if euc_dist > self.eps and 0 < geo_dist < self.max_geodesic:
                        ratio = geo_dist / euc_dist
                        if 0.5 < ratio < 100:
                            ratios_list.append(ratio)

        if len(ratios_list) == 0:
            return np.array([1.0])

        return np.array(ratios_list)


class GeodesicMetric:
    """Метрика для анализа геодезических расстояний."""

    def __init__(self, n_neighbors=15, subsample_size=2000):
        self.computer = GeodesicDistanceComputer(
            n_neighbors=n_neighbors,
            subsample_size=subsample_size
        )

    def compute_global_stats(self, embeddings):
        """Вычисление глобальной статистики."""
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        n_samples = embeddings.shape[0]

        if n_samples < self.computer.n_neighbors + 2:
            return self._empty_stats(n_samples)

        try:
            ratios = self.computer.compute_ratios_efficient(embeddings)

            if len(ratios) == 0 or np.all(np.abs(ratios - 1.0) < 1e-6):
                print("[WARNING] All ratios are 1.0 - this shouldn't happen with non-neighbors!")
                return self._empty_stats(n_samples)

            return {
                'mean': float(np.mean(ratios)),
                'std': float(np.std(ratios)),
                'min': float(np.min(ratios)),
                'max': float(np.max(ratios)),
                'median': float(np.median(ratios)),
                'num_samples': n_samples,
                'num_valid_pairs': int(len(ratios))
            }

        except Exception as e:
            print(f"[ERROR] Global stats: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_stats(n_samples)

    @staticmethod
    def _empty_stats(n_samples):
        return {
            'mean': 1.0,
            'std': 0.0,
            'min': 1.0,
            'max': 1.0,
            'median': 1.0,
            'num_samples': n_samples,
            'num_valid_pairs': 0
        }


def compute_geodesic_summary(embeddings, labels, n_neighbors=15, subsample_size=2000):
    """Вычисление геодезической статистики."""
    print("\n[INFO] Computing geodesic summary...")

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()

    try:
        geo_metric = GeodesicMetric(n_neighbors=n_neighbors, subsample_size=subsample_size)
        geo_global = geo_metric.compute_global_stats(embeddings)

        return {
            'global': geo_global,
            'n_neighbors': n_neighbors
        }

    except Exception as e:
        print(f"[ERROR] Failed to compute geodesic metrics: {e}")
        import traceback
        traceback.print_exc()
        return {
            'global': GeodesicMetric._empty_stats(len(embeddings)),
            'n_neighbors': n_neighbors
        }