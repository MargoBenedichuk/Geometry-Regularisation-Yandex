# src/metrics/geodesic_metrics.py

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import shortest_path


class GeodesicDistanceComputer:
    """
    Базовый класс для вычисления геодезических и евклидовых расстояний.
    Используется как метриками, так и регуляризаторами.
    """

    def __init__(self, n_neighbors=10, eps=1e-6, max_geodesic=1e6):
        """
        Args:
            n_neighbors: количество соседей для построения графа
            eps: epsilon для избежания деления на ноль
            max_geodesic: максимальное геодезическое расстояние (обрезка inf)
        """
        self.n_neighbors = n_neighbors
        self.eps = eps
        self.max_geodesic = max_geodesic

    def compute_euclidean(self, X):
        """
        Вычисление матрицы евклидовых расстояний.

        Args:
            X: np.ndarray (n_samples, feature_dim)

        Returns:
            np.ndarray: матрица расстояний (n_samples, n_samples)
        """
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))

    def compute_geodesic(self, X):
        """
        Вычисление матрицы геодезических расстояний через k-NN граф.

        Args:
            X: np.ndarray (n_samples, feature_dim)

        Returns:
            np.ndarray: матрица геодезических расстояний (n_samples, n_samples)
        """
        n_samples = X.shape[0]

        # Построение графа k-ближайших соседей
        k = min(self.n_neighbors + 1, n_samples - 1)
        if k < 2:
            # Если слишком мало точек, возвращаем евклидовы расстояния
            return self.compute_euclidean(X)

        nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(X)
        distances, indices = nbrs.kneighbors(X)

        # Построение матрицы смежности графа
        graph = np.full((n_samples, n_samples), np.inf)

        for i in range(n_samples):
            for j, neighbor_idx in enumerate(indices[i]):
                if neighbor_idx != i:
                    graph[i, neighbor_idx] = distances[i, j]
                    graph[neighbor_idx, i] = distances[i, j]

        # Вычисление кратчайших путей
        geodesic_dist = shortest_path(graph, method='auto', directed=False)

        # Обрезка бесконечных расстояний
        geodesic_dist[geodesic_dist == np.inf] = self.max_geodesic

        return geodesic_dist

    def compute_ratios(self, X):
        """
        Вычисление матрицы отношений геодезических к евклидовым расстояниям.

        Args:
            X: np.ndarray (n_samples, feature_dim)

        Returns:
            tuple: (ratios_matrix, mask_valid)
                - ratios_matrix: np.ndarray с отношениями
                - mask_valid: np.ndarray булева маска валидных пар
        """
        euclidean_dist = self.compute_euclidean(X)
        geodesic_dist = self.compute_geodesic(X)

        # Маска для валидных пар (не ноль, не бесконечность)
        mask = (euclidean_dist > self.eps) & (geodesic_dist < self.max_geodesic)

        # Вычисление отношений
        ratios = np.ones_like(euclidean_dist)
        ratios[mask] = geodesic_dist[mask] / euclidean_dist[mask]

        return ratios, mask


class GeodesicMetric:
    """
    Метрика для анализа геодезических расстояний в пространстве представлений.
    Используется ТОЛЬКО для оценки (не влияет на обучение).
    """

    def __init__(self, n_neighbors=10):
        """
        Args:
            n_neighbors: количество соседей для графа
        """
        self.computer = GeodesicDistanceComputer(n_neighbors=n_neighbors)

    def compute_global_stats(self, embeddings):
        """
        Вычисляет глобальную статистику отношения расстояний.

        Args:
            embeddings: np.ndarray (n_samples, embedding_dim) или torch.Tensor

        Returns:
            dict: статистика
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        n_samples = embeddings.shape[0]

        # Проверка минимального размера
        if n_samples < self.computer.n_neighbors + 2:
            return self._empty_stats(n_samples)

        ratios, mask = self.computer.compute_ratios(embeddings)

        # Извлекаем только верхний треугольник (уникальные пары)
        upper_triangle = np.triu_indices(n_samples, k=1)
        valid_ratios = ratios[upper_triangle][mask[upper_triangle]]

        if len(valid_ratios) == 0:
            return self._empty_stats(n_samples)

        return {
            'mean': float(np.mean(valid_ratios)),
            'std': float(np.std(valid_ratios)),
            'min': float(np.min(valid_ratios)),
            'max': float(np.max(valid_ratios)),
            'median': float(np.median(valid_ratios)),
            'q25': float(np.percentile(valid_ratios, 25)),
            'q75': float(np.percentile(valid_ratios, 75)),
            'num_samples': n_samples,
            'num_valid_pairs': int(len(valid_ratios))
        }

    def compute_class_wise_stats(self, embeddings, labels):
        """
        Вычисляет статистику для каждого класса отдельно.

        Args:
            embeddings: np.ndarray (n_samples, embedding_dim)
            labels: np.ndarray (n_samples,)

        Returns:
            dict: статистика по классам
        """
        unique_labels = np.unique(labels)
        class_stats = {}

        for label in unique_labels:
            mask = labels == label
            class_embeddings = embeddings[mask]

            if len(class_embeddings) >= self.computer.n_neighbors + 2:
                stats = self.compute_global_stats(class_embeddings)
                class_stats[f'class_{int(label)}'] = stats
            else:
                class_stats[f'class_{int(label)}'] = self._empty_stats(len(class_embeddings))

        return class_stats

    def compute_inter_class_stats(self, embeddings, labels, sample_size=50):
        """
        Вычисляет статистику между разными классами.

        Args:
            embeddings: np.ndarray (n_samples, embedding_dim)
            labels: np.ndarray (n_samples,)
            sample_size: размер подвыборки из каждого класса

        Returns:
            dict: межклассовая статистика
        """
        unique_labels = np.unique(labels)
        inter_class_ratios = []

        for i, label1 in enumerate(unique_labels):
            for label2 in unique_labels[i + 1:]:
                mask1 = labels == label1
                mask2 = labels == label2

                # Подвыборка
                emb1 = embeddings[mask1][:sample_size]
                emb2 = embeddings[mask2][:sample_size]

                combined = np.vstack([emb1, emb2])

                if len(combined) >= self.computer.n_neighbors + 2:
                    ratios, mask = self.computer.compute_ratios(combined)

                    # Пары между классами
                    n1 = len(emb1)
                    for ii in range(n1):
                        for jj in range(n1, len(combined)):
                            if mask[ii, jj]:
                                inter_class_ratios.append(float(ratios[ii, jj]))

        if len(inter_class_ratios) == 0:
            return {
                'mean': 1.0,
                'std': 0.0,
                'min': 1.0,
                'max': 1.0,
                'num_pairs': 0
            }

        return {
            'mean': float(np.mean(inter_class_ratios)),
            'std': float(np.std(inter_class_ratios)),
            'min': float(np.min(inter_class_ratios)),
            'max': float(np.max(inter_class_ratios)),
            'num_pairs': int(len(inter_class_ratios))
        }

    @staticmethod
    def _empty_stats(n_samples):
        """Возвращает пустую статистику (для слишком малого числа точек)."""
        return {
            'mean': 1.0,
            'std': 0.0,
            'min': 1.0,
            'max': 1.0,
            'median': 1.0,
            'q25': 1.0,
            'q75': 1.0,
            'num_samples': n_samples,
            'num_valid_pairs': 0
        }
