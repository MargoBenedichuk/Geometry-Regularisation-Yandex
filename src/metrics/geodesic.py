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


def compute_geodesic_summary(embeddings, labels, n_neighbors=10):
    """
    Вычисляет полную статистику геодезических расстояний.
    Отдельная функция для анализа геодезических метрик.

    Args:
        embeddings: np.ndarray (n_samples, embedding_dim) или torch.Tensor
        labels: np.ndarray (n_samples,) или torch.Tensor
        n_neighbors: количество соседей для построения графа

    Returns:
        dict: полная статистика геодезических метрик
    """
    # Конвертация в numpy если нужно
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    summary = {}

    try:
        # Создаем метрику
        geo_metric = GeodesicMetric(n_neighbors=n_neighbors)

        # === 1. Глобальная статистика ===
        geo_global = geo_metric.compute_global_stats(embeddings)
        summary['global'] = geo_global

        # === 2. Поклассовая статистика ===
        geo_class = geo_metric.compute_class_wise_stats(embeddings, labels)
        summary['class_wise'] = geo_class

        # === 3. Межклассовая статистика ===
        geo_inter = geo_metric.compute_inter_class_stats(embeddings, labels)
        summary['inter_class'] = geo_inter

        # === 4. Дополнительная аналитика ===
        summary['analysis'] = _compute_additional_analysis(
            geo_global, geo_class, geo_inter
        )

    except Exception as e:
        print(f"Warning: Failed to compute geodesic metrics: {e}")
        summary = _get_empty_summary(len(embeddings), len(np.unique(labels)))

    return summary


def _compute_additional_analysis(global_stats, class_stats, inter_stats):
    """
    Вычисляет дополнительные аналитические метрики.

    Args:
        global_stats: dict - глобальная статистика
        class_stats: dict - поклассовая статистика
        inter_stats: dict - межклассовая статистика

    Returns:
        dict: дополнительная аналитика
    """
    analysis = {}

    # === Анализ отклонения от плоскости ===
    global_mean = global_stats.get('mean', 1.0)
    analysis['flatness_score'] = float(1.0 / max(global_mean, 0.1))
    # Чем ближе к 1.0, тем более "плоское" пространство

    # === Анализ однородности между классами ===
    if class_stats:
        class_means = [
            stats.get('mean', 1.0)
            for stats in class_stats.values()
            if stats.get('num_valid_pairs', 0) > 0
        ]
        if class_means:
            analysis['class_uniformity'] = {
                'mean': float(np.mean(class_means)),
                'std': float(np.std(class_means)),
                'coefficient_of_variation': float(np.std(class_means) / (np.mean(class_means) + 1e-8))
            }
        else:
            analysis['class_uniformity'] = {'mean': 1.0, 'std': 0.0, 'coefficient_of_variation': 0.0}

    # === Анализ разделимости классов ===
    intra_class_mean = global_stats.get('mean', 1.0)
    inter_class_mean = inter_stats.get('mean', 1.0)

    if intra_class_mean > 0:
        analysis['class_separability_ratio'] = float(inter_class_mean / intra_class_mean)
        # > 1.0 означает, что между классами расстояния больше, чем внутри (хорошо)
    else:
        analysis['class_separability_ratio'] = 1.0

    # === Общая оценка качества ===
    # Комбинированная метрика: насколько хорошо организовано пространство
    quality_score = (
            analysis['flatness_score'] * 0.3 +  # 30% - плоскость
            (1.0 - min(analysis.get('class_uniformity', {}).get('coefficient_of_variation', 0.5),
                       1.0)) * 0.3 +  # 30% - однородность
            min(analysis['class_separability_ratio'] / 2.0, 1.0) * 0.4  # 40% - разделимость
    )
    analysis['overall_quality_score'] = float(quality_score)

    return analysis


def _get_empty_summary(n_samples, n_classes):
    """
    Возвращает пустую статистику при ошибке.

    Args:
        n_samples: количество образцов
        n_classes: количество классов

    Returns:
        dict: пустая статистика
    """
    return {
        'global': {
            'mean': 1.0,
            'std': 0.0,
            'min': 1.0,
            'max': 1.0,
            'median': 1.0,
            'q25': 1.0,
            'q75': 1.0,
            'num_samples': n_samples,
            'num_valid_pairs': 0
        },
        'class_wise': {},
        'inter_class': {
            'mean': 1.0,
            'std': 0.0,
            'min': 1.0,
            'max': 1.0,
            'num_pairs': 0
        },
        'analysis': {
            'flatness_score': 1.0,
            'class_uniformity': {'mean': 1.0, 'std': 0.0, 'coefficient_of_variation': 0.0},
            'class_separability_ratio': 1.0,
            'overall_quality_score': 0.5
        }
    }


def compute_geodesic_summary_with_config(embeddings, labels, cfg):
    """
    Вычисляет геодезическую статистику с параметрами из конфига.

    Args:
        embeddings: np.ndarray (n_samples, embedding_dim)
        labels: np.ndarray (n_samples,)
        cfg: OmegaConf config с параметрами

    Returns:
        dict: полная геодезическая статистика
    """
    # Извлекаем n_neighbors из конфига
    n_neighbors = 10  # Значение по умолчанию

    if hasattr(cfg, 'regularization'):
        reg_cfg = cfg.regularization
        if hasattr(reg_cfg, 'geodesic_ratio'):
            n_neighbors = reg_cfg.geodesic_ratio.get('n_neighbors', 10)

    return compute_geodesic_summary(embeddings, labels, n_neighbors=n_neighbors)
