# src/metrics/geodesic.py - ULTRA-FAST VERSION
"""
Ultra-optimized geodesic metrics with LAZY evaluation and caching.
Skips expensive computations when not needed.
"""

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
import warnings

warnings.filterwarnings('ignore')


class GeodesicDistanceComputer:
    """
    Оптимизированный вычислитель геодезических расстояний.
    ВНИМАНИЕ: для больших датасетов используйте только с маленькими подвыборками!
    """

    def __init__(self, n_neighbors=5, eps=1e-6, max_geodesic=1e6):
        """
        Args:
            n_neighbors: количество соседей (УМЕНЬШЕНО с 10 на 5!)
            eps: epsilon для избежания деления на ноль
            max_geodesic: максимальное геодезическое расстояние
        """
        self.n_neighbors = n_neighbors
        self.eps = eps
        self.max_geodesic = max_geodesic

    def compute_euclidean_sparse(self, X):
        """
        Вычисление расстояний ТОЛЬКО до соседей (не полная матрица!).
        Экономит память: O(n*k) вместо O(n²)
        """
        n_samples = X.shape[0]
        k = min(self.n_neighbors + 1, n_samples - 1)

        if k < 2:
            return np.array([]), np.array([])

        nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=-1).fit(X)
        distances, indices = nbrs.kneighbors(X)

        return distances, indices

    def compute_geodesic(self, X):
        """
        Быстрое вычисление геодезических расстояний.
        Использует ТОЛЬКО соседей, не полную матрицу.
        """
        n_samples = X.shape[0]

        # Получаем только соседей
        distances, indices = self.compute_euclidean_sparse(X)
        k = distances.shape[1]

        if k < 2:
            return np.ones((n_samples, n_samples))

        # Строим РАЗРЕЖЕННЫЙ граф
        row, col, data = [], [], []

        for i in range(n_samples):
            for j, neighbor_idx in enumerate(indices[i]):
                if neighbor_idx != i:
                    row.append(i)
                    col.append(neighbor_idx)
                    data.append(distances[i, j])

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
            return geodesic_dist

        except Exception as e:
            print(f"[WARNING] Geodesic computation failed: {e}")
            return np.ones((n_samples, n_samples))

    def compute_ratios_efficient(self, X):
        """
        ULTRA-БЫСТРОЕ вычисление отношений.
        НЕ вычисляет полную матрицу расстояний!
        """
        n_samples = X.shape[0]

        # Только соседи
        distances, indices = self.compute_euclidean_sparse(X)
        geodesic_dist = self.compute_geodesic(X)

        # Вычисляем отношения только для соседей
        ratios_list = []

        for i in range(n_samples):
            for j, neighbor_idx in enumerate(indices[i]):
                if neighbor_idx != i:
                    euc_dist = distances[i, j]
                    geo_dist = geodesic_dist[i, neighbor_idx]

                    if euc_dist > self.eps and geo_dist < self.max_geodesic:
                        ratio = geo_dist / euc_dist
                        ratios_list.append(ratio)

        return np.array(ratios_list) if ratios_list else np.array([1.0])


class GeodesicMetric:
    """
    ULTRA-БЫСТРАЯ метрика для анализа геодезических расстояний.
    Пропускает дорогостоящие вычисления.
    """

    def __init__(self, n_neighbors=5, skip_expensive=True):
        """
        Args:
            n_neighbors: количество соседей
            skip_expensive: пропускать дорогостоящие вычисления
        """
        self.computer = GeodesicDistanceComputer(n_neighbors=n_neighbors)
        self.skip_expensive = skip_expensive

    def compute_global_stats(self, embeddings):
        """
        БЫСТРОЕ вычисление глобальной статистики.
        Не вычисляет полные матрицы расстояний.
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        n_samples = embeddings.shape[0]

        if n_samples < self.computer.n_neighbors + 2:
            return self._empty_stats(n_samples)

        # ULTRA-оптимизация: берем еще меньше
        if n_samples > 2000:
            print(f"[INFO] Computing fast geodesic stats on {n_samples} samples...")
            np.random.seed(42)
            sample_idx = np.random.choice(n_samples, size=min(1000, n_samples), replace=False)
            embeddings_sample = embeddings[sample_idx]
            print(f"[INFO] Using subsample of 1000 points for speed")
        else:
            embeddings_sample = embeddings

        try:
            ratios = self.computer.compute_ratios_efficient(embeddings_sample)

            if len(ratios) == 0:
                return self._empty_stats(n_samples)

            return {
                'mean': float(np.mean(ratios)),
                'std': float(np.std(ratios)),
                'min': float(np.min(ratios)),
                'max': float(np.max(ratios)),
                'median': float(np.median(ratios)),
                'q25': float(np.percentile(ratios, 25)),
                'q75': float(np.percentile(ratios, 75)),
                'num_samples': n_samples,
                'num_valid_pairs': int(len(ratios))
            }
        except Exception as e:
            print(f"[WARNING] Fast geodesic computation failed: {e}, using defaults")
            return self._empty_stats(n_samples)

    def compute_class_wise_stats(self, embeddings, labels):
        """
        БЫСТРОЕ вычисление статистики по классам.
        Пропускает классы если их мало.
        """
        unique_labels = np.unique(labels)
        class_stats = {}

        for label in unique_labels:
            mask = labels == label
            class_embeddings = embeddings[mask]

            # Пропускаем если класс слишком маленький
            if len(class_embeddings) < self.computer.n_neighbors + 2:
                class_stats[f'class_{int(label)}'] = self._empty_stats(len(class_embeddings))
                continue

            # Если класс большой - берем подвыборку
            if len(class_embeddings) > 1000:
                np.random.seed(42)
                sample_idx = np.random.choice(len(class_embeddings), size=500, replace=False)
                class_embeddings = class_embeddings[sample_idx]

            try:
                ratios = self.computer.compute_ratios_efficient(class_embeddings)
                if len(ratios) > 0:
                    class_stats[f'class_{int(label)}'] = {
                        'mean': float(np.mean(ratios)),
                        'std': float(np.std(ratios)),
                        'num_valid_pairs': len(ratios)
                    }
                else:
                    class_stats[f'class_{int(label)}'] = self._empty_stats(len(class_embeddings))
            except:
                class_stats[f'class_{int(label)}'] = self._empty_stats(len(class_embeddings))

        return class_stats

    def compute_inter_class_stats(self, embeddings, labels, sample_size=20):
        """
        БЫСТРОЕ вычисление межклассовых статистик.
        Использует ОЧЕНЬ маленькие подвыборки.
        """
        unique_labels = np.unique(labels)
        inter_class_ratios = []

        # Берем только ПЕРВУЮ пару классов для быстрости
        if len(unique_labels) > 2:
            unique_labels = unique_labels[:2]

        try:
            for i, label1 in enumerate(unique_labels):
                for label2 in unique_labels[i + 1:]:
                    mask1 = labels == label1
                    mask2 = labels == label2

                    # ULTRA-маленькая подвыборка
                    emb1 = embeddings[mask1][:sample_size]
                    emb2 = embeddings[mask2][:sample_size]

                    if len(emb1) < 2 or len(emb2) < 2:
                        continue

                    combined = np.vstack([emb1, emb2])
                    ratios = self.computer.compute_ratios_efficient(combined)
                    inter_class_ratios.extend(ratios)

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
                'num_pairs': len(inter_class_ratios)
            }
        except Exception as e:
            print(f"[WARNING] Inter-class stats failed: {e}")
            return {
                'mean': 1.0,
                'std': 0.0,
                'min': 1.0,
                'max': 1.0,
                'num_pairs': 0
            }

    @staticmethod
    def _empty_stats(n_samples):
        """Пустая статистика."""
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


def compute_geodesic_summary(embeddings, labels, n_neighbors=5):
    """
    ULTRA-БЫСТРОЕ вычисление геодезической статистики.
    Оптимизировано для больших датасетов.
    """
    print("\n[INFO] Computing FAST geodesic summary (optimized for speed)...")

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    summary = {}

    try:
        geo_metric = GeodesicMetric(n_neighbors=n_neighbors, skip_expensive=True)

        print("[1/3] Computing global geodesic statistics (fast)...")
        geo_global = geo_metric.compute_global_stats(embeddings)
        summary['global'] = geo_global

        print("[2/3] Computing class-wise geodesic statistics (fast)...")
        geo_class = geo_metric.compute_class_wise_stats(embeddings, labels)
        summary['class_wise'] = geo_class

        print("[3/3] Computing inter-class geodesic statistics (fast)...")
        geo_inter = geo_metric.compute_inter_class_stats(embeddings, labels)
        summary['inter_class'] = geo_inter

        print("[4/4] Computing additional analysis...")
        summary['analysis'] = _compute_additional_analysis(
            geo_global, geo_class, geo_inter
        )

        print("[INFO] FAST geodesic summary completed!\n")

    except Exception as e:
        print(f"[WARNING] Failed to compute geodesic metrics: {e}")
        summary = _get_empty_summary(len(embeddings), len(np.unique(labels)))

    return summary


def _compute_additional_analysis(global_stats, class_stats, inter_stats):
    """Дополнительный анализ."""
    analysis = {}

    global_mean = global_stats.get('mean', 1.0)
    analysis['flatness_score'] = float(1.0 / max(global_mean, 0.1))

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

    intra_class_mean = global_stats.get('mean', 1.0)
    inter_class_mean = inter_stats.get('mean', 1.0)

    if intra_class_mean > 0:
        analysis['class_separability_ratio'] = float(inter_class_mean / intra_class_mean)
    else:
        analysis['class_separability_ratio'] = 1.0

    quality_score = (
            analysis['flatness_score'] * 0.3 +
            (1.0 - min(analysis.get('class_uniformity', {}).get('coefficient_of_variation', 0.5), 1.0)) * 0.3 +
            min(analysis['class_separability_ratio'] / 2.0, 1.0) * 0.4
    )
    analysis['overall_quality_score'] = float(quality_score)

    return analysis


def _get_empty_summary(n_samples, n_classes):
    """Пустая статистика."""
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
    n_neighbors = 5  # ULTRA-маленькое значение!

    if hasattr(cfg, 'regularization'):
        reg_cfg = cfg.regularization
        if hasattr(reg_cfg, 'geodesic_ratio'):
            n_neighbors = getattr(reg_cfg.geodesic_ratio, 'n_neighbors', 5)

    return compute_geodesic_summary(embeddings, labels, n_neighbors=n_neighbors)
