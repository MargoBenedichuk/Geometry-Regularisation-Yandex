# src/dataset/datasets.py
import random
from collections import defaultdict
from typing import Callable, Iterable, Optional

from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    """
    Общая обёртка для любого torch-compatible датасета, где объектом является (x, y).
    Позволяет задать трансформации и поддерживает доступ к меткам через .targets
    """
    def __init__(
        self,
        base_dataset: Dataset,
        transform: Optional[Callable] = None
    ):
        self.ds = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.ds[index]
        x = self.transform(x) if self.transform else x
        return x, y

    def __len__(self):
        return len(self.ds)

    @property
    def targets(self):
        if hasattr(self.ds, 'targets'):
            return self.ds.targets
        elif hasattr(self.ds, 'labels'):
            return self.ds.labels
        else:
            raise AttributeError("Underlying dataset has no 'targets' or 'labels' attribute")


class BalancedClassificationDataset(Dataset):
    """Balanced wrapper that enforces equal samples from each class per batch."""

    def __init__(
        self,
        base_dataset: Dataset,
        n_classes: int,
        n_samples: int,
        seed: int = 42,
        class_labels: Optional[Iterable[int]] = None,
    ):
        if n_classes <= 0 or n_samples <= 0:
            raise ValueError("n_classes and n_samples must be positive integers")

        self.ds = base_dataset
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = n_classes * n_samples
        self.rng = random.Random(seed)

        labels = self._extract_labels(base_dataset)
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_to_indices[label].append(idx)

        available_classes = sorted(self.class_to_indices.keys())
        if class_labels is not None:
            class_labels = list(class_labels)
            if len(class_labels) < n_classes:
                raise ValueError("class_labels must include at least n_classes entries")
            missing = set(class_labels) - set(available_classes)
            if missing:
                raise ValueError(f"Requested classes {missing} not present in dataset")
            self.classes = class_labels[:n_classes]
        else:
            if len(available_classes) < n_classes:
                raise ValueError("Dataset has fewer classes than requested n_classes")
            self.classes = available_classes[:n_classes]

        per_class_batches = []
        for cls in self.classes:
            count = len(self.class_to_indices[cls])
            if count < self.n_samples:
                raise ValueError(f"Class {cls} has fewer than {self.n_samples} samples")
            per_class_batches.append(count // self.n_samples)
        self._num_batches = min(per_class_batches)
        if self._num_batches == 0:
            raise ValueError("Not enough data to form a balanced batch")

        self.indices = []
        self.reshuffle()

    def _extract_labels(self, dataset: Dataset):
        if hasattr(dataset, "targets"):
            targets = dataset.targets
        elif hasattr(dataset, "labels"):
            targets = dataset.labels
        elif hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
            base_labels = self._extract_labels(dataset.dataset)
            targets = [base_labels[idx] for idx in dataset.indices]
        else:
            raise AttributeError("Dataset must expose 'targets' or 'labels' for BalancedClassificationDataset")

        if hasattr(targets, "tolist"):
            targets = targets.tolist()
        return [int(t.item()) if hasattr(t, "item") else int(t) for t in targets]

    def reshuffle(self):
        for cls in self.classes:
            self.rng.shuffle(self.class_to_indices[cls])

        indices = []
        for batch_id in range(self._num_batches):
            batch = []
            for cls in self.classes:
                start = batch_id * self.n_samples
                end = start + self.n_samples
                batch.extend(self.class_to_indices[cls][start:end])
            self.rng.shuffle(batch)
            indices.extend(batch)
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return self.ds[real_idx]
