
# src/dataset/loaders.py
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Callable
from collections import defaultdict


def collate_by_class(batch):
    """
    Collate-функция, группирующая данные по меткам в батче:
    возвращает словарь вида {class_label: [x1, x2, ...]}
    """
    grouped = defaultdict(list)
    for x, y in batch:
        grouped[y].append(x)
    return dict(grouped)


def make_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    collate_fn: Optional[Callable] = None
) -> DataLoader:
    """
    Стандартный PyTorch-style DataLoader конструктор с поддержкой
    collate_fn и батчинга, пригодного для topological analysis.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn
    )
