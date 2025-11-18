# src/dataset/datasets.py
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from typing import Optional, Callable


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

