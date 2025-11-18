
import pytest
import torch
from torch.utils.data import Dataset, DataLoader
from src.dataset.datasets import ClassificationDataset
from src.dataset.loaders import make_dataloader, collate_by_class
from src.dataset.splits import make_classification_splits
from torchvision.datasets import MNIST
from torchvision import transforms
import os


def get_dummy_dataset():
    return ClassificationDataset(
        MNIST(root="./data", train=True, download=True),
        transform=transforms.ToTensor()
    )


def test_classification_dataset():
    dataset = get_dummy_dataset()
    x, y = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, int)
    assert hasattr(dataset, 'targets')


def test_make_dataloader():
    dataset = get_dummy_dataset()
    loader = make_dataloader(dataset, batch_size=8, shuffle=False)
    batch = next(iter(loader))
    x, y = batch
    assert x.shape[0] == 8
    assert isinstance(y, torch.Tensor)


def test_collate_by_class():
    dataset = get_dummy_dataset()
    loader = make_dataloader(dataset, batch_size=10, shuffle=False, collate_fn=collate_by_class)
    batch = next(iter(loader))
    assert isinstance(batch, dict)
    for k, v in batch.items():
        assert isinstance(k, int)
        assert isinstance(v, list)
        assert isinstance(v[0], torch.Tensor)


def test_make_classification_splits():
    dataset = get_dummy_dataset()
    train, val = make_classification_splits(dataset, val_ratio=0.1, stratified=True, target_type='int', seed=123)
    assert isinstance(train, Dataset)
    assert isinstance(val, Dataset)
    assert len(train) + len(val) == len(dataset)


def test_non_stratified_split():
    dataset = get_dummy_dataset()
    train, val = make_classification_splits(dataset, val_ratio=0.2, stratified=False, seed=123)
    assert len(train) + len(val) == len(dataset)
