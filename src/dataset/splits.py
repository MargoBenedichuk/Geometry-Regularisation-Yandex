from torch.utils.data import Subset, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Literal


def make_classification_splits(
    dataset: Dataset,
    val_ratio: float = 0.2,
    stratified: bool = True,
    target_type: Literal['int'] 'int',
    seed: int = 42,
    shuffle: bool = True
):
    """
    Делит dataset на train/val выборки с возможностью стратификации по меткам.

    :param dataset: любой torch Dataset с .targets или .labels
    :param val_ratio: доля валидационной выборки
    :param stratified: использовать ли стратификацию
    :param target_type: тип таргета: int (classification) или float (regression)
    :param seed: random seed для повторяемости
    :param shuffle: перемешивать ли перед разбиением (True по умолчанию)
    :return: Subset(train), Subset(val)
    """
    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        targets = np.array(dataset.labels)
    else:
        raise AttributeError("Dataset must have 'targets' or 'labels' attribute.")

    indices = np.arange(len(dataset))

    if stratified:
        if target_type != 'int':
            raise NotImplementedError("Stratified split only supported for classification (int targets).")
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_ratio,
            random_state=seed,
            shuffle=shuffle,
            stratify=targets
        )
    else:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_ratio,
            random_state=seed,
            shuffle=shuffle
        )

    return Subset(dataset, train_idx), Subset(dataset, val_idx)
