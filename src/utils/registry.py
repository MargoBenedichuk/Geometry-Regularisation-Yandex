from __future__ import annotations

import importlib
from functools import partial
from typing import Any, Callable, Dict, Optional, Union

from omegaconf import DictConfig, OmegaConf
from torchvision import datasets, transforms

def get_mnist(
    root: str = "./data",
    train: bool = True,
    download: bool = True,
    transform=None,
):
    return datasets.MNIST(root=root, train=train, download=download, transform=transform)

def get_cifar10(
    root: str = "./data",
    train: bool = True,
    download: bool = True,
    transform=None,
):
    return datasets.CIFAR10(root=root, train=train, download=download, transform=transform)

def get_imagenet(
    root: str = "./data/imagenet",
    train: bool = True,
    transform=None,
):
    split = "train" if train else "val"
    return datasets.ImageNet(root=root, split=split, transform=transform)

def default_mnist_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])



def resolve_target(target_like: Union[str, Callable, Dict[str, Any], DictConfig, None]) -> Optional[Callable]:
    """Convert a string/DictConfig reference to a callable.

    Supports dotted-path strings (``package.module.fn``) as well as objects
    that already are callables. Dict-like specs should contain ``target`` and
    optionally ``params`` describing keyword arguments.
    """
    if target_like is None:
        return None
    if callable(target_like):
        return target_like

    if isinstance(target_like, (str, bytes)):
        module_path, _, attr = target_like.rpartition(".")
        if not module_path:
            raise ValueError(f"Target '{target_like}' must be a dotted path")
        module = importlib.import_module(module_path)
        return getattr(module, attr)

    if isinstance(target_like, DictConfig):
        target_like = OmegaConf.to_container(target_like, resolve=True)  # type: ignore

    if isinstance(target_like, dict):
        if "target" not in target_like:
            raise ValueError("Dict config must contain 'target'")
        target = resolve_target(target_like["target"])
        params = target_like.get("params")
        if params is None:
            params = {k: v for k, v in target_like.items() if k != "target"}
        return partial(target, **params)

    raise TypeError(f"Unsupported target specification: {type(target_like)!r}")

__all__ = [
    "get_mnist",
    "get_cifar10",
    "get_imagenet",
    "default_mnist_transform",
    "resolve_target",
]
