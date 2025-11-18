import pytest
import torch
from src.models.cnns import SimpleCNN
from src.models.head_factory import get_head
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


def test_simplecnn_forward():
    model = SimpleCNN(input_shape=(1, 28, 28), hidden_dim=64, num_classes=10)
    x = torch.randn(8, 1, 28, 28)
    logits = model(x)
    assert logits.shape == (8, 10)


def test_simplecnn_return_features():
    model = SimpleCNN(input_shape=(1, 28, 28), hidden_dim=64, num_classes=10)
    x = torch.randn(4, 1, 28, 28)
    logits, z = model(x, return_features=True)
    assert logits.shape == (4, 10)
    assert len(z.shape) == 2  # [B, D]


def test_head_factory_linear():
    head = get_head("linear", in_dim=64, num_classes=10)
    assert isinstance(head, nn.Linear)
    x = torch.randn(2, 64)
    y = head(x)
    assert y.shape == (2, 10)


def test_head_factory_mlp():
    head = get_head("mlp", in_dim=64, num_classes=10, hidden_dim=32)
    assert isinstance(head, nn.Sequential)
    x = torch.randn(2, 64)
    y = head(x)
    assert y.shape == (2, 10)


def test_head_factory_projection():
    head = get_head("projection", in_dim=64, num_classes=10, hidden_dim=32)
    assert isinstance(head, nn.Sequential)
    x = torch.randn(2, 64)
    y = head(x)
    assert y.shape == (2, 10)


def test_head_factory_invalid():
    with pytest.raises(ValueError):
        _ = get_head("unsupported", in_dim=64, num_classes=10)
