import torch
import pytest
from src.regularization.geometric_loss import compute_geometric_loss


class DummyCfg:
    def __init__(self, metric):
        self.metric = metric


@pytest.mark.parametrize("metric", ["info_nce", "contrastive"])
def test_info_nce_loss(metric):
    z = torch.randn(8, 16)
    y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    cfg = DummyCfg(metric)
    loss = compute_geometric_loss(z, y, cfg)
    assert loss.dim() == 0 and torch.is_tensor(loss)


def test_compactness():
    z = torch.randn(10, 8)
    y = torch.randint(0, 3, (10,))
    cfg = DummyCfg("compactness")
    loss = compute_geometric_loss(z, y, cfg)
    assert loss > 0


def test_spectral_entropy():
    z = torch.randn(20, 10)
    y = torch.randint(0, 2, (20,))
    cfg = DummyCfg("spectral_entropy")
    loss = compute_geometric_loss(z, y, cfg)
    assert loss > 0


def test_empty_batch():
    z = torch.randn(0, 8)
    y = torch.tensor([])
    cfg = DummyCfg("compactness")
    loss = compute_geometric_loss(z, y, cfg)
    assert loss.numel() == 1 and loss.item() == 0.0


def test_unsupported_metric():
    z = torch.randn(5, 5)
    y = torch.randint(0, 2, (5,))
    cfg = DummyCfg("unknown")
    with pytest.raises(ValueError):
        _ = compute_geometric_loss(z, y, cfg)