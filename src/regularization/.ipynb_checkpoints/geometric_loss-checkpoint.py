from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def info_nce_sigmoid_regularizer(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    z = F.normalize(embeddings, dim=1)
    sim = torch.matmul(z, z.T) / temperature
    labels = labels.view(-1, 1)
    same = labels == labels.T
    diag_mask = torch.eye(same.size(0), dtype=torch.bool, device=same.device)
    same = same & ~diag_mask
    diff = (~same) & ~diag_mask

    pos_logits = sim[same]
    neg_logits = sim[diff]

    if pos_logits.numel() == 0 or neg_logits.numel() == 0:
        return torch.zeros(1, device=embeddings.device, dtype=embeddings.dtype)

    logits = torch.cat([pos_logits, neg_logits], dim=0)
    targets = torch.cat([
        torch.ones_like(pos_logits),
        torch.zeros_like(neg_logits),
    ])
    return F.binary_cross_entropy_with_logits(logits, targets)


def _class_compactness(embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    losses = []
    for cls in labels.unique():
        mask = labels == cls
        cls_vecs = embeddings[mask]
        if cls_vecs.size(0) < 2:
            continue
        centered = cls_vecs - cls_vecs.mean(dim=0, keepdim=True)
        losses.append((centered.pow(2).sum(dim=1)).mean())
    if not losses:
        return torch.zeros(1, device=embeddings.device, dtype=embeddings.dtype)
    return torch.stack(losses).mean()


def _spectral_entropy(embeddings: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / max(1, centered.size(0) - 1)
    eigvals = torch.linalg.eigvalsh(cov)
    eigvals = torch.clamp(eigvals, min=0)
    probs = eigvals / (eigvals.sum() + eps)
    entropy = -(probs * torch.log(probs + eps)).sum()
    return entropy


def compute_geometric_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    cfg: Optional[object] = None,
) -> torch.Tensor:
    """Dispatch geometric regularizers based on the config block."""

    if cfg is None:
        return torch.zeros(1, device=embeddings.device, dtype=embeddings.dtype)

    metric = getattr(cfg, "metric", "info_nce") or "info_nce"
    temperature = getattr(cfg, "temperature", 0.1)

    if metric in {"info_nce", "contrastive", "geodesic"} or getattr(cfg, "type", "") == "geometric":
        return info_nce_sigmoid_regularizer(embeddings, labels, temperature=temperature)
    if metric == "spectral_entropy":
        return _spectral_entropy(embeddings)
    if metric in {"compactness", "local_dim"}:
        return _class_compactness(embeddings, labels)

    raise ValueError(f"Unsupported regularization metric: {metric}")
