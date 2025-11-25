from typing import Optional, List
import torch
from torch_topological.nn import VietorisRipsComplex, PersistenceInformation

def _topological_density_regularizer(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    beta: float = 0.1,
    weight: float = 0.1
) -> torch.Tensor:
    vr = VietorisRipsComplex(dim=0)
    total_loss = torch.tensor(0.0, requires_grad=True)
    for i, label in enumerate(labels):
        class_embeddings = embeddings[labels == label]
        info: List[PersistenceInformation] = vr(class_embeddings)
        death_times = info[0].diagram[:, 1]
        total_loss = total_loss + (death_times - beta).abs().sum()
    return total_loss * weight / embeddings.shape[0]

def compute_topological_loss(
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        cfg: Optional[object] = None
) -> torch.Tensor:
    if cfg is None:
        return torch.zeros(1, device=embeddigs.device, dtype=embeddings.dtype)
    metric = getattr(cfg, "metric", "density")

    if metric is None or metric == "none" or metric == "":
        return torch.zeros(1, device=embeddings.device, dtype=embeddings.dtype)

    weight = getattr(cfg, "weight", 0.1)
    if weight == 0:
        return torch.zeros(1, device=embeddigs.device, dtype=embeddings.dtype)

    if metric in {"density", "density_topological"}:
        beta = getattr(cfg, "beta", 0.1)
        return _topological_density_regularizer(embeddings, labels, beta, weight)

    return torch.zeros(1, device=embeddings.device, dtype=embeddings.dtype)

