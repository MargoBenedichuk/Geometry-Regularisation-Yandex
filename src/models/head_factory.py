import torch.nn as nn
from typing import Literal


def get_head(
    type: Literal["linear", "mlp", "projection"],
    in_dim: int,
    num_classes: int,
    hidden_dim: int = 128
) -> nn.Module:
    """
    Фабрика классификационных голов для модели.

    :param type: тип головы — linear | mlp | projection
    :param in_dim: размерность входа (latent size)
    :param num_classes: количество классов
    :param hidden_dim: промежуточная размерность (для mlp и projection)
    :return: nn.Module
    """
    if type == "linear":
        return nn.Linear(in_dim, num_classes)

    elif type == "mlp":
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    elif type == "projection":
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    else:
        raise ValueError(f"Unsupported head type: {type}")
