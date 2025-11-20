import torch.nn as nn
from typing import Literal

from src.models.cnns import SimpleCNN, ResNetBackbone

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



def build_model(cfg):
    """
    Универсальный конструктор моделей:
    - simple_cnn
    - resnet50
    - resnet101

    Возвращает модель с методом:
        forward(x, return_features=True) → (logits, z)
    """

    name = cfg.model.name.lower()
    num_classes = cfg.model.num_classes
    hidden_dim = cfg.model.hidden_dim
    input_shape = tuple(cfg.model.input_shape)

    # ---------------------------
    # SIMPLE CNN
    # ---------------------------
    if name == "simple_cnn":
        return SimpleCNN(
            input_shape=input_shape,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )

    # ---------------------------
    # RESNET BACKBONES
    # ---------------------------
    if name in ["resnet50", "resnet101"]:
        model = torchvision.models.__dict__[model_name](weights=none)
        # заменить классификатор
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, cfg.model.num_classes)
        return model

    # ---------------------------
    raise ValueError(f"Unknown model type: {name}")
