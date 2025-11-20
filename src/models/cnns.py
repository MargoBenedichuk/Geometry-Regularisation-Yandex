import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101
from typing import Tuple, Union


class SimpleCNN(nn.Module):
    """
    Простой сверточный backbone с извлечением признаков (latent features).
    Возвращает либо logits, либо (logits, z) при return_features=True.
    """
    def __init__(self, input_shape: Tuple[int, int, int], hidden_dim: int, num_classes: int):
        super().__init__()
        c, h, w = input_shape

        self.encoder = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Вычислим размерность выхода encoder'а
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            z_shape = self.encoder(dummy).view(1, -1).shape[1]

        self.fc = nn.Linear(z_shape, hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        logits = self.head(z)
        if return_features:
            return logits, z
        return logits


class ResNetBackbone(nn.Module):
    """
    ResNet-based backbone (50 or 101), без последнего классификатора.
    """
    def __init__(self, variant: str = "resnet50", hidden_dim: int = 128, num_classes: int = 10):
        super().__init__()
        if variant == "resnet50":
            base = resnet50(weights=None)
        elif variant == "resnet101":
            base = resnet101(weights=None)
        else:
            raise ValueError("Unsupported ResNet variant. Choose 'resnet50' or 'resnet101'.")

        self.encoder = nn.Sequential(*list(base.children())[:-1])  # Remove final FC
        self.fc = nn.Linear(base.fc.in_features, hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        z = self.fc(x)
        logits = self.head(z)
        if return_features:
            return logits, z
        return logits
