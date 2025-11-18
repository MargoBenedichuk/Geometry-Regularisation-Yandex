
import torch
import torch.nn as nn
import torch.nn.functional as F
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