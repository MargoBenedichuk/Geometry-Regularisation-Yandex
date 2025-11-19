import json
import torch
from pathlib import Path
from typing import Any, Dict

# --- 1. Работа с путями ---

def create_dir_if_not_exists(path: str | Path) -> Path:
    """Создает директорию, если она еще не существует."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

# --- 2. Сохранение/Загрузка конфигов (для отслеживания) ---

def save_config_to_json(config_data: Dict, path: str | Path):
    """Сохраняет словарь конфигурации в JSON файл."""
    path = Path(path)
    with open(path, 'w') as f:
        json.dump(config_data, f, indent=4)

# --- 3. Работа с PyTorch Checkpoints ---

def save_checkpoint(
    state: Dict[str, Any], 
    filepath: str | Path
):
    """Сохраняет состояние модели, оптимизатора и т.д. в PyTorch-файл."""
    filepath = Path(filepath)
    create_dir_if_not_exists(filepath.parent)
    torch.save(state, filepath)

def load_checkpoint(filepath: str | Path, device: str = "cpu") -> Dict[str, Any]:
    """Загружает PyTorch checkpoint с указанного пути."""
    return torch.load(filepath, map_location=device)