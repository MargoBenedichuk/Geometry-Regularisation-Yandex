from omegaconf import OmegaConf, DictConfig
from typing import Union
import yaml
import os


def load_config(cfg_path: Union[str, os.PathLike]) -> DictConfig:
    """
    Загружает конфигурацию эксперимента из YAML и возвращает OmegaConf.DictConfig.
    Поддерживает доступ по точке и ссылки типа ${var.subvar}.
    
    :param cfg_path: путь до defaults.yaml
    :return: конфиг в виде DictConfig
    """
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r") as f:
        raw = yaml.safe_load(f)

    cfg = OmegaConf.create(raw)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)
    return cfg
