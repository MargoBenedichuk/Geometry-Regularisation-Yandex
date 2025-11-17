from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# === 1️ IO / Input-Output block ===
class IOConfig(BaseModel):
    data_path_train: Optional[str]
    data_path_test: Optional[str]
    seed: int = 42


class ClassificationConfig(BaseModel):
    model: str = "logreg"
    class_weight: str = "balanced"
    calibration: CalibrationConfig = CalibrationConfig()
    threshold: ThresholdConfig = ThresholdConfig()
    model_params: Optional[Dict[str, Any]] = None


# === 6️ Preprocessing ===
class PreprocessStep(BaseModel):
    name: str
    params: Optional[Dict[str, Any]] = None



# === 8️ Metrics ===
class MetricsConfig(BaseModel):
    enabled: bool = True
    per_domain: bool = True
    decision_curve: bool = True
    save_plots: bool = True


# === 9️ Reverse-test (для UDA) ===
class ReverseTestConfig(BaseModel):
    enabled: bool = False
    pseudo_label_source: Optional[str] = "train"
    n_repeats: int = 1
    consistency_metric: str = "delta_auc"  # delta_auc / spearman


# === 10 RootConfig — единый для всех пайплайнов ===
class RootConfig(BaseModel):
    # Основные
    mode: str
    experiment_name: str
    random_seed: int = 42

    # Блоки
    io: IOConfig
    classification: ClassificationConfig
    preprocess: PreprocessConfig

    # Дополнительные
    metrics: Optional[MetricsConfig] = MetricsConfig()


# ===  Helper ===
def load_and_validate_config(path: str) -> RootConfig:
    import yaml
    from omegaconf import OmegaConf

    if path.endswith(".yaml"):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    else:
        data = OmegaConf.load(path)

    return RootConfig(**data)
