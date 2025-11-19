# src/utils/logging.py
import logging
import sys
from typing import Dict, Any

# По умолчанию используется MLflow, согласно конфигу
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def setup_console_logging(level=logging.INFO):
    """Настраивает базовый консольный логгер."""
    logger = logging.getLogger('project_logger')
    logger.setLevel(level)

    if not logger.handlers:
        # Убираем стандартные хэндлеры, если есть, и добавляем наш
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False # Предотвращаем дублирование логов
    
    return logger

# Предоставляем готовый логгер для импорта
logger = setup_console_logging()


# --- MLflow Хелперы (согласно конфигу) ---

def initialize_mlflow(project_name: str, run_name: str, config: Dict):
    """Инициализирует MLflow эксперимент и логирует параметры."""
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow не установлен. Пропускаем логирование.")
        return

    mlflow.set_experiment(project_name)
    mlflow.start_run(run_name=run_name)
    logger.info(f"MLflow запущен. Эксперимент: {project_name}, Run: {run_name}")

    # Логируем все параметры конфигурации
    mlflow.log_params(config)

def log_metric(key: str, value: Any, step: int):
    """Логирует метрику в MLflow и консоль."""
    logger.info(f"Step {step}: {key} = {value:.4f}")
    if MLFLOW_AVAILABLE and mlflow.active_run():
        mlflow.log_metric(key, value, step=step)

def end_mlflow_run():
    """Завершает текущий MLflow run."""
    if MLFLOW_AVAILABLE and mlflow.active_run():
        mlflow.end_run()