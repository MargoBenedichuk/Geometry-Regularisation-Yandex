# src/runners/hooks.py
import os
import json
import mlflow
import torch
import numpy as np
from typing import Dict, Any, Optional


def log_metrics_mlflow(metrics: Dict[str, float], step: int = 0):
    for key, val in metrics.items():
        mlflow.log_metric(key, val, step=step)


def save_metrics_json(metrics: Dict[str, float], path: str):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def save_model_checkpoint(model: torch.nn.Module, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def save_latents_npz(latents: torch.Tensor, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, latents=latents.detach().cpu().numpy())


def maybe_log_and_save(
    metrics: Dict[str, float],
    cfg: Any,
    model: torch.nn.Module,
    stage: str = "val",
    latents: Optional[torch.Tensor] = None,
):
    out_dir = cfg.output.dir
    os.makedirs(out_dir, exist_ok=True)

    # Save metrics to JSON
    save_path = os.path.join(out_dir, f"{stage}_metrics.json")
    save_metrics_json(metrics, save_path)

    # Log to MLflow if enabled
    if cfg.logging.mlflow:
        log_metrics_mlflow(metrics)

    # Save checkpoint only after val
    if stage == "val":
        ckpt_path = os.path.join(out_dir, "model.pt")
        save_model_checkpoint(model, ckpt_path)

    # Save latent features if requested
    if stage == "val" and cfg.output.save_latents and latents is not None:
        lat_path = os.path.join(out_dir, "val_latents.npz")
        save_latents_npz(latents, lat_path)


class EarlyStopping:
    def __init__(self, patience: int = 5, delta: float = 0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, current_score: float):
        if self.best_score is None or current_score > self.best_score + self.delta:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
