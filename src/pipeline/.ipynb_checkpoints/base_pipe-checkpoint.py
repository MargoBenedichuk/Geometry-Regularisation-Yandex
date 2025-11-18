# src/runners/run_clf_base.py

import os
import json
import mlflow
import torch
import numpy as np
from omegaconf import OmegaConf
from src.utils.config import load_config
from src.dataset.datasets import ClassificationDataset
from src.dataset.loaders import make_dataloader
from src.dataset.splits import make_classification_splits
from src.models.cnns import SimpleCNN
from src.models.head_factory import get_head
from src.metrics.task import compute_classification_metrics
from src.regularizers.geometric_loss import compute_geometric_loss
from src.metrics.geometry import compute_geometry_summary
from src.vizualisation.vizualisator import save_umap_projection


def run_experiment(cfg_path: str, exp_dir: str):
    os.makedirs(exp_dir, exist_ok=True)
    cfg = load_config(cfg_path)
    OmegaConf.save(cfg, os.path.join(exp_dir, "config_used.yaml"))
    mlflow.set_experiment(cfg.experiment_name)

    with mlflow.start_run():
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        # === 1. Датасет и загрузка ===
        base_ds = cfg.dataset.base()
        dataset = ClassificationDataset(base_ds, transform=cfg.dataset.transform())
        train_ds, val_ds = make_classification_splits(dataset, val_ratio=cfg.dataset.val_ratio, seed=cfg.seed)

        train_loader = make_dataloader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
        val_loader = make_dataloader(val_ds, batch_size=cfg.train.batch_size, shuffle=False)

        # === 2. Модель ===
        model = SimpleCNN(cfg.model).to(cfg.device)
        head = get_head(cfg.model.head_type, cfg.model.hidden_dim, cfg.model.num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

        # === 3. Тренировка ===
        for epoch in range(cfg.train.epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(cfg.device), yb.to(cfg.device)
                logits, latents = model(xb, return_features=True)
                loss_ce = torch.nn.functional.cross_entropy(logits, yb)
                loss_geom = compute_geometric_loss(latents, yb, cfg.regularization)
                loss = loss_ce + cfg.regularization.weight * loss_geom
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # === 4. Валидация ===
        model.eval()
        all_logits, all_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(cfg.device)
                logits, _ = model(xb, return_features=True)
                all_logits.append(logits.cpu())
                all_targets.append(yb)

        logits = torch.cat(all_logits, dim=0)
        targets = torch.cat(all_targets, dim=0)
        metrics = compute_classification_metrics(logits, targets, save_path=os.path.join(exp_dir, "metrics.json"))

        # === 5. Геометрический анализ ===
        latent_vectors = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(cfg.device)
                _, z = model(xb, return_features=True)
                latent_vectors.append(z.cpu())

        latent_all = torch.cat(latent_vectors, dim=0).numpy()
        geometry_summary = compute_geometry_summary(latent_all, targets.numpy())
        with open(os.path.join(exp_dir, "geometry_summary.json"), "w") as f:
            json.dump(geometry_summary, f, indent=2)

        # === 6. Визуализация ===
        save_umap_projection(latent_all, targets.numpy(), os.path.join(exp_dir, "umap.png"))

        # === 7. Логирование ===
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))
        mlflow.log_artifact(os.path.join(exp_dir, "metrics.json"))
        mlflow.log_artifact(os.path.join(exp_dir, "geometry_summary.json"))
        mlflow.log_artifact(os.path.join(exp_dir, "umap.png"))


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python -m src.runners.run_clf_base <CONFIG_PATH> <EXP_DIR>")
        exit(1)
    run_experiment(sys.argv[1], sys.argv[2])
