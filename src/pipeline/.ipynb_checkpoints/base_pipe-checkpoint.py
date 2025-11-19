
import os
import json
import mlflow
import torch
import numpy as np
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from src.utils.config import load_config
from src.dataset.datasets import BalancedClassificationDataset, ClassificationDataset
from src.dataset.dataloader import make_dataloader
from src.dataset.splits import make_classification_splits
from src.models.cnns import SimpleCNN
from src.metrics.task import compute_classification_metrics
from src.regularization.geometric_loss import compute_geometric_loss
from src.utils.registry import default_mnist_transform, resolve_target
from src.vizualisation.vizualisator import save_interactive_projection, save_umap_projection


def run_experiment(cfg_path: str, exp_dir: str):
    os.makedirs(exp_dir, exist_ok=True)
    cfg = load_config(cfg_path)
    OmegaConf.save(cfg, os.path.join(exp_dir, "config_used.yaml"))
    mlflow.set_experiment(cfg.experiment_name)

    with mlflow.start_run():
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        device = torch.device(
            cfg.device if ("cuda" not in str(cfg.device).lower() or torch.cuda.is_available()) else "cpu"
        )
        if device.type == "cpu" and str(cfg.device).lower().startswith("cuda"):
            print("[INFO] CUDA not available, using CPU instead.")

        # === 1. Датасет и загрузка ===
        dataset_factory = resolve_target(cfg.dataset.base)
        if dataset_factory is None:
            raise ValueError("cfg.dataset.base must point to a dataset constructor")

        dataset_kwargs = dict(root=cfg.dataset.root, train=True, download=True)
        try:
            base_ds = dataset_factory(**dataset_kwargs)
        except TypeError:
            base_ds = dataset_factory()

        transform_resolver = resolve_target(getattr(cfg.dataset, "transform", None))
        transform = (transform_resolver() if transform_resolver else default_mnist_transform())
        dataset = ClassificationDataset(base_ds, transform=transform)
        train_ds, val_ds = make_classification_splits(dataset, val_ratio=cfg.dataset.val_ratio, seed=cfg.seed)

        balanced_cfg = getattr(cfg.dataset, "balanced_dataset", None)
        train_balanced = False
        if balanced_cfg is not None:
            seed = getattr(balanced_cfg, "seed", cfg.seed)
            class_labels = getattr(balanced_cfg, "class_labels", None)
            train_ds = BalancedClassificationDataset(
                train_ds,
                n_classes=balanced_cfg.n_classes,
                n_samples=balanced_cfg.n_samples,
                seed=seed,
                class_labels=class_labels,
            )
            train_balanced = True
            expected_batch_size = train_ds.batch_size
            if cfg.train.batch_size != expected_batch_size:
                raise ValueError(
                    f"cfg.train.batch_size ({cfg.train.batch_size}) must equal n_classes * n_samples ({expected_batch_size})"
                )

        train_loader = make_dataloader(
            train_ds,
            batch_size=cfg.train.batch_size,
            shuffle=not train_balanced,
        )
        val_loader = make_dataloader(val_ds, batch_size=cfg.train.batch_size, shuffle=False)

        # === 2. Модель ===
        model = SimpleCNN(
            input_shape=tuple(cfg.model.input_shape),
            hidden_dim=cfg.model.hidden_dim,
            num_classes=cfg.model.num_classes,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

        # === 3. Тренировка ===
        for epoch in range(cfg.train.epochs):
            model.train()
            running_loss = 0.0
            running_ce = 0.0
            running_geom = 0.0
            steps = 0
            if train_balanced and hasattr(train_ds, "reshuffle"):
                train_ds.reshuffle()
            progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.train.epochs}", leave=False)
            for xb, yb in progress:
                xb, yb = xb.to(device), yb.to(device)
                logits, latents = model(xb, return_features=True)
                loss_ce = torch.nn.functional.cross_entropy(logits, yb)
                loss_geom = compute_geometric_loss(latents, yb, cfg.regularization)
                loss = loss_ce + cfg.regularization.weight * loss_geom
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                steps += 1
                running_loss += loss.item()
                running_ce += loss_ce.item()
                running_geom += loss_geom.item()
                progress.set_postfix(
                    loss=f"{running_loss / steps:.4f}",
                    ce=f"{loss_ce.item():.3f}",
                    geom=f"{loss_geom.item():.3f}",
                )

            epoch_loss = running_loss / max(steps, 1)
            epoch_ce = running_ce / max(steps, 1)
            epoch_geom = running_geom / max(steps, 1)
            mlflow.log_metric("train/loss", epoch_loss, step=epoch)
            mlflow.log_metric("train/ce_loss", epoch_ce, step=epoch)
            mlflow.log_metric("train/geom_loss", epoch_geom, step=epoch)

        # === 4. Валидация ===
        model.eval()
        all_logits, all_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits, _ = model(xb, return_features=True)
                all_logits.append(logits.cpu())
                all_targets.append(yb)

        logits = torch.cat(all_logits, dim=0)
        targets = torch.cat(all_targets, dim=0)
        metrics_path = os.path.join(exp_dir, "metrics.json")
        metrics = compute_classification_metrics(logits, targets, save_path=metrics_path)

        # === 5. Геометрический анализ ===
        latent_vectors = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device)
                _, z = model(xb, return_features=True)
                latent_vectors.append(z.cpu())

        latent_all = torch.cat(latent_vectors, dim=0).numpy()
        # === 5. Визуализация ===
        umap_img_path = os.path.join(exp_dir, "umap.png")
        save_umap_projection(latent_all, targets.numpy(), umap_img_path)
        interactive_path = os.path.join(exp_dir, "latents_interactive.html")
        save_interactive_projection(latent_all, targets.numpy(), interactive_path, fallback_image=umap_img_path)
        npz_path = os.path.join(exp_dir, "latents.npz")
        np.savez(npz_path, latents=latent_all, labels=targets.numpy())

        # === 6. Логирование ===
        for k, v in metrics.items():
            mlflow.log_metric(f"val/{k}", float(v))
        mlflow.log_artifact(metrics_path)
        mlflow.log_artifact(umap_img_path)
        mlflow.log_artifact(interactive_path)
        mlflow.log_artifact(npz_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python -m src.runners.run_clf_base <CONFIG_PATH> <EXP_DIR>")
        exit(1)
    run_experiment(sys.argv[1], sys.argv[2])
