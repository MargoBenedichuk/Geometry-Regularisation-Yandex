# src/runners/run_clf_base_v2.py
"""
Enhanced version with detailed metrics collection for training analysis.
"""

import os
import json
import mlflow
import torch
import numpy as np
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from collections import defaultdict

from src.regularization.geometry_mark_loss import compute_geometric_loss
from src.utils.config import load_config
from src.dataset.datasets import BalancedClassificationDataset, ClassificationDataset
from src.dataset.dataloader import make_dataloader
from src.dataset.splits import make_classification_splits
from src.models.cnns import SimpleCNN
from src.metrics.task import compute_classification_metrics
from src.regularization.geodesic_loss import compute_geodesic_loss
from src.utils.registry import default_mnist_transform, resolve_target
from src.vizualisation.vizualisator import save_interactive_projection, save_umap_projection
from src.metrics.geodesic import compute_geodesic_summary
from src.metrics.geometry_mark import compute_geometry_summary


def compute_geodesic_loss_on_epoch(all_latents, all_targets, cfg, device):
    """
    Вычислить geodesic loss на ВСЕ embeddings эпохи, а не на маленьких батчах.

    Это дает правильные ratios, потому что граф имеет достаточно точек.
    """
    if len(all_latents) == 0:
        return torch.tensor(0.0, device=device)

    latents_full = torch.cat(all_latents, dim=0)
    targets_full = torch.cat(all_targets, dim=0)

    loss_geod = compute_geodesic_loss(latents_full, labels=targets_full, cfg=cfg)

    return loss_geod


def run_experiment(cfg_path: str, exp_dir: str, experiment_name: str = None):
    """
    Run experiment with geodesic regularizer using compute_geodesic_loss() dispatcher.
    
    Args:
        cfg_path: path to config YAML file
        exp_dir: directory to save results
        experiment_name: optional experiment name override
    """
    os.makedirs(exp_dir, exist_ok=True)
    cfg = load_config(cfg_path)
    OmegaConf.save(cfg, os.path.join(exp_dir, "config_used.yaml"))

    exp_name = experiment_name or cfg.experiment_name
    mlflow.set_experiment(exp_name)

    with mlflow.start_run():
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        device = torch.device(
            cfg.device if ("cuda" not in str(cfg.device).lower() or torch.cuda.is_available()) else "cpu"
        )
        if device.type == "cpu" and str(cfg.device).lower().startswith("cuda"):
            print("[INFO] CUDA not available, using CPU instead.")

        # === 1. Dataset Loading ===
        print("[INFO] Loading dataset...")
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

        # === 2. Model ===
        print("[INFO] Creating model...")
        model = SimpleCNN(
            input_shape=tuple(cfg.model.input_shape),
            hidden_dim=cfg.model.hidden_dim,
            num_classes=cfg.model.num_classes,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

        # === 3. Metrics Collection ===
        metrics_history = defaultdict(list)  # Store metrics for all epochs

        # === 4. Training Loop ===
        print("[INFO] Starting training...")
        for epoch in range(cfg.train.epochs):
            model.train()
            running_loss = 0.0
            running_ce = 0.0
            running_geom = 0.0
            steps = 0

            epoch_latents = []
            epoch_targets = []
            if train_balanced and hasattr(train_ds, "reshuffle"):
                train_ds.reshuffle()

            progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.train.epochs}", leave=False)
            for xb, yb in progress:
                xb, yb = xb.to(device), yb.to(device)
                logits, latents = model(xb, return_features=True)

                # CE loss
                loss_ce = torch.nn.functional.cross_entropy(logits, yb)

                loss_geom = compute_geometric_loss(latents, labels=yb, cfg=cfg.regularization)

                loss = loss_ce + loss_geom

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                steps += 1
                running_loss += loss.item()
                running_ce += loss_ce.item()
                running_geom += loss_geom.item()

                epoch_latents.append(latents.detach().cpu())
                epoch_targets.append(yb.detach().cpu())
                progress.set_postfix(
                    loss=f"{running_loss / steps:.4f}",
                    ce=f"{loss_ce.item():.3f}",
                    geom=f"{loss_geom.item():.3f}",
                )

            print(f"[INFO] Computing geodesic loss for epoch {epoch + 1} on accumulated embeddings...")
            epoch_latents_full = torch.cat(epoch_latents, dim=0)
            epoch_targets_full = torch.cat(epoch_targets, dim=0)
            loss_geod_epoch = compute_geodesic_loss(epoch_latents_full, labels=epoch_targets_full,
                                                    cfg=cfg.regularization)
            epoch_geod = loss_geod_epoch.item() if loss_geod_epoch.numel() > 0 else 0.0

            epoch_loss = running_loss / max(steps, 1)
            epoch_ce = running_ce / max(steps, 1)
            epoch_geom = running_geom / max(steps, 1)

            metrics_history["train_loss"].append(epoch_loss)
            metrics_history["train_ce_loss"].append(epoch_ce)
            metrics_history["train_geodesic_loss"].append(epoch_geod)
            metrics_history["train_geom_loss"].append(epoch_geom)

            mlflow.log_metric("train/loss", epoch_loss, step=epoch)
            mlflow.log_metric("train/ce_loss", epoch_ce, step=epoch)
            mlflow.log_metric("train/geodesic_loss", epoch_geod, step=epoch)
            mlflow.log_metric("train/geom_loss", epoch_geom, step=epoch)

            # === 5. Validation ===
            model.eval()
            all_logits_val, all_targets_val, all_latents_val = [], [], []
            val_loss_total = 0.0
            val_steps = 0

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits, latents = model(xb, return_features=True)
                    all_logits_val.append(logits.cpu())
                    all_targets_val.append(yb.cpu())
                    all_latents_val.append(latents.cpu())

                    loss_ce = torch.nn.functional.cross_entropy(logits, yb)
                    loss_geom = compute_geometric_loss(latents, labels=yb, cfg=cfg.regularization)
                    val_loss = loss_ce + loss_geom
                    val_loss_total += val_loss.item()
                    val_steps += 1

            if len(all_latents_val) > 0:
                val_latents_full = torch.cat(all_latents_val, dim=0)
                val_targets_full = torch.cat(all_targets_val, dim=0)
                val_loss_geod = compute_geodesic_loss(val_latents_full, labels=val_targets_full, cfg=cfg.regularization)
                val_geod = val_loss_geod.item() if val_loss_geod.numel() > 0 else 0.0
            else:
                val_geod = 0.0

            val_loss_avg = val_loss_total / max(val_steps, 1)
            metrics_history["val_loss"].append(val_loss_avg)
            metrics_history["val_geodesic_loss"].append(val_geod)

            # Compute accuracy
            logits_val = torch.cat(all_logits_val, dim=0)
            targets_val = torch.cat(all_targets_val, dim=0)
            val_accuracy = (logits_val.argmax(1) == targets_val).float().mean().item()
            metrics_history["val_accuracy"].append(val_accuracy)

            mlflow.log_metric("val/loss", val_loss_avg, step=epoch)
            mlflow.log_metric("val/accuracy", val_accuracy, step=epoch)
            mlflow.log_metric("val/geodesic_loss", val_geod, step=epoch)

            print(
                f"  Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss_avg:.4f}, Val Acc: {val_accuracy:.4f}, Geod: {epoch_geod:.6f}")

        # === 6. Final Metrics ===
        print("[INFO] Computing final metrics...")
        model.eval()
        all_logits, all_targets, all_latents = [], [], []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits, latents = model(xb, return_features=True)
                all_logits.append(logits.cpu())
                all_targets.append(yb)
                all_latents.append(latents.cpu())

        logits = torch.cat(all_logits, dim=0)
        targets = torch.cat(all_targets, dim=0)
        latents = torch.cat(all_latents, dim=0).numpy()

        metrics_path = os.path.join(exp_dir, "metrics.json")
        metrics = compute_classification_metrics(logits, targets, save_path=metrics_path)

        # === 7. Geometric Analysis ===
        print("[INFO] Computing geometry summary...")
        geometry_summary = compute_geometry_summary(latents, targets.numpy())
        geometry_path = os.path.join(exp_dir, "geometry_summary.json")
        with open(geometry_path, "w") as f:
            json.dump(geometry_summary, f, indent=2)

        # === 8. Geodesic Analysis ===
        print("[INFO] Computing geodesic summary...")
        n_neighbors = getattr(
            getattr(cfg.regularization, 'geodesic_ratio', None),
            'n_neighbors',
            15
        )
        geodesic_summary = compute_geodesic_summary(latents, targets.numpy(), n_neighbors=n_neighbors,
                                                    subsample_size=2000)
        geodesic_path = os.path.join(exp_dir, "geodesic_summary.json")
        with open(geodesic_path, "w") as f:
            json.dump(geodesic_summary, f, indent=2)

        # === 9. Save Training History ===
        history_path = os.path.join(exp_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump({k: v for k, v in metrics_history.items()}, f, indent=2)

        # === 10. Visualizations ===
        print("[INFO] Creating visualizations...")
        umap_img_path = os.path.join(exp_dir, "umap.png")
        save_umap_projection(latents, targets.numpy(), umap_img_path)

        interactive_path = os.path.join(exp_dir, "latents_interactive.html")
        save_interactive_projection(latents, targets.numpy(), interactive_path, fallback_image=umap_img_path)

        npz_path = os.path.join(exp_dir, "latents.npz")
        np.savez(npz_path, latents=latents, labels=targets.numpy())

        # === 11. MLflow Logging ===
        for k, v in metrics.items():
            mlflow.log_metric(f"final/{k}", float(v))

        if 'local_dimension' in geometry_summary:
            mlflow.log_metric("geometry/local_dim_mean", geometry_summary['local_dimension']['mean'])
            mlflow.log_metric("geometry/local_dim_std", geometry_summary['local_dimension']['std'])

        if 'silhouette_score' in geometry_summary:
            mlflow.log_metric("geometry/silhouette_score", geometry_summary['silhouette_score'])

        if 'global' in geodesic_summary:
            mlflow.log_metric("geodesic/global_mean", geodesic_summary['global']['mean'])
            mlflow.log_metric("geodesic/global_std", geodesic_summary['global']['std'])
            mlflow.log_metric("geodesic/global_median", geodesic_summary['global']['median'])

        if 'analysis' in geodesic_summary:
            mlflow.log_metric("geodesic/flatness_score", geodesic_summary['analysis']['flatness_score'])
            mlflow.log_metric("geodesic/separability_ratio", geodesic_summary['analysis']['class_separability_ratio'])
            mlflow.log_metric("geodesic/quality_score", geodesic_summary['analysis']['overall_quality_score'])

        # Log artifacts
        mlflow.log_artifact(metrics_path)
        mlflow.log_artifact(geometry_path)
        mlflow.log_artifact(geodesic_path)
        mlflow.log_artifact(history_path)
        mlflow.log_artifact(umap_img_path)
        mlflow.log_artifact(interactive_path)
        mlflow.log_artifact(npz_path)

        print(f"[INFO] Experiment completed. Results saved to {exp_dir}")

        return {
            "metrics": metrics,
            "geometry": geometry_summary,
            "geodesic": geodesic_summary,
            "history": dict(metrics_history),
            "exp_dir": exp_dir
        }


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python -m src.runners.run_clf_base_v2 <config_path> <exp_dir>")
        sys.exit(1)
    run_experiment(sys.argv[1], sys.argv[2])
