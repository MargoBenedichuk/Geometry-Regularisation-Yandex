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

from sklearn.metrics import classification_report
from src.utils.config import load_config
from src.dataset.datasets import BalancedClassificationDataset, ClassificationDataset
from src.dataset.dataloader import make_dataloader
from src.dataset.splits import make_classification_splits
from src.models.head_factory import build_model
from src.metrics.task import compute_classification_metrics
from src.regularization.geodesic_loss import compute_geodesic_loss
from src.utils.registry import default_mnist_transform, resolve_target
from src.vizualisation.vizualisator import save_interactive_projection, save_umap_projection
from src.metrics.geodesic import compute_geodesic_summary
from src.metrics.geometry_mark import compute_geometry_summary
import matplotlib.pyplot as plt


def plot_metric(metric_name, history, save_path):
    plt.figure()
    plt.plot(history[metric_name], label='Train')
    val_key = f"val_{metric_name}" if f"val_{metric_name}" in history else None
    if val_key:
        plt.plot(history[val_key], label='Validation')
    plt.title(f'{metric_name.capitalize()} over epochs')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name.capitalize())
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()


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
        model = build_model(cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
        
        # === 3. Metrics Collection ===
        metrics_history = defaultdict(list)  # Store metrics for all epochs
        
        # === 4. Training Loop ===
        print("[INFO] Starting training...")
        progress = tqdm(range(cfg.train.epochs), desc=f"Epoch: ", leave=False)
        for epoch in  progress:
            model.train()
            running_loss = 0.0
            running_ce = 0.0
            running_geod = 0.0
            steps = 0
            
            if train_balanced and hasattr(train_ds, "reshuffle"):
                train_ds.reshuffle()
            # progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.train.epochs}", leave=False)
            for xb, yb in train_loader:
                # print(f"Epoch {epoch + 1}/{cfg.train.epochs} - Batch Step")
                xb, yb = xb.to(device), yb.to(device)
                logits, latents = model(xb, return_features=True)
                
                # CE loss
                loss_ce = torch.nn.functional.cross_entropy(logits, yb)
                
                # GEODESIC loss используя диспетчер
                loss_geod = compute_geodesic_loss(latents, labels=yb, cfg=cfg.regularization)
                
                # Total loss
                loss = loss_ce + cfg.regularization.weight * loss_geod
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                steps += 1
                running_loss += loss.item()
                running_ce += loss_ce.item()
                running_geod += loss_geod.item()
                
                progress.set_postfix(
                    loss=f"{running_loss / steps:.4f}",
                    ce=f"{loss_ce.item():.3f}",
                    geod=f"{loss_geod.item():.3f}",
                )
            
            epoch_loss = running_loss / max(steps, 1)
            epoch_ce = running_ce / max(steps, 1)
            epoch_geod = running_geod / max(steps, 1)
            
            metrics_history["train_loss"].append(epoch_loss)
            metrics_history["train_ce_loss"].append(epoch_ce)
            metrics_history["train_geodesic_loss"].append(epoch_geod)
            
            mlflow.log_metric("train/loss", epoch_loss, step=epoch)
            mlflow.log_metric("train/ce_loss", epoch_ce, step=epoch)
            mlflow.log_metric("train/geodesic_loss", epoch_geod, step=epoch)
            
            # === 5. Validation ===
            model.eval()
            all_logits_val, all_targets_val = [], []
            val_loss_total = 0.0
            val_steps = 0
            
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits, latents = model(xb, return_features=True)
                    all_logits_val.append(logits.cpu())
                    all_targets_val.append(yb.cpu())
                    
                    # Compute validation loss
                    loss_ce = torch.nn.functional.cross_entropy(logits, yb)
                    loss_geod = compute_geodesic_loss(latents, labels=yb, cfg=cfg.regularization)
                    val_loss = loss_ce + cfg.regularization.weight * loss_geod
                    val_loss_total += val_loss.item()
                    val_steps += 1
            
            val_loss_avg = val_loss_total / max(val_steps, 1)
            metrics_history["val_loss"].append(val_loss_avg)
            
            # Compute accuracy
            logits_val = torch.cat(all_logits_val, dim=0)
            targets_val = torch.cat(all_targets_val, dim=0)
            val_accuracy = (logits_val.argmax(1) == targets_val).float().mean().item()
            metrics_history["val_accuracy"].append(val_accuracy)
            
            mlflow.log_metric("val/loss", val_loss_avg, step=epoch)
            mlflow.log_metric("val/accuracy", val_accuracy, step=epoch)
            
            print(f"  Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss_avg:.4f}, Val Acc: {val_accuracy:.4f}")
        
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
            10
        )
        geodesic_summary = compute_geodesic_summary(latents, targets.numpy(), n_neighbors=n_neighbors)
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


        # === 12. Save Train/Val Curves ===
        plot_metric("loss", metrics_history, os.path.join(exp_dir, "train_val_loss.png"))
        plot_metric("accuracy", metrics_history, os.path.join(exp_dir, "train_val_accuracy.png"))


        # === 13. Classification Report ===
        report_dict = classification_report(
            targets.numpy(), logits.argmax(1).numpy(),
            output_dict=True, digits=4
        )
        report_path = os.path.join(exp_dir, "classification_report.json")
        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=2)

        # log classification report as artifact
        mlflow.log_artifact(report_path)


        
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
