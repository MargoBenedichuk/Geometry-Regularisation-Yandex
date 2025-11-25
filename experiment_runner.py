# experiment_runner.py
"""
Main experiment runner for comparing geodesic regularizer vs baseline (CE only).

Usage:
    python experiment_runner.py
    
This script:
1. Runs experiment WITH geodesic regularizer
2. Runs experiment WITHOUT regularization (CE baseline)
3. Compares results and generates analysis
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).parent))

from src.runner.run_clf_base_v2 import run_experiment


def create_comparison_report(results_with_custom, results_baseline, output_dir):
    """
    Create comparison report between two experiments.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "comparison": {},
        "analysis": {}
    }
    
    # === Classification Metrics Comparison ===
    with_custom_metrics = results_with_custom["metrics"]
    baseline_metrics = results_baseline["metrics"]
    
    comparison_data = []
    metric_keys = set(with_custom_metrics.keys()) | set(baseline_metrics.keys())
    
    for metric in sorted(metric_keys):
        with_custom_val = with_custom_metrics.get(metric, "N/A")
        baseline_val = baseline_metrics.get(metric, "N/A")
        
        if isinstance(with_custom_val, (int, float)) and isinstance(baseline_val, (int, float)):
            diff = with_custom_val - baseline_val
            diff_pct = (diff / baseline_val * 100) if baseline_val != 0 else 0
            comparison_data.append({
                "Metric": metric,
                f"With {regularization}": f"{with_custom_val:.4f}",
                "Baseline (CE)": f"{baseline_val:.4f}",
                "Difference": f"{diff:+.4f}",
                "% Change": f"{diff_pct:+.2f}%"
            })
        else:
            comparison_data.append({
                "Metric": metric,
                f"With {regularization}": str(with_custom_val),
                "Baseline (CE)": str(baseline_val),
                "Difference": "N/A",
                "% Change": "N/A"
            })
    
    report["comparison"]["classification_metrics"] = comparison_data
    
    # === Geometric Metrics Comparison ===
    geo_with = results_with_custom.get("geometry", {})
    geo_base = results_baseline.get("geometry", {})
    
    geo_comparison = []
    
    if 'local_dimension' in geo_with and 'local_dimension' in geo_base:
        ld_with = geo_with['local_dimension']
        ld_base = geo_base['local_dimension']
        geo_comparison.append({
            "Property": "Local Dimension (Mean)",
            "With Geodesic": f"{ld_with.get('mean', 'N/A'):.4f}",
            "Baseline": f"{ld_base.get('mean', 'N/A'):.4f}"
        })
    
    if 'silhouette_score' in geo_with and 'silhouette_score' in geo_base:
        geo_comparison.append({
            "Property": "Silhouette Score",
            "With Geodesic": f"{geo_with['silhouette_score']:.4f}",
            "Baseline": f"{geo_base['silhouette_score']:.4f}"
        })
    
    report["comparison"]["geometric_metrics"] = geo_comparison
    
    # === Geodesic Metrics Comparison ===
    geo_reg_with = results_with_custom.get("geodesic", {})
    geo_reg_base = results_baseline.get("geodesic", {})
    
    geo_reg_comparison = []
    
    if 'global' in geo_reg_with and 'global' in geo_reg_base:
        with_global = geo_reg_with['global']
        base_global = geo_reg_base['global']
        
        geo_reg_comparison.append({
            "Metric": "Global Geodesic Mean",
            "With Geodesic": f"{with_global.get('mean', 'N/A'):.4f}",
            "Baseline": f"{base_global.get('mean', 'N/A'):.4f}"
        })
        
        geo_reg_comparison.append({
            "Metric": "Global Geodesic Std",
            "With Geodesic": f"{with_global.get('std', 'N/A'):.4f}",
            "Baseline": f"{base_global.get('std', 'N/A'):.4f}"
        })
    
    if 'analysis' in geo_reg_with and 'analysis' in geo_reg_base:
        with_analysis = geo_reg_with['analysis']
        base_analysis = geo_reg_base['analysis']
        
        geo_reg_comparison.append({
            "Metric": "Flatness Score",
            "With Geodesic": f"{with_analysis.get('flatness_score', 'N/A'):.4f}",
            "Baseline": f"{base_analysis.get('flatness_score', 'N/A'):.4f}"
        })
        
        geo_reg_comparison.append({
            "Metric": "Class Separability Ratio",
            "With Geodesic": f"{with_analysis.get('class_separability_ratio', 'N/A'):.4f}",
            "Baseline": f"{base_analysis.get('class_separability_ratio', 'N/A'):.4f}"
        })
        
        geo_reg_comparison.append({
            "Metric": "Overall Quality Score",
            "With Geodesic": f"{with_analysis.get('overall_quality_score', 'N/A'):.4f}",
            "Baseline": f"{base_analysis.get('overall_quality_score', 'N/A'):.4f}"
        })
    
    report["comparison"]["geodesic_metrics"] = geo_reg_comparison
    
    # === Training Dynamics Analysis ===
    history_with = results_with_custom.get("history", {})
    history_base = results_baseline.get("history", {})
    
    analysis = {
        "final_train_loss": {
            "with_geodesic": history_with.get("train_loss", [])[-1] if history_with.get("train_loss") else "N/A",
            "baseline": history_base.get("train_loss", [])[-1] if history_base.get("train_loss") else "N/A"
        },
        "final_val_loss": {
            "with_geodesic": history_with.get("val_loss", [])[-1] if history_with.get("val_loss") else "N/A",
            "baseline": history_base.get("val_loss", [])[-1] if history_base.get("val_loss") else "N/A"
        },
        "final_val_accuracy": {
            "with_geodesic": history_with.get("val_accuracy", [])[-1] if history_with.get("val_accuracy") else "N/A",
            "baseline": history_base.get("val_accuracy", [])[-1] if history_base.get("val_accuracy") else "N/A"
        },
        "final_geodesic_loss": {
            "with_geodesic": history_with.get("train_geodesic_loss", [])[-1] if history_with.get("train_geodesic_loss") else "N/A",
            "baseline": 0.0
        },
        "overfitting_gap_geodesic": (
            history_with.get("val_loss", [])[-1] - history_with.get("train_loss", [])[-1]
            if history_with.get("val_loss") and history_with.get("train_loss")
            else "N/A"
        ),
        "overfitting_gap_baseline": (
            history_base.get("val_loss", [])[-1] - history_base.get("train_loss", [])[-1]
            if history_base.get("val_loss") and history_base.get("train_loss")
            else "N/A"
        )
    }
    
    report["analysis"]["training_dynamics"] = analysis
    
    # === Save Report ===
    report_path = os.path.join(output_dir, "comparison_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    return report, comparison_data, geo_comparison, geo_reg_comparison, analysis


def plot_training_curves(results_with_custom, results_baseline, output_dir):
    """
    Plot training curves comparison.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    history_with = results_with_custom.get("history", {})
    history_base = results_baseline.get("history", {})
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Training Loss
    if history_with.get("train_loss") and history_base.get("train_loss"):
        axes[0, 0].plot(history_with["train_loss"], label="With Geodesic", marker='o', linewidth=2)
        axes[0, 0].plot(history_base["train_loss"], label="Baseline (CE)", marker='s', linewidth=2)
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss
    if history_with.get("val_loss") and history_base.get("val_loss"):
        axes[0, 1].plot(history_with["val_loss"], label="With Geodesic", marker='o', linewidth=2)
        axes[0, 1].plot(history_base["val_loss"], label="Baseline (CE)", marker='s', linewidth=2)
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].set_title("Validation Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Validation Accuracy
    if history_with.get("val_accuracy") and history_base.get("val_accuracy"):
        axes[1, 0].plot(history_with["val_accuracy"], label="With Geodesic", marker='o', linewidth=2)
        axes[1, 0].plot(history_base["val_accuracy"], label="Baseline (CE)", marker='s', linewidth=2)
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Accuracy")
        axes[1, 0].set_title("Validation Accuracy")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Overfitting Gap
    if history_with.get("train_loss") and history_with.get("val_loss"):
        overfitting_with = [v - t for v, t in zip(history_with["val_loss"], history_with["train_loss"])]
        overfitting_base = [v - t for v, t in zip(history_base["val_loss"], history_base["train_loss"])]
        
        axes[1, 1].plot(overfitting_with, label="With Geodesic", marker='o', linewidth=2)
        axes[1, 1].plot(overfitting_base, label="Baseline (CE)", marker='s', linewidth=2)
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Val Loss - Train Loss")
        axes[1, 1].set_title("Overfitting Gap (Val - Train)")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_curves_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Training curves saved to {plot_path}")
    plt.close()


regularization = "density_topological" ### "geodesic"

def main():
    """
    Main experiment runner.
    """
    print("=" * 80)
    print(f"{regularization.upper()} REGULARIZER vs BASELINE (CE only) EXPERIMENT")
    print("=" * 80)
    
    # Create experiment directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiments_dir = f"exps/experiments_{timestamp}"
    
    exp_with_custom_dir = os.path.join(experiments_dir, f"with_{regularization}")
    exp_baseline_dir = os.path.join(experiments_dir, "baseline_ce")
    comparison_dir = os.path.join(experiments_dir, "comparison")
    
    # Get config paths
    config_with_custom = f"configs/config_with_{regularization}.yaml"
    config_baseline = "configs/config_without_regularization.yaml"
    
    if not os.path.exists(config_with_custom):
        print(f"[ERROR] Config file not found: {config_with_custom}")
        sys.exit(1)
    
    if not os.path.exists(config_baseline):
        print(f"[ERROR] Config file not found: {config_baseline}")
        sys.exit(1)
    
    # Run experiments
    print(f"\n[1/3] Running experiment WITH {regularization} regularizer...")
    print(f"      Config: {config_with_custom}")
    print(f"      Output: {exp_with_custom_dir}")
    results_with_custom = run_experiment(config_with_custom, exp_with_custom_dir)
    
    print(f"\n[2/3] Running baseline experiment (CE only)...")
    print(f"      Config: {config_baseline}")
    print(f"      Output: {exp_baseline_dir}")
    results_baseline = run_experiment(config_baseline, exp_baseline_dir)
    
    # Compare results
    print(f"\n[3/3] Creating comparison report...")
    report, comp_metrics, comp_geom, comp_geod, analysis = create_comparison_report(
        results_with_custom, results_baseline, comparison_dir
    )
    
    # Print tables
    print("\n" + "=" * 80)
    print("CLASSIFICATION METRICS COMPARISON")
    print("=" * 80)
    print(tabulate(comp_metrics, headers="keys", tablefmt="grid"))
    
    if comp_geom:
        print("\n" + "=" * 80)
        print("GEOMETRIC METRICS COMPARISON")
        print("=" * 80)
        print(tabulate(comp_geom, headers="keys", tablefmt="grid"))
    
    if comp_geod:
        print("\n" + "=" * 80)
        print("GEODESIC METRICS COMPARISON")
        print("=" * 80)
        print(tabulate(comp_geod, headers="keys", tablefmt="grid"))
    
    print("\n" + "=" * 80)
    print("TRAINING DYNAMICS ANALYSIS")
    print("=" * 80)
    for key, value in analysis.items():
        if isinstance(value, dict):
            print(f"\n{key.upper().replace('_', ' ')}:")
            for k, v in value.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.6f}")
                else:
                    print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    
    # Plot training curves
    plot_training_curves(results_with_custom, results_baseline, comparison_dir)
    
    # Summary
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED")
    print("=" * 80)
    print(f"All results saved to: {experiments_dir}")
    print(f"  - With {regularization}: {exp_with_custom_dir}")
    print(f"  - Baseline (CE): {exp_baseline_dir}")
    print(f"  - Comparison:    {comparison_dir}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Experiment interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Experiment failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
