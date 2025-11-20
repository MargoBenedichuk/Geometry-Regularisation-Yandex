import os
import json
import hashlib
import yaml
import subprocess
from datetime import datetime
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from filelock import FileLock

# === Каталоги ===
BASE_DIR = Path(".").resolve()
CONFIG_DIR = BASE_DIR / "configs/auto_generated"
RESULTS_DIR = BASE_DIR / "experiments_result"
REGISTRY_PATH = RESULTS_DIR / "experiment_registry.json"
LOCK_PATH = REGISTRY_PATH.with_suffix(".lock")

os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# === Параметры перебора ===
DATASETS = ["mnist", "cifar10", "imagenet"]
MODELS = ["simple_cnn", "resnet50", "resnet101"]
SEEDS = [42, 2025]
LOSSES = ["none", "geodesic", "geometry", "combined"]  # "combined" = geodesic + geometry


# === Генерация сигнатуры ===
def make_signature(dataset, model, seed, loss):
    return {
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "loss": loss
    }

def hash_signature(sig):
    payload = json.dumps(sig, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()

# === Генерация конфига ===
def generate_config(sig, hash_id):
    reg_cfg = {
        "weight": 0.1 if sig["loss"] != "none" else 0.0,
        "metric": sig["loss"]
    }
    if sig["loss"] in ["geodesic", "combined"]:
        reg_cfg["geodesic_ratio"] = {
            "n_neighbors": 10,
            "target_ratio": 1.0,
            "lambda_reg": 0.1
        }
    if sig["loss"] in ["geometry", "combined"]:
        reg_cfg["geometry_mark"] = {
            "margin": 0.1,
            "lambda_reg": 0.1
        }

    if sig["dataset"] == "imagenet":
        transform = None  # assume handled inside model or dataset wrapper
        input_shape = [3, 224, 224]
        num_classes = 1000
    elif sig["dataset"] == "cifar10":
        transform = "src.utils.registry.default_mnist_transform"
        input_shape = [3, 32, 32]
        num_classes = 10
    else:  # mnist
        transform = "src.utils.registry.default_mnist_transform"
        input_shape = [1, 28, 28]
        num_classes = 10

    cfg = {
        "experiment_name": f"clf_{sig['dataset']}_{sig['model']}_{sig['loss']}_{sig['seed']}",
        "dataset": {
            "base": f"src.utils.registry.get_{sig['dataset']}",
            "root": "./data",
            "val_ratio": 0.2,
        },
        "model": {
            "name": sig["model"],
            "input_shape": input_shape,
            "hidden_dim": 128,
            "num_classes": num_classes
        },
        "train": {
            "epochs": 10,
            "batch_size": 64,
            "lr": 0.001
        },
        "device": "cuda",
        "seed": sig["seed"],
        "regularization": reg_cfg
    }

    if transform:
        cfg["dataset"]["transform"] = transform

    cfg_path = CONFIG_DIR / f"{hash_id}.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)
    return str(cfg_path)


# === Запуск эксперимента ===
def run_experiment(entry):
    sig, hash_id = entry["signature"], entry["hash"]
    cfg_path = generate_config(sig, hash_id)
    out_dir = RESULTS_DIR / f"{sig['dataset']}/{sig['model']}/seed_{sig['seed']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["python", "-m", "src.runners.run_clf_base_v3", cfg_path, str(out_dir)]
    try:
        subprocess.run(cmd, check=True)
        status = "done"
    except Exception as e:
        status = f"failed: {e}"

    with FileLock(LOCK_PATH):
        registry = json.load(open(REGISTRY_PATH)) if REGISTRY_PATH.exists() else []
        for r in registry:
            if r["hash"] == hash_id:
                r["status"] = status
        with open(REGISTRY_PATH, "w") as f:
            json.dump(registry, f, indent=2)

    print(f"[STATUS] {sig['dataset']}/{sig['model']}/seed_{sig['seed']} → {status}")

def is_valid_combo(dataset, model):
    if model == "simple_cnn":
        return dataset in ["mnist", "cifar10"]
    if model.startswith("resnet"):
        return dataset == "imagenet"
    return True


# === Основная функция ===
def main():
    all_signatures = []
    for ds, mdl, seed, loss in product(DATASETS, MODELS, SEEDS, LOSSES):
        if not is_valid_combo(ds, mdl):
            continue
        sig = make_signature(ds, mdl, seed, loss)
        h = hash_signature(sig)
        all_signatures.append({"signature": sig, "hash": h, "status": "pending"})

    if not REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "w") as f:
            json.dump(all_signatures, f, indent=2)
    else:
        with open(REGISTRY_PATH) as f:
            existing = json.load(f)
        merged = {r["hash"]: r for r in existing}
        for r in all_signatures:
            if r["hash"] not in merged:
                merged[r["hash"]] = r
        with open(REGISTRY_PATH, "w") as f:
            json.dump(list(merged.values()), f, indent=2)

    with open(REGISTRY_PATH) as f:
        entries = [r for r in json.load(f) if r["status"] != "done"]

    print(f"[INFO] Запуск {len(entries)} экспериментов...")
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 2)) as pool:
        futures = [pool.submit(run_experiment, entry) for entry in entries]
        for fut in as_completed(futures):
            fut.result()

if __name__ == "__main__":
    main()
