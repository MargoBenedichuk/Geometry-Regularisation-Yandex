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

# === Генерация сигнатуры ===
def make_signature(dataset, model, seed):
    return {
        "dataset": dataset,
        "model": model,
        "seed": seed,
    }

def hash_signature(sig):
    payload = json.dumps(sig, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()

# === Генерация конфига ===
def generate_config(sig, hash_id):
    cfg = {
        "experiment_name": f"clf_{sig['dataset']}_{sig['model']}_{sig['seed']}",
        "dataset": {
            "base": f"src.utils.registry.get_{sig['dataset']}",
            "root": "./data",
            "val_ratio": 0.2,
            "transform": "src.utils.registry.default_mnist_transform"
        },
        "model": {
            "name": sig["model"],
            "input_shape": [1, 28, 28] if sig["dataset"] == "mnist" else [3, 32, 32],
            "hidden_dim": 128,
            "num_classes": 10
        },
        "train": {
            "epochs": 10,
            "batch_size": 64,
            "lr": 0.001
        },
        "device": "cuda",
        "seed": sig["seed"],
        "regularization": {
            "weight": 0.0,
            "metric": "info_nce"
        }
    }
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

    cmd = ["python", "-m", "src.runners.run_clf_base_v2", cfg_path, str(out_dir)]
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

# === Основная функция ===
def main():
    all_signatures = []
    for ds, mdl, seed in product(DATASETS, MODELS, SEEDS):
        sig = make_signature(ds, mdl, seed)
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
