"""
Evalue le baseline naif k=7 (2 variantes) sur toute la grille :
5 datasets x 6 missing_ratio (0.0-0.5) x 5 seeds. Aucun entrainement --
juste une lecture directe + log MLflow. Parallelise sur 2 GPU.


uv run run_naive_benchmark.py
"""

import uuid
import multiprocessing as mp

import torch
import numpy as np
from omegaconf import OmegaConf

from datasets.loader import load_market_dataloader
from utils.metrics import compute_metrics
from utils.mlflow_logger import MLflowLogger
from naive_k7 import compute_naive_k7

DATASETS = ["FR", "NP", "PJM", "BE", "DE"]
MISSING_RATIOS = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5"]
SEEDS = [0, 1, 2, 3, 4]

LOOKBACK = 168
HORIZON = 24
BATCH_SIZE = 64

N_GPUS = 1
WORKERS_PER_GPU = 1


def evaluate_naive(dataset_name, missing_ratio, seed, device):
    _, _, test_loader, scaler = load_market_dataloader(
        dataset_name, batch_size=BATCH_SIZE, missing_rate=float(missing_ratio), seed=seed
    )

    all_naive_interp, all_naive_raw, all_target = [], [], []

    for batch in test_loader:
        naive_interp, naive_raw = compute_naive_k7(batch, device, LOOKBACK, HORIZON)
        target = batch["y_target_no_mask"][:, LOOKBACK:].to(device).unsqueeze(-1)

        all_naive_interp.append(naive_interp.cpu().numpy())
        all_naive_raw.append(naive_raw.cpu().numpy())
        all_target.append(target.cpu().numpy())

    naive_interp = np.concatenate(all_naive_interp, axis=0)
    naive_raw = np.concatenate(all_naive_raw, axis=0)
    target = np.concatenate(all_target, axis=0)

    if scaler is not None:
        shape = target.shape
        naive_interp = scaler.inverse_transform(naive_interp.reshape(-1, shape[-1])).reshape(shape)
        naive_raw = scaler.inverse_transform(naive_raw.reshape(-1, shape[-1])).reshape(shape)
        target = scaler.inverse_transform(target.reshape(-1, shape[-1])).reshape(shape)

    return naive_interp.squeeze(-1), naive_raw.squeeze(-1), target.squeeze(-1)


def run_one(dataset_name, missing_ratio, seed, device):
    naive_interp, naive_raw, target = evaluate_naive(dataset_name, missing_ratio, seed, device)

    rmse_i, mae_i, mapf_i, smf_i, rmaef_i = compute_metrics(naive_interp, target, naive_ref=naive_interp)
    rmse_r, mae_r, mapf_r, smf_r, rmaef_r = compute_metrics(naive_raw, target, naive_ref=naive_interp)

    model_uid = uuid.uuid4().hex[:8]
    cfg = OmegaConf.create({
        "registry": "naive_k7",
        "seed": seed,
        "model_uid": model_uid,
        "dataset": {"name": dataset_name, "batch_size": BATCH_SIZE, "missing_rate": float(missing_ratio)},
        "model": {"lookback": LOOKBACK, "horizon": HORIZON},
        "mlflow": {
            "experiment_name": f"icassp-{dataset_name}-{missing_ratio}",
            "run_name": f"naivek7_{model_uid}",
        },
    })

    with MLflowLogger(cfg) as logger:
        logger.log_metrics(
            {"RMSE": rmse_i, "MAE": mae_i, "SMAPE": smf_i, "rMAE": rmaef_i},
            epoch=0, prefix="test_naive_interp",
        )
        logger.log_metrics(
            {"RMSE": rmse_r, "MAE": mae_r, "SMAPE": smf_r, "rMAE": rmaef_r},
            epoch=0, prefix="test_naive_raw",
        )
        logger.tester_flag = True

    print(f"[PID {mp.current_process().pid} / {device}] naive | {dataset_name}/{missing_ratio}/seed{seed} "
          f"interp: MAE={mae_i:.4f} SMAPE={smf_i:.4f} | raw: MAE={mae_r:.4f} SMAPE={smf_r:.4f}")
    return dataset_name, missing_ratio, seed


_WORKER_DEVICE = None


def _init_worker(counter, lock):
    global _WORKER_DEVICE
    with lock:
        idx = counter.value
        counter.value += 1
    _WORKER_DEVICE = f"cuda:{idx % N_GPUS}"
    print(f"[Worker PID {mp.current_process().pid}] epingle a {_WORKER_DEVICE}")


def _worker_entry(task):
    dataset_name, missing_ratio, seed = task
    return run_one(dataset_name, missing_ratio, seed, _WORKER_DEVICE)


def main():
    tasks = [
        (dataset_name, missing_ratio, seed)
        for dataset_name in DATASETS
        for missing_ratio in MISSING_RATIOS
        for seed in SEEDS
    ]
    print(f"Total : {len(tasks)} evaluations naive a effectuer.")

    manager = mp.Manager()
    counter = manager.Value("i", 0)
    lock = manager.Lock()

    n_workers = N_GPUS * WORKERS_PER_GPU
    with mp.Pool(processes=n_workers, initializer=_init_worker, initargs=(counter, lock)) as pool:
        pool.map(_worker_entry, tasks)

    print("\nTermine.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()