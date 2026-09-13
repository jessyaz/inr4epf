"""
Ablation "avec/sans contexte exogene" (use_exog), a missing_ratio=0.0
uniquement, 5 datasets x 5 seeds x 2 variantes = 50 runs.

Parallelisation : 2 GPU, 1 run par GPU en parallele.
"""

import uuid
from pathlib import Path
import multiprocessing as mp

import torch
from omegaconf import OmegaConf

from runner import instantiate_model
from datasets.loader import load_market_dataloader
from utils.trainer import train as trainer
from utils.tester import test as tester
from utils.mlflow_logger import MLflowLogger

DATASETS = ["FR", "NP", "PJM", "BE", "DE"]
MISSING_RATIO = "0.0"
SEEDS = [0, 1, 2, 3, 4]

BEST_CONFIG_DIR = "best_configs"
N_GPUS = max(1, torch.cuda.device_count())
WORKERS_PER_GPU = 2


def build_cfg(dataset_name, seed, use_exog, model_uid, run_dir):
    cfg = OmegaConf.load(f"{BEST_CONFIG_DIR}/best_config_{dataset_name}.yaml")

    cfg.seed = seed
    cfg.model_uid = model_uid
    cfg.run_dir = str(run_dir)
    cfg.dataset.missing_rate = float(MISSING_RATIO)
    cfg.model.use_exog = use_exog

    variant = "full" if use_exog else "noexog"
    cfg.mlflow.experiment_name = f"ablation-{dataset_name}-{MISSING_RATIO}"
    cfg.mlflow.run_name = f"{variant}_{model_uid}"

    return cfg


def run_one(dataset_name, seed, use_exog, device):
    model_uid = uuid.uuid4().hex[:8]
    variant = "full" if use_exog else "noexog"
    run_dir = Path("runs") / f"{model_uid}_{variant}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(dataset_name, seed, use_exog, model_uid, run_dir)
    torch.manual_seed(cfg.seed)

    train_loader, val_loader, test_loader, scaler = load_market_dataloader(
        cfg.dataset.name, batch_size=cfg.dataset.batch_size,
        missing_rate=cfg.dataset.missing_rate, seed=cfg.seed,
    )

    model = instantiate_model(cfg).to(device)

    with MLflowLogger(cfg) as logger:
        optimizer = model.configure_optimizer()
        loaders = {"train_loader": train_loader, "val_loader": val_loader}
        trainer(model, loaders, optimizer, device, logger)

        ckpt_path = run_dir / "model.pth"
        torch.save(model.state_dict(), ckpt_path)
        logger.log_checkpoint(str(ckpt_path))

        loss_dict_test = tester(model, test_loader, scaler, device, logger)
        logger.tester_flag = True

    print(f"[PID {mp.current_process().pid} / {device}] {variant} | {dataset_name} seed={seed} -> {loss_dict_test}")
    return dataset_name, variant, seed, loss_dict_test


_WORKER_DEVICE = None


def _init_worker(counter, lock):
    global _WORKER_DEVICE
    with lock:
        idx = counter.value
        counter.value += 1
    _WORKER_DEVICE = f"cuda:{idx % N_GPUS}"
    print(f"[Worker PID {mp.current_process().pid}] epingle a {_WORKER_DEVICE}")


def _worker_entry(task):
    dataset_name, seed, use_exog = task
    return run_one(dataset_name, seed, use_exog, _WORKER_DEVICE)


def main():
    tasks = [
        (dataset_name, seed, use_exog)
        for dataset_name in DATASETS
        for seed in SEEDS
        for use_exog in [True, False]
    ]
    print(f"Total : {len(tasks)} runs d'ablation a effectuer.")

    manager = mp.Manager()
    counter = manager.Value("i", 0)
    lock = manager.Lock()

    n_workers = N_GPUS * WORKERS_PER_GPU
    with mp.Pool(processes=n_workers, initializer=_init_worker, initargs=(counter, lock)) as pool:
        results = pool.map(_worker_entry, tasks)

    print("\nTermine.")
    return results


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()