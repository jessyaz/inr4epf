import os
import uuid
import multiprocessing as mp
from pathlib import Path

import torch
import mlflow
from omegaconf import OmegaConf
from dotenv import load_dotenv

from runner import instantiate_model
from datasets.loader import load_market_dataloader
from utils.trainer import train as trainer
from utils.tester import test as tester
from utils.mlflow_logger import MLflowLogger

DATASETS = ["FR", "NP", "PJM", "BE", "DE"]
MISSING_RATIOS = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5"]
SEEDS = [0, 1, 2, 3, 4]

N_GPUS = max(1, torch.cuda.device_count())
WORKERS_PER_GPU = 2


def build_cfg(dataset_name, missing_ratio, seed, model_uid, run_dir):
    cfg = OmegaConf.create({
        "registry": "dlinear",
        "seed": seed, "device": "auto", "run_dir": str(run_dir), "model_uid": model_uid,
        "dataset": {"name": dataset_name, "batch_size": 64, "missing_rate": float(missing_ratio)},
        "model": {
            "backbone_name": "DLinear",
            "lookback": 168, "horizon": 24, "enc_in": 3,
            "d_model": 512, "d_ff": 2048, "e_layers": 2,
            "moving_avg": 25, "individual": False,
            "dropout": 0.1, "activation": "gelu",
            "lr": 1e-3, "num_epoch": 50,
        },
        "mlflow": {
            "experiment_name": f"icassp2-dlinear_{dataset_name}_{missing_ratio}",
            "run_name": f"dlinearv2_{model_uid}",
        },
    })
    return cfg


def run_one(dataset_name, missing_ratio, seed, device):
    model_uid = uuid.uuid4().hex[:8]
    run_dir = Path("runs") / f"{model_uid}_dlinearv2"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(dataset_name, missing_ratio, seed, model_uid, run_dir)
    torch.manual_seed(cfg.seed)

    train_loader, val_loader, test_loader, scaler = load_market_dataloader(
        cfg.dataset.name, batch_size=cfg.dataset.batch_size,
        missing_rate=cfg.dataset.missing_rate, seed=cfg.seed,
    )

    model = instantiate_model(cfg).to(device)

    with MLflowLogger(cfg) as logger:
        mlflow.set_tag("seed", seed)
        mlflow.set_tag("missing_ratio", missing_ratio)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.lr)
        loaders = {"train_loader": train_loader, "val_loader": val_loader}
        trainer(model, loaders, optimizer, device, logger)

        ckpt_path = run_dir / "model.pth"
        torch.save(model.state_dict(), ckpt_path)
        logger.log_checkpoint(str(ckpt_path))

        loss_dict_test = tester(model, test_loader, scaler, device, logger)
        logger.tester_flag = True

    print(f"[PID {mp.current_process().pid} / {device}] DLinearv2 | {dataset_name} ratio={missing_ratio} seed={seed} -> {loss_dict_test}")
    return dataset_name, missing_ratio, seed, loss_dict_test


_WORKER_DEVICE = None


def _init_worker(counter, lock, tracking_uri):
    global _WORKER_DEVICE
    with lock:
        idx = counter.value
        counter.value += 1
    _WORKER_DEVICE = f"cuda:{idx % N_GPUS}"
    load_dotenv("./.env", override=True)
    mlflow.set_tracking_uri(tracking_uri)
    print(f"[Worker PID {os.getpid()}] epingle a {_WORKER_DEVICE}")


def _worker_entry(task):
    dataset_name, missing_ratio, seed = task
    return run_one(dataset_name, missing_ratio, seed, _WORKER_DEVICE)


def main():
    load_dotenv("./.env", override=True)
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(tracking_uri)

    tasks = [(d, r, s) for d in DATASETS for r in MISSING_RATIOS for s in SEEDS]
    print(f"Total : {len(tasks)} runs DLinearv2 a effectuer.")

    n_workers = N_GPUS * WORKERS_PER_GPU
    manager = mp.Manager()
    counter = manager.Value("i", 0)
    lock = manager.Lock()

    with mp.Pool(processes=n_workers, initializer=_init_worker, initargs=(counter, lock, tracking_uri)) as pool:
        pool.map(_worker_entry, tasks)

    print("\nTermine.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()