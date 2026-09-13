"""
Entraine/evalue 5 modeles legers (LEAR, XGBoost, DNN, DLinear, CNN-LSTM) sur
5 datasets x missing_ratio croissant x 5 seeds, avec les hyperparametres
trouves par Optuna (missing_ratio=0.0).

Loggue dans une nouvelle convention de nom d'experience, separee du reste :
icassp-lightmodels_{dataset}_{missing_ratio}

Parallelisation : toujours 2 (modele, dataset, ratio, seed) en cours
d'entrainement en meme temps, un par GPU.
"""

import os
import uuid
import multiprocessing as mp
from pathlib import Path

import mlflow
import numpy as np
import torch
from omegaconf import OmegaConf
from dotenv import load_dotenv

from datasets.loader import load_market_dataloader
from utils.trainer import train as trainer
from utils.metrics import compute_metrics, get_naive_reference
from utils.mlflow_logger import MLflowLogger
from light_models_torch import TORCH_MODEL_REGISTRY
from light_models_sklearn import SKLEARN_MODEL_BUILDERS, fit_and_predict

MODELS = ["lear", "xgboost", "dnn", "dlinear", "cnnlstm"]
DATASETS = ["FR", "NP", "PJM", "BE", "DE"]
MISSING_RATIOS = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5"]
SEEDS = [0, 1, 2, 3, 4]

LOOKBACK, HORIZON = 168, 24
BATCH_SIZE = 64

N_GPUS = max(1, torch.cuda.device_count())
WORKERS_PER_GPU = 4  # modeles petits, 50 Go de VRAM disponible


def load_best_params(model_name, dataset_name):
    import mlflow as mlf
    exp = mlf.get_experiment_by_name(f"optim_light_{model_name}_{dataset_name}")
    if exp is None:
        raise ValueError(f"Pas de recherche Optuna trouvee pour {model_name}/{dataset_name}")
    runs_df = mlf.search_runs(experiment_ids=[exp.experiment_id],
                              filter_string="tags.role = 'optuna_parent' and attributes.status = 'FINISHED'")
    if len(runs_df) == 0:
        raise ValueError(f"Aucun run Optuna FINISHED pour {model_name}/{dataset_name}")
    row = runs_df.sort_values("start_time", ascending=False).iloc[0]
    param_cols = [c for c in runs_df.columns if c.startswith("params.")]
    return {c.replace("params.", ""): row[c] for c in param_cols if row[c] is not None}


def run_torch_model(model_name, dataset_name, missing_ratio, seed, params, device, model_uid, run_dir):
    cfg = OmegaConf.create({
        "model": {
            "lookback": LOOKBACK, "horizon": HORIZON, "exog_dim": 2,
            "num_epoch": 50,
            **{k: (float(v) if k == "lr" or k == "dropout" else int(v)) for k, v in params.items()},
        },
        "dataset": {"name": dataset_name, "batch_size": BATCH_SIZE, "missing_rate": float(missing_ratio)},
        "seed": seed, "model_uid": model_uid, "run_dir": str(run_dir),
        "mlflow": {
            "experiment_name": f"icassp-lightmodels_{dataset_name}_{missing_ratio}",
            "run_name": f"{model_name}_{model_uid}",
        },
    })

    torch.manual_seed(seed)
    train_loader, val_loader, test_loader, scaler = load_market_dataloader(
        dataset_name, batch_size=BATCH_SIZE, missing_rate=float(missing_ratio), seed=seed
    )

    model = TORCH_MODEL_REGISTRY[model_name](cfg).to(device)
    optimizer = model.configure_optimizer()

    with MLflowLogger(cfg) as logger:
        mlflow.set_tag("seed", seed)
        mlflow.set_tag("missing_ratio", missing_ratio)

        loaders = {"train_loader": train_loader, "val_loader": val_loader}
        trainer(model, loaders, optimizer, device, logger)

        # ---- Test manuel (pas de tester.py generique pour ces archis ad hoc) ----
        model.eval()
        all_pred, all_target, all_naive = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                pred = model.forward_step(batch, device)
                y_target = batch["y_target"].to(device)
                target = batch["y_target_no_mask"][:, LOOKBACK:].unsqueeze(-1).to(device)
                naive = get_naive_reference(y_target, LOOKBACK, HORIZON, mode="naive1").unsqueeze(-1)
                all_pred.append(pred.cpu().numpy())
                all_target.append(target.cpu().numpy())
                all_naive.append(naive.cpu().numpy())

        preds = np.concatenate(all_pred, axis=0)
        targets = np.concatenate(all_target, axis=0)
        naive_ref = np.concatenate(all_naive, axis=0)

        if scaler is not None:
            shape = preds.shape
            preds = scaler.inverse_transform(preds.reshape(-1, shape[-1])).reshape(shape)
            targets = scaler.inverse_transform(targets.reshape(-1, shape[-1])).reshape(shape)
            naive_ref = scaler.inverse_transform(naive_ref.reshape(-1, shape[-1])).reshape(shape)

        rmse, mae, mapf, smape, rmae = compute_metrics(preds.squeeze(-1), targets.squeeze(-1), naive_ref=naive_ref.squeeze(-1))
        logger.log_metrics({"RMSE": rmse, "MAE": mae, "SMAPE": smape, "rMAE": rmae}, epoch=0, prefix="test")

        # Sauvegarde des predictions brutes (necessaire pour DM test / analyses futures)
        preds_path = run_dir / "predictions.npz"
        np.savez(preds_path, preds=preds.squeeze(-1), targets=targets.squeeze(-1))
        logger.log_checkpoint(str(preds_path))

        logger.tester_flag = True

    print(f"[{model_name}/{dataset_name}/{missing_ratio}/seed{seed}] MAE={mae:.4f} SMAPE={smape:.4f}")
    return mae


def run_sklearn_model(model_name, dataset_name, missing_ratio, seed, params, model_uid, run_dir):
    builder = SKLEARN_MODEL_BUILDERS[model_name]
    params_cast = {}
    for k, v in params.items():
        if k in ("n_estimators", "max_depth", "calibration_window"):
            params_cast[k] = int(v)
        else:
            params_cast[k] = float(v)
    model = builder(**params_cast)

    train_loader, val_loader, test_loader, scaler = load_market_dataloader(
        dataset_name, batch_size=BATCH_SIZE, missing_rate=float(missing_ratio), seed=seed
    )

    preds, targets = fit_and_predict(model, train_loader, val_loader, test_loader, "cpu", LOOKBACK, HORIZON)

    # naive ref recalculee separement pour le rMAE (mode naive1, coherent avec le reste)
    all_naive = []
    for batch in test_loader:
        y_target = batch["y_target"]
        naive = get_naive_reference(y_target, LOOKBACK, HORIZON, mode="naive1")
        all_naive.append(naive.numpy())
    naive_ref = np.concatenate(all_naive, axis=0)

    if scaler is not None:
        shape = preds.shape
        preds = scaler.inverse_transform(preds.reshape(-1, shape[-1])).reshape(shape)
        targets = scaler.inverse_transform(targets.reshape(-1, shape[-1])).reshape(shape)
        naive_ref = scaler.inverse_transform(naive_ref.reshape(-1, shape[-1])).reshape(shape)

    rmse, mae, mapf, smape, rmae = compute_metrics(preds, targets, naive_ref=naive_ref)

    cfg = OmegaConf.create({
        "model": {"lookback": LOOKBACK, "horizon": HORIZON},
        "dataset": {"name": dataset_name, "batch_size": BATCH_SIZE, "missing_rate": float(missing_ratio)},
        "seed": seed, "model_uid": model_uid, "run_dir": str(run_dir),
        "mlflow": {
            "experiment_name": f"icassp-lightmodels_{dataset_name}_{missing_ratio}",
            "run_name": f"{model_name}_{model_uid}",
        },
    })

    with MLflowLogger(cfg) as logger:
        mlflow.set_tag("seed", seed)
        mlflow.set_tag("missing_ratio", missing_ratio)
        logger.log_metrics({"RMSE": rmse, "MAE": mae, "SMAPE": smape, "rMAE": rmae}, epoch=0, prefix="test")

        preds_path = run_dir / "predictions.npz"
        np.savez(preds_path, preds=preds, targets=targets)
        logger.log_checkpoint(str(preds_path))

        logger.tester_flag = True

    print(f"[{model_name}/{dataset_name}/{missing_ratio}/seed{seed}] MAE={mae:.4f} SMAPE={smape:.4f}")
    return mae


def run_one(model_name, dataset_name, missing_ratio, seed, device):
    model_uid = uuid.uuid4().hex[:8]
    run_dir = Path("runs") / f"{model_uid}_{model_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    params = load_best_params(model_name, dataset_name)

    if model_name in TORCH_MODEL_REGISTRY:
        return run_torch_model(model_name, dataset_name, missing_ratio, seed, params, device, model_uid, run_dir)
    else:
        return run_sklearn_model(model_name, dataset_name, missing_ratio, seed, params, model_uid, run_dir)


_WORKER_DEVICE = None


def _init_worker(counter, lock, tracking_uri):
    global _WORKER_DEVICE
    with lock:
        idx = counter.value
        counter.value += 1
    _WORKER_DEVICE = f"cuda:{idx % N_GPUS}" if N_GPUS > 0 else "cpu"
    load_dotenv("./.env", override=True)
    mlflow.set_tracking_uri(tracking_uri)
    print(f"[Worker PID {os.getpid()}] epingle a {_WORKER_DEVICE}")


def _worker_entry(task):
    model_name, dataset_name, missing_ratio, seed = task
    return run_one(model_name, dataset_name, missing_ratio, seed, _WORKER_DEVICE)


def main():
    load_dotenv("./.env", override=True)
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(tracking_uri)

    tasks = [
        (model_name, dataset_name, missing_ratio, seed)
        for model_name in MODELS
        for dataset_name in DATASETS
        for missing_ratio in MISSING_RATIOS
        for seed in SEEDS
    ]
    print(f"Total : {len(tasks)} runs (5 modeles x 5 datasets x 6 ratios x 5 seeds).")

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