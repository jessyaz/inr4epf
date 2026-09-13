"""
Recherche Optuna pour 5 modeles (LEAR, XGBoost, DNN, DLinear, CNN-LSTM) x
5 datasets, a missing_ratio=0.0 -- 25 studies independantes.

Parallelisation : toujours 2 (modele, dataset) en cours d'entrainement en
meme temps, un par GPU.
"""

import os
import time
import multiprocessing as mp

import optuna
import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import OmegaConf
import torch
from dotenv import load_dotenv

from datasets.loader import load_market_dataloader
from utils.trainer import train as trainer
from utils.valider import validate
from light_models_torch import TORCH_MODEL_REGISTRY
from light_models_sklearn import SKLEARN_MODEL_BUILDERS, fit_and_predict_with_val_mse

MODELS = ["lear", "xgboost", "dnn", "dlinear", "cnnlstm"]
DATASETS = ["FR", "NP", "PJM", "BE", "DE"]
N_TRIALS = 30
SEED = 42
N_GPUS = max(1, torch.cuda.device_count())
WORKERS_PER_GPU = 4  # modeles petits (DNN/DLinear/CNN-LSTM), 50 Go de VRAM disponible

LOOKBACK, HORIZON = 168, 24
BATCH_SIZE = 64

MAX_RETRIES = 5
INITIAL_DELAY = 2.0
BACKOFF_FACTOR = 2.0


def robust_call(fn, *args, **kwargs):
    delay = INITIAL_DELAY
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            print(f"[RETRY] tentative {attempt}/{MAX_RETRIES} echouee pour {getattr(fn, '__name__', fn)} : {e}")
            if attempt < MAX_RETRIES:
                time.sleep(delay)
                delay *= BACKOFF_FACTOR
    raise last_exc


def get_or_restore_experiment(name):
    client = MlflowClient()
    exp = robust_call(client.get_experiment_by_name, name)
    if exp is not None and exp.lifecycle_stage == "deleted":
        robust_call(client.restore_experiment, exp.experiment_id)
        exp = robust_call(client.get_experiment_by_name, name)

    if exp is None:
        experiment_id = robust_call(client.create_experiment, name)
    else:
        experiment_id = exp.experiment_id

    return experiment_id


# ==========================================================================
# ESPACES DE RECHERCHE PAR MODELE
# ==========================================================================
def suggest_torch_hyperparams(model_name, trial):
    if model_name == "dnn":
        return {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
            "n_layers": trial.suggest_int("n_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        }
    if model_name == "dlinear":
        return {
            "moving_avg": trial.suggest_categorical("moving_avg", [13, 25, 49]),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        }
    if model_name == "cnnlstm":
        return {
            "cnn_channels": trial.suggest_categorical("cnn_channels", [16, 32, 64]),
            "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7]),
            "lstm_hidden": trial.suggest_categorical("lstm_hidden", [32, 64, 128]),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        }
    raise ValueError(model_name)


def suggest_sklearn_hyperparams(model_name, trial):
    if model_name == "lear":
        # calibration_window en JOURS -- adapte selon la taille reelle de ton train set
        # (valeurs usuelles dans la litterature EPF : ~0.5 a 3 ans d'historique)
        return {"calibration_window": trial.suggest_categorical("calibration_window", [182, 364, 728])}
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        }
    raise ValueError(model_name)
# ==========================================================================


def objective_torch(trial, model_name, dataset_name, device):
    params = suggest_torch_hyperparams(model_name, trial)

    cfg = OmegaConf.create({
        "model": {
            "lookback": LOOKBACK, "horizon": HORIZON, "exog_dim": 2,
            "num_epoch": 30, **params,
        },
        "dataset": {"name": dataset_name, "batch_size": BATCH_SIZE, "missing_rate": 0.0},
    })

    torch.manual_seed(SEED)
    train_loader, val_loader, _, _ = load_market_dataloader(
        dataset_name, batch_size=BATCH_SIZE, missing_rate=0.0, seed=SEED
    )

    model = TORCH_MODEL_REGISTRY[model_name](cfg).to(device)
    optimizer = model.configure_optimizer()

    class _DummyLogger:
        def log_metrics(self, *a, **k): pass
        def log_plot(self, *a, **k): pass

    loaders = {"train_loader": train_loader, "val_loader": val_loader}
    result = trainer(model, loaders, optimizer, device, _DummyLogger())
    return result["val_loss"]["MSE"]


def objective_sklearn(trial, model_name, dataset_name):
    params = suggest_sklearn_hyperparams(model_name, trial)
    builder = SKLEARN_MODEL_BUILDERS[model_name]
    model = builder(**params)

    train_loader, val_loader, _, _ = load_market_dataloader(
        dataset_name, batch_size=BATCH_SIZE, missing_rate=0.0, seed=SEED
    )
    val_mse = fit_and_predict_with_val_mse(model, train_loader, val_loader, "cpu", LOOKBACK, HORIZON)
    return val_mse


def run_search(model_name, dataset_name, device, tracking_uri):
    load_dotenv("./.env", override=True)
    mlflow.set_tracking_uri(tracking_uri)
    exp_name = f"optim_light_{model_name}_{dataset_name}"
    experiment_id = get_or_restore_experiment(exp_name)

    print(f"[PID {os.getpid()}] Recherche {model_name}/{dataset_name} sur {device} (experiment_id={experiment_id})")

    is_torch = model_name in TORCH_MODEL_REGISTRY

    # LEAR : espace de recherche minuscule (3 valeurs distinctes, deterministe)
    # -> grid search direct, pas de TPE avec 30 trials redondants
    if model_name == "lear":
        sampler = optuna.samplers.GridSampler({"calibration_window": [182, 364, 728]})
        n_trials = 3
    else:
        sampler = None
        n_trials = N_TRIALS

    study = optuna.create_study(study_name=f"{model_name}_{dataset_name}", direction="minimize", sampler=sampler)

    with mlflow.start_run(experiment_id=experiment_id, run_name=f"optuna_{model_name}_{dataset_name}") as run:
        print(f"[PID {os.getpid()}] Run MLflow demarre : run_id={run.info.run_id}, experiment_id={run.info.experiment_id}")
        robust_call(mlflow.set_tags, {"model": model_name, "dataset": dataset_name, "role": "optuna_parent"})

        if is_torch:
            study.optimize(lambda trial: objective_torch(trial, model_name, dataset_name, device), n_trials=n_trials)
        else:
            study.optimize(lambda trial: objective_sklearn(trial, model_name, dataset_name), n_trials=n_trials)

        robust_call(mlflow.log_params, study.best_params)
        robust_call(mlflow.log_metric, "best_val_MSE", study.best_value)

    print(f"[{model_name}/{dataset_name}] best_params={study.best_params}, best_value={study.best_value:.4f}")
    study.trials_dataframe().to_csv(f"optuna_light_{model_name}_{dataset_name}.csv")
    return model_name, dataset_name, study.best_params


_WORKER_DEVICE = None


def _init_worker(counter, lock):
    global _WORKER_DEVICE
    with lock:
        idx = counter.value
        counter.value += 1
    _WORKER_DEVICE = f"cuda:{idx % N_GPUS}" if N_GPUS > 0 else "cpu"
    # Etale legerement le demarrage pour eviter que tous les workers tapent
    # MLflow (SQLite backend, ecriture concurrente limitee) au meme instant
    time.sleep(idx * 1.5)
    print(f"[Worker PID {os.getpid()}] epingle a {_WORKER_DEVICE}")


def _worker_entry(args):
    model_name, dataset_name, tracking_uri = args
    return run_search(model_name, dataset_name, _WORKER_DEVICE, tracking_uri)


def main():
    load_dotenv("./.env", override=True)
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]

    tasks = [(m, d, tracking_uri) for m in MODELS for d in DATASETS]
    print(f"Total : {len(tasks)} recherches (5 modeles x 5 datasets).")

    n_workers = N_GPUS * WORKERS_PER_GPU

    manager = mp.Manager()
    counter = manager.Value("i", 0)
    lock = manager.Lock()

    with mp.Pool(processes=n_workers, initializer=_init_worker, initargs=(counter, lock)) as pool:
        results = pool.map(_worker_entry, tasks)

    print("\nTermine.")
    for model_name, dataset_name, best_params in results:
        print(f"{model_name}/{dataset_name} : {best_params}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()