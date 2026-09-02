import sys
from pathlib import Path

import mlflow
import torch
from omegaconf import OmegaConf

from dotenv import load_dotenv
load_dotenv("./.env", override=True)

from utils.mlflow_logger import MLflowLogger

from runner import instantiate_model
from datasets.loader import load_market_dataloader
from utils.tester import test as tester


class RetrieverMLflowLogger(MLflowLogger):
    """Reprend un run MLflow existant au lieu d'en créer un nouveau."""

    def __init__(self, cfg, run_id):
        super().__init__(cfg)
        self.run_id = run_id

    def __enter__(self):
        mlflow.end_run()
        mlflow.start_run(run_id=self.run_id)
        return self


def find_run_dir(model_uid):
    matches = list(Path("runs").glob(f"{model_uid}_*"))
    if not matches:
        raise ValueError(f"Aucun dossier trouvé pour model_uid '{model_uid}' dans runs/")
    return matches[0]


def retrieve(model_uid, experiment_name):
    run_dir = find_run_dir(model_uid)
    registry = run_dir.name.split("_", 1)[1]
    run_name = f"{registry}_{model_uid}"

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' introuvable")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        output_format="list",
    )
    if not runs:
        raise ValueError(f"Aucun run MLflow trouvé pour run_name '{run_name}'")
    run_id = runs[0].info.run_id

    print(f"[retriever] model_uid={model_uid} run_id={run_id} run_dir={run_dir}")

    client = mlflow.tracking.MlflowClient()
    config_path = client.download_artifacts(run_id, "config.yaml")
    cfg = OmegaConf.load(config_path)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device

    _, test_loader, scaler = load_market_dataloader(
        cfg.dataset.name,
        batch_size=cfg.dataset.batch_size,
        missing_rate=cfg.dataset.missing_rate,
        seed=cfg.seed,
    )

    model = instantiate_model(cfg).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pth", map_location=device))
    model.eval()

    with RetrieverMLflowLogger(cfg, run_id) as logger:
        print("test en cours...")
        loss_dict_test = tester(model, test_loader, scaler, device, logger)
        print(f"Finish : {loss_dict_test}")
        logger.tester_flag = True

    print(f"[retriever] run {run_id} -> {'FINISHED' if logger.tester_flag else 'FAILED'}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: uv run python -m utils.retriever <model_uid> <experiment_name>")
        sys.exit(1)
    retrieve(sys.argv[1], sys.argv[2])