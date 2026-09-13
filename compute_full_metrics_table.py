"""
Agrege RMSE, MAE, SMAPE, rMAE (relatif au naive_k7 interpole) pour :
- INR, TimeXer, Naive-k7 : 5 datasets x 6 missing_ratio x 5 seeds
- Ablation (full vs noexog) : 5 datasets x missing_ratio=0.0 x 5 seeds

rMAE = MAE_model / MAE_naive_k7_interp (meme baseline pour tout le monde,
recalculee par (dataset, missing_ratio) -- pas par seed, pour eviter le
bruit d'une seule realisation naive comme diviseur).
"""

import os
import tempfile
import mlflow
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from dotenv import load_dotenv

from datasets.loader import load_market_dataloader
from utils.metrics import compute_metrics
from runner import instantiate_model
from naive_k7 import compute_naive_k7

load_dotenv("./.env", override=True)
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

DATASETS = ["FR", "NP", "PJM", "BE", "DE"]
MISSING_RATIOS = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5"]
SEEDS = [0, 1, 2, 3, 4]
METRICS = ["RMSE", "MAE", "SMAPE", "rMAE"]

RUN_PREFIXES = {"inr": "inrv2_", "timexer": "timexer_"}


def get_naive_ref(dataset_name, missing_ratio, device):
    """Naive k=7 interpole, moyenne sur les 5 seeds -- baseline canonique pour rMAE."""
    from run_naive_benchmark import evaluate_naive
    all_naive, all_target = [], []
    for seed in SEEDS:
        naive_interp, _, target = evaluate_naive(dataset_name, missing_ratio, seed, device)
        all_naive.append(naive_interp)
        all_target.append(target)
    return np.concatenate(all_naive, axis=0), np.concatenate(all_target, axis=0)


def get_finished_runs(experiment_name, run_prefix):
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return []
    runs_df = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"attributes.status = 'FINISHED' and tags.`mlflow.runName` LIKE '{run_prefix}%'",
    )
    return runs_df["run_id"].tolist() if len(runs_df) > 0 else []


def run_inference(run_id, device):
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="config.yaml", dst_path=tmpdir)
        ckpt_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model.pth", dst_path=tmpdir)
        cfg = OmegaConf.load(config_path)
        state_dict = torch.load(ckpt_path, map_location=device)

    _, _, test_loader, scaler = load_market_dataloader(
        cfg.dataset.name, batch_size=cfg.dataset.batch_size,
        missing_rate=cfg.dataset.missing_rate, seed=cfg.seed,
    )
    model = instantiate_model(cfg).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    lookback, horizon = model.cfg.lookback, model.cfg.horizon
    all_pred, all_target = [], []
    with torch.no_grad():
        for batch in test_loader:
            pred = model.forward_step(batch, device)
            target = batch["y_target_no_mask"][:, lookback:, ...].to(device).unsqueeze(-1)
            all_pred.append(pred.detach().cpu().numpy())
            all_target.append(target.detach().cpu().numpy())

    preds = np.concatenate(all_pred, axis=0)
    targets = np.concatenate(all_target, axis=0)
    if scaler is not None:
        shape = preds.shape
        preds = scaler.inverse_transform(preds.reshape(-1, shape[-1])).reshape(shape)
        targets = scaler.inverse_transform(targets.reshape(-1, shape[-1])).reshape(shape)

    return preds.squeeze(-1), targets.squeeze(-1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []

    for dataset_name in DATASETS:
        for missing_ratio in MISSING_RATIOS:
            naive_ref, _ = get_naive_ref(dataset_name, missing_ratio, device)

            exp_name = f"icassp-{dataset_name}-{missing_ratio}"
            for model_key, prefix in RUN_PREFIXES.items():
                run_ids = get_finished_runs(exp_name, prefix)
                for rid in run_ids:
                    preds, targets = run_inference(rid, device)
                    n = min(len(preds), len(naive_ref))
                    rmse, mae, mapf, smape, rmae = compute_metrics(preds[:n], targets[:n], naive_ref=naive_ref[:n])
                    rows.append({
                        "model": model_key, "dataset": dataset_name, "missing_ratio": missing_ratio,
                        "run_id": rid, "RMSE": rmse, "MAE": mae, "SMAPE": smape, "rMAE": rmae,
                    })
                    print(f"{model_key}/{dataset_name}/{missing_ratio} : MAE={mae:.4f}")

    # ---- Ablation (ratio=0.0 uniquement) ----
    for dataset_name in DATASETS:
        naive_ref, _ = get_naive_ref(dataset_name, "0.0", device)
        exp_name = f"ablation-{dataset_name}-0.0"
        for variant, prefix in [("full", "full_"), ("noexog", "noexog_")]:
            run_ids = get_finished_runs(exp_name, prefix)
            for rid in run_ids:
                preds, targets = run_inference(rid, device)
                n = min(len(preds), len(naive_ref))
                rmse, mae, mapf, smape, rmae = compute_metrics(preds[:n], targets[:n], naive_ref=naive_ref[:n])
                rows.append({
                    "model": f"ablation_{variant}", "dataset": dataset_name, "missing_ratio": "0.0",
                    "run_id": rid, "RMSE": rmse, "MAE": mae, "SMAPE": smape, "rMAE": rmae,
                })

    df = pd.DataFrame(rows)
    df.to_csv("full_metrics_raw.csv", index=False)

    summary = df.groupby(["model", "dataset", "missing_ratio"])[METRICS].agg(["mean", "std"])
    summary.to_csv("full_metrics_summary.csv")

    print("\nSauvegarde : full_metrics_raw.csv (par seed), full_metrics_summary.csv (mean/std)")


if __name__ == "__main__":
    main()