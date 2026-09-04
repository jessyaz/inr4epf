"""
Liste tous les runs d'une experiment MLflow, avec leurs metriques (val/test)
et la config dataset utilisee -- utile pour comparer plusieurs modeles/datasets
sans avoir a ouvrir l'UI manuellement.

Usage:
    uv run utils/print_mlflow_results.py --experiment icassp-dev
"""

import argparse
import json
import tempfile
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from mlflow.entities import ViewType

from dotenv import load_dotenv

load_dotenv("./.env", override=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="icassp-dev")
    args = parser.parse_args()

    client = MlflowClient()
    experiment = client.get_experiment_by_name(args.experiment)
    if experiment is None:
        print(f"Experiment '{args.experiment}' introuvable.")
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attribute.start_time DESC"],
        run_view_type=ViewType.ACTIVE_ONLY,
    )

    print(f"{len(runs)} runs trouves dans '{args.experiment}'\n")

    rows = []
    for run in runs:
        run_id = run.info.run_id
        status = run.info.status
        metrics = run.data.metrics  # dernier point logge pour chaque cle (mlflow ne garde que le dernier via search_runs)

        # recupere la config complete (artifact config.yaml) pour lire dataset.name / registry / model.*
        dataset_name = "?"
        registry = "?"
        lookback = "?"
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                local_path = client.download_artifacts(run_id, "config.yaml", tmp_dir)
                with open(local_path) as f:
                    cfg = json.load(f) if local_path.endswith(".json") else None
                if cfg is None:
                    import yaml
                    with open(local_path) as f:
                        cfg = yaml.safe_load(f)
                dataset_name = cfg.get("dataset", {}).get("name", "?")
                registry = cfg.get("registry", "?")
                lookback = cfg.get("model", {}).get("lookback", "?")
        except Exception:
            pass

        rows.append({
            "run_id": run_id[:8],
            "status": status,
            "registry": registry,
            "dataset": dataset_name,
            "lookback": lookback,
            "val/MAE": metrics.get("val/MAE"),
            "val/rMAE": metrics.get("val/rMAE"),
            "test/MAE": metrics.get("test/MAE"),
            "test/RMSE": metrics.get("test/RMSE"),
            "test/SMAPE": metrics.get("test/SMAPE"),
            "test/rMAE": metrics.get("test/rMAE"),
        })

    # affichage groupe par (registry, dataset) pour reperer facilement les changements de config
    rows.sort(key=lambda r: (r["registry"], r["dataset"]))

    header = f"{'run_id':>10} {'status':>10} {'registry':>18} {'dataset':>8} {'lookback':>9} {'val/MAE':>9} {'val/rMAE':>9} {'test/MAE':>9} {'test/RMSE':>10} {'test/SMAPE':>11} {'test/rMAE':>10}"
    print(header)
    print("-" * len(header))
    last_key = None
    for r in rows:
        key = (r["registry"], r["dataset"])
        if key != last_key:
            print()  # ligne vide a chaque changement de (registry, dataset)
            last_key = key
        print(
            f"{r['run_id']:>10} {r['status']:>10} {r['registry']:>18} {r['dataset']:>8} {r['lookback']:>9} "
            f"{fmt(r['val/MAE'])} {fmt(r['val/rMAE'])} {fmt(r['test/MAE'])} {fmt(r['test/RMSE'])} "
            f"{fmt(r['test/SMAPE'])} {fmt(r['test/rMAE'])}"
        )


def fmt(x):
    if x is None:
        return f"{'--':>9}"
    return f"{x:>9.4f}"


if __name__ == "__main__":
    main()