"""
Calcule les baselines LEAR et DNN directement via epftoolbox, en utilisant ses
fonctions haut niveau officielles (recalibration quotidienne native, cf. Lago
et al. 2021, Section 5.6).

IMPORTANT : ce script est INDEPENDANT du pipeline runner.py/trainer.py/valider.py/
tester.py. LEAR et DNN ne suivent pas le contrat forward_step(batch, device) --
ils operent directement sur des DataFrames pandas avec leur propre boucle de
recalibration interne, pas sur des batchs/epochs PyTorch. Les faire rentrer de
force dans MODEL_REGISTRY casserait cette logique native ; on les calcule donc
a part, avec ce script dedie.

Usage:
    # LEAR seul, un marche
    uv run models/epftoolbox_baselines.py --model lear --market NP

    # DNN : necessite d'abord une recherche d'hyperparametres (une fois par marche)
    uv run models/epftoolbox_baselines.py --model dnn_hyperopt --market NP --max-evals 1500
    uv run models/epftoolbox_baselines.py --model dnn --market NP

    # Tous les marches d'un coup
    uv run models/epftoolbox_baselines.py --model lear --market ALL
"""

import argparse
import os

import numpy as np
import pandas as pd

from epftoolbox.data import read_data
from epftoolbox.evaluation import MAE, RMSE, sMAPE, rMAE
from epftoolbox.models import evaluate_lear_in_test_dataset
from epftoolbox.models._dnn import evaluate_dnn_in_test_dataset
from epftoolbox.models._dnn_hyperopt import hyperparameter_optimizer

MARKETS = ("NP", "PJM", "BE", "FR", "DE")

DATASETS_DIR = "./datasets/raw"
LEAR_RESULTS_DIR = "./results/lear"
DNN_HYPERPARAMS_DIR = "./results/dnn_hyperparameters"
DNN_RESULTS_DIR = "./results/dnn"


def compute_metrics(real_values: pd.DataFrame, forecast: pd.DataFrame) -> dict:
    """Metriques standard du papier Lago et al. 2021 (Section 5.4), rMAE avec m='W' (naive2)."""
    return {
        "MAE": float(MAE(real_values.values, forecast.values)),
        "RMSE": float(RMSE(real_values.values, forecast.values)),
        "SMAPE": float(sMAPE(real_values.values, forecast.values) * 100),
        "rMAE": float(rMAE(real_values.values, forecast.values, m="W", freq="1h")),
    }


def run_lear(market: str, years_test: int = 2, calibration_window: int = 364 * 3) -> dict:
    os.makedirs(LEAR_RESULTS_DIR, exist_ok=True)

    forecast = evaluate_lear_in_test_dataset(
        path_datasets_folder=DATASETS_DIR,
        path_recalibration_folder=LEAR_RESULTS_DIR,
        dataset=market,
        years_test=years_test,
        calibration_window=calibration_window,
    )

    _, df_test = read_data(path=DATASETS_DIR, dataset=market, years_test=years_test)
    real_values = pd.DataFrame(
        df_test["Price"].values.reshape(-1, 24),
        index=forecast.index,
        columns=forecast.columns,
    )

    metrics = compute_metrics(real_values, forecast)
    print(f"[LEAR][{market}] {metrics}")
    return metrics


def run_dnn_hyperopt(
        market: str,
        years_test: int = 2,
        calibration_window: int = 4,
        max_evals: int = 1500,
        nlayers: int = 2,
) -> None:
    os.makedirs(DNN_HYPERPARAMS_DIR, exist_ok=True)

    experiment_id = f"{market}_YT{years_test}_CW{calibration_window}"
    print(f"[DNN][{market}] Recherche hyperparametres (experiment_id={experiment_id}, max_evals={max_evals})...")

    hyperparameter_optimizer(
        path_datasets_folder=DATASETS_DIR,
        path_hyperparameters_folder=DNN_HYPERPARAMS_DIR,
        new_hyperopt=1,
        max_evals=max_evals,
        nlayers=nlayers,
        dataset=market,
        years_test=years_test,
        calibration_window=calibration_window,
        experiment_id=experiment_id,
    )
    print(f"[DNN][{market}] Hyperparametres sauvegardes dans {DNN_HYPERPARAMS_DIR}")


def run_dnn(
        market: str,
        years_test: int = 2,
        calibration_window: int = 4,
        nlayers: int = 2,
) -> dict:
    os.makedirs(DNN_RESULTS_DIR, exist_ok=True)

    experiment_id = f"{market}_YT{years_test}_CW{calibration_window}"

    forecast = evaluate_dnn_in_test_dataset(
        experiment_id=experiment_id,
        path_datasets_folder=DATASETS_DIR,
        path_hyperparameter_folder=DNN_HYPERPARAMS_DIR,
        path_recalibration_folder=DNN_RESULTS_DIR,
        nlayers=nlayers,
        dataset=market,
        years_test=years_test,
        calibration_window=calibration_window,
    )

    _, df_test = read_data(path=DATASETS_DIR, dataset=market, years_test=years_test)
    real_values = pd.DataFrame(
        df_test["Price"].values.reshape(-1, 24),
        index=forecast.index,
        columns=forecast.columns,
    )

    metrics = compute_metrics(real_values, forecast)
    print(f"[DNN][{market}] {metrics}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lear", "dnn", "dnn_hyperopt"], required=True)
    parser.add_argument("--market", default="NP", help="NP|PJM|BE|FR|DE ou ALL pour les 5")
    parser.add_argument("--years-test", type=int, default=2)
    parser.add_argument("--calibration-window", type=int, default=None)
    parser.add_argument("--max-evals", type=int, default=1500, help="uniquement pour dnn_hyperopt")
    parser.add_argument("--nlayers", type=int, default=2, help="uniquement pour dnn/dnn_hyperopt")
    args = parser.parse_args()

    markets = list(MARKETS) if args.market == "ALL" else [args.market]

    results = {}
    for market in markets:
        if args.model == "lear":
            cw = args.calibration_window or (364 * 3)
            results[market] = run_lear(market, years_test=args.years_test, calibration_window=cw)
        elif args.model == "dnn_hyperopt":
            cw = args.calibration_window or 4
            run_dnn_hyperopt(
                market, years_test=args.years_test, calibration_window=cw,
                max_evals=args.max_evals, nlayers=args.nlayers,
            )
        elif args.model == "dnn":
            cw = args.calibration_window or 4
            results[market] = run_dnn(
                market, years_test=args.years_test, calibration_window=cw, nlayers=args.nlayers,
            )

    if results:
        print("\n--- Resume ---")
        for market, metrics in results.items():
            print(f"{market}: {metrics}")


if __name__ == "__main__":
    main()