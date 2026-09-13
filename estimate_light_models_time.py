"""
Chronometre un seul run par modele (meme config que le benchmark complet),
pour extrapoler le temps total des 750 runs avant de tout lancer.
"""

import time
import torch

from run_light_models_benchmark import run_one, MODELS

DATASET = "FR"
MISSING_RATIO = "0.0"
SEED = 0


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    n_gpus_parallel = max(1, torch.cuda.device_count())

    total_estimate = 0.0
    for model_name in MODELS:
        start = time.time()
        try:
            run_one(model_name, DATASET, MISSING_RATIO, SEED, device)
        except Exception as e:
            print(f"[ERREUR] {model_name} : {e}")
            continue
        elapsed = time.time() - start

        n_runs_this_model = 150  # 5 datasets x 6 ratios x 5 seeds
        estimated_total = elapsed * n_runs_this_model
        total_estimate += estimated_total

        print(f"{model_name} : {elapsed:.1f}s/run -> estimation {n_runs_this_model} runs = "
              f"{estimated_total/60:.1f} min ({estimated_total/3600:.2f}h)")

    print(f"\nEstimation totale (sequentiel) : {total_estimate/3600:.2f}h")
    print(f"Estimation avec {n_gpus_parallel} en parallele : {total_estimate/3600/n_gpus_parallel:.2f}h")


if __name__ == "__main__":
    main()