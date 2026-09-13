"""
Reproduit le split EXACT de Dataset_Custom (TimeXer officiel) : 70/10/20%
generique, au lieu du split standard Lago et al. -- UNIQUEMENT PJM, pour
valider l'hypothese avant de generaliser.

Sauvegarde dans un dossier COMPLETEMENT SEPARE (./test_timexersplit/processed/)
pour ne rien ecraser des donnees existantes.
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from epftoolbox.data import read_data, scaling

# Chemin absolu vers la racine du projet (parent de test_timexersplit/),
# pour ne pas dependre du dossier depuis lequel le script est lance.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = str(PROJECT_ROOT / "datasets")  # <-- ajuste "datasets" si le vrai sous-dossier differe
OUTPUT_DIR = str(Path(__file__).resolve().parent / "processed")  # meme dossier que le script, pas de prefixe en trop
MARKET = "PJM"
WINDOW_SIZE = 192                     # lookback(168) + horizon(24)
NORMALIZE_METHOD = "Invariant"        # valeurs valides : 'Norm', 'Norm1', 'Std', 'Median', 'Invariant'


def create_sliding_windows_multivar(df, window_size, stride):
    dates = df['Date'].values if 'Date' in df.columns else df.index.values
    prices = df['Price'].values
    exog = df[['Grid load forecast', 'Wind power forecast']].values

    X_exog_windows, y_price_windows, date_windows = [], [], []
    for i in range(0, len(df) - window_size + 1, stride):
        X_exog_windows.append(exog[i:(i + window_size)])
        y_price_windows.append(prices[i:(i + window_size)])
        date_windows.append(dates[i:(i + window_size)])

    return np.array(X_exog_windows), np.array(y_price_windows), np.array(date_windows)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_train_full, df_test_full = read_data(path=DATA_DIR, dataset=MARKET)

    cols = ['Price', 'Grid load forecast', 'Wind power forecast']
    df_train_full.columns = cols
    df_test_full.columns = cols

    # Recombine en une seule serie complete, puis re-decoupe avec le split
    # generique 70/10/20 de Dataset_Custom (TimeXer officiel).
    df_full = pd.concat([df_train_full, df_test_full], axis=0)
    n = len(df_full)

    seq_len = 168
    num_train = int(n * 0.7)
    num_test = int(n * 0.2)
    num_vali = n - num_train - num_test

    border1s = [0, num_train - seq_len, n - num_test - seq_len]
    border2s = [num_train, num_train + num_vali, n]

    df_train_raw = df_full.iloc[border1s[0]:border2s[0]]
    df_val_raw = df_full.iloc[border1s[1]:border2s[1]]
    df_test_raw = df_full.iloc[border1s[2]:border2s[2]]

    print(f"{MARKET} : train={len(df_train_raw)}, val={len(df_val_raw)}, "
          f"test={len(df_test_raw)} (total={n})")

    scaled_datasets, scaler = scaling(
        [df_train_raw.values, df_val_raw.values, df_test_raw.values],
        normalize=NORMALIZE_METHOD
    )
    train_scaled_arr, val_scaled_arr, test_scaled_arr = scaled_datasets

    df_train_scaled = pd.DataFrame(train_scaled_arr, columns=cols)
    df_val_scaled = pd.DataFrame(val_scaled_arr, columns=cols)
    df_test_scaled = pd.DataFrame(test_scaled_arr, columns=cols)

    df_train_scaled['Date'] = df_train_raw.index
    df_val_scaled['Date'] = df_val_raw.index
    df_test_scaled['Date'] = df_test_raw.index

    X_exog_train, y_price_train, dates_train = create_sliding_windows_multivar(df_train_scaled, WINDOW_SIZE, stride=1)
    X_exog_val, y_price_val, dates_val = create_sliding_windows_multivar(df_val_scaled, WINDOW_SIZE, stride=24)
    X_exog_test, y_price_test, dates_test = create_sliding_windows_multivar(df_test_scaled, WINDOW_SIZE, stride=24)

    market_data = {
        'X_exogenous_train': X_exog_train, 'Y_target_train': y_price_train, 'dates_train': dates_train,
        'X_exogenous_val': X_exog_val, 'Y_target_val': y_price_val, 'dates_val': dates_val,
        'X_exogenous_test': X_exog_test, 'Y_target_test': y_price_test, 'dates_test': dates_test,
        'scaler': scaler,
    }

    file_path = os.path.join(OUTPUT_DIR, f"{MARKET}_data.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(market_data, f)
    print(f"-> sauvegarde : {file_path}")


if __name__ == "__main__":
    main()