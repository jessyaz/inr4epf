import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
from epftoolbox.data import read_data, scaling
from tqdm import tqdm


def create_sliding_windows_multivar(df: pd.DataFrame, window_size: int, stride : int):
    dates = df['Date'].values if 'Date' in df.columns else df.index.values

    prices = df['Price'].values
    exog = df[['Grid load forecast', 'Wind power forecast']].values

    X_exog_windows, y_price_windows, date_windows = [], [], []

    for i in range(0, len(df) - window_size + 1, stride):
        X_exog_windows.append(exog[i:(i + window_size)])
        y_price_windows.append(prices[i:(i + window_size)])
        date_windows.append(dates[i:(i + window_size)])

    return (
        np.array(X_exog_windows),
        np.array(y_price_windows),
        np.array(date_windows)
    )


def process_markets_raw_windows(data_dir: str, markets: list, normalize_method: str, window_size: int, processed_dir: str):
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    processed_data = {}

    print(f"--- Running data processing (Raw Windows) with '{normalize_method}' ---")

    for market in tqdm(markets, desc="Progression des marchés"):
        df_train, df_test = read_data(path=data_dir, dataset=market)

        cols = ['Price', 'Grid load forecast', 'Wind power forecast']
        df_train.columns = cols
        df_test.columns = cols

        # Lago et al. 2021 section 4.3.2 ("Training dataset")
        val_weeks = 42
        val_hours = val_weeks * 7 * 24

        df_train_raw = df_train.iloc[:-val_hours]
        df_val_raw = df_train.iloc[-val_hours:]
        df_test_raw = df_test

        scaled_datasets, scaler = scaling( # # Lago et al. 2021 Fonction (ok for data leakage)
            [df_train_raw.values, df_val_raw.values, df_test_raw.values],
            normalize=normalize_method
        )
        train_scaled_arr, val_scaled_arr, test_scaled_arr = scaled_datasets

        df_train_scaled = pd.DataFrame(train_scaled_arr, columns=cols)
        df_val_scaled = pd.DataFrame(val_scaled_arr, columns=cols)
        df_test_scaled = pd.DataFrame(test_scaled_arr, columns=cols)

        df_train_scaled['Date'] = df_train_raw.index
        df_val_scaled['Date'] = df_val_raw.index
        df_test_scaled['Date'] = df_test_raw.index

        X_exog_train, y_price_train, dates_train = create_sliding_windows_multivar(df_train_scaled, window_size, stride = 1) # Hardcoded from the paper Lago et al. 2021, Applied Energy — "Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark"
        X_exog_val, y_price_val, dates_val = create_sliding_windows_multivar(df_val_scaled, window_size, stride=24)
        X_exog_test, y_price_test, dates_test = create_sliding_windows_multivar(df_test_scaled, window_size, stride = 24)

        market_data = {
            'X_exogenous_train': X_exog_train,
            'Y_target_train': y_price_train,
            'dates_train': dates_train,

            'X_exogenous_val': X_exog_val,
            'Y_target_val': y_price_val,
            'dates_val': dates_val,

            'X_exogenous_test': X_exog_test,
            'Y_target_test': y_price_test,
            'dates_test': dates_test,

            'scaler': scaler
        }

        processed_data[market] = market_data

        file_path = os.path.join(processed_dir, f"{market}_data.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(market_data, f)

    print(f"\nSucceed! Processed data saved in '{processed_dir}/'")
    return processed_data


@hydra.main(version_base=None, config_path="./", config_name="datasets")
def main(cfg: DictConfig):
    processed_dir = "./datasets/processed"
    datasets = process_markets_raw_windows(
        data_dir=cfg.data.data_dir,
        markets=cfg.data.markets,
        normalize_method=cfg.data.normalize_method,
        window_size=cfg.data.window_duration_hours,
        processed_dir=processed_dir
    )

    print("\nForme de X_exogenous_train (FR) :", datasets['FR']['X_exogenous_train'].shape)
    print("Forme de Y_target_train (FR) :", datasets['FR']['Y_target_train'].shape)


if __name__ == '__main__':
    main()