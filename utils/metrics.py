import numpy as np

def compute_metrics(p, t, naive_ref):
    rmse = np.sqrt(np.mean((p - t) ** 2))
    mae = np.mean(np.abs(p - t))
    mape = np.mean(np.abs((t - p) / (np.abs(t) + 1e-8))) * 100
    smape = (
            np.mean(2.0 * np.abs(p - t) / (np.abs(p) + np.abs(t) + 1e-8)) * 100
    )
    naive_mae = np.mean(np.abs(t - naive_ref))
    rmae = mae / (naive_mae + 1e-8)
    return rmse, mae, mape, smape, rmae