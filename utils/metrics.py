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



def get_naive_reference(Y_target, lookback, horizon, mode="naive1"):
    """
    Renvoie le naive_ref (meme shape que target_future) selon le mode choisi.
    Lago et al. 2021, Section 5.4.2, Eq. 10.
    """
    if mode == "naive1":
        if lookback < horizon:
            raise ValueError(f"naive1 necessite lookback >= horizon (lookback={lookback}, horizon={horizon})")
        return Y_target[:, lookback - horizon : lookback, ...]

    elif mode == "naive2":
        required = 7 * horizon
        if lookback < required:
            raise ValueError(
                f"naive2 necessite lookback >= 7*horizon={required}h (lookback actuel={lookback}h)."
            )
        start = lookback - required
        return Y_target[:, start : start + horizon, ...]

    elif mode == "naive3":
        raise NotImplementedError(
            "naive3 necessite le jour de la semaine, non disponible dans le pipeline actuel."
        )

    else:
        raise ValueError(f"mode naive inconnu : '{mode}'")