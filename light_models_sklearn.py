"""
LEAR et XGBoost -- pas de boucle d'entrainement par epoch (fit/predict direct),
donc pas de reutilisation de utils.trainer.train ici.

LEAR utilise ICI l'implementation OFFICIELLE d'epftoolbox (Lago et al.),
via un wrapper .fit()/.predict() qui delegue a .recalibrate()/.predict().
Seule difference avec l'algorithme original du papier : les features
utilisees sont NOTRE representation tabulaire (lookback aplati + imputation),
pas exactement le feature engineering d'origine (lags calendaires precis
jour-1/jour-2/jour-3/jour-7 + dummies de jour de semaine) -- necessaire pour
rester coherent avec notre propre gestion du missing_ratio/imputation.
La regularisation Lasso elle-meme (selection automatique du niveau de
sparsite) reste bien celle d'epftoolbox, pas reimplementee.

Prerequis : pip install epftoolbox xgboost scikit-learn
"""

import numpy as np
import torch
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
from epftoolbox.models import LEAR

from utils.interpolate import linear_interpolate_masked


def get_imputed_past(batch, device, lookback):
    y_target = batch["y_target"].to(device)
    mask = batch["mask"].to(device)

    mask_past = mask[:, :lookback].unsqueeze(-1).float()
    y_past = y_target[:, :lookback].unsqueeze(-1)
    return linear_interpolate_masked(y_past, mask_past)  # (batch, lookback)


def extract_tabular_torch(batch, device, lookback, horizon):
    X_exog = batch["X_exog"].to(device)
    y_target_no_mask = batch["y_target_no_mask"].to(device)

    y_past_interp = get_imputed_past(batch, device, lookback)
    exog_past = X_exog[:, :lookback]
    exog_future = X_exog[:, lookback:]
    target = y_target_no_mask[:, lookback:]

    batch_size = y_past_interp.shape[0]
    X = torch.cat([
        y_past_interp,
        exog_past.reshape(batch_size, -1),
        exog_future.reshape(batch_size, -1),
    ], dim=-1)

    return X, target


class EpftoolboxLEARWrapper:
    """Adapte epftoolbox.models.LEAR a une interface fit(X,y)/predict(X)
    compatible avec le reste du pipeline (meme usage que XGBoost)."""

    def __init__(self, calibration_window):
        self.model = LEAR(calibration_window=calibration_window)

    def fit(self, X, y):
        # recalibrate() ajuste les coefficients (regularisation choisie en interne)
        self.model.recalibrate(Xtrain=X, Ytrain=y)
        return self

    def predict(self, X):
        return self.model.predict(X)


def collect_tabular_dataset(loader, device, lookback, horizon):
    """Convertit un DataLoader entier en (X, y) numpy, pour sklearn/xgboost/LEAR."""
    all_X, all_y = [], []
    for batch in loader:
        X, y = extract_tabular_torch(batch, device, lookback, horizon)
        all_X.append(X.cpu().numpy())
        all_y.append(y.cpu().numpy())
    return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)


def build_lear(calibration_window):
    return EpftoolboxLEARWrapper(calibration_window=int(calibration_window))


import os as _os

def build_xgboost(n_estimators, max_depth, learning_rate):
    # n_jobs limite pour eviter la contention CPU quand plusieurs XGBoost
    # tournent en parallele (1 GPU x 4 workers = 4 processus concurrents)
    n_cpus = _os.cpu_count() or 4
    n_jobs = max(1, n_cpus // 4)

    base = xgb.XGBRegressor(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        tree_method="hist", n_jobs=n_jobs,
    )
    return MultiOutputRegressor(base, n_jobs=1)


def fit_and_predict(model, train_loader, val_loader, test_loader, device, lookback, horizon):
    """Entraine sur train (+val concatenes), predit sur test. Retourne (preds, targets) numpy."""
    X_train, y_train = collect_tabular_dataset(train_loader, device, lookback, horizon)
    X_val, y_val = collect_tabular_dataset(val_loader, device, lookback, horizon)
    X_train_full = np.concatenate([X_train, X_val], axis=0)
    y_train_full = np.concatenate([y_train, y_val], axis=0)

    model.fit(X_train_full, y_train_full)

    X_test, y_test = collect_tabular_dataset(test_loader, device, lookback, horizon)
    preds = model.predict(X_test)

    return preds, y_test


def fit_and_predict_with_val_mse(model, train_loader, val_loader, device, lookback, horizon):
    """Pour Optuna : entraine sur train seul, retourne le MSE de validation."""
    X_train, y_train = collect_tabular_dataset(train_loader, device, lookback, horizon)
    X_val, y_val = collect_tabular_dataset(val_loader, device, lookback, horizon)

    model.fit(X_train, y_train)
    preds_val = model.predict(X_val)
    val_mse = float(np.mean((preds_val - y_val) ** 2))
    return val_mse


SKLEARN_MODEL_BUILDERS = {
    "lear": build_lear,
    "xgboost": build_xgboost,
}