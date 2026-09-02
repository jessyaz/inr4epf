import torch
import torch.nn as nn


LAG_BY_MODE = {
    "last_day": 24,
    "last_week": 168,
}


class Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg.model
        self.cfg_data = cfg.dataset
        self.name = "naive_model"

        mode = getattr(self.cfg, "mode", "last_day")
        if mode not in LAG_BY_MODE:
            raise ValueError(f"mode inconnu : '{mode}'. Disponibles : {list(LAG_BY_MODE)}")
        self.mode = mode
        self.lag = LAG_BY_MODE[mode]

        lookback = self.cfg.lookback
        horizon = self.cfg.horizon
        if lookback != horizon:
            raise ValueError(
                f"naive_model nécessite lookback == horizon (reçu lookback={lookback}, horizon={horizon})"
            )
        if lookback != self.lag:
            raise ValueError(
                f"mode='{mode}' nécessite lookback == horizon == {self.lag} "
                f"(reçu lookback={lookback}). Adapte la config."
            )

        # paramètre factice pour que .to(device)/.parameters() restent cohérents
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward_step(self, batch, device, debug=False):
        X_exog, Y_target = batch
        Y_target = Y_target.to(device)

        lookback = self.cfg.lookback
        y_past = Y_target[:, :lookback].unsqueeze(-1)

        if debug:
            print(f"[naive:{self.mode}, lag={self.lag}] y_past {tuple(y_past.shape)}")

        pred_past = y_past          # naïf : le passé "prédit" est le passé lui-même
        pred_future = y_past        # naïf saisonnier : future[t] = past[t] (t-lag)

        return pred_future, pred_past, Y_target