import torch
import torch.nn as nn

from utils.interpolate import linear_interpolate_masked


class Model(nn.Module):
    """
    DLinear (Zeng et al., 2022) -- decomposition trend/seasonal (moving
    average) + une couche lineaire par composante, plus un terme lineaire
    pour les exogenes (passees + futures connues).

    Pas de dependance externe : implementation directe, pas de vendor a cloner.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.model
        self.name = "dlinear"

        self.lookback = self.cfg.lookback
        self.horizon = self.cfg.horizon
        self.kernel = self.cfg.moving_avg  # taille du noyau de moyenne mobile (ex: 25)
        exog_dim = self.cfg.exog_dim

        self.linear_trend = nn.Linear(self.lookback, self.horizon)
        self.linear_seasonal = nn.Linear(self.lookback, self.horizon)

        exog_input_dim = exog_dim * (self.lookback + self.horizon)
        self.exog_proj = nn.Linear(exog_input_dim, self.horizon)

    def set_epoch(self, epoch):
        pass

    def configure_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.lr)

    def _decompose(self, x):
        """x : [B, lookback] -> (trend, seasonal), meme shape."""
        pad = self.kernel // 2
        x_padded = torch.nn.functional.pad(x.unsqueeze(1), (pad, pad), mode="replicate")
        trend = torch.nn.functional.avg_pool1d(x_padded, kernel_size=self.kernel, stride=1).squeeze(1)
        trend = trend[:, : x.shape[1]]
        seasonal = x - trend
        return trend, seasonal

    def forward_step(self, batch, device, debug: bool = False):
        X_exog, mask, y_target = batch["X_exog"], batch["mask"], batch["y_target"]
        X_exog, mask, y_target = X_exog.to(device), mask.to(device), y_target.to(device)

        lookback, horizon = self.lookback, self.horizon

        mask_past = mask[:, :lookback].unsqueeze(-1).float()
        y_past = y_target[:, :lookback].unsqueeze(-1)
        y_past = linear_interpolate_masked(y_past, mask_past)  # [B, lookback], interpole

        exog_past = X_exog[:, :lookback]
        exog_future = X_exog[:, lookback:]

        trend, seasonal = self._decompose(y_past)

        batch_size = y_past.shape[0]
        exog_flat = torch.cat([exog_past.reshape(batch_size, -1), exog_future.reshape(batch_size, -1)], dim=-1)

        out = self.linear_trend(trend) + self.linear_seasonal(seasonal) + self.exog_proj(exog_flat)

        return out.unsqueeze(-1)  # [B, horizon, 1]