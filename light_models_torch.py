"""
DNN, DLinear, CNN-LSTM -- implementes avec la MEME interface que le modele
INR (forward_step, configure_optimizer) pour reutiliser tel quel
utils.trainer.train / utils.tester.test / MLflowLogger, sans dupliquer
la boucle d'entrainement.
"""

import torch
import torch.nn as nn

from utils.interpolate import linear_interpolate_masked


def get_imputed_past(batch, device, lookback):
    y_target = batch["y_target"].to(device)
    mask = batch["mask"].to(device)

    mask_past = mask[:, :lookback].unsqueeze(-1).float()
    y_past = y_target[:, :lookback].unsqueeze(-1)
    return linear_interpolate_masked(y_past, mask_past)  # (batch, lookback)


def extract_tabular_torch(batch, device, lookback, horizon):
    """Pour DNN : tout aplati en un vecteur par echantillon."""
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


def extract_sequence_torch(batch, device, lookback, horizon):
    """Pour DLinear / CNN-LSTM : garde la structure sequentielle."""
    X_exog = batch["X_exog"].to(device)
    y_target_no_mask = batch["y_target_no_mask"].to(device)

    y_past_interp = get_imputed_past(batch, device, lookback)
    exog_past = X_exog[:, :lookback]
    exog_future = X_exog[:, lookback:]
    target = y_target_no_mask[:, lookback:]

    return y_past_interp, exog_past, exog_future, target


class DNNModel(nn.Module):
    def __init__(self, cfg, logger=None):
        super().__init__()
        self.cfg = cfg.model
        self.name = "dnn"

        lookback, horizon, exog_dim = self.cfg.lookback, self.cfg.horizon, self.cfg.exog_dim
        input_dim = lookback + exog_dim * (lookback + horizon)

        layers = []
        d = input_dim
        for _ in range(self.cfg.n_layers):
            layers += [nn.Linear(d, self.cfg.hidden_dim), nn.ReLU(), nn.Dropout(self.cfg.dropout)]
            d = self.cfg.hidden_dim
        layers += [nn.Linear(d, horizon)]
        self.net = nn.Sequential(*layers)

    def set_epoch(self, epoch):
        pass

    def forward_step(self, batch, device, debug=False):
        X, _ = extract_tabular_torch(batch, device, self.cfg.lookback, self.cfg.horizon)
        pred = self.net(X)
        return pred.unsqueeze(-1)

    def configure_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.lr)


class DLinearModel(nn.Module):
    """Decomposition trend/seasonal (moving average) + Linear, + terme exogene lineaire."""

    def __init__(self, cfg, logger=None):
        super().__init__()
        self.cfg = cfg.model
        self.name = "dlinear"

        self.lookback, self.horizon = self.cfg.lookback, self.cfg.horizon
        self.kernel = self.cfg.moving_avg
        exog_dim = self.cfg.exog_dim

        self.linear_trend = nn.Linear(self.lookback, self.horizon)
        self.linear_seasonal = nn.Linear(self.lookback, self.horizon)

        exog_input_dim = exog_dim * (self.lookback + self.horizon)
        self.exog_proj = nn.Linear(exog_input_dim, self.horizon)

    def set_epoch(self, epoch):
        pass

    def _decompose(self, x):
        pad = self.kernel // 2
        x_padded = torch.nn.functional.pad(x.unsqueeze(1), (pad, pad), mode="replicate")
        trend = torch.nn.functional.avg_pool1d(x_padded, kernel_size=self.kernel, stride=1).squeeze(1)
        trend = trend[:, : x.shape[1]]
        seasonal = x - trend
        return trend, seasonal

    def forward_step(self, batch, device, debug=False):
        y_past, exog_past, exog_future, _ = extract_sequence_torch(batch, device, self.lookback, self.horizon)
        trend, seasonal = self._decompose(y_past)

        batch_size = y_past.shape[0]
        exog_flat = torch.cat([exog_past.reshape(batch_size, -1), exog_future.reshape(batch_size, -1)], dim=-1)

        out = self.linear_trend(trend) + self.linear_seasonal(seasonal) + self.exog_proj(exog_flat)
        return out.unsqueeze(-1)

    def configure_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.lr)


class CNNLSTMModel(nn.Module):
    def __init__(self, cfg, logger=None):
        super().__init__()
        self.cfg = cfg.model
        self.name = "cnnlstm"

        lookback, horizon, exog_dim = self.cfg.lookback, self.cfg.horizon, self.cfg.exog_dim
        input_channels = 1 + exog_dim

        self.conv1 = nn.Conv1d(input_channels, self.cfg.cnn_channels,
                               kernel_size=self.cfg.kernel_size, padding=self.cfg.kernel_size // 2)
        self.lstm = nn.LSTM(self.cfg.cnn_channels, self.cfg.lstm_hidden, batch_first=True)

        exog_future_dim = exog_dim * horizon
        self.head = nn.Linear(self.cfg.lstm_hidden + exog_future_dim, horizon)

    def set_epoch(self, epoch):
        pass

    def forward_step(self, batch, device, debug=False):
        y_past, exog_past, exog_future, _ = extract_sequence_torch(batch, device, self.cfg.lookback, self.cfg.horizon)

        x = torch.cat([y_past.unsqueeze(-1), exog_past], dim=-1)  # (batch, lookback, 1+exog_dim)
        x = x.transpose(1, 2)                                      # (batch, channels, lookback)
        x = torch.relu(self.conv1(x))
        x = x.transpose(1, 2)                                      # (batch, lookback, cnn_channels)

        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]

        exog_future_flat = exog_future.reshape(exog_future.shape[0], -1)
        combined = torch.cat([h_last, exog_future_flat], dim=-1)
        out = self.head(combined)
        return out.unsqueeze(-1)

    def configure_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.lr)


TORCH_MODEL_REGISTRY = {
    "dnn": DNNModel,
    "dlinear": DLinearModel,
    "cnnlstm": CNNLSTMModel,
}