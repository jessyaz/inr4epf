import torch
import torch.nn as nn





class INR(nn.Module):
    def __init__(self, cfg, logger=None):
        super().__init__()
        cfg = cfg.inr
        self.num_layers = cfg.num_layers
        self.hidden_dim = cfg.hidden_dim

        self.fourier = PE(cfg.num_frequencies, logger)
        fourier_dim = 2 * cfg.num_frequencies

        dims = [fourier_dim] + [cfg.hidden_dim] * cfg.num_layers
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1], bias=False)
            for i in range(cfg.num_layers)
        ])
        self.output_layer = nn.Linear(cfg.hidden_dim, cfg.output_dim, bias=False)

    def set_epoch(self, epoch):
        self.fourier.set_epoch(epoch)

    def forward(self, t, film):
        gamma, beta = film  # chacun (batch, num_layers * hidden_dim)
        batch_size = gamma.shape[0]
        gamma = gamma.view(batch_size, self.num_layers, self.hidden_dim)
        beta = beta.view(batch_size, self.num_layers, self.hidden_dim)

        x = self.fourier(t)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = gamma[:, i, :] * x + beta[:, i, :]
            x = torch.nn.functional.gelu(x)
        return self.output_layer(x)


FREEZE_EPOCH = 50
PE_GRAD_CLIP_NORM = 0.01

class PE(nn.Module):
    def __init__(self, num_frequencies, logger=None):
        super().__init__()
        #self.freqs = nn.Parameter(2.0 ** torch.arange(num_frequencies, dtype=torch.float32))
        self.freqs = nn.Parameter(
            torch.logspace(0, torch.log2(torch.tensor(6.0)), num_frequencies, base=2.0)
        )
        #self.freqs = nn.Parameter(torch.randn(num_frequencies))
        self.freqs.register_hook(self._clip_grad)
        self._pe_frozen = False
        self.current_epoch = 0

        self.logger = logger

    def forward(self, t):
        if not self._pe_frozen and self.current_epoch >= FREEZE_EPOCH:
            self.freqs.requires_grad = False
            self._pe_frozen = True

        angles = t.unsqueeze(-1)
        return torch.cat([torch.sin(2*angles* self.freqs), torch.cos(2*angles* self.freqs)], dim=-1)

    def _clip_grad(self, grad):
        norm = grad.norm()

        if norm > PE_GRAD_CLIP_NORM:


            grad = grad * (PE_GRAD_CLIP_NORM / (norm + 1e-6))
        return grad

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if self.logger is not None:
            freqs_values = self.freqs.detach().cpu().numpy()
            freq_dict = {f"pe_freq_{i}": float(val) for i, val in enumerate(freqs_values)}
            self.logger.log_metrics(freq_dict, epoch, prefix="pe")


class FiLMGenerator(nn.Module):
    def __init__(self, z_dim, hidden_dim, feature_dims):
        super().__init__()
        self.feature_dims = feature_dims  # entier : taille totale (num_layers * hidden_dim)

        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * feature_dims),
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[:feature_dims] = 1.0  # gamma initial = 1, beta reste à 0

    def forward(self, z):
        out = self.net(z)  # (batch, 2 * feature_dims)
        gamma, beta = out.chunk(2, dim=-1)  # chacun (batch, feature_dims)
        return gamma, beta


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.cell = nn.LSTMCell(input_dim, hidden_dim)

    def forward(self, exog_past):
        _, (h, c) = self.lstm(exog_past)
        return h.squeeze(0), c.squeeze(0)

class DeepSetsEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, aggregation="mean"):
        super().__init__()
        self.aggregation = aggregation
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, elements, mask):
        embeddings = self.phi(elements)  # (batch, lookback, hidden_dim)

        mask_float = mask.unsqueeze(-1).to(embeddings.dtype)
        embeddings = embeddings * mask_float

        if self.aggregation == "mean":
            aggregated = embeddings.sum(dim=1) / mask_float.sum(dim=1).clamp(min=1.0)
        elif self.aggregation == "max":
            aggregated, _ = embeddings.max(dim=1)
        else:
            aggregated = embeddings.sum(dim=1)

        return aggregated  # (batch, hidden_dim)


def make_time_scale(lookback, horizon, device=None):
    indices = torch.arange(-lookback, horizon, device=device, dtype=torch.float32)
    t = indices / lookback * torch.pi

    t_past, t_future = t[:lookback], t[lookback:]
    return t_past, t_future


def build_past_elements(t_past, y_past):
    batch_size = y_past.shape[0]
    t_expanded = t_past.unsqueeze(0).expand(batch_size, -1)  # (batch, lookback)
    return torch.stack([t_expanded, y_past], dim=-1)         # (batch, lookback, 2)


class Model(nn.Module):
    def __init__(self, cfg, logger=None):
        super().__init__()
        self.cfg = cfg.model
        self.cfg_data = cfg.dataset
        self.name = "inr_deepsets"

        self.deepsets_encoder = DeepSetsEncoder(
            input_dim=self.cfg.deepsets.input_dim,
            hidden_dim=self.cfg.deepsets.hidden_dim,
            aggregation=self.cfg.deepsets.aggregation,
        )

        self.lstm_encoder = LSTMEncoder(self.cfg.lstm.input_dim, self.cfg.lstm.hidden_dim)

        self.inr = INR(self.cfg, logger)

        z_dim = self.cfg.deepsets.hidden_dim + self.cfg.lstm.hidden_dim
        self.film_generator = FiLMGenerator(
            z_dim=z_dim,
            hidden_dim=self.cfg.inr.film_hidden_dim,
            feature_dims=self.cfg.inr.num_layers * self.cfg.inr.hidden_dim,
        )

    def set_epoch(self, epoch):
        self.inr.set_epoch(epoch)

    def forward_step(self, batch, device, debug=False):
        X_exog, mask, y_target = batch["X_exog"], batch["mask"], batch["y_target"]
        X_exog, mask, y_target = X_exog.to(device), mask.to(device), y_target.to(device)

        lookback = self.cfg.lookback
        horizon = self.cfg.horizon

        mask_past = mask[:, :lookback]
        y_past = y_target[:, :lookback]

        exog_past, exog_future = X_exog[:, :lookback], X_exog[:, lookback:]
        t_past, t_future = make_time_scale(lookback, horizon, device=device)

        elems_past = build_past_elements(t_past, y_past)
        z_lb = self.deepsets_encoder(elems_past, mask_past)  # (batch, deepsets.hidden_dim)

        h_t, c_t = self.lstm_encoder(exog_past)

        batch_size = z_lb.shape[0]
        predictions = []

        for t in range(horizon):
            t_ = t_future[t].expand(batch_size)
            x_t = exog_future[:, t, :]
            h_t, c_t = self.lstm_encoder.cell(x_t, (h_t, c_t))

            z_t = torch.cat([z_lb, h_t], dim=-1)
            films = self.film_generator(z_t)

            pred_t = self.inr(t_, films)
            predictions.append(pred_t)

        return torch.stack(predictions, dim=1)  # (batch, horizon, output_dim)
