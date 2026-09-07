import torch
import torch.nn as nn





class INR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        cfg = cfg.inr

        self.fourier = PE(cfg.num_frequencies)
        fourier_dim = 2 * cfg.num_frequencies

        dims = [fourier_dim] + [cfg.hidden_dim] * cfg.num_layers
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1], bias=False)
            for i in range(cfg.num_layers)
        ])

        self.output_layer = nn.Linear(cfg.hidden_dim, cfg.output_dim, bias=False)

    def forward(self, t, films, progress=1.0):
        x = self.fourier(t, progress=progress)
        for layer, (gamma, beta) in zip(self.layers, films):
            x = layer(x)


            x = gamma * x + beta
            # x = torch.exp(-torch.square(torch.cos(x))) Lead to 2.45 no bias
            x = torch.exp(-torch.square(x)) # With variable - pe -> lead to 2.42 -- best no bias
           #  x = torch.exp(-torch.square(torch.cos(x))) # lead to 2.58 no bias
            #x = torch.exp(-torch.square(torch.cos(x))) #  lead to 2.49 bias
          #  x = torch.exp(-torch.square(x)) #  lead to 3.06 bias


            #x = torch.relu(x)
        return self.output_layer(x)

class PE(nn.Module):
    def __init__(self, num_frequencies):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(num_frequencies))
        #self.register_buffer("freqs", torch.randn(num_frequencies))

    def forward(self, t, progress = 1.0):
        angles = t.unsqueeze(-1) * self.freqs                       # (batch, num_frequencies)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (batch, 2*num_frequencies)



class FiLMGenerator(nn.Module):
    def __init__(self, z_dim, hidden_dim, feature_dims):

        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(z_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2 * feature_dim),
            )
            for feature_dim in feature_dims
        ])

    def forward(self, z):
        films = []
        for head in self.heads:
            gamma, beta = head(z).chunk(2, dim=-1)
            films.append((gamma, beta))
        return films



class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.cell = nn.LSTMCell(input_dim, hidden_dim)

    def forward(self, exog_past):
        _, (h, c) = self.lstm(exog_past)
        return h.squeeze(0), c.squeeze(0)

class DeepSetsEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, aggregation="mean"):
        super().__init__()
        self.aggregation = aggregation
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
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

        return self.rho(aggregated)


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
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.model
        self.cfg_data = cfg.dataset
        self.name = "inr_deepsets"

        self.deepsets_encoder = DeepSetsEncoder(
            input_dim=self.cfg.deepsets.input_dim,
            hidden_dim=self.cfg.deepsets.hidden_dim,
            output_dim=self.cfg.deepsets.output_dim,
        )
        self.predictor = nn.Linear(self.cfg.deepsets.output_dim, self.cfg.horizon)

        self.lstm_encoder = LSTMEncoder(self.cfg.lstm.input_dim, self.cfg.lstm.hidden_dim)

        self.inr = INR(self.cfg)

        z_dim = self.cfg.deepsets.output_dim + self.cfg.lstm.hidden_dim
        self.film_generator = FiLMGenerator(
            z_dim=z_dim,
            hidden_dim=self.cfg.inr.film_hidden_dim,
            feature_dims=[self.cfg.inr.hidden_dim] * self.cfg.inr.num_layers,
        )


    def forward_step(self, batch, device, prg=1.0, debug=False):
        X_exog, mask, y_target = batch["X_exog"], batch["mask"], batch["y_target"]
        X_exog, mask, y_target = X_exog.to(device), mask.to(device), y_target.to(device)

        lookback = self.cfg.lookback
        horizon = self.cfg.horizon

        mask_past = mask[:, :lookback]
        mask_future = mask[:, lookback:]

        y_past = y_target[:, :lookback]

        exog_past, exog_future = X_exog[:, :lookback], X_exog[:, lookback:]

        t_past, t_future = make_time_scale(lookback, horizon, device=device)

        elems_past = build_past_elements(t_past, y_past)
        z_lb = self.deepsets_encoder(elems_past, mask_past)  # (batch, deepsets.output_dim)

        h_t, c_t = self.lstm_encoder(exog_past)

        batch_size = z_lb.shape[0]
        predictions = []

        for t in range(horizon):
            t_ = t_future[t].expand(batch_size)

            x_t = exog_future[:, t, :]
            h_t, c_t = self.lstm_encoder.cell(x_t, (h_t, c_t))

            z_t = torch.cat([z_lb, h_t], dim=-1)
            films = self.film_generator(z_t)

            pred_t = self.inr(t_, films, prg)
            predictions.append(pred_t)

        pred_future = torch.stack(predictions, dim=1)  # (batch, horizon, output_dim)

        return pred_future
