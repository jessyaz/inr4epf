import torch
import torch.nn as nn





class INR(nn.Module):
    def __init__(self, cfg, logger=None):
        super().__init__()
        cfg = cfg.inr
        self.num_layers = cfg.num_layers
        self.hidden_dim = cfg.hidden_dim
        self.current_epoch = 0

        self.fourier = PE(cfg.num_frequencies, learnable=cfg.pe_learnable, logger=logger)
        fourier_dim = 2 * cfg.num_frequencies

        dims = [fourier_dim] + [cfg.hidden_dim] * cfg.num_layers
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1], bias=True)
            for i in range(cfg.num_layers)
        ])
        self.output_layer = nn.Linear(cfg.hidden_dim, cfg.output_dim, bias=True)

        if cfg.activation == "silu":
            self.activation = torch.nn.functional.silu
        elif cfg.activation == "gelu":
            self.activation = torch.nn.functional.gelu
        else:
            raise ValueError(f"Unknown activation: {cfg.activation}")

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self, t, film):
        gamma, beta = film

        x = self.fourier(t)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = gamma[:, i, :] * x + beta[:, i, :]
            x = self.activation(x)   # <- utilise self.activation, déjà résolu à l'init

        return self.output_layer(x)


FREQ_MIN = 0.875
FREQ_MAX = 84.0

class PE(nn.Module):
    def __init__(self, num_frequencies, learnable=True, logger=None):
        super().__init__()
        freqs_init = torch.logspace(
            torch.log2(torch.tensor(FREQ_MIN)),
            torch.log2(torch.tensor(FREQ_MAX)),
            num_frequencies,
            base=2.0,
        )

        self.learnable = learnable
        if learnable:
            self.freqs = nn.Parameter(freqs_init)
        else:
            self.register_buffer("freqs", freqs_init)

        self.current_epoch = 0
        self.logger = logger

    def forward(self, t):
        angles = t.unsqueeze(-1)
        return torch.cat([torch.sin(2*angles*self.freqs), torch.cos(2*angles*self.freqs)], dim=-1)

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if self.logger is not None:
            freqs_values = self.freqs.detach().cpu().numpy()
            freq_dict = {f"pe_freq_{i}": float(val) for i, val in enumerate(freqs_values)}
            self.logger.log_metrics(freq_dict, epoch, prefix="pe")



class FiLMGenerator(nn.Module):
    def __init__(self, z_dim, hidden_dim, num_layers, layer_dim):
        super().__init__()
        self.num_layers = num_layers
        self.layer_dim = layer_dim

        self.shared = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.GELU(),
        )

        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, 2 * layer_dim) for _ in range(num_layers)
        ])

        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            with torch.no_grad():
                head.bias[:layer_dim] = 1.0  # gamma init = 1, beta = 0

    def forward(self, z):
        h = self.shared(z)
        gammas, betas = [], []
        for head in self.heads:
            out = head(h)
            g, b = out.chunk(2, dim=-1)
            gammas.append(g)
            betas.append(b)
        gamma = torch.stack(gammas, dim=1)  # (batch, num_layers, layer_dim)
        beta = torch.stack(betas, dim=1)
        return gamma, beta



class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = nn.LSTMCell(input_dim, hidden_dim)

    def forward(self, exog_past):
        batch_size, seq_len, _ = exog_past.shape
        h_t = torch.zeros(batch_size, self.hidden_dim, device=exog_past.device)
        c_t = torch.zeros(batch_size, self.hidden_dim, device=exog_past.device)

        for t in range(seq_len):
            h_t, c_t = self.cell(exog_past[:, t, :], (h_t, c_t))

        return h_t, c_t




class DeepSetsEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, elements, mask):
        embeddings = self.phi(elements)
        mask_float = mask.unsqueeze(-1).to(embeddings.dtype)

        e_mean = embeddings * mask_float
        mean_pool = e_mean.sum(dim=1) / mask_float.sum(dim=1).clamp(min=1.0)

        e_max = embeddings.masked_fill(mask_float == 0, float('-inf'))
        max_pool, _ = e_max.max(dim=1)
        max_pool = torch.nan_to_num(max_pool, neginf=0.0)

        return torch.cat([mean_pool, max_pool], dim=-1)




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
        self.name = "inr"

        self.use_exog = getattr(self.cfg, "use_exog", True)

        self.deepsets_encoder = DeepSetsEncoder(
            input_dim=self.cfg.deepsets.input_dim,
            hidden_dim=self.cfg.deepsets.hidden_dim,
        )

        #self.lstm_encoder = LSTMEncoder(self.cfg.lstm.input_dim, self.cfg.lstm.hidden_dim)

        if self.use_exog:
            self.lstm_encoder = LSTMEncoder(self.cfg.lstm.input_dim, self.cfg.lstm.hidden_dim)
            z_dim = (2 * self.cfg.deepsets.hidden_dim) + self.cfg.lstm.hidden_dim
        else:
            self.lstm_encoder = None
            z_dim = 2 * self.cfg.deepsets.hidden_dim

        self.inr = INR(self.cfg, logger)

       # z_dim = (2 * self.cfg.deepsets.hidden_dim) + self.cfg.lstm.hidden_dim

        self.film_generator = FiLMGenerator(
            z_dim=z_dim,
            hidden_dim=self.cfg.inr.film_hidden_dim,
            num_layers=self.cfg.inr.num_layers,
            layer_dim=self.cfg.inr.hidden_dim,
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

        t_past, t_future = make_time_scale(lookback, horizon, device=device)

        elems_past = build_past_elements(t_past, y_past)
        z_lb = self.deepsets_encoder(elems_past, mask_past)  # (batch, 2*deepsets.hidden_dim)

        batch_size = z_lb.shape[0]
        predictions = []

        if self.use_exog:
            exog_past, exog_future = X_exog[:, :lookback], X_exog[:, lookback:]
            h_t, c_t = self.lstm_encoder(exog_past)

        for t in range(horizon):
            t_ = t_future[t].expand(batch_size)

            if self.use_exog:
                x_t = exog_future[:, t, :]
                h_t, c_t = self.lstm_encoder.cell(x_t, (h_t, c_t))
                z_t = torch.cat([z_lb, h_t], dim=-1)
            else:
                z_t = z_lb

            films = self.film_generator(z_t)
            pred_t = self.inr(t_, films)
            predictions.append(pred_t)

        return torch.stack(predictions, dim=1)  # (batch, horizon, output_dim)

    def configure_optimizer(self):

        cfg_opt = self.cfg.optim

        pe_freq_params = []
        inr_layer_params = []
        other_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "fourier.freqs" in name:
                pe_freq_params.append(param)
            elif name.startswith("inr.layers") or name.startswith("inr.output_layer"):
                inr_layer_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {"params": other_params, "lr": cfg_opt.lr},
            {"params": inr_layer_params, "lr": cfg_opt.lr_inr},
            {"params": pe_freq_params, "lr": cfg_opt.lr_pe},
        ]
        param_groups = [g for g in param_groups if len(g["params"]) > 0]

        return torch.optim.AdamW(param_groups, weight_decay=0.0)