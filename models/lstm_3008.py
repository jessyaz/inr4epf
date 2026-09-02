import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.ih = nn.Linear(input_dim, hidden_dim * 4)
        self.hh = nn.Linear(hidden_dim, hidden_dim * 4)

    def forward(self, xt, state):
        h_prev, c_prev = state
        gates = self.ih(xt) + self.hh(h_prev)
        f, i, g, o = gates.chunk(4, dim=-1)
        ft, it, ot = torch.sigmoid(f), torch.sigmoid(i), torch.sigmoid(o)
        c_bar = torch.tanh(g)
        c_next = ft * c_prev + it * c_bar
        h_next = ot * torch.tanh(c_next)
        return h_next, (h_next, c_next)


class Encoder(nn.Module):
    """input_dim = exog_dim + target_dim (X_exog passé concaténé à Y passé)."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.cell = LSTMCell(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        b, seq_len, _ = x.shape
        h = torch.zeros(b, self.hidden_dim, device=x.device)
        c = torch.zeros(b, self.hidden_dim, device=x.device)
        x = self.proj_in(x)
        for t in range(seq_len):
            _, (h, c) = self.cell(x[:, t], (h, c))
        return (h, c)


class Decoder(nn.Module):
    """
    A chaque pas de décodage, l'entrée est [x_exog_futur[t], y_prev].
    input_dim = exog_dim + target_dim.
    """

    def __init__(self, input_dim, hidden_dim, target_dim):
        super().__init__()
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.cell = LSTMCell(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, target_dim)

    def forward(self, exog_t, y_prev, state):

        xt = torch.cat([exog_t, y_prev], dim=-1)

        xt = self.proj_in(xt)
        h, state = self.cell(xt, state)
        return self.fc_out(h), state


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, y0_dim):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.y0 = nn.Parameter(torch.zeros(y0_dim))


    def forward(self, exog_past, y_past, exog_future, y_future=None, teacher_forcing=0.5):
        batch_size = exog_past.size(0)
        lookback = exog_past.size(1)
        horizon = exog_future.size(1)

        enc_in = torch.cat([exog_past, y_past.unsqueeze(-1)], dim=-1)
        state = self.encoder(enc_in)

        past_outputs = []
        y_prev = self.y0.expand(batch_size, -1)

        for t in range(lookback):

            exog_t = exog_past[:, t, ...]
            y_hat, state = self.decoder(exog_t, y_prev, state)
            past_outputs.append(y_hat)
            y_prev = y_past[:, t].view(-1, 1)


        pred_past = torch.stack(past_outputs, dim=1)


        futur_outputs = []
        for t in range(horizon):
            exog_t = exog_future[:, t,...]
            y_hat, state = self.decoder(exog_t, y_prev, state)
            futur_outputs.append(y_hat)
            if y_future is not None and random.random() < teacher_forcing:
                y_prev = y_future[:, t]
            else:
                y_prev = y_hat

        pred_future = torch.stack(futur_outputs, dim=1)

        return pred_future, pred_past


class Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg.model
        self.cfg_data = cfg.dataset

        self.name = "lstm_3008"
        print(f"Config: {cfg}")

        batch_size = self.cfg_data.batch_size
        input_dim = self.cfg.input_dim
        exog_dim = self.cfg.exog_dim
        target_dim = self.cfg.target_dim
        hidden_dim = self.cfg.hidden_dim
        lookback = self.cfg.lookback
        horizon = self.cfg.horizon

        encoder = Encoder(input_dim, hidden_dim)
        decoder = Decoder(exog_dim + target_dim, hidden_dim, target_dim)

        self.model = Seq2Seq(encoder, decoder, y0_dim=(1, target_dim))

    def forward(self, exog_past, y_past, exog_future, y_future=None, teacher_forcing=None):
        tf = self.cfg.teacher_forcing if teacher_forcing is None else teacher_forcing
        return self.model(exog_past, y_past, exog_future, y_future=y_future, teacher_forcing=tf)

    def forward_step(self, batch, device, debug=False):

        X_exog, Y_target = batch
        X_exog, Y_target = X_exog.to(device), Y_target.to(device)

        lookback = self.cfg.lookback

        exog_past, exog_future = X_exog[:, :lookback], X_exog[:, lookback:]
        y_past, y_future = Y_target[:, :lookback], Y_target[:, lookback:].unsqueeze(-1)


        if debug:
            print(f"exog_past {tuple(exog_past.shape)} y_past {tuple(y_past.shape)} "
                  f"exog_future {tuple(exog_future.shape)} y_future {tuple(y_future.shape)}")

        pred_future, pred_past = self(exog_past, y_past, exog_future, y_future=y_future)

        return pred_future, pred_past, Y_target