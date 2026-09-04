import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg.model
        self.cfg_data = cfg.dataset

        self.name = "inr_3108"
        print(f"Config: {cfg}")





    def forward(self, exog_past, y_past, exog_future, y_future=None, teacher_forcing=None):
        pred_future = 0
        return pred_future

    def forward_step(self, batch, device, debug=False):

        X_exog, y_target = batch
        X_exog, y_target = X_exog.to(device), y_target.to(device)

        lookback = self.cfg.lookback

        exog_past, exog_future = X_exog[:, :lookback], X_exog[:, lookback:]
        y_past, y_future = y_target[:, :lookback], y_target[:, lookback:].unsqueeze(-1)



        pred_future = self(exog_past, y_past, exog_future, y_future=y_future)

        return pred_future, y_target