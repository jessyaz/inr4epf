import sys
from pathlib import Path
import torch
import torch.nn as nn

_VENDOR_PATH = Path(__file__).resolve().parent.parent / "git_src" / "epf-transformers"
sys.path.insert(0, str(_VENDOR_PATH))

try:
    from src.models import BaseDailyElectricTransformer
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        f"[epf_transformer] Import error : 'src.models' from '{_VENDOR_PATH}'.\n"
        f"  Error : {e}"
    ) from e


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.model
        self.name = "epf_transformer"

        self.backbone = BaseDailyElectricTransformer(
            embedding_dim=self.cfg.embedding_dim,
            num_heads=self.cfg.num_heads,
            dim_feedforward=self.cfg.dim_feedforward,
            num_layers=self.cfg.num_layers,
            normalize_first=self.cfg.normalize_first,
            dropout=self.cfg.dropout,
            activation=self.cfg.activation,
        )

    def forward(self, values: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return self.backbone(values, features)

    def forward_step(self, batch, device, debug: bool = False):
        X_exog, Y_target = batch          # X_exog: [B, 360, 2], Y_target: [B, 360]
        X_exog, Y_target = X_exog.to(device), Y_target.to(device)
        lookback = self.cfg.lookback

        inputs = torch.cat([Y_target.unsqueeze(-1), X_exog], dim=-1)   # [B, 360, 3] -- toute la fenetre
        horizon = inputs.size(1) - lookback

        values = inputs[:, :lookback, 0].unsqueeze(2)     # [B, 336, 1]  prix passes (jours 1-14)
        features = inputs[:, horizon:, 1:]                # [B, 336, 2]  exogenes (jours 2-15, decale d'1 jour)

        pred_future = self(values, features)[:, -horizon:].unsqueeze(-1)

        return pred_future, Y_target