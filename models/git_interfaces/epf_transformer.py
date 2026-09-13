import sys
from pathlib import Path
import torch
import torch.nn as nn

from utils.interpolate import linear_interpolate_masked

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
    """
    Interface pour BaseDailyElectricTransformer (Llorente & Portela).

    IMPORTANT (verifie dans src/train_functions.py officiel, pas devine) :
    - `values`   : SL heures de prix passes -> reshape interne en (SL/24) "jours-tokens"
    - `features` : SL heures d'EXOGENES decalees de 24h (PAS juste les 24h de l'horizon !)
                   -> meme longueur que `values`, sinon le concat interne du modele
                   (torch.concat sur dim=2, qui exige un nombre de "jours-tokens" identique
                   entre values_embeddings et features_embeddings) plante.
    - Le modele retourne une sequence complete de longueur SL ; seules les 24
      dernieres heures sont utilisees comme prevision "jour-ahead" (comme fait
      dans leur propre test() officiel : outputs[:, -24:]).
    """

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

    def set_epoch(self, epoch):
        pass

    def configure_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.lr)

    def forward(self, values: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return self.backbone(values, features)

    def forward_step(self, batch, device, debug: bool = False):
        X_exog, mask, y_target = batch["X_exog"], batch["mask"], batch["y_target"]
        X_exog, mask, y_target = X_exog.to(device), mask.to(device), y_target.to(device)

        lookback = self.cfg.lookback   # = SL (168 chez nous)
        horizon = self.cfg.horizon     # = 24

        mask_past = mask[:, :lookback].unsqueeze(-1).float()
        y_past = y_target[:, :lookback].unsqueeze(-1)
        values = linear_interpolate_masked(y_past, mask_past).unsqueeze(-1)  # [B, SL, 1]

        # features : SL heures d'exogenes, DECALEES de horizon(24)h par rapport
        # au debut de la fenetre totale (lookback+horizon) -- PAS juste les
        # 24h de l'horizon. Meme longueur que `values` (SL), sinon le concat
        # interne du modele echoue (nombre de "jours-tokens" different).
        features = X_exog[:, horizon:, :]  # [B, SL, exog_dim] (indices 24..191 chez nous)

        pred_full = self(values, features)          # [B, SL] -- sequence complete
        pred_future = pred_full[:, -horizon:]        # ne garde que les 24 dernieres heures
        pred_future = pred_future.unsqueeze(-1)      # [B, horizon, 1]

        return pred_future