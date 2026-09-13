import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from utils.interpolate import linear_interpolate_masked

_VENDOR_PATH = Path(__file__).resolve().parent.parent / "git_src" / "TimeXer"

_CONFLICTING_TOP_LEVEL = ["models", "utils", "layers", "data_provider", "exp"]


def _load_timexer_backbone():
    stashed = {}
    for name in list(sys.modules.keys()):
        if name.split(".")[0] in _CONFLICTING_TOP_LEVEL:
            stashed[name] = sys.modules.pop(name)

    sys.path.insert(0, str(_VENDOR_PATH))
    try:
        import models.TimeXer as _vendored_module
        return _vendored_module.Model
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"[timexer] Import error : 'models.TimeXer' from '{_VENDOR_PATH}'.\n"
            f"  -> Verifie que le repo est bien clone : ls {_VENDOR_PATH}/models/TimeXer.py\n"
            f"  -> Sinon, lance : ./clone_models.sh\n"
            f"  Erreur d'origine : {e}"
        ) from e
    finally:
        sys.path.remove(str(_VENDOR_PATH))
        for name in list(sys.modules.keys()):
            if name.split(".")[0] in _CONFLICTING_TOP_LEVEL and name not in stashed:
                del sys.modules[name]
        sys.modules.update(stashed)


TimeXerBackbone = _load_timexer_backbone()


class Model(nn.Module):
    """
    TimeXer, fidele a l'architecture officielle (Wang et al.) : enc_in = 2
    exogenes + 1 cible = 3, AUCUN canal masque. Le manque de donnees est gere
    uniquement via l'interpolation lineaire de l'endogene avant l'entree du
    modele -- TimeXer ne recoit aucune information explicite sur quels points
    sont observes vs interpoles.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.model
        self.name = "timexer"

        configs = SimpleNamespace(
            task_name="long_term_forecast",
            features="MS",
            seq_len=self.cfg.lookback,
            pred_len=self.cfg.horizon,
            use_norm=self.cfg.use_norm,
            patch_len=self.cfg.patch_len,
            enc_in=self.cfg.enc_in,
            dec_in=self.cfg.enc_in,
            c_out=1,
            d_model=self.cfg.d_model,
            d_ff=self.cfg.d_ff,
            n_heads=self.cfg.n_heads,
            e_layers=self.cfg.e_layers,
            dropout=self.cfg.dropout,
            activation=self.cfg.activation,
            embed=self.cfg.embed,
            freq=self.cfg.freq,
            factor=self.cfg.factor,
        )
        self.backbone = TimeXerBackbone(configs)

    def set_epoch(self, epoch):
        pass

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        dummy_dec = torch.zeros(
            x_enc.size(0), self.cfg.horizon, x_enc.size(2), device=x_enc.device
        )
        return self.backbone(x_enc, None, dummy_dec, None)

    def forward_step(self, batch, device, debug: bool = False):
        X_exog, mask, y_target = batch["X_exog"], batch["mask"], batch["y_target"]
        X_exog, mask, y_target = X_exog.to(device), mask.to(device), y_target.to(device)

        lookback = self.cfg.lookback
        horizon = self.cfg.horizon

        mask_past = mask[:, :lookback].unsqueeze(-1).float()
        exog_past = X_exog[:, :lookback]
        y_past = y_target[:, :lookback].unsqueeze(-1)

        # mask_past utilise UNIQUEMENT pour l'interpolation, jamais transmis au modele
        y_past = linear_interpolate_masked(y_past, mask_past).unsqueeze(-1)

        # x_enc : target en DERNIERE colonne, cf. convention officielle 'features=MS'
        # enc_in = exog_dim + 1, AUCUN canal masque -- architecture officielle
        x_enc = torch.cat([exog_past, y_past], dim=-1)

        pred_future = self(x_enc)

        return pred_future