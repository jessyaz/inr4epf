import sys
from pathlib import Path
from types import SimpleNamespace


import importlib.util

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
        # nettoie les modules TimeXer fraichement charges sous ces noms generiques
        for name in list(sys.modules.keys()):
            if name.split(".")[0] in _CONFLICTING_TOP_LEVEL and name not in stashed:
                del sys.modules[name]
        # restaure les modules originaux du projet (utils.trainer, etc.)
        sys.modules.update(stashed)


TimeXerBackbone = _load_timexer_backbone()

class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.model
        self.name = "timexer"

        # Si absent de la config, on garde le comportement historique (masque inclus)
        # pour ne rien casser sur des configs déjà existantes qui ne connaissent pas ce champ.
        self.include_mask_channel = getattr(self.cfg, "include_mask_channel", True)

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
        # No-op : TimeXer n'a pas de logique dépendante de l'epoch (contrairement au PE de l'INR),
        # mais trainer.py appelle model.set_epoch(epoch) sans garde hasattr() -- cette méthode
        # doit donc exister pour éviter un AttributeError.
        pass

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        # x_mark_enc=None gere nativement par DataEmbedding_inverted (cf. Embed.py)
        # x_dec/x_mark_dec ignores en interne par forecast() en mode MS -> placeholders vides
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



        exog_past = X_exog[:, :lookback]              # [B, lookback, exog_dim]
        y_past = y_target[:, :lookback].unsqueeze(-1)  # [B, lookback, 1]

        y_past = linear_interpolate_masked(y_past, mask_past).unsqueeze(-1)

        # x_enc : target en DERNIERE colonne, cf. convention officielle 'features=MS'
        if self.include_mask_channel:
            # enc_in attendu = exog_dim + 1 (cible) + 1 (masque)
            x_enc = torch.cat([mask_past, exog_past, y_past], dim=-1)
        else:
            # enc_in attendu = exog_dim + 1 (cible), SANS masque
            # -> architecture strictement identique à celle du papier TimeXer (enc_in=3 sur EPF)
            x_enc = torch.cat([exog_past, y_past], dim=-1)

        pred_future = self(x_enc)  # [B, horizon, 1] -- deja au bon format

        return pred_future