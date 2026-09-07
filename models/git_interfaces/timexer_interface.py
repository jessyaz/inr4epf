"""
Interface entre l'entrainement (trainer.py/valider.py/tester.py) et le modele
vendored TimeXer (thuml/TimeXer, NeurIPS 2024), clone dans models/git-src/
via clone_models.sh.

Convention native du modele vendored (cf. dataset/EPF/ + scripts/forecast_exogenous/EPF/TimeXer.sh
du repo original -- deja evalue sur les 5 marches epftoolbox par les auteurs eux-memes) :

    x_enc      : [B, seq_len, enc_in]   toute la fenetre passee, TARGET EN DERNIERE COLONNE
                 (convention 'features=MS' : x_enc[:, :, -1] = Price, x_enc[:, :, :-1] = exogenes)
    x_mark_enc : covariables temporelles (peut etre None, gere nativement par le modele)
    x_dec, x_mark_dec : requis par la signature partagee de la lib, mais IGNORES en interne
                 par Model.forecast() en mode 'MS' -- on peut passer des tenseurs vides.

Le modele ne travaille que sur le PASSE (x_enc) pour produire le futur -- contrairement a
epf-transformers, il n'a PAS besoin des exogenes futures en entree explicite (le mecanisme
d'attention croisee endogene/exogene se fait uniquement sur la fenetre d'observation passee).
pred_past n'existe pas ici non plus -- modele purement forecast-only, comme epf_transformer.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

import importlib.util

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

_VENDOR_PATH = Path(__file__).resolve().parent.parent / "git_src" / "TimeXer"

# TimeXer utilise des noms de package generiques (models, layers, utils, data_provider,
# exp) qui collisionnent avec la structure du projet (notamment notre propre 'utils/').
# On isole temporairement ces noms de sys.modules pendant le chargement, pour forcer
# Python a resoudre les imports internes de TimeXer contre SON PROPRE code vendored,
# puis on restaure les modules originaux du projet juste apres.
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

        # TimeXer attend un objet "configs" type argparse.Namespace (voir run.py du repo original)
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

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        # x_mark_enc=None gere nativement par DataEmbedding_inverted (cf. Embed.py)
        # x_dec/x_mark_dec ignores en interne par forecast() en mode MS -> placeholders vides
        dummy_dec = torch.zeros(
            x_enc.size(0), self.cfg.horizon, x_enc.size(2), device=x_enc.device
        )
        return self.backbone(x_enc, None, dummy_dec, None)

    def forward_step(self, batch, device, debug: bool = False):
        X_exog, Y_target = batch
        X_exog, Y_target = X_exog.to(device), Y_target.to(device)
        lookback = self.cfg.lookback

        exog_past = X_exog[:, :lookback]              # [B, lookback, exog_dim]
        y_past = Y_target[:, :lookback].unsqueeze(-1)  # [B, lookback, 1]

        # x_enc : target en DERNIERE colonne, cf. convention officielle 'features=MS'
        x_enc = torch.cat([exog_past, y_past], dim=-1)  # [B, lookback, exog_dim+1]

        pred_future = self(x_enc)  # [B, horizon, 1] -- deja au bon format

        return pred_future, Y_target