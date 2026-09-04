"""
Interface generique entre l'entrainement (trainer.py/valider.py/tester.py) et
n'importe quel modele de la "Time-Series-Library" vendored depuis thuml/TimeXer
(qui partage TimeXer, PatchTST, DLinear, iTransformer, Autoformer, Informer,
Crossformer, TiDE, TimesNet, etc. -- tous pilotes par la meme interface
Model(configs) / forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)).

cfg.model.backbone_name choisit quelle classe importer (ex: "TimeXer", "PatchTST",
"DLinear", "iTransformer"), sans dupliquer un fichier d'interface par architecture.

Convention d'entree commune a toute la librairie (mode 'features=MS') :
    x_enc : [B, seq_len, enc_in]   toute la fenetre passee, TARGET EN DERNIERE COLONNE
    x_mark_enc / x_dec / x_mark_dec : geres nativement (None ou zeros acceptes,
        cf. verification faite sur DataEmbedding_inverted et les forward() de
        TimeXer/PatchTST/DLinear/iTransformer)

pred_past n'existe pour aucun de ces modeles -- tous purement forecast-only.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

_VENDOR_PATH = Path(__file__).resolve().parent.parent / "git_src" / "TimeXer"

# Ces noms de package sont generiques et collisionnent avec la structure du projet
# (notamment notre propre 'utils/' et 'models/'). On les isole temporairement de
# sys.modules pendant le chargement du backbone vendored, puis on restaure l'etat
# original du projet juste apres -- cf. discussion complete sur ce point.
_CONFLICTING_TOP_LEVEL = ["models", "utils", "layers", "data_provider", "exp"]


def _load_backbone_class(backbone_name: str):
    stashed = {}
    for name in list(sys.modules.keys()):
        if name.split(".")[0] in _CONFLICTING_TOP_LEVEL:
            stashed[name] = sys.modules.pop(name)

    sys.path.insert(0, str(_VENDOR_PATH))
    try:
        module = __import__(f"models.{backbone_name}", fromlist=["Model"])
        return module.Model
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"[timeseries_lib] Import error : 'models.{backbone_name}' from '{_VENDOR_PATH}'.\n"
            f"  -> Verifie que le repo est bien clone : ls {_VENDOR_PATH}/models/{backbone_name}.py\n"
            f"  -> Sinon, lance : ./clone_models.sh\n"
            f"  Erreur d'origine : {e}"
        ) from e
    finally:
        sys.path.remove(str(_VENDOR_PATH))
        for name in list(sys.modules.keys()):
            if name.split(".")[0] in _CONFLICTING_TOP_LEVEL and name not in stashed:
                del sys.modules[name]
        sys.modules.update(stashed)


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.model
        self.name = f"timeseries_lib_{self.cfg.backbone_name.lower()}"

        Backbone = _load_backbone_class(self.cfg.backbone_name)

        configs = SimpleNamespace(
            task_name="long_term_forecast",
            features="MS",
            seq_len=self.cfg.lookback,
            pred_len=self.cfg.horizon,
            label_len=self.cfg.get("label_len", self.cfg.lookback // 2),
            use_norm=self.cfg.get("use_norm", 1),
            patch_len=self.cfg.get("patch_len", 16),
            stride=self.cfg.get("stride", 8),
            enc_in=self.cfg.enc_in,
            dec_in=self.cfg.enc_in,
            c_out=1,
            d_model=self.cfg.d_model,
            d_ff=self.cfg.d_ff,
            n_heads=self.cfg.get("n_heads", 8),
            e_layers=self.cfg.e_layers,
            d_layers=self.cfg.get("d_layers", 1),
            moving_avg=self.cfg.get("moving_avg", 25),
            individual=self.cfg.get("individual", False),
            dropout=self.cfg.dropout,
            activation=self.cfg.activation,
            embed=self.cfg.get("embed", "timeF"),
            freq=self.cfg.get("freq", "h"),
            factor=self.cfg.get("factor", 1),
            output_attention=False,
        )
        self.backbone = Backbone(configs)

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        dummy_dec = torch.zeros(
            x_enc.size(0), self.cfg.horizon, x_enc.size(2), device=x_enc.device
        )
        out = self.backbone(x_enc, None, dummy_dec, None)   # [B, horizon, n_channels_sortie]
        return out[:, :, -1:]                                # garde uniquement le canal cible (Price)

    def forward_step(self, batch, device, debug: bool = False):
        X_exog, Y_target = batch
        X_exog, Y_target = X_exog.to(device), Y_target.to(device)
        lookback = self.cfg.lookback

        exog_past = X_exog[:, :lookback]
        y_past = Y_target[:, :lookback].unsqueeze(-1)

        x_enc = torch.cat([exog_past, y_past], dim=-1)  # target en derniere colonne

        pred_future = self(x_enc)

        return pred_future, Y_target