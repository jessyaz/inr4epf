"""
Baseline naif EPF, k=7 jours : predit l'heure h de l'horizon par la valeur
observee 7 jours avant, meme heure. Comme lookback=168=7*24, cette valeur
correspond exactement a y_past[:, h] pour h dans 0..23 -- aucun modele,
aucun entrainement necessaire.

Deux variantes calculees :
- naive_interp : y_past interpole (linear_interpolate_masked) avant extraction
  -> subit le meme handicap face au manque de donnees que les autres modeles
     (TimeXer, INR). C'est LA baseline canonique utilisee pour le rMAE.
- naive_raw    : y_past NON masque (acces aux vraies valeurs historiques)
                 -> reference diagnostique "historique parfait", pas la
                    baseline officielle pour le rMAE.
"""

import torch
from utils.interpolate import linear_interpolate_masked


def compute_naive_k7(batch, device, lookback, horizon):
    assert lookback == horizon * 7, (
        f"naive k=7 suppose lookback == 7*horizon (ici lookback={lookback}, horizon={horizon})"
    )

    y_target = batch["y_target"].to(device)
    y_target_no_mask = batch["y_target_no_mask"].to(device)
    mask = batch["mask"].to(device)

    y_past = y_target[:, :lookback].unsqueeze(-1)
    y_past_raw = y_target_no_mask[:, :lookback].unsqueeze(-1)
    mask_past = mask[:, :lookback].unsqueeze(-1).float()

    y_past_interp = linear_interpolate_masked(y_past, mask_past).unsqueeze(-1)

    naive_interp = y_past_interp[:, :horizon, :]  # (batch, horizon, 1)
    naive_raw = y_past_raw[:, :horizon, :]        # (batch, horizon, 1)

    return naive_interp, naive_raw