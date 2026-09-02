import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils.metrics import compute_metrics

def test(model, loader, scaler, device, logger):
    model.eval()

    lookback = model.cfg.lookback

    loss_dict = {'MSE': 0.0, 'RMSE': 0.0, 'MAE': 0.0, 'MAPE': 0.0, 'SMAPE': 0.0, 'rMAE': 0.0}

    n = 0
    all_pred_f, all_target_f = [], []
    all_pred_p, all_target_p = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            n += 1

            pred_future, pred_past, Y_target = model.forward_step(batch, device)

            target_past = Y_target[:, :lookback, ...].unsqueeze(-1)
            target_future = Y_target[:, lookback:, ...].unsqueeze(-1)

            pf_np = pred_future.detach().cpu().numpy()
            tf_np = target_future.detach().cpu().numpy()
            pp_np = pred_past.detach().cpu().numpy()
            tp_np = target_past.detach().cpu().numpy()

            if scaler is not None:
                orig_shape_f = pf_np.shape
                orig_shape_p = pp_np.shape

                pf_np = scaler.inverse_transform(pf_np.reshape(-1, pf_np.shape[-1])).reshape(
                    orig_shape_f
                )
                tf_np = scaler.inverse_transform(tf_np.reshape(-1, tf_np.shape[-1])).reshape(
                    orig_shape_f
                )
                pp_np = scaler.inverse_transform(pp_np.reshape(-1, pp_np.shape[-1])).reshape(
                    orig_shape_p
                )
                tp_np = scaler.inverse_transform(tp_np.reshape(-1, tp_np.shape[-1])).reshape(
                    orig_shape_p
                )

            all_pred_f.append(pf_np)
            all_target_f.append(tf_np)
            all_pred_p.append(pp_np)
            all_target_p.append(tp_np)

    preds_f = np.concatenate(all_pred_f, axis=0)
    targets_f = np.concatenate(all_target_f, axis=0)
    preds_p = np.concatenate(all_pred_p, axis=0)
    targets_p = np.concatenate(all_target_p, axis=0)


    rf, maf, mapf, smf, rmaef = compute_metrics(preds_f, targets_f, naive_ref=targets_p)
    rp, map_, mapp, smp, _ = compute_metrics(preds_p, targets_p, naive_ref=targets_p)

    err = ((preds_f - targets_f) ** 2).mean(axis=(1, 2))
    ids = np.argsort(err)
    worst = ids[-3:]
    best = ids[:3]

    print("worst : ", worst, "best" , best)


    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, i in zip(axes.flatten(), list(worst) + list(best)):
        ax.plot(np.concatenate([preds_p[i], preds_f[i]]).squeeze(), label="Pred")
        ax.plot(np.concatenate([targets_p[i], targets_f[i]]).squeeze(), label="Target")
        ax.axvline(lookback, color='r', linestyle='--')
        ax.legend()
    plt.tight_layout()
    logger.log_plot(fig, artifact_path="plot_test/test.png")
    plt.close(fig)


    loss_dict['MSE'] = float(np.mean((preds_f - targets_f) ** 2))
    loss_dict['RMSE'] = rf
    loss_dict['MAE'] = maf
    loss_dict['MAPE'] = mapf
    loss_dict['SMAPE'] = smf
    loss_dict['rMAE'] = rmaef

    logger.log_metrics(loss_dict, epoch=0, prefix="test")

    return {"test_loss": loss_dict}