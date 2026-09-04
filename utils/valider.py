from utils.metrics import compute_metrics, get_naive_reference

def validate(model, loader, device):

    model.eval()

    loss_dict = {'MSE': 0.0, 'RMSE': 0.0, 'MAE': 0.0, 'rMAE': 0.0}

    lookback = model.cfg.lookback
    horizon  = model.cfg.horizon

    n = 0
    for batch_idx, batch in enumerate(loader):
        n += 1

        pred_future, y_target = model.forward_step(batch, device)
        target_future = y_target[:, lookback:, ...].unsqueeze(-1)
        target_past = get_naive_reference(y_target, lookback, horizon, mode="naive1").unsqueeze(-1) #y_target[:, :lookback, ...].unsqueeze(-1)

        loss = ((pred_future - target_future) ** 2).mean()

        loss_dict['MSE'] += loss.item()
        rmse, mae, _, _, rmae = compute_metrics(pred_future.detach().cpu().numpy(), target_future.detach().cpu().numpy(), naive_ref=target_past.detach().cpu().numpy())
        loss_dict['RMSE'] += rmse
        loss_dict['MAE'] += mae
        loss_dict['rMAE'] += rmae


    for k in loss_dict:
        loss_dict[k] /= n

    return {
        "val_loss" : loss_dict
    }