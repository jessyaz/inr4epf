from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

from utils.valider import validate


def train(model, loaders, optimizer, device, logger):
    train_loader, val_loader = loaders['train_loader'], loaders['val_loader']

    # loss_dict = {'MSE': 0.0, 'RMSE': 0.0, 'MAE': 0.0, 'MAPE': 0.0, 'SMAPE': 0.0, 'rMAE': 0.0}
    loss_dict = {'MSE': 0.0}

    lookback = model.cfg.lookback

    num_epochs = model.cfg.num_epoch

    for epoch in range(num_epochs):
        n = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            n += 1

            pred_future, y_target = model.forward_step(batch, device)

            target_past          = y_target[:, :lookback, ...].unsqueeze(-1)
            target_future        = y_target[:, lookback:, ...].unsqueeze(-1)

            loss_future = ((pred_future - target_future) ** 2).mean()

            loss = loss_future

            if loss.requires_grad:
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            loss_dict['MSE'] += loss.item()

            if batch_idx == 5:
                fig, axes = plt.subplots(2, 3, figsize=(15, 8))
                axes = axes.flatten()

                for i in range(6):

                    axes[i].plot(pred_future[i].detach().cpu().squeeze(), label="Pred")
                    axes[i].plot(target_future[i].detach().cpu().squeeze(), label="Target")
                    axes[i].axvline(x=lookback, color='r', linestyle='--')
                    axes[i].set_title(f"Batch Sample {i}")
                    axes[i].legend()

                plt.tight_layout()
                logger.log_plot(fig, artifact_path=f"plots/epoch_{epoch+1}.png")
                plt.close(fig)

        #
        # CALL VALIDATION
        #
        val_loss = validate(model, val_loader, device)
        val_loss_dict = val_loss['val_loss']

        for k in loss_dict:
            loss_dict[k] /= n

        logger.log_metrics(loss_dict, epoch=epoch, prefix="train")
        logger.log_metrics(val_loss_dict, epoch=epoch, prefix="val")

    return {
        "train_loss" : loss_dict,
        "val_loss" : val_loss_dict,
    }