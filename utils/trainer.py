
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch


def train(model, loader, optimizer, device, logger):

    model.train()

    loss_dict = {'MSE_future': 0.0, 'MSE_past': 0.0, 'MSE_total': 0.0}
    lookback = model.cfg.lookback

    num_epochs = model.cfg.num_epoch

    for epoch in range(num_epochs):
        n = 0
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
            n += 1

            pred_future, pred_past, Y_target = model.forward_step(batch, device)
            target_past, target_future = Y_target[:, :lookback, ...].unsqueeze(-1), Y_target[:, lookback:, ...].unsqueeze(-1)


            loss_future = ((pred_future - target_future) ** 2).mean()
            loss_past = ((pred_past - target_past) ** 2).mean()

            loss = loss_future + loss_past

            if loss.requires_grad:
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            loss_dict['MSE_future'] += loss_future.item()
            loss_dict['MSE_past'] += loss_past.item()
            loss_dict['MSE_total'] += loss.item()

            if batch_idx == 5:
                fig, axes = plt.subplots(2, 3, figsize=(15, 8))
                axes = axes.flatten()

                for i in range(6):
                    full_pred = torch.cat([pred_past[i].detach().cpu(), pred_future[i].detach().cpu()], dim=0)
                    full_target = Y_target[i].detach().cpu()

                    axes[i].plot(full_pred.squeeze(), label="Pred")
                    axes[i].plot(full_target.squeeze(), label="Target")
                    axes[i].axvline(x=lookback, color='r', linestyle='--')
                    axes[i].set_title(f"Batch Sample {i}")
                    axes[i].legend()

                plt.tight_layout()
                logger.log_plot(fig, artifact_path=f"plots/epoch_{epoch+1}.png")
                plt.close(fig)


        for k in loss_dict:
            loss_dict[k] /= n

        logger.log_metrics(loss_dict, epoch=epoch, prefix="train")

    return {
        "loss" : loss_dict
    }