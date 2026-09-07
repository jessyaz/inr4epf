from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

from utils.valider import validate


def train(model, loaders, optimizer, device, logger):
    train_loader, val_loader = loaders['train_loader'], loaders['val_loader']

    # loss_dict = {'MSE': 0.0, 'RMSE': 0.0, 'MAE': 0.0, 'MAPE': 0.0, 'SMAPE': 0.0, 'rMAE': 0.0}
    loss_dict = {'MSE': 0.0}

    patience = 20
    grad_clip_norm = model.cfg.grad_clip_norm

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    lookback = model.cfg.lookback

    num_epochs = model.cfg.num_epoch

    best_val_loss = float("inf")
    best_state_dict = None
    patience_counter = 0

    #single_batch = next(iter(train_loader))
    #single_batch = {k: v[0:1].to(device) for k, v in single_batch.items()}  # une seule série, fixe

    for epoch in range(num_epochs):
        model.train()
        n = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            n += 1


            #batch = single_batch

            prg = epoch / max(1, 10)

            pred_future = model.forward_step(batch, device, prg)

            y_target = batch["y_target"].unsqueeze(-1).to(device)
            mask_future = batch["mask"][:, lookback:, ...].unsqueeze(-1).to(device)

            target_past          = y_target[:, :lookback, ...]
            target_future        = y_target[:, lookback:, ...]
            mask_future_expanded = mask_future.float()

            #loss_future = ((pred_future - target_future) ** 2).mean()

            squared_error = (pred_future - target_future) ** 2

            loss_future = (squared_error * mask_future_expanded).sum() / mask_future_expanded.sum().clamp(min=1.0)


            loss = loss_future

            if loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            loss_dict['MSE'] += loss.item()


            if batch_idx == 5:
                fig, axes = plt.subplots(2, 3, figsize=(15, 8))
                axes = axes.flatten()
                n_plots = min(6, pred_future.shape[0])

                for i in range(n_plots):
                    pred = pred_future[i].detach().cpu().squeeze().numpy()
                    target = target_future[i].detach().cpu().squeeze().numpy()
                    mask = mask_future[i].detach().cpu().squeeze().numpy().astype(bool)

                    # Remplace les positions masquées par NaN -> matplotlib laisse un vrai trou, pas un 0
                    target_display = target.copy()
                    target_display[~mask] = float("nan")

                    x = range(len(target))

                    axes[i].plot(x, pred, label="Pred")
                    axes[i].plot(x, target_display, label="Target", marker="o", markersize=3)

                    # Ombre les zones masquées pour les repérer visuellement
                    in_masked_zone = False
                    start = None
                    for j, valid in enumerate(mask):
                        if not valid and not in_masked_zone:
                            start = j
                            in_masked_zone = True
                        elif valid and in_masked_zone:
                            axes[i].axvspan(start - 0.5, j - 0.5, color="red", alpha=0.15)
                            in_masked_zone = False
                    if in_masked_zone:
                        axes[i].axvspan(start - 0.5, len(mask) - 0.5, color="red", alpha=0.15)


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

        current_val_loss = val_loss_dict["MSE"]
        scheduler.step(current_val_loss)
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping à l'epoch {epoch+1} (meilleure val loss: {best_val_loss:.6f})")
            break

        loss_dict = {k: 0.0 for k in loss_dict}

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {
        "train_loss" : loss_dict,
        "val_loss" : val_loss_dict,
    }