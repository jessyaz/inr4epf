






def validate(model, loader, optimizer, device):

    model.train()

    loss_dict = {}

    loss_dict['MSE'] = 0.0

    n = 0
    for batch_idx, batch in enumerate(loader):
        n += 1

        pred, target = model.forward_step(batch, device)

        loss = ((pred - target) ** 2).mean()

        loss_dict['MSE'] += loss.item()

    loss_dict['MSE'] /= n

    return {
        "loss" : loss_dict
    }