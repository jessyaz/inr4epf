import torch


def linear_interpolate_masked(y, mask):
    mask = mask.bool()
    if mask.dim() == 3:
        mask = mask.squeeze(-1)  # (batch, seq_len, 1) -> (batch, seq_len)
    if y.dim() == 3:
        y = y.squeeze(-1)

    batch_size, seq_len = y.shape
    device = y.device
    y_interp = y.clone()

    idx = torch.arange(seq_len, device=device).float().unsqueeze(0).expand(batch_size, -1)

    for b in range(batch_size):
        valid_idx = idx[b][mask[b]]
        valid_val = y[b][mask[b]]

        if valid_idx.numel() == 0:
            continue
        if valid_idx.numel() == 1:
            y_interp[b] = valid_val.item()
            continue

        missing_idx = idx[b][~mask[b]]
        if missing_idx.numel() == 0:
            continue

        interp_val = torch_interp(missing_idx, valid_idx, valid_val)
        y_interp[b][~mask[b]] = interp_val

    return y_interp
def torch_interp(x, xp, fp):

    idx = torch.searchsorted(xp, x, right=True).clamp(1, len(xp) - 1)

    x0 = xp[idx - 1]
    x1 = xp[idx]
    y0 = fp[idx - 1]
    y1 = fp[idx]

    slope = (y1 - y0) / (x1 - x0).clamp(min=1e-8)
    result = y0 + slope * (x - x0)

    result = torch.where(x < xp[0], fp[0], result)
    result = torch.where(x > xp[-1], fp[-1], result)

    return result