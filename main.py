
from datasets.loader import load_market_dataloader
train_loader, test_loader, scaler = load_market_dataloader('FR', batch_size=16, missing_rate=0.1, seed=42)
for x, y in train_loader:
    print('Shape X_exog:', x.shape)
    print('Shape Y_target:', y.shape)
    break
