import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader


class EPFDataset(Dataset):
    def __init__(self, X_exog, y_target, missing_rate=0.0, seed=42):
        self.X_exog = torch.tensor(X_exog, dtype=torch.float32)
        self.y_target = torch.tensor(y_target, dtype=torch.float32)
        self.missing_rate = missing_rate
        self.seed = seed

        if self.missing_rate > 0.0:
            self._apply_corruption()

    def _apply_corruption(self):
        g = torch.Generator()
        g.manual_seed(self.seed)

        mask = torch.rand(self.X_exog.shape[:2], generator=g) > self.missing_rate
        self.X_exog = self.X_exog * mask.unsqueeze(-1)

    def __len__(self):
        return len(self.X_exog)

    def __getitem__(self, idx):
        return self.X_exog[idx], self.y_target[idx]


def load_market_dataloader(
        market: str,
        batch_size: int = 64,
        processed_dir: str = "./datasets/processed",
        missing_rate: float = 0.0,
        seed: int = 42
):
    file_path = os.path.join(processed_dir, f"{market}_data.pkl")
    print(file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier pour le marché {market} n'existe pas dans {processed_dir}.")

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    train_dataset = EPFDataset(
        data['X_exogenous_train'],
        data['Y_target_train'],
        missing_rate=missing_rate,
        seed=seed
    )
    test_dataset = EPFDataset(
        data['X_exogenous_test'],
        data['Y_target_test'],
        missing_rate=0.0,
        seed=seed
    )

    g_loader = torch.Generator()
    g_loader.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g_loader
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader, data['scaler']