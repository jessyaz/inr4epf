import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.name = "test_model"
        print(cfg.test)

    def forward_step(self, batch, device, debug=False):
        print(f"Test data dimension from {self.name} =", batch.shape)