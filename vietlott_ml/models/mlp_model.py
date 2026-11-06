"""MLP baseline: multi-hot kỳ trước → xác suất xuất hiện kỳ kế tiếp."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import TorchBinaryModel, get_device


class MLPModel(TorchBinaryModel):
    arch = "mlp"
    name = "mlp"

    def __init__(
        self,
        n_features: int,
        hidden1: int = 256,
        hidden2: int = 128,
        dropout: float = 0.2,
        seed: int = 42,
    ):
        super().__init__(n_features=n_features, seed=seed)
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.dropout = dropout

    def _build(self):
        from torch import nn

        return nn.Sequential(
            nn.Linear(self.n_features, self.hidden1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden1, self.hidden2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden2, self.n_features),
        )

    def get_config(self) -> dict:
        return {
            "n_features": self.n_features,
            "hidden1": self.hidden1,
            "hidden2": self.hidden2,
            "dropout": self.dropout,
            "seed": self.seed,
        }

    @classmethod
    def load(cls, path: str | Path, prefer_cuda: bool = False) -> "MLPModel":
        import torch

        device = get_device(prefer_cuda)
        payload = torch.load(path, map_location=device, weights_only=False)
        model = cls(**payload["config"])
        model.device = device
        model._ensure_built()
        model.model.load_state_dict(payload["state_dict"])
        model.model.eval()
        return model
