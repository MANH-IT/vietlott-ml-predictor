"""LSTM / BiLSTM cho dữ liệu multi-hot cửa sổ trượt."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import TorchBinaryModel, get_device


class _LSTMNet:
    @staticmethod
    def build(n_features: int, hidden1: int, hidden2: int, dropout: float, bidirectional: bool):
        import torch
        from torch import nn

        directions = 2 if bidirectional else 1

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm1 = nn.LSTM(
                    n_features,
                    hidden1,
                    batch_first=True,
                    bidirectional=bidirectional,
                )
                self.drop1 = nn.Dropout(dropout)
                self.lstm2 = nn.LSTM(
                    hidden1 * directions,
                    hidden2,
                    batch_first=True,
                    bidirectional=bidirectional,
                )
                self.drop2 = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden2 * directions, n_features)

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                out, _ = self.lstm1(x)
                out = self.drop1(out)
                out, _ = self.lstm2(out)
                out = self.drop2(out[:, -1, :])
                return self.fc(out)

        return Net()


class LSTMModel(TorchBinaryModel):
    """LSTM hai lớp (tuỳ chọn Bidirectional) → sigmoid cho từng số."""

    arch = "lstm"

    def __init__(
        self,
        n_features: int,
        hidden1: int = 128,
        hidden2: int = 64,
        dropout: float = 0.2,
        bidirectional: bool = False,
        seed: int = 42,
    ):
        super().__init__(n_features=n_features, seed=seed)
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.dropout = dropout
        self.bidirectional = bidirectional

    @property
    def name(self) -> str:
        return "bilstm" if self.bidirectional else "lstm"

    def _build(self):
        return _LSTMNet.build(
            self.n_features,
            self.hidden1,
            self.hidden2,
            self.dropout,
            self.bidirectional,
        )

    def get_config(self) -> dict:
        return {
            "n_features": self.n_features,
            "hidden1": self.hidden1,
            "hidden2": self.hidden2,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional,
            "seed": self.seed,
        }

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Chưa huấn luyện, không thể lưu")
        super().save(path)

    @classmethod
    def load(cls, path: str | Path, prefer_cuda: bool = False) -> "LSTMModel":
        import torch

        device = get_device(prefer_cuda)
        payload = torch.load(path, map_location=device, weights_only=False)
        model = cls(**payload["config"])
        model.device = device
        model._ensure_built()
        model.model.load_state_dict(payload["state_dict"])
        model.model.eval()
        return model
