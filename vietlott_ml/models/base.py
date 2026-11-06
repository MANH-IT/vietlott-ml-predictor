"""Hạ tầng dùng chung cho các mô hình PyTorch và sklearn."""

from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np


def positive_probability(clf, X: np.ndarray) -> np.ndarray:
    """Lấy xác suất lớp dương, xử lý trường hợp chỉ có một lớp."""
    classes = list(getattr(clf, "classes_", []))
    if not classes:
        return np.zeros(len(X), dtype=float)
    if len(classes) == 1:
        value = 1.0 if classes[0] == 1 else 0.0
        return np.full(len(X), value, dtype=float)
    index = classes.index(1)
    return clf.predict_proba(X)[:, index]


def get_device(prefer_cuda: bool = True):
    import torch

    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TorchBinaryModel:
    """Mô hình neural network với đầu ra sigmoid cho từng số (multi-label)."""

    arch = "torch"

    def __init__(self, n_features: int, seed: int = 42):
        self.n_features = n_features
        self.seed = seed
        self.model = None
        self.device = get_device()

    def _build(self):
        raise NotImplementedError

    def get_config(self) -> dict:
        raise NotImplementedError

    def _ensure_built(self) -> None:
        if self.model is None:
            set_seed(self.seed)
            self.model = self._build().to(self.device)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_fraction: float = 0.1,
        epochs: int = 120,
        batch_size: int = 64,
        lr: float = 1e-3,
        patience: int = 15,
        weight_decay: float = 1e-5,
        verbose: bool = False,
    ) -> "TorchBinaryModel":
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        self._ensure_built()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        n = len(X)
        if n == 0:
            raise ValueError("Không có dữ liệu huấn luyện")

        val_size = max(1, int(n * val_fraction)) if n >= 10 else 0
        split = n - val_size
        train_ds = TensorDataset(
            torch.from_numpy(X[:split]), torch.from_numpy(y[:split])
        )
        loader = DataLoader(
            train_ds, batch_size=min(batch_size, max(1, split)), shuffle=True
        )

        if val_size:
            X_val = torch.from_numpy(X[split:]).to(self.device)
            y_val = torch.from_numpy(y[split:]).to(self.device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        best_state = copy.deepcopy(self.model.state_dict())
        best_loss = float("inf")
        wait = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            epoch_loss /= max(1, split)

            if val_size:
                self.model.eval()
                with torch.no_grad():
                    val_loss = criterion(self.model(X_val), y_val).item()
                monitor = val_loss
            else:
                monitor = epoch_loss

            if monitor < best_loss - 1e-6:
                best_loss = monitor
                best_state = copy.deepcopy(self.model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print(f"  early stop tại epoch {epoch}, best={best_loss:.5f}")
                    break
            if verbose and epoch % 20 == 0:
                print(f"  epoch {epoch}: train={epoch_loss:.5f} monitor={monitor:.5f}")

        self.model.load_state_dict(best_state)
        self.model.eval()
        self.best_loss = best_loss
        return self

    def predict_proba(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        import torch

        if self.model is None:
            raise RuntimeError("Mô hình chưa được huấn luyện")
        self.model.eval()
        X = np.asarray(X, dtype=np.float32)
        outputs = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                batch = torch.from_numpy(X[start : start + batch_size]).to(self.device)
                outputs.append(torch.sigmoid(self.model(batch)).cpu().numpy())
        if not outputs:
            return np.empty((0, self.n_features), dtype=np.float32)
        return np.vstack(outputs)

    def save(self, path: str | Path) -> None:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"config": self.get_config(), "state_dict": self.model.state_dict()},
            path,
        )

    def _load_state(self, path: str | Path) -> None:
        import torch

        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.__init__(**payload["config"])
        self._ensure_built()
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
