"""Hyperparameters mặc định, không gian tìm kiếm và lưu tham số đã tune.

Tách rõ hai loại tham số:
- Tham số kiến trúc (truyền vào constructor của model).
- Tham số huấn luyện (chỉ dùng khi gọi ``fit`` của model PyTorch).
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import MODELS_DIR

DEFAULT_PARAMS: dict[str, dict] = {
    "lstm": {
        "hidden1": 128,
        "hidden2": 64,
        "dropout": 0.2,
        "bidirectional": False,
        "pooling": "auto",
        "lr": 1e-3,
        "epochs": 120,
        "batch_size": 64,
        "patience": 15,
        "weight_decay": 1e-5,
        "val_fraction": 0.1,
    },
    "bilstm": {
        "hidden1": 128,
        "hidden2": 64,
        "dropout": 0.2,
        "bidirectional": True,
        "pooling": "auto",
        "lr": 1e-3,
        "epochs": 120,
        "batch_size": 64,
        "patience": 15,
        "weight_decay": 1e-5,
        "val_fraction": 0.1,
    },
    "mlp": {
        "hidden1": 256,
        "hidden2": 128,
        "dropout": 0.2,
        "lr": 1e-3,
        "epochs": 120,
        "batch_size": 64,
        "patience": 15,
        "weight_decay": 1e-5,
        "val_fraction": 0.1,
    },
    "logistic": {"C": 1.0, "max_iter": 1000},
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 12,
        "min_samples_leaf": 10,
        "n_jobs": 1,
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": 1,
    },
    "markov": {"alpha": 1.0, "transition_weight": 0.8},
}

# Không gian tìm kiếm khi chạy ``tune``. Giữ nhỏ để chạy được trên CPU.
SEARCH_SPACES: dict[str, dict] = {
    "lstm": {
        "hidden1": [64, 128],
        "dropout": [0.15, 0.3],
        "lr": [1e-3, 5e-4],
    },
    "bilstm": {
        "hidden1": [64, 128],
        "dropout": [0.15, 0.3],
        "lr": [1e-3, 5e-4],
    },
    "mlp": {
        "hidden1": [128, 256],
        "hidden2": [64, 128],
        "dropout": [0.1, 0.3],
    },
    "logistic": {"C": [0.1, 1.0, 10.0]},
    "random_forest": {
        "n_estimators": [100, 200],
        "max_depth": [8, 12, 20],
        "min_samples_leaf": [5, 10],
    },
    "xgboost": {
        "n_estimators": [100, 200],
        "max_depth": [4, 6],
        "subsample": [0.8],
    },
    "markov": {"alpha": [0.5, 1.0, 2.0], "transition_weight": [0.5, 0.8, 0.95]},
}

# Tham số chỉ dùng khi huấn luyện mô hình PyTorch.
FIT_KEYS = {"lr", "epochs", "batch_size", "patience", "weight_decay", "val_fraction"}

# Rút ngắn huấn luyện khi tune để giới hạn thời gian (không lưu vào kết quả).
TUNING_FIT_OVERRIDES = {"epochs": 60, "patience": 8}


def arch_params(name: str, params: dict) -> dict:
    """Lọc ra các tham số kiến trúc (truyền cho constructor)."""
    return {k: v for k, v in params.items() if k not in FIT_KEYS}


def fit_params(name: str, params: dict) -> dict:
    """Lọc ra các tham số huấn luyện (truyền cho ``fit`` của mô hình PyTorch)."""
    return {k: v for k, v in params.items() if k in FIT_KEYS}


def tuned_path(lottery: str) -> Path:
    return MODELS_DIR / lottery / "hyperparams.json"


def load_tuned(lottery: str) -> dict:
    path = tuned_path(lottery)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_tuned(lottery: str, payload: dict) -> Path:
    path = tuned_path(lottery)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def resolve_params(name: str, lottery: str | None = None) -> dict:
    """Gộp tham số mặc định với tham số đã tune (nếu có) cho một mô hình."""
    params = dict(DEFAULT_PARAMS[name])
    if lottery:
        entry = load_tuned(lottery).get("models", {}).get(name, {})
        params.update(entry.get("params", {}))
    return params
