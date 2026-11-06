"""Registry các mô hình và tiện ích lưu/đọc."""

from __future__ import annotations

from pathlib import Path

from ..config import model_dir
from .forest_model import ForestModel
from .logistic_model import LogisticModel
from .lstm_model import LSTMModel
from .markov_model import MarkovModel
from .mlp_model import MLPModel

MODEL_NAMES = ["lstm", "bilstm", "logistic", "random_forest", "markov", "mlp"]

DISPLAY_NAMES = {
    "lstm": "LSTM",
    "bilstm": "BiLSTM",
    "logistic": "Logistic Regression",
    "random_forest": "Random Forest",
    "markov": "Markov",
    "mlp": "MLP",
    "random": "Random baseline",
}

SEQUENCE_MODELS = {"lstm", "bilstm"}
TABULAR_MODELS = {"logistic", "random_forest"}
TORCH_MODELS = {"lstm", "bilstm", "mlp"}


def build_model(name: str, n_features: int, seed: int = 42):
    if name == "lstm":
        return LSTMModel(n_features=n_features, seed=seed)
    if name == "bilstm":
        return LSTMModel(n_features=n_features, bidirectional=True, seed=seed)
    if name == "logistic":
        return LogisticModel(n_features=n_features, seed=seed)
    if name == "random_forest":
        return ForestModel(n_features=n_features, seed=seed)
    if name == "markov":
        return MarkovModel(n_features=n_features, seed=seed)
    if name == "mlp":
        return MLPModel(n_features=n_features, seed=seed)
    raise ValueError(f"Mô hình không hỗ trợ: {name}")


def model_path(lottery: str, name: str) -> Path:
    suffix = ".pt" if name in TORCH_MODELS else ".joblib"
    return model_dir(lottery) / f"{name}{suffix}"


def save_model(model, lottery: str, name: str) -> Path:
    path = model_path(lottery, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    return path


def load_model(lottery: str, name: str):
    path = model_path(lottery, name)
    if not path.exists():
        raise FileNotFoundError(
            f"Chưa có mô hình '{name}' cho {lottery} tại {path}. "
            f"Hãy chạy: python -m vietlott_ml.cli train --lottery {lottery}"
        )
    if name == "lstm" or name == "bilstm":
        return LSTMModel.load(path)
    if name == "mlp":
        return MLPModel.load(path)
    if name == "logistic":
        return LogisticModel.load(path)
    if name == "random_forest":
        return ForestModel.load(path)
    if name == "markov":
        return MarkovModel.load(path)
    raise ValueError(f"Mô hình không hỗ trợ: {name}")
