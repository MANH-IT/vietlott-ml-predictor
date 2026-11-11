"""Registry các mô hình và tiện ích lưu/đọc."""

from __future__ import annotations

from pathlib import Path

from ..config import model_dir
from ..hyperparams import arch_params, resolve_params
from .forest_model import ForestModel
from .logistic_model import LogisticModel
from .lstm_model import LSTMModel
from .markov_model import MarkovModel
from .mlp_model import MLPModel
from .xgboost_model import XGBoostModel

MODEL_NAMES = ["lstm", "bilstm", "logistic", "random_forest", "markov", "mlp", "xgboost"]

DISPLAY_NAMES = {
    "lstm": "LSTM",
    "bilstm": "BiLSTM",
    "logistic": "Logistic Regression",
    "random_forest": "Random Forest",
    "markov": "Markov",
    "mlp": "MLP",
    "xgboost": "XGBoost",
    "random": "Random baseline",
    "repeat_last": "Lặp kỳ trước",
    "frequency": "Top tần suất (mở rộng)",
    "frequency_train": "Top tần suất (train)",
}

SEQUENCE_MODELS = {"lstm", "bilstm"}
TABULAR_MODELS = {"logistic", "random_forest", "xgboost"}
TORCH_MODELS = {"lstm", "bilstm", "mlp"}


def build_model(
    name: str,
    n_features: int,
    seed: int = 42,
    params: dict | None = None,
    lottery: str | None = None,
):
    """Tạo model. ``params`` mặc định lấy từ DEFAULT_PARAMS + kết quả tune."""
    resolved = resolve_params(name, lottery) if params is None else dict(params)
    kwargs = arch_params(name, resolved)
    if name == "lstm":
        return LSTMModel(n_features=n_features, seed=seed, **kwargs)
    if name == "bilstm":
        return LSTMModel(n_features=n_features, seed=seed, **kwargs)
    if name == "logistic":
        return LogisticModel(n_features=n_features, seed=seed, **kwargs)
    if name == "random_forest":
        return ForestModel(n_features=n_features, seed=seed, **kwargs)
    if name == "markov":
        return MarkovModel(n_features=n_features, seed=seed, **kwargs)
    if name == "mlp":
        return MLPModel(n_features=n_features, seed=seed, **kwargs)
    if name == "xgboost":
        return XGBoostModel(n_features=n_features, seed=seed, **kwargs)
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
    if name == "xgboost":
        return XGBoostModel.load(path)
    raise ValueError(f"Mô hình không hỗ trợ: {name}")
