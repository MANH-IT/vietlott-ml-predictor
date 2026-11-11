"""Xây dựng ma trận multi-hot, cửa sổ trượt và đặc trưng cho từng mô hình."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MIN_HISTORY, WINDOW_SIZE, lottery_config
from .data_collection import load_canonical
from .data_validation import Draw

FEATURE_NAMES = [
    "freq_5",
    "freq_10",
    "freq_20",
    "freq_50",
    "gap",
    "is_last",
    "freq_all",
    "avg_gap",
    "n_appear",
]


@dataclass
class SequenceDataset:
    X: np.ndarray  # (n_samples, window, n_features)
    y: np.ndarray  # (n_samples, n_features)
    target_index: np.ndarray  # vị trí kỳ đích trong ma trận gốc


@dataclass
class TabularDataset:
    X: np.ndarray  # (n_samples * n_features, n_features_engineered)
    y: np.ndarray  # (n_samples * n_features,)
    target_index: np.ndarray  # (n_samples * n_features,) kỳ đích lặp lại
    number: np.ndarray  # (n_samples * n_features,) số tương ứng 1..F
    feature_names: list[str]


def load_matrix(lottery: str) -> tuple[list[Draw], np.ndarray]:
    """Trả về danh sách Draw và ma trận multi-hot (n_draws, n_features)."""
    cfg = lottery_config(lottery)
    draws = load_canonical(lottery)
    n_features = cfg["n_features"]
    matrix = np.zeros((len(draws), n_features), dtype=np.float32)
    for row, draw in enumerate(draws):
        for number in draw.numbers:
            matrix[row, number - 1] = 1.0
    return draws, matrix


def make_sequences(
    matrix: np.ndarray,
    window: int = WINDOW_SIZE,
    min_target: int = MIN_HISTORY,
) -> SequenceDataset:
    """Tạo cửa sổ trượt: X = window kỳ trước, y = kỳ kế tiếp."""
    n_draws, n_features = matrix.shape
    targets = np.arange(max(window, min_target), n_draws)
    if len(targets) == 0:
        return SequenceDataset(
            np.empty((0, window, n_features), dtype=np.float32),
            np.empty((0, n_features), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    X = np.stack([matrix[t - window : t] for t in targets]).astype(np.float32)
    y = matrix[targets].astype(np.float32)
    return SequenceDataset(X, y, targets.astype(np.int64))


def _features_at(
    matrix: np.ndarray, t: int, last_seen: np.ndarray, cumulative: np.ndarray
) -> np.ndarray:
    """Đặc trưng của từng số tại thời điểm t (dùng lịch sử 0..t)."""

    def window_sum(size: int) -> np.ndarray:
        start = max(0, t - size + 1)
        return matrix[start : t + 1].sum(axis=0)

    freq_5 = window_sum(5)
    freq_10 = window_sum(10)
    freq_20 = window_sum(20)
    freq_50 = window_sum(50)
    is_last = matrix[t]
    freq_all = cumulative[t] / float(t + 1)
    gap = np.where(last_seen >= 0, t - last_seen, t + 1).astype(np.float32)
    n_appear = cumulative[t]
    avg_gap = (t + 1.0) / (n_appear + 1.0)
    return np.column_stack(
        [
            freq_5 / 5.0,
            freq_10 / 10.0,
            freq_20 / 20.0,
            freq_50 / 50.0,
            gap / 50.0,
            is_last,
            freq_all,
            avg_gap / 50.0,
            n_appear / 50.0,
        ]
    ).astype(np.float32)


def make_tabular(
    matrix: np.ndarray, min_history: int = MIN_HISTORY
) -> TabularDataset:
    """Tạo đặc trưng dạng bảng: mỗi (kỳ, số) là một mẫu phân loại nhị phân."""
    n_draws, n_features = matrix.shape
    cumulative = np.cumsum(matrix, axis=0)
    last_seen = np.full(n_features, -1, dtype=np.int64)

    feature_rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    target_indices: list[int] = []

    for t in range(n_draws - 1):
        last_seen = np.where(matrix[t] > 0, t, last_seen)
        if t < min_history - 1:
            continue
        feature_rows.append(_features_at(matrix, t, last_seen, cumulative))
        targets.append(matrix[t + 1])
        target_indices.append(t + 1)

    n_samples = len(feature_rows)
    if n_samples == 0:
        return TabularDataset(
            np.empty((0, len(FEATURE_NAMES)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            FEATURE_NAMES,
        )

    X = np.stack(feature_rows)  # (n_samples, F, n_feat)
    y = np.stack(targets)  # (n_samples, F)
    tgt = np.asarray(target_indices, dtype=np.int64)
    numbers = np.tile(np.arange(1, n_features + 1, dtype=np.int64), n_samples)
    return TabularDataset(
        X=X.reshape(-1, X.shape[-1]).astype(np.float32),
        y=y.reshape(-1).astype(np.float32),
        target_index=np.repeat(tgt, n_features),
        number=numbers,
        feature_names=FEATURE_NAMES,
    )


def prediction_features(matrix: np.ndarray) -> np.ndarray:
    """Đặc trưng tại kỳ gần nhất để dự đoán kỳ kế tiếp (chưa biết kết quả)."""
    n_draws, n_features = matrix.shape
    cumulative = np.cumsum(matrix, axis=0)
    last_seen = np.full(n_features, -1, dtype=np.int64)
    for t in range(n_draws):
        last_seen = np.where(matrix[t] > 0, t, last_seen)
    return _features_at(matrix, n_draws - 1, last_seen, cumulative)


def prediction_window(matrix: np.ndarray, window: int = WINDOW_SIZE) -> np.ndarray:
    """Cửa sổ cuối cùng để dự đoán kỳ kế tiếp: (1, window, F)."""
    if len(matrix) < window:
        raise ValueError(f"Cần tối thiểu {window} kỳ, hiện có {len(matrix)}")
    return matrix[-window:][None, :, :].astype(np.float32)


def split_by_target(
    target_index: np.ndarray, train_ratio: float
) -> tuple[np.ndarray, np.ndarray]:
    """Chia train/test theo thời gian, trả về mask train và test."""
    unique_targets = np.unique(target_index)
    split_pos = int(len(unique_targets) * train_ratio) - 1
    if not 0 <= split_pos < len(unique_targets):
        raise ValueError(
            f"Không đủ dữ liệu để chia train/test với ratio {train_ratio} "
            f"({len(unique_targets)} kỳ đích)"
        )
    cutoff = unique_targets[split_pos]
    train_mask = target_index <= cutoff
    test_mask = target_index > cutoff
    return train_mask, test_mask
