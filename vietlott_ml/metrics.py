"""Đánh giá trung thực hiệu suất mô hình so với baseline ngẫu nhiên."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np


def top_k(scores: np.ndarray, k: int) -> list[int]:
    """Chọn k số có điểm/xác suất cao nhất (trả về số 1-based)."""
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="stable")[:k]
    return [int(i) + 1 for i in order]


def _at_least_one_probability(n_features: int, k: int) -> float:
    if n_features - k < k:
        return 1.0
    return 1.0 - math.comb(n_features - k, k) / math.comb(n_features, k)


def evaluate_predictions(
    predictions: list[list[int]],
    actuals: list[list[int]],
    n_features: int,
    probabilities: np.ndarray | None = None,
    targets: np.ndarray | None = None,
    k: int = 6,
) -> dict:
    """Tính hit rate, lift và kiểm định so với kỳ vọng ngẫu nhiên.

    predictions/actuals: danh sách bộ số (1-based) cho từng kỳ test.
    probabilities/targets: ma trận xác suất và nhãn nhị phân (nếu có) để
    tính Brier score và log-loss.
    """
    if len(predictions) != len(actuals):
        raise ValueError("predictions và actuals phải cùng độ dài")
    if not predictions:
        return {"n_test": 0}

    hits = np.array(
        [len(set(p) & set(a)) for p, a in zip(predictions, actuals)], dtype=float
    )
    mean_hits = float(hits.mean())
    expected_hits = k * k / n_features
    std_hits = float(hits.std(ddof=1)) if len(hits) > 1 else 0.0

    # Phương sai lý thuyết của số trúng khi chọn ngẫu nhiên k số từ F số
    # (phân phối siêu bội), dùng để kiểm định so với kỳ vọng ngẫu nhiên.
    if n_features > 1 and n_features > k:
        hyper_var = (
            k
            * (k / n_features)
            * ((n_features - k) / n_features)
            * ((n_features - k) / (n_features - 1))
        )
    else:
        hyper_var = 0.0
    stderr = math.sqrt(hyper_var / len(hits)) if hyper_var > 0 else 0.0
    z_score = (mean_hits - expected_hits) / stderr if stderr > 0 else 0.0
    p_value = (
        2.0 * (1.0 - NormalDist().cdf(abs(z_score))) if stderr > 0 else 1.0
    )

    result = {
        "n_test": int(len(hits)),
        "avg_hits": mean_hits,
        "median_hits": float(np.median(hits)),
        "std_hits": std_hits,
        "precision_at_6": mean_hits / k,
        "recall_at_6": mean_hits / k,
        "expected_random_hits": expected_hits,
        "lift_vs_random": mean_hits / expected_hits,
        "z_score": float(z_score),
        "p_value": float(p_value),
        "significant_5pct": bool(p_value < 0.05 and mean_hits > expected_hits),
        "coverage_at_least_one": float((hits >= 1).mean()),
        "expected_coverage": _at_least_one_probability(n_features, k),
        "hit_distribution": {
            str(i): int((hits == i).sum()) for i in range(k + 1)
        },
        "max_hits": int(hits.max()),
    }

    if probabilities is not None and targets is not None:
        probs = np.asarray(probabilities, dtype=float).ravel()
        labels = np.asarray(targets, dtype=float).ravel()
        if len(probs) != len(labels):
            raise ValueError("probabilities và targets phải cùng kích thước")
        probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
        result["brier_score"] = float(np.mean((probs - labels) ** 2))
        result["log_loss"] = float(
            -np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs))
        )

    return result


def compare_report(results: dict[str, dict]) -> str:
    """Tạo bảng markdown so sánh các mô hình."""
    header = (
        "| Mô hình | Avg hits | Lift | Precision@6 | p-value | "
        "Có ý nghĩa (5%) |\n"
        "|---|---:|---:|---:|---:|:---:|\n"
    )
    rows = []
    for name, metrics in results.items():
        rows.append(
            f"| {name} | {metrics['avg_hits']:.3f} | "
            f"{metrics['lift_vs_random']:.3f} | "
            f"{metrics['precision_at_6']:.3f} | "
            f"{metrics['p_value']:.3f} | "
            f"{'Có' if metrics['significant_5pct'] else 'Không'} |"
        )
    return header + "\n".join(rows) + "\n"
