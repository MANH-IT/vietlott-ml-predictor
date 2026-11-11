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


def hits_from_predictions(
    predictions: list[list[int]], actuals: list[list[int]]
) -> np.ndarray:
    """Số số trúng cho từng kỳ (1-based)."""
    if len(predictions) != len(actuals):
        raise ValueError("predictions và actuals phải cùng độ dài")
    return np.array(
        [len(set(p) & set(a)) for p, a in zip(predictions, actuals, strict=True)], dtype=float
    )


def summarize_hits(hits: np.ndarray, n_features: int, k: int = 6) -> dict:
    """Tóm tắt phân bố số trúng và kiểm định so với kỳ vọng ngẫu nhiên.

    Nhận thẳng mảng hits để có thể gộp nhiều lần lặp (ví dụ nhiều seed của
    baseline ngẫu nhiên) thành một phân bố chung.
    """
    hits = np.asarray(hits, dtype=float)
    if len(hits) == 0:
        return {"n_test": 0}
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

    # Khoảng tin cậy 95% cho avg_hits (xấp xỉ chuẩn).
    sample_std_error = std_hits / math.sqrt(len(hits)) if len(hits) > 1 else 0.0
    ci95_low = mean_hits - 1.96 * sample_std_error
    ci95_high = mean_hits + 1.96 * sample_std_error

    result = {
        "n_test": int(len(hits)),
        "avg_hits": mean_hits,
        "median_hits": float(np.median(hits)),
        "std_hits": std_hits,
        "std_error_hits": sample_std_error,
        "ci95_low": float(ci95_low),
        "ci95_high": float(ci95_high),
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
    return result


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

    result = summarize_hits(hits_from_predictions(predictions, actuals), n_features, k)

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


def holm_bonferroni(p_values: dict[str, float]) -> dict[str, float]:
    """Hiệu chỉnh Holm–Bonferroni cho nhiều phép so sánh.

    Trả về p-value đã hiệu chỉnh (cùng khóa). Thứ tự xử lý theo p tăng dần;
    giá trị hiệu chỉnh được giữ đơn điệu không giảm để đảm bảo tính hợp lệ.
    """
    items = sorted(p_values.items(), key=lambda item: item[1])
    total = len(items)
    adjusted: dict[str, float] = {}
    previous = 0.0
    for rank, (name, p_value) in enumerate(items):
        value = min(1.0, (total - rank) * float(p_value))
        value = max(value, previous)
        adjusted[name] = value
        previous = value
    return adjusted


def compare_report(results: dict[str, dict]) -> str:
    """Tạo bảng markdown so sánh các mô hình.

    Nếu metrics có ``p_value_holm`` (đã hiệu chỉnh đa so sánh) thì hiển thị
    thêm cột và dùng nó cho cột "Có ý nghĩa".
    """
    header = (
        "| Mô hình | Avg hits | 95% CI | Lift | Precision@6 | p-value | p Holm | "
        "Có ý nghĩa (Holm 5%) |\n"
        "|---|---:|---:|---:|---:|---:|---:|:---:|\n"
    )
    rows = []
    for name, metrics in results.items():
        ci = (
            f"[{metrics['ci95_low']:.3f}, {metrics['ci95_high']:.3f}]"
            if "ci95_low" in metrics
            else "-"
        )
        p_holm = metrics.get("p_value_holm")
        p_holm_text = f"{p_holm:.3f}" if p_holm is not None else "-"
        significant = metrics.get(
            "significant_holm_5pct", metrics.get("significant_5pct", False)
        )
        rows.append(
            f"| {name} | {metrics['avg_hits']:.3f} | {ci} | "
            f"{metrics['lift_vs_random']:.3f} | "
            f"{metrics['precision_at_6']:.3f} | "
            f"{metrics['p_value']:.3f} | "
            f"{p_holm_text} | "
            f"{'Có' if significant else 'Không'} |"
        )
    return header + "\n".join(rows) + "\n"
