"""Sinh bộ số gợi ý từ mô hình đã huấn luyện (chỉ phục vụ học tập)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from .config import DISCLAIMER, RANDOM_SEED, WINDOW_SIZE, lottery_config
from .dataset import (
    load_matrix,
    prediction_features,
    prediction_window,
)
from .metrics import top_k
from .models import (
    DISPLAY_NAMES,
    SEQUENCE_MODELS,
    TABULAR_MODELS,
    load_model,
    model_path,
)


def special_distribution(draws, n_features: int, window: int = 200) -> dict[int, float]:
    """Phân phối số đặc biệt (làm mượt Laplace) trên các kỳ gần nhất."""
    specials = [d.special for d in draws[-window:] if d.special is not None]
    counts = np.full(n_features, 1.0)
    for value in specials:
        counts[value - 1] += 1.0
    return {i + 1: float(counts[i] / counts.sum()) for i in range(n_features)}


def next_draw_date(lottery: str, latest: str) -> str | None:
    """Ước lượng ngày quay kế tiếp dựa trên lịch cố định."""
    cfg = lottery_config(lottery)
    current = date.fromisoformat(latest[:10])
    for step in range(1, 8):
        candidate = current + timedelta(days=step)
        if candidate.weekday() in cfg["draw_days"]:
            return candidate.isoformat()
    return None


def generate_numbers(
    lottery: str,
    model_name: str,
    window: int = WINDOW_SIZE,
    seed: int | None = None,
) -> dict:
    """Tạo bộ số gợi ý cho kỳ kế tiếp từ dữ liệu mới nhất."""
    cfg = lottery_config(lottery)
    n_features = cfg["n_features"]
    k = cfg["pick"]
    draws, matrix = load_matrix(lottery)
    if len(draws) < window:
        raise ValueError(
            f"{lottery}: cần tối thiểu {window} kỳ, hiện có {len(draws)}"
        )

    if model_name == "random":
        rng = np.random.default_rng(seed if seed is not None else RANDOM_SEED)
        probs = np.full(n_features, 1.0 / n_features)
        picked = sorted(
            (rng.choice(n_features, size=k, replace=False) + 1).tolist()
        )
        ranked = [
            {"number": number, "probability": 1.0 / n_features}
            for number in picked
        ]
    else:
        model = load_model(lottery, model_name)
        if model_name in SEQUENCE_MODELS:
            probs = model.predict_proba(prediction_window(matrix, window))[0]
        elif model_name == "mlp":
            probs = model.predict_proba(matrix[-1:])[0]
        elif model_name in TABULAR_MODELS:
            probs = model.predict_proba(prediction_features(matrix))[0]
        elif model_name == "markov":
            probs = model.predict_proba(matrix, [len(matrix)])[0]
        else:
            raise ValueError(f"Mô hình không hỗ trợ: {model_name}")
        picked = top_k(probs, k)
        ranked = sorted(
            [
                {"number": int(number), "probability": float(probs[number - 1])}
                for number in picked
            ],
            key=lambda item: -item["probability"],
        )

    numbers = sorted(picked)
    result = {
        "lottery": lottery,
        "lottery_name": cfg["name"],
        "model": model_name,
        "model_name": DISPLAY_NAMES.get(model_name, model_name),
        "numbers": numbers,
        "ranked": ranked,
        "latest_draw": {
            "draw_id": draws[-1].draw_id,
            "date": draws[-1].date,
            "numbers": list(draws[-1].numbers),
            "special": draws[-1].special,
        },
        "next_draw_estimate": next_draw_date(lottery, draws[-1].date),
        "n_draws_used": len(draws),
        "window": window,
        "disclaimer": DISCLAIMER,
    }

    if cfg["has_special"]:
        distribution = special_distribution(draws, n_features)
        ordered = sorted(distribution.items(), key=lambda item: -item[1])
        chosen = next(
            (
                (number, probability)
                for number, probability in ordered
                if number not in numbers
            ),
            ordered[0],
        )
        result["special"] = chosen[0]
        result["special_probability"] = chosen[1]
        result["special_top3"] = [
            {"number": number, "probability": probability}
            for number, probability in ordered[:3]
        ]

    return result
