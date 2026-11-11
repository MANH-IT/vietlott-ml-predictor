from datetime import date, timedelta

import numpy as np

import vietlott_ml.training as training
from vietlott_ml.data_validation import Draw


def synthetic_lottery(n_draws=120, n_features=45, pick=6, seed=7):
    rng = np.random.default_rng(seed)
    matrix = np.zeros((n_draws, n_features), dtype=np.float32)
    draws = []
    start = date(2020, 1, 1)
    for row in range(n_draws):
        picks = sorted(rng.choice(n_features, size=pick, replace=False) + 1)
        matrix[row, np.array(picks) - 1] = 1.0
        draws.append(
            Draw(
                lottery="mega645",
                draw_id=f"{row:05d}",
                date=(start + timedelta(days=row)).isoformat(),
                numbers=tuple(int(n) for n in picks),
            )
        )
    return draws, matrix


def test_prepare_split_and_actuals(monkeypatch):
    draws, matrix = synthetic_lottery()
    monkeypatch.setattr(training, "load_matrix", lambda lottery: (draws, matrix))

    data = training._prepare("mega645", window=10, train_ratio=0.8)

    assert data.n_features == 45
    assert len(data.actuals) == len(data.test_targets)
    assert data.test_targets.min() > data.cutoff
    assert data.sequence.target_index.max() == len(draws) - 1


def test_random_baseline_multiseed_has_ci(monkeypatch):
    draws, matrix = synthetic_lottery()
    monkeypatch.setattr(training, "load_matrix", lambda lottery: (draws, matrix))

    data = training._prepare("mega645", window=10, train_ratio=0.8)
    baseline = training._random_baseline(data, k=6, n_repeats=10)

    assert baseline["n_test"] == len(data.test_targets) * 10
    assert baseline["ci95_low"] <= baseline["avg_hits"] <= baseline["ci95_high"]
    assert abs(baseline["avg_hits"] - 6 * 6 / 45) < 0.1
