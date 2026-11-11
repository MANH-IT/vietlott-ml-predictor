from datetime import date, timedelta

import numpy as np
import pytest

import vietlott_ml.backtest as backtest
import vietlott_ml.hyperparams as hyperparams
import vietlott_ml.training as training
from vietlott_ml.config import RANDOM_BASELINE_REPEATS
from vietlott_ml.data_validation import Draw
from vietlott_ml.metrics import holm_bonferroni


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


def test_holm_bonferroni_monotonic_and_capped():
    adjusted = holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.03})

    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["c"] == pytest.approx(0.06)
    assert adjusted["b"] == pytest.approx(0.06)
    assert all(0.0 <= value <= 1.0 for value in adjusted.values())
    assert holm_bonferroni({}) == {}


def test_folds_expand_and_cover_tail():
    targets = np.arange(30, 120)
    folds = backtest._folds(targets, n_splits=4, initial_train_ratio=0.6)

    assert len(folds) == 4
    assert all(folds[i][0] < folds[i + 1][0] for i in range(3))
    covered = np.concatenate([fold_targets for _, fold_targets in folds])
    assert np.array_equal(covered, targets[int(len(targets) * 0.6) :])


def test_baseline_predictions():
    matrix = np.zeros((4, 6), dtype=np.float32)
    matrix[0, [0, 1]] = 1
    matrix[1, [2, 3]] = 1
    matrix[2, [0, 4]] = 1
    matrix[3, [1, 5]] = 1
    targets = np.array([3])

    assert backtest.baseline_predictions(matrix, targets, 2, "repeat_last") == [[1, 5]]
    assert backtest.baseline_predictions(matrix, targets, 2, "frequency") == [[1, 2]]
    assert backtest.baseline_predictions(
        matrix, targets, 2, "frequency_train", cutoff=1
    ) == [[1, 2]]
    with pytest.raises(ValueError):
        backtest.baseline_predictions(matrix, targets, 2, "unknown")


def test_walk_forward_writes_reports(monkeypatch, tmp_path):
    draws, matrix = synthetic_lottery()
    monkeypatch.setattr(training, "load_matrix", lambda lottery: (draws, matrix))
    monkeypatch.setattr(
        hyperparams, "tuned_path", lambda lottery: tmp_path / "missing.json"
    )
    monkeypatch.setattr(backtest, "REPORTS_DIR", tmp_path)

    payload = backtest.walk_forward_lottery(
        "mega645",
        model_names=["markov", "logistic"],
        n_splits=2,
        initial_train_ratio=0.6,
        verbose=False,
    )

    assert payload["n_splits"] == 2
    assert len(payload["folds"]) == 2
    assert payload["folds"][0]["train_end_index"] < payload["folds"][1]["train_end_index"]
    n_test = sum(fold["n_test"] for fold in payload["folds"])
    expected_models = {
        "markov",
        "logistic",
        "random",
        "repeat_last",
        "frequency",
        "frequency_train",
    }
    assert expected_models <= set(payload["models"])
    for name in expected_models - {"random"}:
        assert payload["models"][name]["metrics"]["n_test"] == n_test
    assert payload["models"]["random"]["metrics"]["n_test"] == (
        n_test * RANDOM_BASELINE_REPEATS
    )
    assert payload["multiple_comparison"]["n_tests"] == 5
    assert "p_value_holm" in payload["models"]["markov"]["metrics"]
    assert "p_value_holm" in payload["models"]["logistic"]["metrics"]

    assert (tmp_path / "mega645_walkforward_metrics.json").exists()
    assert (tmp_path / "mega645_walkforward_report.md").exists()
