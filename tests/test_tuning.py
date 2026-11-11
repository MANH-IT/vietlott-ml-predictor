import json
from datetime import date, timedelta

import numpy as np

import vietlott_ml.hyperparams as hp
import vietlott_ml.training as training
import vietlott_ml.tuning as tuning
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


def test_candidates_cover_search_space():
    candidates = tuning._candidates("markov")
    assert len(candidates) == 9
    for params in candidates:
        assert set(hp.DEFAULT_PARAMS["markov"]) <= set(params)


def test_candidates_without_space_returns_default():
    candidates = tuning._candidates("lstm")
    assert len(candidates) == 8
    assert all(set(hp.DEFAULT_PARAMS["lstm"]) <= set(p) for p in candidates)


def test_resolve_params_merges_tuned(tmp_path, monkeypatch):
    path = tmp_path / "hyperparams.json"
    monkeypatch.setattr(hp, "tuned_path", lambda lottery: path)
    path.write_text(
        json.dumps({"models": {"markov": {"params": {"alpha": 2.0}}}}),
        encoding="utf-8",
    )
    params = hp.resolve_params("markov", "mega645")
    assert params["alpha"] == 2.0
    assert params["transition_weight"] == hp.DEFAULT_PARAMS["markov"]["transition_weight"]


def test_resolve_params_without_tuned_file(tmp_path, monkeypatch):
    monkeypatch.setattr(hp, "tuned_path", lambda lottery: tmp_path / "missing.json")
    params = hp.resolve_params("logistic", "mega645")
    assert params == hp.DEFAULT_PARAMS["logistic"]


def test_tune_lottery_markov_small(tmp_path, monkeypatch):
    draws, matrix = synthetic_lottery()
    monkeypatch.setattr(training, "load_matrix", lambda lottery: (draws, matrix))
    path = tmp_path / "hyperparams.json"
    monkeypatch.setattr(hp, "tuned_path", lambda lottery: path)
    monkeypatch.setitem(hp.SEARCH_SPACES, "markov", {"alpha": [0.5, 1.0]})

    result = tuning.tune_lottery(
        "mega645", model_names=["markov"], val_ratio=0.2, verbose=False
    )

    assert "markov" in result["models"]
    assert result["models"]["markov"]["n_candidates"] == 2
    assert path.exists()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["models"]["markov"]["params"]["alpha"] in (0.5, 1.0)


def test_tune_lottery_logistic_small(tmp_path, monkeypatch):
    draws, matrix = synthetic_lottery()
    monkeypatch.setattr(training, "load_matrix", lambda lottery: (draws, matrix))
    path = tmp_path / "hyperparams.json"
    monkeypatch.setattr(hp, "tuned_path", lambda lottery: path)
    monkeypatch.setitem(hp.SEARCH_SPACES, "logistic", {"C": [0.5]})

    result = tuning.tune_lottery(
        "mega645", model_names=["logistic"], val_ratio=0.2, verbose=False
    )

    assert result["models"]["logistic"]["params"]["C"] == 0.5
