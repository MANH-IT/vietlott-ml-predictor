import numpy as np

from vietlott_ml.metrics import evaluate_predictions, top_k
from vietlott_ml.models import build_model
from vietlott_ml.models.markov_model import MarkovModel


def random_matrix(n_draws=120, n_features=6, k=2, seed=3):
    rng = np.random.default_rng(seed)
    matrix = np.zeros((n_draws, n_features), dtype=np.float32)
    for row in range(n_draws):
        matrix[row, rng.choice(n_features, size=k, replace=False)] = 1.0
    return matrix


def test_top_k():
    scores = np.array([0.1, 0.9, 0.5, 0.8, 0.2, 0.7])
    assert top_k(scores, 2) == [2, 4]
    assert sorted(top_k(scores, 3)) == [2, 4, 6]


def test_markov_learns_transition():
    matrix = np.zeros((40, 6), dtype=np.float32)
    for t in range(39):
        matrix[t, 0] = 1.0
        matrix[t + 1, 1] = 1.0
    matrix[0, 0] = 1.0
    model = MarkovModel(n_features=6).fit(matrix)
    scores = model.predict_proba(matrix, np.array([40]))[0]
    assert set(top_k(scores, 2)) == {1, 2}
    assert abs(scores.sum() - 1.0) < 1e-9


def test_logistic_shapes():
    matrix = random_matrix()
    from vietlott_ml.dataset import make_tabular

    dataset = make_tabular(matrix, min_history=20)
    model = build_model("logistic", 6).fit(dataset.X, dataset.y)
    probabilities = model.predict_proba(dataset.X[:60])
    assert probabilities.shape == (10, 6)
    assert np.all(probabilities >= 0) and np.all(probabilities <= 1)


def test_mlp_smoke():
    from vietlott_ml.dataset import make_sequences

    matrix = random_matrix(n_draws=80)
    dataset = make_sequences(matrix, window=5, min_target=20)
    model = build_model("mlp", 6)
    import torch

    model.device = torch.device("cpu")
    model.fit(dataset.X[:, -1, :], dataset.y, epochs=1, verbose=False)
    probabilities = model.predict_proba(dataset.X[:, -1, :])
    assert probabilities.shape == dataset.y.shape


def test_lstm_smoke():
    from vietlott_ml.dataset import make_sequences

    matrix = random_matrix(n_draws=80)
    dataset = make_sequences(matrix, window=5, min_target=20)
    model = build_model("lstm", 6)
    model.hidden1 = 8
    model.hidden2 = 4
    import torch

    model.device = torch.device("cpu")
    model.fit(dataset.X, dataset.y, epochs=1, verbose=False)
    probabilities = model.predict_proba(dataset.X)
    assert probabilities.shape == dataset.y.shape


def test_metrics_perfect_predictions():
    actuals = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
    metrics = evaluate_predictions(
        predictions=actuals, actuals=actuals, n_features=45, k=6
    )
    assert metrics["avg_hits"] == 6.0
    assert metrics["lift_vs_random"] == 6.0 / (36 / 45)
    assert metrics["n_test"] == 2


def test_metrics_no_hits():
    actuals = [[1, 2, 3, 4, 5, 6]] * 10
    predictions = [[7, 8, 9, 10, 11, 12]] * 10
    metrics = evaluate_predictions(predictions, actuals, n_features=45, k=6)
    assert metrics["avg_hits"] == 0.0
    assert metrics["p_value"] < 0.05
    assert not metrics["significant_5pct"]
