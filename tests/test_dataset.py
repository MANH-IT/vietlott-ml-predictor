import numpy as np

from vietlott_ml.config import MIN_HISTORY
from vietlott_ml.dataset import (
    FEATURE_NAMES,
    make_sequences,
    make_tabular,
    prediction_features,
    prediction_window,
    split_by_target,
)


def synthetic_matrix(n_draws=80, n_features=10, seed=1):
    rng = np.random.default_rng(seed)
    matrix = np.zeros((n_draws, n_features), dtype=np.float32)
    for row in range(n_draws):
        picks = rng.choice(n_features, size=6, replace=False)
        matrix[row, picks] = 1.0
    return matrix


def test_sequence_shapes():
    matrix = synthetic_matrix()
    dataset = make_sequences(matrix, window=5, min_target=MIN_HISTORY)
    expected = len(matrix) - max(5, MIN_HISTORY)
    assert dataset.X.shape == (expected, 5, 10)
    assert dataset.y.shape == (expected, 10)
    assert dataset.target_index[0] == max(5, MIN_HISTORY)
    np.testing.assert_array_equal(dataset.X[0], matrix[MIN_HISTORY - 5 : MIN_HISTORY])


def test_tabular_shapes():
    matrix = synthetic_matrix()
    dataset = make_tabular(matrix, min_history=MIN_HISTORY)
    n_samples = len(matrix) - MIN_HISTORY
    assert dataset.X.shape == (n_samples * 10, len(FEATURE_NAMES))
    assert dataset.y.shape == (n_samples * 10,)
    assert dataset.number[0] == 1
    assert dataset.number[9] == 10


def test_no_future_leakage_in_features():
    matrix = synthetic_matrix()
    tampered = matrix.copy()
    cutoff = 60
    tampered[cutoff + 1 :] = 0.0
    original = make_tabular(matrix, min_history=MIN_HISTORY)
    changed = make_tabular(tampered, min_history=MIN_HISTORY)

    n_features = matrix.shape[1]
    samples_original = original.X.reshape(-1, n_features, len(FEATURE_NAMES))
    samples_changed = changed.X.reshape(-1, n_features, len(FEATURE_NAMES))
    position = cutoff - MIN_HISTORY + 1
    np.testing.assert_allclose(
        samples_original[position], samples_changed[position], rtol=0, atol=0
    )


def test_split_is_chronological():
    matrix = synthetic_matrix()
    dataset = make_sequences(matrix, window=5, min_target=MIN_HISTORY)
    train, test = split_by_target(dataset.target_index, 0.8)
    assert train.sum() + test.sum() == len(dataset.target_index)
    assert dataset.target_index[train].max() < dataset.target_index[test].min()


def test_prediction_inputs():
    matrix = synthetic_matrix()
    window = prediction_window(matrix, 5)
    assert window.shape == (1, 5, 10)
    features = prediction_features(matrix)
    assert features.shape == (10, len(FEATURE_NAMES))
