"""Chuỗi Markov: xác suất số j xuất hiện nếu số i đã xuất hiện kỳ trước."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class MarkovModel:
    name = "markov"

    def __init__(
        self,
        n_features: int,
        alpha: float = 1.0,
        transition_weight: float = 0.8,
        seed: int = 42,
    ):
        self.n_features = n_features
        self.alpha = alpha
        self.transition_weight = transition_weight
        self.seed = seed
        self.transition: np.ndarray | None = None
        self.marginal: np.ndarray | None = None

    def fit(self, matrix: np.ndarray, **_: object) -> "MarkovModel":
        matrix = np.asarray(matrix, dtype=np.float64)
        n_features = self.n_features
        counts = np.zeros((n_features, n_features), dtype=np.float64)
        for t in range(len(matrix) - 1):
            counts += np.outer(matrix[t], matrix[t + 1])

        row_sums = counts.sum(axis=1, keepdims=True)
        self.transition = (counts + self.alpha) / (
            row_sums + self.alpha * n_features
        )
        col_sums = matrix.sum(axis=0)
        self.marginal = (col_sums + self.alpha) / (
            len(matrix) + self.alpha * n_features
        )
        return self

    def predict_proba(
        self, matrix: np.ndarray, target_indices: np.ndarray | None = None
    ) -> np.ndarray:
        if self.transition is None or self.marginal is None:
            raise RuntimeError("Mô hình chưa được huấn luyện")
        matrix = np.asarray(matrix, dtype=np.float64)
        if target_indices is None:
            target_indices = np.array([len(matrix)])
        target_indices = np.asarray(target_indices)
        scores = np.zeros((len(target_indices), self.n_features), dtype=np.float64)
        w = self.transition_weight
        for row, target in enumerate(target_indices):
            previous = matrix[target - 1]
            transition_score = previous @ self.transition
            score = w * transition_score + (1.0 - w) * self.marginal
            total = score.sum()
            scores[row] = score / total if total > 0 else 1.0 / self.n_features
        return scores

    def save(self, path: str | Path) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "name": self.name,
                "n_features": self.n_features,
                "alpha": self.alpha,
                "transition_weight": self.transition_weight,
                "seed": self.seed,
                "transition": self.transition,
                "marginal": self.marginal,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "MarkovModel":
        import joblib

        payload = joblib.load(path)
        model = cls(
            n_features=payload["n_features"],
            alpha=payload["alpha"],
            transition_weight=payload["transition_weight"],
            seed=payload["seed"],
        )
        model.transition = payload["transition"]
        model.marginal = payload["marginal"]
        return model
