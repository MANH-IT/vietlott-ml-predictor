"""Hồi quy logistic: một bộ phân loại nhị phân cho mỗi số."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import positive_probability


class LogisticModel:
    name = "logistic"

    def __init__(
        self,
        n_features: int,
        C: float = 1.0,
        max_iter: int = 1000,
        seed: int = 42,
    ):
        self.n_features = n_features
        self.C = C
        self.max_iter = max_iter
        self.seed = seed
        self.scaler = None
        self.models: list = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticModel":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples = len(X) // self.n_features
        n_feat = X.shape[1]
        X3 = X.reshape(n_samples, self.n_features, n_feat)
        y2 = y.reshape(n_samples, self.n_features)

        self.scaler = StandardScaler()
        flat = X3.reshape(-1, n_feat)
        self.scaler.fit(flat)
        scaled = self.scaler.transform(flat).reshape(n_samples, self.n_features, n_feat)

        self.models = []
        for number in range(self.n_features):
            labels = y2[:, number]
            clf = LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                solver="liblinear",
                class_weight="balanced",
                random_state=self.seed,
            )
            clf.fit(scaled[:, number, :], labels)
            self.models.append(clf)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("Mô hình chưa được huấn luyện")
        X = np.asarray(X, dtype=np.float64)
        n_samples = len(X) // self.n_features
        n_feat = X.shape[1]
        scaled = self.scaler.transform(X).reshape(
            n_samples, self.n_features, n_feat
        )
        probabilities = np.zeros((n_samples, self.n_features), dtype=np.float64)
        for number, clf in enumerate(self.models):
            probabilities[:, number] = positive_probability(
                clf, scaled[:, number, :]
            )
        return probabilities

    def save(self, path: str | Path) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "name": self.name,
                "n_features": self.n_features,
                "C": self.C,
                "max_iter": self.max_iter,
                "seed": self.seed,
                "scaler": self.scaler,
                "models": self.models,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "LogisticModel":
        import joblib

        payload = joblib.load(path)
        model = cls(
            n_features=payload["n_features"],
            C=payload["C"],
            max_iter=payload["max_iter"],
            seed=payload["seed"],
        )
        model.scaler = payload["scaler"]
        model.models = payload["models"]
        return model
