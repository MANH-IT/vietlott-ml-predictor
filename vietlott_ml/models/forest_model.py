"""Random Forest: một khu rừng nhị phân cho mỗi số."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import positive_probability


class ForestModel:
    name = "random_forest"

    def __init__(
        self,
        n_features: int,
        n_estimators: int = 100,
        max_depth: int | None = 12,
        min_samples_leaf: int = 10,
        n_jobs: int = 1,
        seed: int = 42,
    ):
        self.n_features = n_features
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.seed = seed
        self.models: list = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ForestModel":
        from sklearn.ensemble import RandomForestClassifier

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        n_samples = len(X) // self.n_features
        n_feat = X.shape[1]
        X3 = X.reshape(n_samples, self.n_features, n_feat)
        y2 = y.reshape(n_samples, self.n_features)

        self.models = []
        for number in range(self.n_features):
            clf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                n_jobs=self.n_jobs,
                random_state=self.seed + number,
                class_weight="balanced_subsample",
            )
            clf.fit(X3[:, number, :], y2[:, number])
            self.models.append(clf)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.models:
            raise RuntimeError("Mô hình chưa được huấn luyện")
        X = np.asarray(X, dtype=np.float32)
        n_samples = len(X) // self.n_features
        n_feat = X.shape[1]
        X3 = X.reshape(n_samples, self.n_features, n_feat)
        probabilities = np.zeros((n_samples, self.n_features), dtype=np.float64)
        for number, clf in enumerate(self.models):
            probabilities[:, number] = positive_probability(
                clf, X3[:, number, :]
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
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "min_samples_leaf": self.min_samples_leaf,
                "n_jobs": self.n_jobs,
                "seed": self.seed,
                "models": self.models,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ForestModel":
        import joblib

        payload = joblib.load(path)
        model = cls(
            n_features=payload["n_features"],
            n_estimators=payload["n_estimators"],
            max_depth=payload["max_depth"],
            min_samples_leaf=payload["min_samples_leaf"],
            n_jobs=payload.get("n_jobs", 1),
            seed=payload["seed"],
        )
        model.models = payload["models"]
        return model
