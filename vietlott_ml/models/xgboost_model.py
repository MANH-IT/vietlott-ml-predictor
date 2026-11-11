"""Hồi quy XGBoost: một khu rừng Gradient Boosting cho mỗi số."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import n_samples_from_rows, positive_probability

# Thử import xgboost, nếu không có thì báo lỗi rõ ràng
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class XGBoostModel:
    name = "xgboost"

    def __init__(
        self,
        n_features: int,
        n_estimators: int = 100,
        max_depth: int | None = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        n_jobs: int = 1,
        seed: int = 42,
    ):
        self.n_features = n_features
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        # Mặc định 1 luồng: n_jobs=-1 từng chậm hơn ~12x trong môi trường nhiều
        # nhân do tranh chấp OpenMP khi huấn luyện hàng chục bộ phân loại nhỏ.
        self.n_jobs = n_jobs
        self.seed = seed
        self.models: list = []
        self.constants: np.ndarray | None = None

    def _check_available(self):
        if not XGBOOST_AVAILABLE:
            raise RuntimeError(
                "Thư viện xgboost chưa được cài đặt. "
                "Vui lòng chạy: pip install xgboost"
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostModel":
        self._check_available()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        n_samples = n_samples_from_rows(len(X), self.n_features)
        n_feat = X.shape[1]
        X3 = X.reshape(n_samples, self.n_features, n_feat)
        y2 = y.reshape(n_samples, self.n_features)

        self.models = []
        self.constants = np.zeros(self.n_features, dtype=np.float64)
        for number in range(self.n_features):
            labels = y2[:, number]
            unique = np.unique(labels)
            # XGBoost không huấn luyện được khi chỉ có một lớp (số chưa từng
            # xuất hiện hoặc luôn xuất hiện); dùng hằng số làm fallback.
            if unique.size < 2:
                self.constants[number] = float(labels.mean()) if labels.size else 0.0
                self.models.append(None)
                continue
            clf = XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                n_jobs=self.n_jobs,
                random_state=self.seed + number,
                eval_metric="logloss",
            )
            clf.fit(X3[:, number, :], labels)
            self.models.append(clf)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.constants is None:
            raise RuntimeError("Mô hình chưa được huấn luyện")
        self._check_available()
        X = np.asarray(X, dtype=np.float32)
        n_samples = n_samples_from_rows(len(X), self.n_features)
        n_feat = X.shape[1]
        X3 = X.reshape(n_samples, self.n_features, n_feat)
        probabilities = np.zeros((n_samples, self.n_features), dtype=np.float64)
        for number, clf in enumerate(self.models):
            if clf is None:
                probabilities[:, number] = self.constants[number]
            else:
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
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "n_jobs": self.n_jobs,
                "seed": self.seed,
                "models": self.models,
                "constants": self.constants,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostModel":
        import joblib

        payload = joblib.load(path)
        model = cls(
            n_features=payload["n_features"],
            n_estimators=payload["n_estimators"],
            max_depth=payload["max_depth"],
            subsample=payload["subsample"],
            colsample_bytree=payload["colsample_bytree"],
            n_jobs=payload.get("n_jobs", 1),
            seed=payload["seed"],
        )
        model.models = payload["models"]
        model.constants = payload.get("constants")
        if model.constants is None:
            model.constants = np.zeros(model.n_features, dtype=np.float64)
        return model
