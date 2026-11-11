"""Tìm hyperparameters trên validation (thuộc phần train), không dùng test.

Nguyên tắc chống rò rỉ: chỉ dùng các kỳ nằm trong phần train (``target <= cutoff``).
Phần train lại được chia thành fit (đầu) và validation (cuối). Test set hoàn toàn
không được chạm tới trong quá trình tune.
"""

from __future__ import annotations

import itertools
import time
from datetime import datetime, timedelta, timezone

import numpy as np

from .config import RANDOM_SEED, TRAIN_RATIO, WINDOW_SIZE, lottery_config
from .hyperparams import (
    DEFAULT_PARAMS,
    SEARCH_SPACES,
    TUNING_FIT_OVERRIDES,
    fit_params,
    load_tuned,
    save_tuned,
)
from .metrics import evaluate_predictions, top_k
from .models import MODEL_NAMES, TABULAR_MODELS, TORCH_MODELS, build_model
from .training import _prepare

VN_TZ = timezone(timedelta(hours=7))


def _candidates(name: str) -> list[dict]:
    """Sinh danh sách ứng viên: mặc định gộp với từng tổ hợp trong search space."""
    space = SEARCH_SPACES.get(name, {})
    if not space:
        return [dict(DEFAULT_PARAMS[name])]
    keys = list(space)
    out = []
    for values in itertools.product(*(space[key] for key in keys)):
        params = dict(DEFAULT_PARAMS[name])
        params.update(dict(zip(keys, values, strict=True)))
        out.append(params)
    return out


def _split_train(data, val_ratio: float) -> tuple[int, np.ndarray]:
    """Chia phần train thành fit/validation theo thời gian; trả về fit_cutoff."""
    targets = np.unique(
        data.sequence.target_index[data.sequence.target_index <= data.cutoff]
    )
    n_val = max(1, int(len(targets) * val_ratio))
    fit_targets = targets[: len(targets) - n_val]
    val_targets = targets[len(targets) - n_val :]
    return int(fit_targets[-1]), val_targets


def _fit_predict_candidate(
    name: str, data, fit_cutoff: int, val_targets: np.ndarray, params: dict
) -> np.ndarray:
    n_features = data.n_features

    if name in TORCH_MODELS:
        seq = data.sequence
        fit_mask = seq.target_index <= fit_cutoff
        val_mask = np.isin(seq.target_index, val_targets)
        X_fit = seq.X[fit_mask]
        y_fit = seq.y[fit_mask]
        X_val = seq.X[val_mask]
        if name == "mlp":
            X_fit = X_fit[:, -1, :]
            X_val = X_val[:, -1, :]
        model = build_model(name, n_features, seed=RANDOM_SEED, params=params)
        kwargs = fit_params(name, params)
        kwargs.update(TUNING_FIT_OVERRIDES)
        model.fit(X_fit, y_fit, verbose=False, **kwargs)
        return model.predict_proba(X_val)

    if name in TABULAR_MODELS:
        n = len(data.tabular.target_index) // n_features
        X_all = data.tabular.X.reshape(n, n_features, -1)
        y_all = data.tabular.y.reshape(n, n_features)
        sample_targets = data.tabular.target_index.reshape(n, n_features)[:, 0]
        fit_mask = sample_targets <= fit_cutoff
        val_mask = np.isin(sample_targets, val_targets)
        X_fit = X_all[fit_mask].reshape(-1, X_all.shape[-1])
        y_fit = y_all[fit_mask].reshape(-1)
        X_val = X_all[val_mask].reshape(-1, X_all.shape[-1])
        model = build_model(name, n_features, seed=RANDOM_SEED, params=params)
        model.fit(X_fit, y_fit)
        return model.predict_proba(X_val).reshape(len(val_targets), n_features)

    if name == "markov":
        model = build_model(name, n_features, seed=RANDOM_SEED, params=params)
        model.fit(data.matrix[: fit_cutoff + 1])
        return model.predict_proba(data.matrix, val_targets)

    raise ValueError(f"Mô hình không hỗ trợ: {name}")


def _score(metrics: dict) -> tuple[float, float]:
    """Ưu tiên avg_hits; hoà thì ưu tiên log-loss thấp hơn."""
    return (metrics["avg_hits"], -metrics.get("log_loss", 0.0))


def tune_lottery(
    lottery: str,
    model_names: list[str] | None = None,
    window: int = WINDOW_SIZE,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = 0.2,
    verbose: bool = True,
) -> dict:
    """Grid search hyperparameters cho từng mô hình trên validation."""
    cfg = lottery_config(lottery)
    k = cfg["pick"]
    data = _prepare(lottery, window=window, train_ratio=train_ratio)
    fit_cutoff, val_targets = _split_train(data, val_ratio)
    names = model_names or MODEL_NAMES

    actuals = [
        (np.flatnonzero(data.matrix[t] > 0) + 1).astype(int).tolist()
        for t in val_targets
    ]
    truth = np.stack([data.matrix[t] for t in val_targets])

    if verbose:
        print(
            f"\n=== Tune {cfg['name']} | fit tới index {fit_cutoff} | "
            f"validation {len(val_targets)} kỳ ==="
        )

    results: dict[str, dict] = {}
    for name in names:
        candidates = _candidates(name)
        best = None
        started = time.perf_counter()
        for index, params in enumerate(candidates, start=1):
            probs = _fit_predict_candidate(
                name, data, fit_cutoff, val_targets, params
            )
            predictions = [top_k(row, k) for row in probs]
            metrics = evaluate_predictions(
                predictions,
                actuals,
                n_features=data.n_features,
                probabilities=probs,
                targets=truth,
                k=k,
            )
            score = _score(metrics)
            if best is None or score > best["score"]:
                best = {"params": params, "metrics": metrics, "score": score}
            if verbose:
                print(
                    f"  [{name}] {index}/{len(candidates)} "
                    f"avg_hits={metrics['avg_hits']:.3f} "
                    f"log_loss={metrics.get('log_loss', float('nan')):.4f}"
                )
        elapsed = time.perf_counter() - started
        results[name] = {
            "params": best["params"],
            "val_metrics": best["metrics"],
            "n_candidates": len(candidates),
            "tuning_seconds": round(elapsed, 2),
        }
        if verbose:
            print(
                f"  -> [{name}] best avg_hits={best['metrics']['avg_hits']:.3f} "
                f"| {elapsed:.1f}s\n     {best['params']}"
            )

    merged = load_tuned(lottery).get("models", {})
    merged.update(results)
    payload = {
        "lottery": lottery,
        "generated_at": datetime.now(VN_TZ).isoformat(timespec="seconds"),
        "window": window,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "n_val_targets": int(len(val_targets)),
        "models": merged,
    }
    path = save_tuned(lottery, payload)
    if verbose:
        print(f"  Đã lưu hyperparameters: {path}")
    return payload
