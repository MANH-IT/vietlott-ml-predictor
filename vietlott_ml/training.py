"""Huấn luyện, đánh giá và lưu báo cáo cho từng loại xổ số."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from .config import (
    DISCLAIMER,
    MIN_HISTORY,
    RANDOM_BASELINE_REPEATS,
    RANDOM_SEED,
    REPORTS_DIR,
    TRAIN_RATIO,
    WINDOW_SIZE,
    lottery_config,
    report_path,
)
from .data_validation import Draw
from .dataset import (
    SequenceDataset,
    TabularDataset,
    load_matrix,
    make_sequences,
    make_tabular,
)
from .hyperparams import fit_params, resolve_params
from .metrics import (
    compare_report,
    evaluate_predictions,
    hits_from_predictions,
    holm_bonferroni,
    summarize_hits,
    top_k,
)
from .models import (
    DISPLAY_NAMES,
    MODEL_NAMES,
    SEQUENCE_MODELS,
    TABULAR_MODELS,
    TORCH_MODELS,
    build_model,
    load_model,
    model_path,
    save_model,
)

VN_TZ = timezone(timedelta(hours=7))

BASELINE_NAMES = ("repeat_last", "frequency", "frequency_train")


@dataclass
class PreparedData:
    lottery: str
    draws: list[Draw]
    matrix: np.ndarray
    sequence: SequenceDataset
    tabular: TabularDataset
    cutoff: int
    test_targets: np.ndarray
    X_seq_test: np.ndarray
    X_tab_test_samples: np.ndarray
    X_last_test: np.ndarray
    actuals: list[list[int]]

    @property
    def n_features(self) -> int:
        return self.matrix.shape[1]


def build_datasets(
    lottery: str, window: int
) -> tuple[list[Draw], np.ndarray, SequenceDataset, TabularDataset]:
    """Nạp dữ liệu và dựng sequence/tabular dataset dùng chung cho train/backtest."""
    draws, matrix = load_matrix(lottery)
    if len(draws) < MIN_HISTORY + window + 20:
        raise ValueError(
            f"{lottery}: cần tối thiểu {MIN_HISTORY + window + 20} kỳ, "
            f"hiện có {len(draws)}. Hãy chạy bootstrap trước."
        )
    sequence = make_sequences(matrix, window=window, min_target=MIN_HISTORY)
    tabular = make_tabular(matrix, min_history=MIN_HISTORY)
    return draws, matrix, sequence, tabular


def _prepare(lottery: str, window: int, train_ratio: float) -> PreparedData:
    cfg = lottery_config(lottery)
    draws, matrix, sequence, tabular = build_datasets(lottery, window)
    n_features = cfg["n_features"]

    unique_targets = np.unique(sequence.target_index)
    split_pos = int(len(unique_targets) * train_ratio) - 1
    if split_pos < 1:
        raise ValueError("Không đủ dữ liệu để chia train/test")
    cutoff = int(unique_targets[split_pos])

    seq_test_mask = sequence.target_index > cutoff
    test_targets = sequence.target_index[seq_test_mask]

    n_tab_samples = len(tabular.target_index) // n_features
    tab_sample_targets = tabular.target_index.reshape(n_tab_samples, n_features)[:, 0]
    tab_test_mask = tab_sample_targets > cutoff
    tab_test_targets = tab_sample_targets[tab_test_mask]
    if not np.array_equal(test_targets, tab_test_targets):
        raise RuntimeError("Lệch chỉ số kỳ test giữa sequence và tabular")

    X_seq_test = sequence.X[seq_test_mask]
    X_tab_all = tabular.X.reshape(n_tab_samples, n_features, -1)
    X_tab_test = X_tab_all[tab_test_mask]
    X_last_test = X_seq_test[:, -1, :]

    actuals = [
        (np.flatnonzero(matrix[t] > 0) + 1).astype(int).tolist()
        for t in test_targets
    ]
    return PreparedData(
        lottery=lottery,
        draws=draws,
        matrix=matrix,
        sequence=sequence,
        tabular=tabular,
        cutoff=cutoff,
        test_targets=test_targets,
        X_seq_test=X_seq_test,
        X_tab_test_samples=X_tab_test,
        X_last_test=X_last_test,
        actuals=actuals,
    )


def _fit_predict_sequence(
    model_name: str,
    data: PreparedData,
    window: int,
    verbose: bool,
) -> tuple[np.ndarray, float]:
    X_train = data.sequence.X[data.sequence.target_index <= data.cutoff]
    y_train = data.sequence.y[data.sequence.target_index <= data.cutoff]
    params = resolve_params(model_name, data.lottery)
    model = build_model(model_name, data.n_features, params=params)
    started = time.perf_counter()
    if verbose:
        print(f"  [{model_name}] train trên {len(X_train)} mẫu...")
    fit_kwargs = fit_params(model_name, params)
    if model_name == "mlp":
        model.fit(X_train[:, -1, :], y_train, verbose=verbose, **fit_kwargs)
        probabilities = model.predict_proba(data.X_last_test)
    else:
        model.fit(X_train, y_train, verbose=verbose, **fit_kwargs)
        probabilities = model.predict_proba(data.X_seq_test)
    elapsed = time.perf_counter() - started
    save_model(model, data.lottery, model_name)
    return probabilities, elapsed


def _fit_predict_tabular(
    model_name: str, data: PreparedData, verbose: bool
) -> tuple[np.ndarray, float]:
    n_features = data.n_features
    n_tab_samples = len(data.tabular.target_index) // n_features
    train_mask = (
        data.tabular.target_index.reshape(n_tab_samples, n_features)[:, 0]
        <= data.cutoff
    )
    X_all = data.tabular.X.reshape(n_tab_samples, n_features, -1)
    y_all = data.tabular.y.reshape(n_tab_samples, n_features)
    X_train = X_all[train_mask].reshape(-1, X_all.shape[-1])
    y_train = y_all[train_mask].reshape(-1)

    model = build_model(model_name, n_features, lottery=data.lottery)
    started = time.perf_counter()
    if verbose:
        print(f"  [{model_name}] train trên {len(X_train)} mẫu...")
    model.fit(X_train, y_train)
    X_test = data.X_tab_test_samples.reshape(-1, X_all.shape[-1])
    probabilities = model.predict_proba(X_test)
    elapsed = time.perf_counter() - started
    save_model(model, data.lottery, model_name)
    return probabilities, elapsed


def _fit_predict_markov(
    data: PreparedData, verbose: bool
) -> tuple[np.ndarray, float]:
    model = build_model("markov", data.n_features, lottery=data.lottery)
    started = time.perf_counter()
    if verbose:
        print("  [markov] ước lượng ma trận chuyển đổi...")
    model.fit(data.matrix[: data.cutoff + 1])
    probabilities = model.predict_proba(data.matrix, data.test_targets)
    elapsed = time.perf_counter() - started
    save_model(model, data.lottery, "markov")
    return probabilities, elapsed


def fit_predict_fold(
    model_name: str,
    data: PreparedData,
    fit_cutoff: int,
    target_indices: np.ndarray,
    params: dict | None = None,
    fit_overrides: dict | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Fit trên dữ liệu tới ``fit_cutoff`` và dự đoán cho ``target_indices``.

    Dùng chung cho train/test, tuning và walk-forward để tránh lệch logic.
    Không lưu model (caller quyết định).
    """
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Mô hình không hợp lệ: {model_name}")
    resolved = params if params is not None else resolve_params(model_name, data.lottery)
    target_indices = np.asarray(target_indices, dtype=int)
    if len(target_indices) == 0:
        raise ValueError("target_indices rỗng")

    if model_name in TORCH_MODELS:
        X_train = data.sequence.X[data.sequence.target_index <= fit_cutoff]
        y_train = data.sequence.y[data.sequence.target_index <= fit_cutoff]
        X_target = data.sequence.X[np.isin(data.sequence.target_index, target_indices)]
        model = build_model(model_name, data.n_features, params=resolved)
        kwargs = dict(fit_params(model_name, resolved))
        kwargs.update(fit_overrides or {})
        if verbose:
            print(f"  [{model_name}] train trên {len(X_train)} mẫu...")
        if model_name == "mlp":
            model.fit(X_train[:, -1, :], y_train, verbose=verbose, **kwargs)
            return model.predict_proba(X_target[:, -1, :])
        model.fit(X_train, y_train, verbose=verbose, **kwargs)
        return model.predict_proba(X_target)

    if model_name in TABULAR_MODELS:
        n_features = data.n_features
        n_tab_samples = len(data.tabular.target_index) // n_features
        tab_targets = data.tabular.target_index.reshape(n_tab_samples, n_features)[:, 0]
        train_mask = tab_targets <= fit_cutoff
        target_mask = np.isin(tab_targets, target_indices)
        X_all = data.tabular.X.reshape(n_tab_samples, n_features, -1)
        y_all = data.tabular.y.reshape(n_tab_samples, n_features)
        X_train = X_all[train_mask].reshape(-1, X_all.shape[-1])
        y_train = y_all[train_mask].reshape(-1)
        model = build_model(model_name, n_features, lottery=data.lottery, params=resolved)
        kwargs = dict(fit_params(model_name, resolved))
        if verbose:
            print(f"  [{model_name}] train trên {len(X_train)} mẫu...")
        model.fit(X_train, y_train, **kwargs)
        return model.predict_proba(X_all[target_mask].reshape(-1, X_all.shape[-1]))

    model = build_model("markov", data.n_features, lottery=data.lottery)
    if verbose:
        print("  [markov] ước lượng ma trận chuyển đổi...")
    model.fit(data.matrix[: fit_cutoff + 1])
    return model.predict_proba(data.matrix, target_indices)


def _evaluate_probabilities(
    data: PreparedData, probabilities: np.ndarray, k: int
) -> dict:
    predictions = [top_k(row, k) for row in probabilities]
    truth = np.stack([data.matrix[t] for t in data.test_targets])
    return evaluate_predictions(
        predictions,
        data.actuals,
        n_features=data.n_features,
        probabilities=probabilities,
        targets=truth,
        k=k,
    )


def _random_predictions(
    data: PreparedData, k: int, seed: int = RANDOM_SEED
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    return [
        sorted((rng.choice(data.n_features, size=k, replace=False) + 1).tolist())
        for _ in data.test_targets
    ]


def _random_baseline(
    data: PreparedData,
    k: int,
    seed: int = RANDOM_SEED,
    n_repeats: int = RANDOM_BASELINE_REPEATS,
) -> dict:
    """Baseline ngẫu nhiên lặp nhiều seed để giảm nhiễu do một lần rút.

    Gộp số trúng của tất cả các lần lặp thành một phân bố chung, nhờ đó
    avg_hits tiến sát kỳ vọng lý thuyết và khoảng tin cậy phản ánh đúng
    độ bất định của phép chọn ngẫu nhiên.
    """
    all_hits = [
        hits_from_predictions(_random_predictions(data, k, seed + offset), data.actuals)
        for offset in range(max(1, n_repeats))
    ]
    return summarize_hits(np.concatenate(all_hits), n_features=data.n_features, k=k)


def baseline_predictions(
    matrix: np.ndarray,
    target_indices: np.ndarray,
    k: int,
    kind: str,
    cutoff: int | None = None,
) -> list[list[int]]:
    """Dự đoán baseline không cần học.

    - ``repeat_last``: lặp lại đúng bộ số của kỳ ngay trước.
    - ``frequency``: top-k số xuất hiện nhiều nhất tính tới trước kỳ dự đoán.
    - ``frequency_train``: top-k theo tần suất tích lũy tới ``cutoff``.
    """
    predictions: list[list[int]] = []
    for target in np.asarray(target_indices, dtype=int):
        if kind == "repeat_last":
            numbers = np.flatnonzero(matrix[int(target) - 1] > 0) + 1
            predictions.append(sorted(int(n) for n in numbers))
        elif kind in {"frequency", "frequency_train"}:
            end = int(target) if kind == "frequency" else int(cutoff) + 1
            scores = matrix[:end].sum(axis=0)
            predictions.append(sorted(top_k(scores, k)))
        else:
            raise ValueError(f"Baseline không hợp lệ: {kind}")
    return predictions


def _period(draws: list[Draw], indices: np.ndarray) -> dict:
    dates = [draws[int(i)].date for i in indices]
    return {
        "start": min(dates),
        "end": max(dates),
        "n_draws": int(len(indices)),
    }


def _write_predictions(data: PreparedData, predictions: dict[str, list[list[int]]]) -> None:
    path = REPORTS_DIR / f"{data.lottery}_test_predictions.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for position, target in enumerate(data.test_targets):
            row = {
                "target_index": int(target),
                "draw_id": data.draws[int(target)].draw_id,
                "date": data.draws[int(target)].date,
                "actual": data.actuals[position],
                "predictions": {
                    name: preds[position] for name, preds in predictions.items()
                },
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_report(
    data: PreparedData,
    model_results: dict,
    window: int,
    train_ratio: float,
) -> dict:
    cfg = lottery_config(data.lottery)
    k = cfg["pick"]
    train_indices = np.arange(0, data.cutoff + 1)
    test_indices = data.test_targets
    report = {
        "lottery": data.lottery,
        "name": cfg["name"],
        "generated_at": datetime.now(VN_TZ).isoformat(timespec="seconds"),
        "n_draws": len(data.draws),
        "n_features": data.n_features,
        "pick": k,
        "window": window,
        "train_ratio": train_ratio,
        "expected_random_hits": k * k / data.n_features,
        "train_period": _period(data.draws, train_indices),
        "test_period": _period(data.draws, test_indices),
        "n_train_targets": int((data.sequence.target_index <= data.cutoff).sum()),
        "n_test_targets": int(len(test_indices)),
        "random_baseline_repeats": RANDOM_BASELINE_REPEATS,
        "models": model_results,
        "disclaimer": DISCLAIMER,
    }
    return report


def _write_report_files(data: PreparedData, report: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    p_values = {
        name: info["metrics"]["p_value"]
        for name, info in report["models"].items()
        if name != "random" and "p_value" in info["metrics"]
    }
    adjusted = holm_bonferroni(p_values)
    for name, info in report["models"].items():
        metrics = info["metrics"]
        if name in adjusted:
            metrics["p_value_holm"] = adjusted[name]
            metrics["significant_holm_5pct"] = bool(
                adjusted[name] < 0.05
                and metrics["avg_hits"] > metrics["expected_random_hits"]
            )
    report["multiple_comparison"] = {
        "method": "holm-bonferroni",
        "n_tests": len(adjusted),
        "adjusted_p_values": adjusted,
    }
    report_path(data.lottery).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metrics_only = {
        name: info["metrics"] for name, info in report["models"].items()
    }
    table = compare_report(metrics_only)
    best = max(
        report["models"].items(), key=lambda item: item[1]["metrics"]["avg_hits"]
    )
    significant = [
        (name, info)
        for name, info in report["models"].items()
        if name != "random" and info["metrics"].get("significant_holm_5pct")
    ]
    if significant:
        names = ", ".join(
            f"{info['display_name']} (p Holm={info['metrics']['p_value_holm']:.4f})"
            for _, info in significant
        )
        conclusion = (
            f"Sau hiệu chỉnh Holm–Bonferroni cho {len(adjusted)} phép so sánh, các "
            f"mô hình sau vẫn vượt baseline ngẫu nhiên ở mức ý nghĩa 5%: {names}. "
            "Cần kiểm chứng lại bằng walk-forward trên dữ liệu mới trước khi coi là "
            "tín hiệu thật; xổ số vẫn được giả định là ngẫu nhiên."
        )
    else:
        conclusion = (
            f"Không mô hình nào vượt trội có ý nghĩa thống kê sau hiệu chỉnh "
            f"Holm–Bonferroni ({len(adjusted)} phép so sánh) so với baseline ngẫu "
            "nhiên trên tập test. Điều này phù hợp với bản chất ngẫu nhiên của xổ số. "
            "Giá trị của dự án nằm ở pipeline học máy và quy trình đánh giá, không "
            "phải khả năng dự đoán kết quả."
        )
    lines = [
        f"# Báo cáo {report['name']}",
        "",
        f"- Số kỳ: **{report['n_draws']}**",
        f"- Train: {report['train_period']['start']} → {report['train_period']['end']} "
        f"({report['n_train_targets']} kỳ đích)",
        f"- Test: {report['test_period']['start']} → {report['test_period']['end']} "
        f"({report['n_test_targets']} kỳ đích)",
        f"- Cửa sổ: {report['window']} kỳ",
        f"- Kỳ vọng ngẫu nhiên: **{report['expected_random_hits']:.4f}** số trúng/kỳ "
        f"(baseline lặp {report['random_baseline_repeats']} lần)",
        f"- Hiệu chỉnh đa so sánh: Holm–Bonferroni "
        f"({report['multiple_comparison']['n_tests']} phép so sánh)",
        "",
        "## So sánh mô hình",
        "",
        table,
        f"Mô hình có avg hits cao nhất: **{best[1]['display_name']}** "
        f"({best[1]['metrics']['avg_hits']:.3f} số/kỳ).",
        "",
        "## Kết luận trung thực",
        "",
        conclusion,
        "",
        "> **Cảnh báo:** " + DISCLAIMER,
        "",
    ]
    (REPORTS_DIR / f"{data.lottery}_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def train_lottery(
    lottery: str,
    model_names: list[str] | None = None,
    window: int = WINDOW_SIZE,
    train_ratio: float = TRAIN_RATIO,
    verbose: bool = True,
) -> dict:
    """Huấn luyện các mô hình, đánh giá trên tập test và ghi báo cáo."""
    names = model_names or MODEL_NAMES
    data = _prepare(lottery, window=window, train_ratio=train_ratio)
    if verbose:
        cfg = lottery_config(lottery)
        print(
            f"\n=== {cfg['name']} | {len(data.draws)} kỳ | "
            f"train tới index {data.cutoff} | test {len(data.test_targets)} kỳ ==="
        )

    probabilities_by_model: dict[str, np.ndarray] = {}
    results: dict[str, dict] = {}

    for name in names:
        if name in SEQUENCE_MODELS or name == "mlp":
            probs, elapsed = _fit_predict_sequence(name, data, window, verbose)
        elif name in TABULAR_MODELS:
            probs, elapsed = _fit_predict_tabular(name, data, verbose)
        elif name == "markov":
            probs, elapsed = _fit_predict_markov(data, verbose)
        else:
            raise ValueError(f"Mô hình không hỗ trợ: {name}")
        metrics = _evaluate_probabilities(data, probs, k=lottery_config(lottery)["pick"])
        probabilities_by_model[name] = probs
        results[name] = {
            "display_name": DISPLAY_NAMES.get(name, name),
            "training_seconds": round(elapsed, 2),
            "model_file": str(model_path(lottery, name)),
            "params": resolve_params(name, lottery),
            "metrics": metrics,
        }
        if verbose:
            print(
                f"  [{name}] avg_hits={metrics['avg_hits']:.3f} "
                f"(random={metrics['expected_random_hits']:.3f}, "
                f"lift={metrics['lift_vs_random']:.3f}, p={metrics['p_value']:.3f}) "
                f"| {elapsed:.1f}s"
            )

    random_metrics = _random_baseline(data, k=lottery_config(lottery)["pick"])
    results["random"] = {
        "display_name": DISPLAY_NAMES["random"],
        "training_seconds": 0.0,
        "model_file": None,
        "metrics": random_metrics,
    }

    predictions = {
        name: [top_k(row, lottery_config(lottery)["pick"]) for row in probs]
        for name, probs in probabilities_by_model.items()
    }
    predictions["random"] = _random_predictions(
        data, k=lottery_config(lottery)["pick"]
    )
    k = lottery_config(lottery)["pick"]
    for kind in BASELINE_NAMES:
        preds = baseline_predictions(
            data.matrix, data.test_targets, k, kind, cutoff=data.cutoff
        )
        results[kind] = {
            "display_name": DISPLAY_NAMES.get(kind, kind),
            "training_seconds": 0.0,
            "model_file": None,
            "metrics": evaluate_predictions(
                preds, data.actuals, n_features=data.n_features, k=k
            ),
        }
        predictions[kind] = preds
    _write_predictions(data, predictions)

    report = _build_report(data, results, window=window, train_ratio=train_ratio)
    _write_report_files(data, report)
    if verbose:
        print(f"  Đã ghi báo cáo: {report_path(lottery)}")
    return report


def evaluate_saved_lottery(
    lottery: str,
    model_names: list[str] | None = None,
    window: int = WINDOW_SIZE,
    train_ratio: float = TRAIN_RATIO,
    verbose: bool = True,
) -> dict:
    """Đánh giá lại các mô hình đã lưu mà không huấn luyện lại."""
    names = [n for n in (model_names or MODEL_NAMES) if model_path(lottery, n).exists()]
    data = _prepare(lottery, window=window, train_ratio=train_ratio)
    results: dict[str, dict] = {}
    predictions: dict[str, list[list[int]]] = {}

    for name in names:
        model = load_model(lottery, name)
        if name in SEQUENCE_MODELS:
            probs = model.predict_proba(data.X_seq_test)
        elif name == "mlp":
            probs = model.predict_proba(data.X_last_test)
        elif name in TABULAR_MODELS:
            probs = model.predict_proba(
                data.X_tab_test_samples.reshape(
                    -1, data.tabular.X.shape[-1]
                )
            )
        elif name == "markov":
            probs = model.predict_proba(data.matrix, data.test_targets)
        else:
            continue
        metrics = _evaluate_probabilities(
            data, probs, k=lottery_config(lottery)["pick"]
        )
        results[name] = {
            "display_name": DISPLAY_NAMES.get(name, name),
            "training_seconds": None,
            "model_file": str(model_path(lottery, name)),
            "metrics": metrics,
        }
        predictions[name] = [
            top_k(row, lottery_config(lottery)["pick"]) for row in probs
        ]
        if verbose:
            print(f"  [{name}] avg_hits={metrics['avg_hits']:.3f}")

    results["random"] = {
        "display_name": DISPLAY_NAMES["random"],
        "training_seconds": 0.0,
        "model_file": None,
        "metrics": _random_baseline(data, k=lottery_config(lottery)["pick"]),
    }
    k = lottery_config(lottery)["pick"]
    for kind in BASELINE_NAMES:
        preds = baseline_predictions(
            data.matrix, data.test_targets, k, kind, cutoff=data.cutoff
        )
        results[kind] = {
            "display_name": DISPLAY_NAMES.get(kind, kind),
            "training_seconds": 0.0,
            "model_file": None,
            "metrics": evaluate_predictions(
                preds, data.actuals, n_features=data.n_features, k=k
            ),
        }
        predictions[kind] = preds
    if predictions:
        _write_predictions(data, predictions)
    report = _build_report(data, results, window=window, train_ratio=train_ratio)
    _write_report_files(data, report)
    return report
