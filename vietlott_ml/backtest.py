"""Đánh giá walk-forward (rolling-origin) cho từng loại xổ số.

Mỗi fold dùng expanding window: fit trên toàn bộ dữ liệu tới cuối fold trước,
test trên khối kỳ kế tiếp. Kết quả gộp tất cả fold để có phân bố hits ổn định
hơn một lần chia train/test, và p-value được hiệu chỉnh Holm–Bonferroni cho
nhiều phép so sánh.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np

from .config import (
    DISCLAIMER,
    RANDOM_BASELINE_REPEATS,
    RANDOM_SEED,
    REPORTS_DIR,
    WINDOW_SIZE,
    lottery_config,
)
from .hyperparams import resolve_params
from .metrics import (
    compare_report,
    evaluate_predictions,
    hits_from_predictions,
    holm_bonferroni,
    summarize_hits,
    top_k,
)
from .models import DISPLAY_NAMES, MODEL_NAMES
from .training import (
    BASELINE_NAMES,
    baseline_predictions,
    build_datasets,
    fit_predict_fold,
)

VN_TZ = timezone(timedelta(hours=7))


def _folds(
    targets: np.ndarray, n_splits: int, initial_train_ratio: float
) -> list[tuple[int, np.ndarray]]:
    """Chia các kỳ đích thành các fold expanding-window.

    Trả về danh sách ``(fit_cutoff, test_targets)``. ``fit_cutoff`` là index kỳ
    cuối cùng được phép dùng để fit (bao gồm cả kỳ đó).
    """
    targets = np.asarray(targets, dtype=int)
    if n_splits < 1:
        raise ValueError("n_splits phải >= 1")
    if not 0.0 < initial_train_ratio < 1.0:
        raise ValueError("initial_train_ratio phải nằm trong (0, 1)")
    start = int(len(targets) * initial_train_ratio)
    if start < 1 or start >= len(targets):
        raise ValueError("Không đủ dữ liệu để tạo fold walk-forward")
    remaining = len(targets) - start
    if remaining < n_splits:
        raise ValueError(
            f"Cần tối thiểu {n_splits} kỳ test, chỉ còn {remaining} kỳ sau mốc đầu"
        )
    block = remaining // n_splits
    folds: list[tuple[int, np.ndarray]] = []
    for index in range(n_splits):
        test_start = start + index * block
        test_end = len(targets) if index == n_splits - 1 else start + (index + 1) * block
        if test_end <= test_start:
            continue
        fit_cutoff = int(targets[test_start - 1])
        folds.append((fit_cutoff, targets[test_start:test_end]))
    return folds


def _actuals(matrix: np.ndarray, targets: np.ndarray) -> list[list[int]]:
    return [
        (np.flatnonzero(matrix[int(t)] > 0) + 1).astype(int).tolist()
        for t in targets
    ]


def _holm_fields(results: dict, expected_hits: float) -> dict[str, float]:
    p_values = {
        name: info["metrics"]["p_value"]
        for name, info in results.items()
        if name != "random" and "p_value" in info["metrics"]
    }
    adjusted = holm_bonferroni(p_values)
    for name, info in results.items():
        metrics = info["metrics"]
        if name in adjusted:
            metrics["p_value_holm"] = adjusted[name]
            metrics["significant_holm_5pct"] = bool(
                adjusted[name] < 0.05 and metrics["avg_hits"] > expected_hits
            )
    return adjusted


def _write_markdown(payload: dict, adjusted: dict[str, float]) -> None:
    metrics_only = {
        name: info["metrics"] for name, info in payload["models"].items()
    }
    table = compare_report(metrics_only)
    best = max(
        payload["models"].items(), key=lambda item: item[1]["metrics"]["avg_hits"]
    )
    significant = [
        (name, info)
        for name, info in payload["models"].items()
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
            "Kết quả walk-forward nhất quán hơn một lần chia test, nhưng vẫn cần "
            "dữ liệu mới để xác nhận trước khi coi là tín hiệu thật."
        )
    else:
        conclusion = (
            f"Không mô hình nào vượt trội có ý nghĩa thống kê sau hiệu chỉnh "
            f"Holm–Bonferroni ({len(adjusted)} phép so sánh) so với baseline ngẫu "
            "nhiên trong walk-forward. Điều này phù hợp với bản chất ngẫu nhiên của "
            "xổ số; giá trị của dự án nằm ở pipeline và quy trình đánh giá."
        )

    model_names = list(payload["models"])
    fold_header = (
        "| Fold | Train tới | Test | Số kỳ | "
        + " | ".join(DISPLAY_NAMES.get(name, name) for name in model_names)
        + " |"
    )
    fold_sep = "|---:|---|---|---:|" + "---:|" * len(model_names)
    fold_rows = []
    for fold in payload["folds"]:
        cells = " | ".join(
            f"{fold[f'{name}_avg_hits']:.3f}" for name in model_names
        )
        fold_rows.append(
            f"| {fold['fold']} | {fold['train_end_date']} | "
            f"{fold['test_start_date']} → {fold['test_end_date']} | "
            f"{fold['n_test']} | {cells} |"
        )

    lines = [
        f"# Walk-forward {payload['name']}",
        "",
        f"- Số kỳ: **{payload['n_draws']}**",
        f"- Phương pháp: {payload['method']}",
        f"- Số fold: **{payload['n_splits']}** | "
        f"tỉ lệ train ban đầu: {payload['initial_train_ratio']:.2f}",
        f"- Cửa sổ: {payload['window']} kỳ",
        f"- Kỳ vọng ngẫu nhiên: **{payload['expected_random_hits']:.4f}** số trúng/kỳ",
        f"- Hiệu chỉnh đa so sánh: Holm–Bonferroni "
        f"({payload['multiple_comparison']['n_tests']} phép so sánh)",
        "",
        "## Tổng hợp toàn bộ fold",
        "",
        table,
        f"Mô hình có avg hits cao nhất: **{best[1]['display_name']}** "
        f"({best[1]['metrics']['avg_hits']:.3f} số/kỳ).",
        "",
        "## Kết quả từng fold (avg hits)",
        "",
        fold_header,
        fold_sep,
        *fold_rows,
        "",
        "## Kết luận trung thực",
        "",
        conclusion,
        "",
        "> **Cảnh báo:** " + DISCLAIMER,
        "",
    ]
    (REPORTS_DIR / f"{payload['lottery']}_walkforward_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def walk_forward_lottery(
    lottery: str,
    model_names: list[str] | None = None,
    n_splits: int = 4,
    initial_train_ratio: float = 0.6,
    window: int = WINDOW_SIZE,
    verbose: bool = True,
) -> dict:
    """Đánh giá walk-forward và ghi báo cáo JSON + Markdown."""
    cfg = lottery_config(lottery)
    k = cfg["pick"]
    draws, matrix, sequence, tabular = build_datasets(lottery, window)
    data = SimpleNamespace(
        lottery=lottery,
        matrix=matrix,
        sequence=sequence,
        tabular=tabular,
        n_features=matrix.shape[1],
    )
    targets = np.unique(sequence.target_index)
    folds = _folds(targets, n_splits=n_splits, initial_train_ratio=initial_train_ratio)

    names = list(model_names or MODEL_NAMES)
    unknown = [name for name in names if name not in MODEL_NAMES]
    if unknown:
        raise ValueError(f"Mô hình không hỗ trợ: {', '.join(unknown)}")

    all_hits: dict[str, list[np.ndarray]] = {name: [] for name in names}
    for kind in (*BASELINE_NAMES, "random"):
        all_hits[kind] = []

    fold_records: list[dict] = []
    fold_metrics: list[dict] = []
    expected_hits = k * k / matrix.shape[1]

    for fold_index, (fit_cutoff, fold_targets) in enumerate(folds, start=1):
        actuals = _actuals(matrix, fold_targets)
        truth = np.stack([matrix[int(t)] for t in fold_targets])
        if verbose:
            print(
                f"\n--- Fold {fold_index}/{len(folds)} | fit tới index {fit_cutoff} "
                f"| test {len(fold_targets)} kỳ ---"
            )
        fold_record = {
            "fold": fold_index,
            "train_end_index": fit_cutoff,
            "train_end_date": draws[fit_cutoff].date,
            "test_start_date": draws[int(fold_targets[0])].date,
            "test_end_date": draws[int(fold_targets[-1])].date,
            "n_test": int(len(fold_targets)),
        }
        fold_model_metrics: dict[str, dict] = {}

        for name in names:
            probabilities = fit_predict_fold(
                name,
                data,
                fit_cutoff,
                fold_targets,
                params=resolve_params(name, lottery),
                verbose=False,
            )
            predictions = [top_k(row, k) for row in probabilities]
            hits = hits_from_predictions(predictions, actuals)
            all_hits[name].append(hits)
            metrics = evaluate_predictions(
                predictions,
                actuals,
                n_features=matrix.shape[1],
                probabilities=probabilities,
                targets=truth,
                k=k,
            )
            fold_record[f"{name}_avg_hits"] = round(float(metrics["avg_hits"]), 4)
            fold_model_metrics[name] = metrics
            if verbose:
                print(
                    f"  [{name}] avg_hits={metrics['avg_hits']:.3f} "
                    f"p={metrics['p_value']:.3f}"
                )

        for kind in BASELINE_NAMES:
            predictions = baseline_predictions(
                matrix, fold_targets, k, kind, cutoff=fit_cutoff
            )
            hits = hits_from_predictions(predictions, actuals)
            all_hits[kind].append(hits)
            metrics = evaluate_predictions(
                predictions, actuals, n_features=matrix.shape[1], k=k
            )
            fold_record[f"{kind}_avg_hits"] = round(float(metrics["avg_hits"]), 4)
            fold_model_metrics[kind] = metrics

        rng = np.random.default_rng(RANDOM_SEED + fold_index)
        random_hits_list = []
        for _repeat in range(RANDOM_BASELINE_REPEATS):
            random_predictions = [
                sorted(
                    (
                        rng.choice(matrix.shape[1], size=k, replace=False) + 1
                    ).tolist()
                )
                for _ in fold_targets
            ]
            random_hits_list.append(
                hits_from_predictions(random_predictions, actuals)
            )
        random_hits = np.concatenate(random_hits_list)
        all_hits["random"].append(random_hits)
        random_metrics = summarize_hits(
            random_hits, n_features=matrix.shape[1], k=k
        )
        fold_record["random_avg_hits"] = round(
            float(random_metrics["avg_hits"]), 4
        )
        fold_model_metrics["random"] = random_metrics

        fold_records.append(fold_record)
        fold_metrics.append({"fold": fold_index, "models": fold_model_metrics})

    results: dict[str, dict] = {}
    for name, hit_list in all_hits.items():
        hits = np.concatenate(hit_list)
        metrics = summarize_hits(hits, n_features=matrix.shape[1], k=k)
        fold_means = [float(np.mean(fold_hits)) for fold_hits in hit_list]
        metrics["n_folds"] = len(hit_list)
        metrics["fold_mean_avg_hits"] = float(np.mean(fold_means))
        metrics["fold_std_avg_hits"] = (
            float(np.std(fold_means, ddof=1)) if len(fold_means) > 1 else 0.0
        )
        results[name] = {
            "display_name": DISPLAY_NAMES.get(name, name),
            "metrics": metrics,
        }

    adjusted = _holm_fields(results, expected_hits)
    payload = {
        "lottery": lottery,
        "name": cfg["name"],
        "generated_at": datetime.now(VN_TZ).isoformat(timespec="seconds"),
        "method": "walk-forward (expanding window)",
        "window": window,
        "n_splits": len(folds),
        "initial_train_ratio": initial_train_ratio,
        "n_draws": len(draws),
        "n_features": matrix.shape[1],
        "pick": k,
        "expected_random_hits": expected_hits,
        "folds": fold_records,
        "models": results,
        "fold_metrics": fold_metrics,
        "multiple_comparison": {
            "method": "holm-bonferroni",
            "n_tests": len(adjusted),
            "adjusted_p_values": adjusted,
        },
        "disclaimer": DISCLAIMER,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / f"{lottery}_walkforward_metrics.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_markdown(payload, adjusted)
    if verbose:
        print(f"\n  Đã ghi báo cáo walk-forward: {json_path}")
    return payload
