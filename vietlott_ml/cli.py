"""Giao diện dòng lệnh cho pipeline Vietlott ML."""

from __future__ import annotations

import argparse
import logging
import sys
import time

from .config import LOTTERIES, ensure_dirs
from .data_collection import bootstrap, collect_live, configure_logging
from .data_validation import validate_file
from .models import MODEL_NAMES
from .predict import generate_numbers
from .training import evaluate_saved_lottery, train_lottery

LOGGER = logging.getLogger("vietlott.cli")


def _lotteries(value: str) -> list[str]:
    if value == "all":
        return list(LOTTERIES)
    if value not in LOTTERIES:
        raise SystemExit(
            f"Lottery không hợp lệ: {value}. Chọn 'all', {', '.join(LOTTERIES)}"
        )
    return [value]


def _models(value: str | None) -> list[str] | None:
    if not value:
        return None
    names = [n.strip() for n in value.split(",") if n.strip()]
    invalid = [n for n in names if n not in MODEL_NAMES]
    if invalid:
        raise SystemExit(
            f"Mô hình không hợp lệ: {invalid}. Chọn trong: {', '.join(MODEL_NAMES)}"
        )
    return names


def cmd_bootstrap(args: argparse.Namespace) -> None:
    for lottery in _lotteries(args.lottery):
        bootstrap(lottery)


def cmd_collect(args: argparse.Namespace) -> None:
    configure_logging()
    selected = _lotteries(args.lottery)
    running = True
    while running:
        try:
            collect_live(selected)
        except KeyboardInterrupt:
            LOGGER.info("Người dùng dừng thu thập")
            running = False
        except Exception as exc:  # pragma: no cover - phụ thuộc mạng
            LOGGER.error("Thu thập thất bại: %s", exc)
        if not args.watch:
            running = False
        else:
            try:
                time.sleep(args.interval)
            except KeyboardInterrupt:
                LOGGER.info("Người dùng dừng thu thập")
                running = False


def cmd_validate(args: argparse.Namespace) -> None:
    from .config import canonical_path

    failed = False
    for lottery in _lotteries(args.lottery):
        path = canonical_path(lottery)
        if not path.exists():
            print(f"{lottery}: chưa có dữ liệu tại {path}")
            failed = True
            continue
        total, valid, failures = validate_file(path)
        print(f"{lottery}: {valid}/{total} bản ghi hợp lệ")
        for line_no, errors in failures[:10]:
            print(f"  dòng {line_no}: {'; '.join(errors)}")
        failed = failed or bool(failures)
    if failed:
        sys.exit(1)


def cmd_train(args: argparse.Namespace) -> None:
    models = _models(args.models)
    for lottery in _lotteries(args.lottery):
        train_lottery(lottery, model_names=models, verbose=True)


def cmd_evaluate(args: argparse.Namespace) -> None:
    models = _models(args.models)
    for lottery in _lotteries(args.lottery):
        print(f"\n=== Đánh giá lại {lottery} ===")
        evaluate_saved_lottery(lottery, model_names=models, verbose=True)


def cmd_predict(args: argparse.Namespace) -> None:
    import json

    result = generate_numbers(args.lottery, args.model, seed=args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_pipeline(args: argparse.Namespace) -> None:
    selected = _lotteries(args.lottery)
    models = _models(args.models)
    for lottery in selected:
        bootstrap(lottery)
    collect_live(selected)
    for lottery in selected:
        train_lottery(lottery, model_names=models, verbose=True)


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run(
        "vietlott_ml.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vietlott_ml",
        description="Pipeline học máy thực hành trên dữ liệu Vietlott.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("bootstrap", help="Tải dữ liệu lịch sử")
    p.add_argument("--lottery", default="all")
    p.set_defaults(func=cmd_bootstrap)

    p = sub.add_parser("collect", help="Thu thập kết quả mới từ API live")
    p.add_argument("--lottery", default="all")
    p.add_argument("--watch", action="store_true", help="Chạy lặp định kỳ")
    p.add_argument("--interval", type=int, default=300, help="Giây giữa các lần")
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("validate", help="Kiểm tra dữ liệu đã lưu")
    p.add_argument("--lottery", default="all")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("train", help="Huấn luyện và đánh giá mô hình")
    p.add_argument("--lottery", default="all")
    p.add_argument("--models", default=None, help="Danh sách, vd: lstm,logistic")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="Đánh giá lại mô hình đã lưu")
    p.add_argument("--lottery", default="all")
    p.add_argument("--models", default=None)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("predict", help="Tạo bộ số gợi ý")
    p.add_argument("--lottery", required=True, choices=list(LOTTERIES))
    p.add_argument("--model", default="lstm", choices=MODEL_NAMES + ["random"])
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("pipeline", help="bootstrap + collect + train")
    p.add_argument("--lottery", default="all")
    p.add_argument("--models", default=None)
    p.set_defaults(func=cmd_pipeline)

    p = sub.add_parser("serve", help="Chạy web demo FastAPI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> None:
    ensure_dirs()
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
