"""Thu thập dữ liệu Vietlott từ API live, dữ liệu lịch sử và crawler dự phòng."""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterable

import requests

from .config import (
    HISTORY_URLS,
    LIVE_API_TIMEOUT,
    LIVE_API_URL,
    LIVE_JSONL,
    RAW_DIR,
    VIETLOTT_PRODUCT,
    canonical_path,
    ensure_dirs,
    lottery_config,
)
from .data_validation import Draw, normalize_draw, read_jsonl, validate_draw

LOGGER = logging.getLogger("vietlott.collection")
VN_TZ = timezone(timedelta(hours=7))


def _now_iso() -> str:
    return datetime.now(VN_TZ).isoformat(timespec="seconds")


def configure_logging() -> None:
    """Ghi log vào console và data/raw/collection.log."""
    ensure_dirs()
    LOGGER.setLevel(logging.INFO)
    if not LOGGER.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        LOGGER.addHandler(console)
        file_handler = logging.FileHandler(RAW_DIR / "collection.log", encoding="utf-8")
        file_handler.setFormatter(fmt)
        LOGGER.addHandler(file_handler)


def write_jsonl(path: Path, draws: Iterable[Draw], append: bool = False) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    count = 0
    with path.open(mode, encoding="utf-8") as fh:
        for draw in draws:
            fh.write(json.dumps(draw.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def fetch_live_payload(timeout: int = LIVE_API_TIMEOUT) -> dict:
    """Gọi API 383.im và trả về JSON. Ném exception nếu không dùng được."""
    response = requests.get(
        LIVE_API_URL,
        timeout=timeout,
        headers={"User-Agent": "vietlott-ml/1.0 (educational project)"},
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("API live trả về dữ liệu không phải object")
    return payload


def parse_live_payload(payload: dict, collected_at: str | None = None) -> list[Draw]:
    """Chuyển payload live thành danh sách Draw cho các loại được hỗ trợ."""
    collected_at = collected_at or _now_iso()
    draws: list[Draw] = []
    for lottery in ("mega645", "power655"):
        cfg = lottery_config(lottery)
        block = payload.get(cfg["key"])
        if not isinstance(block, dict):
            continue
        if block.get("state") not in (None, "complete"):
            continue
        numbers = block.get("numbers") or []
        special = block.get("special") if cfg["has_special"] else None
        if len(numbers) != cfg["pick"]:
            LOGGER.warning("%s: bỏ qua payload live không đủ số: %s", lottery, numbers)
            continue
        draw_id = str(block.get("ky") or "").strip() or str(block.get("d"))
        try:
            draw = normalize_draw(
                lottery=lottery,
                draw_id=draw_id,
                date=block.get("d"),
                numbers=numbers,
                special=special,
                source="383.im",
                collected_at=collected_at,
            )
        except ValueError as exc:
            LOGGER.warning("%s: bỏ qua payload live lỗi: %s", lottery, exc)
            continue
        errors = validate_draw(draw.to_dict())
        if errors:
            LOGGER.warning("%s: bỏ qua payload live không hợp lệ: %s", lottery, errors)
            continue
        draws.append(draw)
    return draws


def fetch_history(lottery: str, timeout: int = 60) -> list[Draw]:
    """Tải dữ liệu lịch sử NDJSON từ repo của package vietlott-data."""
    url = HISTORY_URLS[lottery]
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "vietlott-ml/1.0 (educational project)"},
    )
    response.raise_for_status()
    cfg = lottery_config(lottery)
    draws: list[Draw] = []
    bad = 0
    for line in response.text.splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        result = row.get("result") or []
        special = result[-1] if cfg["has_special"] and len(result) > cfg["pick"] else None
        main = result[: cfg["pick"]] if len(result) >= cfg["pick"] else result
        try:
            draw = normalize_draw(
                lottery=lottery,
                draw_id=row.get("id") or row.get("date"),
                date=row.get("date"),
                numbers=main,
                special=special,
                source="vietlott-data",
                collected_at=str(row.get("process_time") or ""),
            )
        except ValueError:
            bad += 1
            continue
        if validate_draw(draw.to_dict()):
            bad += 1
            continue
        draws.append(draw)
    LOGGER.info(
        "Tải lịch sử %s: %d kỳ hợp lệ, %d kỳ lỗi bị bỏ qua",
        lottery,
        len(draws),
        bad,
    )
    return draws


def crawl_fallback(lottery: str, timeout: int = 240) -> list[Draw]:
    """Dự phòng: chạy CLI vietlott-crawl rồi đọc file NDJSON của package."""
    product = VIETLOTT_PRODUCT[lottery]
    LOGGER.info("API/history lỗi, thử crawler dự phòng: vietlott-crawl %s", product)
    try:
        subprocess.run(
            ["vietlott-crawl", product],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"vietlott-crawl thất bại cho {product}: {exc}") from exc

    try:
        from vietlott.config.products import product_config_map

        raw_path = Path(product_config_map[product].raw_path)
    except Exception as exc:  # pragma: no cover - chỉ lỗi môi trường
        raise RuntimeError(f"Không đọc được cấu hình vietlott-data: {exc}") from exc

    if not raw_path.exists():
        raise RuntimeError(f"Không tìm thấy dữ liệu crawler tại {raw_path}")

    cfg = lottery_config(lottery)
    draws: list[Draw] = []
    for row in read_jsonl(raw_path):
        result = row.get("result") or []
        special = result[-1] if cfg["has_special"] and len(result) > cfg["pick"] else None
        main = result[: cfg["pick"]] if len(result) >= cfg["pick"] else result
        try:
            draw = normalize_draw(
                lottery=lottery,
                draw_id=row.get("id") or row.get("date"),
                date=row.get("date"),
                numbers=main,
                special=special,
                source="vietlott-crawl",
                collected_at=_now_iso(),
            )
        except ValueError:
            continue
        if not validate_draw(draw.to_dict()):
            draws.append(draw)
    LOGGER.info("Crawler dự phòng trả về %d kỳ hợp lệ", len(draws))
    return draws


def load_canonical(lottery: str) -> list[Draw]:
    """Đọc dataset chuẩn đã lưu, trả về các kỳ hợp lệ, không trùng, đã sắp xếp."""
    path = canonical_path(lottery)
    draws: list[Draw] = []
    invalid = 0
    for row in read_jsonl(path):
        if validate_draw(row):
            invalid += 1
            continue
        draws.append(
            Draw(
                lottery=row["lottery"],
                draw_id=str(row["draw_id"]),
                date=str(row["date"])[:10],
                numbers=tuple(int(n) for n in row["numbers"]),
                special=None if row.get("special") is None else int(row["special"]),
                source=row.get("source", "canonical"),
                collected_at=row.get("collected_at"),
                extra=row.get("extra") or {},
            )
        )
    if invalid:
        LOGGER.warning("%s: bỏ qua %d bản ghi không hợp lệ", lottery, invalid)
    return merge_draws([], draws, prefer_first=False)


def merge_draws(
    existing: Iterable[Draw], new: Iterable[Draw], prefer_first: bool = True
) -> list[Draw]:
    """Gộp và khử trùng lặp theo (lottery, date), ưu tiên bản ghi đầy đủ hơn."""
    merged: dict[tuple[str, str], Draw] = {}
    for draw in list(existing) + list(new):
        key = (draw.lottery, draw.date)
        current = merged.get(key)
        if current is None:
            merged[key] = draw
            continue
        # Ưu tiên bản ghi có số đặc biệt / draw_id thật / nguồn live.
        current_score = _completeness(current)
        new_score = _completeness(draw)
        if new_score > current_score:
            merged[key] = draw
        elif new_score == current_score and not prefer_first:
            merged[key] = draw
    return sorted(merged.values(), key=lambda d: (d.date, d.lottery))


def _completeness(draw: Draw) -> int:
    score = 0
    if draw.special is not None:
        score += 4
    if draw.draw_id and draw.draw_id != draw.date:
        score += 2
    if draw.source == "383.im":
        score += 1
    return score


def bootstrap(lottery: str) -> int:
    """Tải toàn bộ lịch sử và ghi dataset chuẩn."""
    ensure_dirs()
    try:
        history = fetch_history(lottery)
    except Exception as exc:
        LOGGER.warning("Không tải được lịch sử từ GitHub (%s)", exc)
        history = crawl_fallback(lottery)
    existing = load_canonical(lottery)
    merged = merge_draws(existing, history)
    count = write_jsonl(canonical_path(lottery), merged)
    LOGGER.info(
        "Bootstrap %s: %d kỳ (thêm mới %d) -> %s",
        lottery,
        count,
        len(merged) - len(existing),
        canonical_path(lottery),
    )
    return count


def _append_live_file(draws: list[Draw]) -> None:
    """Append các kỳ mới vào vietlott_live.jsonl, bỏ qua bản ghi đã có."""
    existing_keys: set[tuple[str, str]] = set()
    for row in read_jsonl(LIVE_JSONL):
        existing_keys.add((row.get("lottery"), str(row.get("date"))[:10]))
    new_draws = [d for d in draws if (d.lottery, d.date) not in existing_keys]
    if new_draws:
        write_jsonl(LIVE_JSONL, new_draws, append=True)


def collect_live(lotteries: Iterable[str] | None = None) -> dict[str, int]:
    """Lấy kết quả mới nhất từ API, append vào file live và dataset chuẩn."""
    configure_logging()
    ensure_dirs()
    selected = list(lotteries or ("mega645", "power655"))
    collected: list[Draw] = []
    try:
        payload = fetch_live_payload()
        collected = parse_live_payload(payload)
        LOGGER.info(
            "API 383.im OK | updated_at=%s | nhận %d kỳ",
            payload.get("updated_at"),
            len(collected),
        )
    except Exception as exc:
        LOGGER.warning("API 383.im lỗi: %s", exc)
        for lottery in selected:
            try:
                collected.extend(crawl_fallback(lottery))
            except Exception as fallback_exc:
                LOGGER.error("%s: fallback thất bại: %s", lottery, fallback_exc)

    _append_live_file(collected)

    result: dict[str, int] = {}
    for lottery in selected:
        items = [d for d in collected if d.lottery == lottery]
        existing = load_canonical(lottery)
        merged = merge_draws(existing, items)
        added = len(merged) - len(existing)
        write_jsonl(canonical_path(lottery), merged)
        result[lottery] = added
        LOGGER.info(
            "Cập nhật %s: %d kỳ mới, tổng %d, nguồn=%s",
            lottery,
            added,
            len(merged),
            ",".join(sorted({d.source for d in items})) or "không có",
        )
    return result
