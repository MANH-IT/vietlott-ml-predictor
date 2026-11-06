"""Chuẩn hoá và kiểm tra tính hợp lệ của một kỳ quay."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date as Date
from pathlib import Path
from typing import Any, Iterable

from .config import lottery_config


@dataclass(frozen=True)
class Draw:
    lottery: str
    draw_id: str
    date: str
    numbers: tuple[int, ...]
    special: int | None = None
    source: str = "unknown"
    collected_at: str | None = None
    extra: dict[str, Any] = field(default_factory=dict, compare=False)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "lottery": self.lottery,
            "draw_id": self.draw_id,
            "date": self.date,
            "numbers": list(self.numbers),
            "special": self.special,
            "source": self.source,
        }
        if self.collected_at:
            out["collected_at"] = self.collected_at
        if self.extra:
            out["extra"] = self.extra
        return out


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("giá trị boolean không hợp lệ")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise ValueError(f"không phải số nguyên: {value!r}")


def validate_draw(record: dict[str, Any]) -> list[str]:
    """Trả về danh sách lỗi; rỗng nghĩa là bản ghi hợp lệ."""
    errors: list[str] = []
    lottery = record.get("lottery")
    try:
        cfg = lottery_config(str(lottery))
    except ValueError as exc:
        return [str(exc)]

    n_features = cfg["n_features"]
    pick = cfg["pick"]

    draw_id = record.get("draw_id")
    if draw_id is None or str(draw_id).strip() == "":
        errors.append("thiếu draw_id")

    raw_date = record.get("date")
    try:
        Date.fromisoformat(str(raw_date)[:10])
    except (TypeError, ValueError):
        errors.append(f"date không hợp lệ: {raw_date!r}")

    raw_numbers = record.get("numbers")
    if not isinstance(raw_numbers, (list, tuple)):
        errors.append("numbers phải là list")
        return errors

    parsed: list[int] = []
    for item in raw_numbers:
        try:
            parsed.append(_as_int(item))
        except ValueError as exc:
            errors.append(f"số không hợp lệ: {item!r} ({exc})")

    if len(parsed) != pick:
        errors.append(f"cần đúng {pick} số, nhận {len(parsed)}")
    if len(set(parsed)) != len(parsed):
        errors.append("các số bị trùng trong cùng một kỳ")
    out_of_range = [n for n in parsed if not 1 <= n <= n_features]
    if out_of_range:
        errors.append(f"số ngoài khoảng 1..{n_features}: {out_of_range}")

    special = record.get("special")
    if cfg["has_special"]:
        if special is None:
            errors.append("Power 6/55 yêu cầu số đặc biệt")
        else:
            try:
                special_int = _as_int(special)
            except ValueError as exc:
                errors.append(f"số đặc biệt không hợp lệ: {special!r} ({exc})")
            else:
                if not 1 <= special_int <= n_features:
                    errors.append(
                        f"số đặc biệt ngoài khoảng 1..{n_features}: {special_int}"
                    )
                if special_int in parsed:
                    errors.append(f"số đặc biệt {special_int} trùng số chính")
    elif special is not None:
        errors.append("Mega 6/45 không có số đặc biệt")

    return errors


def normalize_draw(
    lottery: str,
    draw_id: str,
    date: str,
    numbers: Iterable[Any],
    special: Any = None,
    source: str = "unknown",
    collected_at: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Draw:
    """Chuyển dữ liệu thô về Draw đã chuẩn hoá (chưa validate)."""
    parsed_numbers = tuple(_as_int(n) for n in numbers)
    parsed_special = None if special is None or special == "" else _as_int(special)
    return Draw(
        lottery=lottery,
        draw_id=str(draw_id).strip(),
        date=str(date)[:10],
        numbers=parsed_numbers,
        special=parsed_special,
        source=source,
        collected_at=collected_at,
        extra=extra or {},
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} JSON lỗi: {exc}") from exc
    return rows


def validate_file(path: Path) -> tuple[int, int, list[tuple[int, list[str]]]]:
    """Kiểm tra một file JSONL. Trả về (tổng, hợp lệ, danh sách lỗi)."""
    rows = read_jsonl(path)
    failures: list[tuple[int, list[str]]] = []
    valid = 0
    for index, row in enumerate(rows, start=1):
        errors = validate_draw(row)
        if errors:
            failures.append((index, errors))
        else:
            valid += 1
    return len(rows), valid, failures
