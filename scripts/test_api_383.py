"""Kiểm tra nhanh API 383.im có hoạt động và trả về kết quả hay không.

Chạy: python scripts/test_api_383.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vietlott_ml.config import LIVE_API_URL  # noqa: E402
from vietlott_ml.data_collection import fetch_live_payload, parse_live_payload  # noqa: E402


def main() -> int:
    print(f"GET {LIVE_API_URL}")
    try:
        payload = fetch_live_payload()
    except Exception as exc:
        print(f"THẤT BẠI: {exc}")
        return 1

    print(f"schema      : {payload.get('schema')}")
    print(f"ngày        : {payload.get('d')}")
    print(f"updated_at  : {payload.get('updated_at')}")
    print(f"active      : {payload.get('active')}")
    print(f"region      : {payload.get('region')}")

    draws = parse_live_payload(payload)
    print(f"kỳ hợp lệ   : {len(draws)}")
    for draw in draws:
        print(
            f"  - {draw.lottery}: kỳ {draw.draw_id} ngày {draw.date} "
            f"số {list(draw.numbers)}"
            + (f" đặc biệt {draw.special}" if draw.special else "")
        )

    if not draws:
        print("CẢNH BÁO: API phản hồi nhưng không có kỳ hợp lệ nào.")
        return 1

    print("\nRaw keys:", json.dumps(sorted(payload.keys()), ensure_ascii=False))
    print("\nKẾT LUẬN: API 383.im đang hoạt động.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
