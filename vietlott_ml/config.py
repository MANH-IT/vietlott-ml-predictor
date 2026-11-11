"""Cấu hình dùng chung cho toàn bộ pipeline."""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

LIVE_JSONL = RAW_DIR / "vietlott_live.jsonl"

# API live (nguồn chính cho kỳ mới nhất). Trả về cả mega645 và power655.
LIVE_API_URL = "https://api.383.im/lottery/live.json"
LIVE_API_TIMEOUT = 20

# Dữ liệu lịch sử do package vietlott-data sinh ra (NDJSON trên GitHub).
HISTORY_URLS = {
    "mega645": "https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl",
    "power655": "https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power655.jsonl",
}

# Tên sản phẩm tương ứng của package vietlott-data (dùng cho fallback crawler).
VIETLOTT_PRODUCT = {
    "mega645": "power_645",
    "power655": "power_655",
}

LOTTERIES = {
    "mega645": {
        "key": "mega645",
        "name": "Mega 6/45",
        "short_name": "Mega",
        "n_features": 45,
        "pick": 6,
        "has_special": False,
        "schedule": "Thứ 4, 6, Chủ nhật",
        "draw_days": (2, 4, 6),  # Python weekday: Mon=0
    },
    "power655": {
        "key": "power655",
        "name": "Power 6/55",
        "short_name": "Power",
        "n_features": 55,
        "pick": 6,
        "has_special": True,
        "schedule": "Thứ 3, 5, 7",
        "draw_days": (1, 3, 5),
    },
}

WINDOW_SIZE = 10
TRAIN_RATIO = 0.8
MIN_HISTORY = 30
RANDOM_SEED = 42
# Số lần lặp baseline ngẫu nhiên để giảm nhiễu của một lần rút.
RANDOM_BASELINE_REPEATS = 50

DISCLAIMER = (
    "Chỉ phục vụ học tập/nghiên cứu. Xổ số là quá trình ngẫu nhiên thuần túy; "
    "các bộ số gợi ý không có khả năng dự đoán kết quả thực tế."
)


def lottery_config(key: str) -> dict:
    if key not in LOTTERIES:
        raise ValueError(
            f"Không hỗ trợ xổ số '{key}'. Chọn một trong: {', '.join(LOTTERIES)}"
        )
    return LOTTERIES[key]


def canonical_path(lottery: str) -> Path:
    return RAW_DIR / f"{lottery}.jsonl"


def model_dir(lottery: str) -> Path:
    return MODELS_DIR / lottery


def report_path(lottery: str) -> Path:
    return REPORTS_DIR / f"{lottery}_metrics.json"


def ensure_dirs() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
