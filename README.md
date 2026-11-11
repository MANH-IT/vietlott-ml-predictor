# Vietlott ML Predictor

Pipeline học máy thực hành trên dữ liệu xổ số Vietlott (**Mega 6/45** và **Power 6/55**).

> **Tuyên bố miễn trừ trách nhiệm:** Dự án **chỉ phục vụ học tập/nghiên cứu**. Xổ số là quá trình ngẫu nhiên thuần túy. Các bộ số gợi ý **không có khả năng dự đoán** kết quả thực tế. Giá trị của dự án nằm ở pipeline học máy hoàn chỉnh và việc đánh giá trung thực giới hạn của mô hình.

---

## Mục lục

1. [Giới thiệu](#1-giới-thiệu)
2. [Tính năng chính](#2-tính-năng-chính)
3. [Yêu cầu hệ thống](#3-yêu-cầu-hệ-thống)
4. [Cài đặt](#4-cài-đặt)
5. [Cấu trúc dự án](#5-cấu-trúc-dự-án)
6. [Bắt đầu nhanh](#6-bắt-đầu-nhanh)
7. [Tham chiếu CLI](#7-tham-chiếu-cli)
8. [Dữ liệu](#8-dữ-liệu)
9. [Tiền xử lý và đặc trưng](#9-tiền-xử-lý-và-đặc-trưng)
10. [Các mô hình](#10-các-mô-hình)
11. [Tuning hyperparameters](#11-tuning-hyperparameters)
12. [Huấn luyện và đánh giá](#12-huấn-luyện-và-đánh-giá)
13. [Dự đoán](#13-dự-đoán)
14. [Web demo và API](#14-web-demo-và-api)
15. [Notebook phân tích](#15-notebook-phân-tích)
16. [Chạy trên Windows với run.bat](#16-chạy-trên-windows-với-runbat)
17. [Docker](#17-docker)
18. [Kiểm thử và CI](#18-kiểm-thử-và-ci)
19. [Tái lập kết quả](#19-tái-lập-kết-quả)
20. [Xử lý sự cố](#20-xử-lý-sự-cố)
21. [Giới hạn và miễn trừ](#21-giới-hạn-và-miễn-trừ)

---

## 1. Giới thiệu

Dự án xây dựng một pipeline học máy **đầu-cuối** cho dữ liệu xổ số:

- Thu thập dữ liệu lịch sử và kỳ mới nhất.
- Chuẩn hoá và kiểm tra dữ liệu.
- Xây dựng đặc trưng (multi-hot, cửa sổ trượt, đặc trưng bảng).
- Huấn luyện 7 mô hình và so sánh với baseline ngẫu nhiên.
- Tuning hyperparameters trên validation (không dùng test).
- Đánh giá bằng kiểm định thống kê, xuất báo cáo trung thực.
- CLI, web demo FastAPI, notebook phân tích, Docker.

Mục tiêu không phải là "dự đoán trúng" — mà là trình diễn một quy trình ML đúng đắn, có
kiểm soát rò rỉ dữ liệu và kết luận trung thực.

## 2. Tính năng chính

- Thu thập dữ liệu real-time từ API `api.383.im`, lịch sử từ `vietlott-data`, crawler dự phòng.
- Chuẩn hoá + kiểm tra dữ liệu (khoảng số, trùng lặp, đủ số, số đặc biệt, ngày).
- 7 mô hình: **LSTM, BiLSTM, Logistic Regression, Random Forest, Markov, MLP, XGBoost** (+ baseline ngẫu nhiên).
- Chia train/test **theo thời gian**, kiểm định z so với baseline ngẫu nhiên.
- Tuning hyperparameters bằng grid search trên validation nội bộ phần train.
- Web demo FastAPI + giao diện tạo bộ số gợi ý.
- CLI đầy đủ: `bootstrap`, `collect`, `validate`, `tune`, `train`, `evaluate`, `predict`, `pipeline`, `serve`.
- 2 notebook Jupyter: EDA và ma trận nhầm lẫn.
- Docker + docker compose, CI (ruff + pytest).

## 3. Yêu cầu hệ thống

- **Python 3.12** (khuyến nghị). Các bản Python quá mới có thể thiếu wheel cho `torch`.
- RAM: tối thiểu ~2 GB (tuning Random Forest/XGBoost dùng thêm).
- GPU: không bắt buộc. PyTorch tự dùng CPU nếu không có CUDA.
- Hệ điều hành: Linux, macOS, Windows (dùng `run.bat`).

## 4. Cài đặt

### 4.1 Conda (khuyến nghị)

```bash
conda create -n vietlott python=3.12 -y
conda activate vietlott
pip install -r requirements.txt
```

### 4.2 venv

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4.3 PyTorch GPU (tuỳ chọn)

Cài PyTorch bản CUDA phù hợp **trước** khi cài requirements:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

### 4.4 Phụ thuộc cho notebook (tuỳ chọn)

```bash
pip install -e ".[notebooks]"    # jupyter, matplotlib, seaborn
```

### 4.5 Phụ thuộc cho phát triển (tuỳ chọn)

```bash
pip install -e ".[dev]"          # pytest, httpx, ruff
```

## 5. Cấu trúc dự án

```text
Vietlott_VN/
├── vietlott_ml/
│   ├── config.py             # cấu hình chung, danh sách lottery, đường dẫn
│   ├── data_collection.py    # API live + lịch sử + crawler dự phòng
│   ├── data_validation.py    # dataclass Draw, validate & normalize
│   ├── dataset.py            # multi-hot, cửa sổ trượt, feature engineering
│   ├── metrics.py            # hit rate, lift, kiểm định z, so sánh mô hình
│   ├── hyperparams.py        # defaults, search space, lưu/đọc tham số đã tune
│   ├── tuning.py             # grid search trên validation (không dùng test)
│   ├── training.py           # train/evaluate/report
│   ├── predict.py            # sinh bộ số gợi ý
│   ├── api.py                # FastAPI web demo
│   ├── cli.py                # giao diện dòng lệnh
│   ├── models/
│   │   ├── base.py           # tiện ích Torch, positive_probability
│   │   ├── lstm_model.py     # LSTM / BiLSTM
│   │   ├── mlp_model.py      # MLP
│   │   ├── logistic_model.py # Logistic Regression
│   │   ├── forest_model.py   # Random Forest
│   │   ├── xgboost_model.py  # XGBoost
│   │   └── markov_model.py   # Markov
│   └── templates/index.html
├── scripts/test_api_383.py   # kiểm tra nhanh API live
├── tests/                    # unit + API + training + tuning tests
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   └── 02_model_confusion_matrices.ipynb
├── data/raw/                 # dữ liệu JSONL (sinh ra khi chạy)
├── models/                   # mô hình đã train (sinh ra khi chạy)
├── reports/                  # metrics + báo cáo markdown
├── run.bat                   # tiện ích chạy trên Windows
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml            # cấu hình project, ruff, pytest
└── requirements.txt
```

Các thư mục `data/raw/`, `models/`, `reports/` và file `models/**/hyperparams.json`
được git-ignore vì là sản phẩm sinh ra khi chạy.

## 6. Bắt đầu nhanh

```bash
# 1. (Tuỳ chọn) kiểm tra API live
python scripts/test_api_383.py

# 2. Tải dữ liệu lịch sử
python -m vietlott_ml.cli bootstrap --lottery all

# 3. Lấy kết quả mới nhất từ API live
python -m vietlott_ml.cli collect --lottery all

# 4. Kiểm tra dữ liệu
python -m vietlott_ml.cli validate --lottery all

# 5. (Tuỳ chọn) Tuning hyperparameters trên validation
python -m vietlott_ml.cli tune --lottery all

# 6. Huấn luyện + đánh giá + xuất báo cáo
python -m vietlott_ml.cli train --lottery all

# 7. Tạo bộ số gợi ý
python -m vietlott_ml.cli predict --lottery power655 --model lstm
python -m vietlott_ml.cli predict --lottery mega645 --model markov

# 8. Web demo
python -m vietlott_ml.cli serve --host 0.0.0.0 --port 8000
# mở http://localhost:8000
```

Chạy toàn bộ pipeline (bootstrap + collect + train) trong một lệnh:

```bash
python -m vietlott_ml.cli pipeline --lottery all
```

## 7. Tham chiếu CLI

```text
python -m vietlott_ml.cli <command> [options]
```

| Lệnh | Mô tả | Tuỳ chọn |
|---|---|---|
| `bootstrap` | Tải dữ liệu lịch sử từ `vietlott-data`, fallback crawler | `--lottery {all,mega645,power655}` |
| `collect` | Lấy kết quả kỳ mới từ API live, append vào JSONL | `--lottery`, `--watch`, `--interval` (giây, mặc định 300) |
| `validate` | Kiểm tra dữ liệu đã lưu, in số bản ghi hợp lệ | `--lottery` |
| `tune` | Grid search hyperparameters trên validation | `--lottery`, `--models`, `--val-ratio` (mặc định 0.2) |
| `train` | Huấn luyện, đánh giá trên test, xuất báo cáo | `--lottery`, `--models` |
| `evaluate` | Đánh giá lại mô hình đã lưu (không train lại) | `--lottery`, `--models` |
| `predict` | Sinh bộ số gợi ý cho kỳ kế tiếp | `--lottery` (bắt buộc), `--model`, `--seed` |
| `pipeline` | `bootstrap` + `collect` + `train` | `--lottery`, `--models` |
| `serve` | Chạy web demo FastAPI | `--host`, `--port`, `--reload` |

Tên mô hình hợp lệ cho `--models` và `--model`:
`lstm`, `bilstm`, `logistic`, `random_forest`, `markov`, `mlp`, `xgboost`.
Riêng `predict` còn hỗ trợ `random`.

Ví dụ:

```bash
# Chỉ train một vài mô hình cho một lottery
python -m vietlott_ml.cli train --lottery mega645 --models lstm,markov,logistic

# Tune riêng nhóm torch với tỉ lệ validation 25%
python -m vietlott_ml.cli tune --lottery power655 --models lstm,bilstm,mlp --val-ratio 0.25

# Predict với seed cố định để tái lập
python -m vietlott_ml.cli predict --lottery power655 --model random --seed 42

# Thu thập lặp mỗi 5 phút
python -m vietlott_ml.cli collect --lottery all --watch --interval 300
```

## 8. Dữ liệu

### 8.1 Nguồn dữ liệu

| Nguồn | Vai trò |
|---|---|
| `https://api.383.im/lottery/live.json` | Kỳ mới nhất (real-time/gần real-time) |
| `vietvudanh/vietlott-data` (NDJSON trên GitHub) | Dữ liệu lịch sử để train |
| `vietlott-data` (package, crawler) | Fallback khi API/lịch sử lỗi |
| `vietlott.vn` | Nguồn gốc cuối cùng của dữ liệu |

### 8.2 Tệp dữ liệu

- `data/raw/mega645.jsonl` — dữ liệu chuẩn Mega 6/45.
- `data/raw/power655.jsonl` — dữ liệu chuẩn Power 6/55.
- `data/raw/vietlott_live.jsonl` — mỗi lần `collect` append kết quả live kèm `source`, `collected_at`.
- `data/raw/collection.log` — log thu thập.

### 8.3 Định dạng JSONL

Mỗi dòng là một kỳ quay:

```json
{
  "lottery": "power655",
  "draw_id": "01422",
  "date": "2026-11-10",
  "numbers": [1, 2, 21, 38, 39, 51],
  "special": 12,
  "source": "383.im",
  "collected_at": "2026-11-10T18:40:00+07:00"
}
```

Quy tắc kiểm tra (`validate`):

- `lottery` thuộc `mega645` / `power655`.
- `draw_id` không rỗng; `date` đúng ISO `YYYY-MM-DD`.
- Đúng `pick` số (6), không trùng, nằm trong `1..n_features` (45 hoặc 55).
- `special` (nếu có) trong `1..n_features`, không trùng 6 số chính.
- Các số được **sắp xếp tăng dần** khi chuẩn hoá.

### 8.4 Lịch quay và thu thập định kỳ

- Mega 6/45: Thứ 4, 6, Chủ nhật.
- Power 6/55: Thứ 3, 5, 7 (Chủ nhật).

Chạy vòng lặp mỗi 5 phút:

```bash
python -m vietlott_ml.cli collect --lottery all --watch --interval 300
```

Hoặc cron (18:05 các ngày có quay; cron: 0=CN, 1=T2, ..., 6=T7):

```cron
5 18 * * 0,2,3,4,5,6 cd /path/to/Vietlott_VN && python -m vietlott_ml.cli collect --lottery all >> data/raw/cron.log 2>&1
```

## 9. Tiền xử lý và đặc trưng

### 9.1 Ma trận multi-hot

`load_matrix(lottery)` trả về `matrix` kích thước `(n_draws, n_features)`:

- Hàng: một kỳ quay.
- Cột: một số (1..F).
- Giá trị `1` nếu số đó xuất hiện trong kỳ, ngược lại `0`.

Số đặc biệt của Power 6/55 **không** nằm trong ma trận chính; nó được dự đoán
riêng bằng phân phối tần suất (xem mục 13).

### 9.2 Cửa sổ trượt (sequence)

`make_sequences(matrix, window=10, min_target=30)`:

- `X`: `(n_samples, window, F)` — 10 kỳ trước.
- `y`: `(n_samples, F)` — multi-hot kỳ kế tiếp.
- `target_index`: vị trí kỳ đích trong ma trận gốc.

Chỉ tạo mẫu khi đã có tối thiểu `MIN_HISTORY = 30` kỳ lịch sử.

### 9.3 Đặc trưng bảng (tabular)

`make_tabular(matrix)` tạo **một mẫu cho mỗi cặp (kỳ, số)**. Tại thời điểm `t`,
9 đặc trưng của từng số (chỉ dùng dữ liệu `0..t`) dùng để dự đoán kỳ `t+1`:

| Đặc trưng | Ý nghĩa |
|---|---|
| `freq_5`, `freq_10`, `freq_20`, `freq_50` | Tần suất trong 5/10/20/50 kỳ gần nhất (đã chuẩn hoá) |
| `gap` | Số kỳ kể từ lần xuất hiện gần nhất |
| `is_last` | Số có xuất hiện ở kỳ `t` không |
| `freq_all` | Tần suất trung bình toàn bộ lịch sử |
| `avg_gap` | Khoảng cách trung bình giữa các lần xuất hiện |
| `n_appear` | Tổng số lần xuất hiện |

### 9.4 Chia train/test theo thời gian

`split_by_target(target_index, train_ratio=0.8)`:

- Sắp theo thời gian, **không xáo trộn**.
- 80% đầu là train, 20% cuối là test.
- `cutoff` là vị trí kỳ cuối của train; mọi đặc trưng của test chỉ dùng dữ liệu
  `<= cutoff`, tránh rò rỉ tương lai.

## 10. Các mô hình

Tất cả mô hình đều dự đoán xác suất xuất hiện của từng số, sau đó chọn **top 6**
(`k = pick`).

| Mô hình | Key | Cách tiếp cận |
|---|---|---|
| **LSTM** | `lstm` | Cửa sổ 10 kỳ multi-hot → 2 lớp LSTM + Dropout + Dense sigmoid; BCE loss, early stopping. |
| **BiLSTM** | `bilstm` | Như LSTM nhưng 2 lớp bidirectional; mặc định mean pooling. |
| **Logistic Regression** | `logistic` | Một bộ phân loại nhị phân cho mỗi số; features bảng, chuẩn hoá StandardScaler, `class_weight=balanced`. |
| **Random Forest** | `random_forest` | Một rừng nhị phân cho mỗi số, cùng bộ features bảng. |
| **Markov** | `markov` | Ma trận chuyển đổi số → số; điểm của số `j` là tổng xác suất chuyển từ các số trong kỳ trước, trộn với phân phối biên. |
| **MLP** | `mlp` | Multi-hot kỳ trước → 2 lớp ẩn → sigmoid cho kỳ kế tiếp. |
| **XGBoost** | `xgboost` | Một bộ phân loại gradient boosting cho mỗi số, cùng bộ features bảng. |
| **Random baseline** | `random` | Chọn ngẫu nhiên `k` số (chỉ dùng cho so sánh/predict). |

### 10.1 Hyperparameters mặc định

Định nghĩa tại `vietlott_ml/hyperparams.py::DEFAULT_PARAMS`.

| Mô hình | Tham số mặc định |
|---|---|
| `lstm` | `hidden1=128, hidden2=64, dropout=0.2, bidirectional=False, pooling=auto, lr=1e-3, epochs=120, batch_size=64, patience=15, weight_decay=1e-5, val_fraction=0.1` |
| `bilstm` | Như LSTM với `bidirectional=True` |
| `mlp` | `hidden1=256, hidden2=128, dropout=0.2, lr=1e-3, epochs=120, batch_size=64, patience=15, weight_decay=1e-5, val_fraction=0.1` |
| `logistic` | `C=1.0, max_iter=1000` |
| `random_forest` | `n_estimators=100, max_depth=12, min_samples_leaf=10, n_jobs=1` |
| `xgboost` | `n_estimators=100, max_depth=6, subsample=0.8, colsample_bytree=0.8, n_jobs=1` |
| `markov` | `alpha=1.0, transition_weight=0.8` |

Ghi chú: `n_jobs=1` được chọn có chủ đích cho Random Forest và XGBoost vì huấn
luyện hàng chục bộ phân loại nhỏ; đa luồng gây tranh chấp và **chậm hơn** đáng kể
(đã đo: Random Forest nhanh gấp ~2 lần với `n_jobs=1`).

## 11. Tuning hyperparameters

### 11.1 Nguyên tắc chống rò rỉ

Tuning **chỉ dùng phần train**, không bao giờ chạm test:

1. Chia train/test theo thời gian như bình thường (80/20).
2. Trong phần train, lấy **20% cuối** (theo thời gian) làm validation.
3. Grid search trên fit (80% đầu của train) → đánh giá trên validation.
4. Chọn bộ tham số tốt nhất theo `(avg_hits, -log_loss)`.
5. Chỉ sau đó `train` mới huấn luyện lại trên **toàn bộ train** và đánh giá test.

### 11.2 Không gian tìm kiếm

Định nghĩa tại `vietlott_ml/hyperparams.py::SEARCH_SPACES`:

| Mô hình | Không gian | Số tổ hợp |
|---|---|---|
| `lstm`, `bilstm` | `hidden1 ∈ {64,128}`, `dropout ∈ {0.15,0.3}`, `lr ∈ {1e-3,5e-4}` | 8 |
| `mlp` | `hidden1 ∈ {128,256}`, `hidden2 ∈ {64,128}`, `dropout ∈ {0.1,0.3}` | 8 |
| `logistic` | `C ∈ {0.1,1.0,10.0}` | 3 |
| `random_forest` | `n_estimators ∈ {100,200}`, `max_depth ∈ {8,12,20}`, `min_samples_leaf ∈ {5,10}` | 12 |
| `xgboost` | `n_estimators ∈ {100,200}`, `max_depth ∈ {4,6}`, `subsample=0.8` | 4 |
| `markov` | `alpha ∈ {0.5,1.0,2.0}`, `transition_weight ∈ {0.5,0.8,0.95}` | 9 |

Khi tune, mô hình PyTorch dùng `epochs=60`, `patience=8` để giới hạn thời gian;
các giá trị này **không** được lưu vào kết quả.

### 11.3 Lệnh

```bash
# Tune tất cả mô hình, cả hai lottery
python -m vietlott_ml.cli tune --lottery all

# Tune một số mô hình với tỉ lệ validation khác
python -m vietlott_ml.cli tune --lottery power655 --models lstm,mlp --val-ratio 0.25
```

Sau khi tune, chạy lại `train` để áp dụng:

```bash
python -m vietlott_ml.cli train --lottery all
```

### 11.4 Tệp kết quả

Lưu tại `models/<lottery>/hyperparams.json` (git-ignore), ví dụ:

```json
{
  "lottery": "mega645",
  "generated_at": "2026-11-11T16:44:27+07:00",
  "window": 10,
  "train_ratio": 0.8,
  "val_ratio": 0.2,
  "n_val_targets": 217,
  "models": {
    "markov": {
      "params": {"alpha": 0.5, "transition_weight": 0.5},
      "val_metrics": {"avg_hits": 0.793, "log_loss": 0.5277},
      "n_candidates": 9,
      "tuning_seconds": 0.1
    }
  }
}
```

`train` tự đọc file này và gộp với `DEFAULT_PARAMS`. Tune một phần (`--models`)
sẽ **giữ nguyên** kết quả của các mô hình không được tune lại.

## 12. Huấn luyện và đánh giá

### 12.1 Chỉ số đánh giá

Metric chính là số số trúng trung bình mỗi kỳ (`avg_hits`) khi chọn top `k` số:

| Chỉ số | Ý nghĩa |
|---|---|
| `avg_hits` | Trung bình số trúng/kỳ |
| `median_hits`, `std_hits` | Trung vị, độ lệch chuẩn |
| `ci95_low`, `ci95_high` | Khoảng tin cậy 95% (xấp xỉ chuẩn) |
| `precision_at_6`, `recall_at_6` | `avg_hits / k` (bằng nhau vì luôn chọn đúng `k` số) |
| `expected_random_hits` | Kỳ vọng ngẫu nhiên `k² / F` |
| `lift_vs_random` | `avg_hits / expected_random_hits` |
| `z_score`, `p_value` | Kiểm định z so với kỳ vọng ngẫu nhiên (phương sai siêu bội) |
| `significant_5pct` | `p < 0.05` và `avg_hits > expected` |
| `coverage_at_least_one` | Tỉ lệ kỳ trúng ít nhất 1 số |
| `hit_distribution` | Phân bố số kỳ theo số trúng (0..k) |
| `brier_score`, `log_loss` | Chất lượng xác suất (dùng cho tuning) |

Kỳ vọng ngẫu nhiên: `6 × 6 / F` → **0.800** cho Mega (F=45), **0.655** cho Power (F=55).

### 12.2 Baseline ngẫu nhiên

Baseline lặp **50 seed**, gộp toàn bộ hit lại thành một phân bố để giảm nhiễu
của một lần rút. Nhờ đó `avg_hits` tiến sát kỳ vọng lý thuyết và khoảng tin cậy
phản ánh đúng độ bất định.

### 12.3 Tệp đầu ra

Sau `train`, mỗi lottery sinh ra:

- `reports/<lottery>_metrics.json` — toàn bộ metrics + hyperparameters đã dùng.
- `reports/<lottery>_report.md` — báo cáo markdown so sánh mô hình + kết luận.
- `reports/<lottery>_test_predictions.jsonl` — dự đoán từng kỳ test của mọi mô hình.
- `models/<lottery>/<model>.pt` hoặc `.joblib` — mô hình đã lưu.

### 12.4 Kết quả hiện tại

Trên dữ liệu mới nhất (Mega: 1.389 kỳ; Power: 1.421 kỳ; test 272/279 kỳ):

| Lottery | Mô hình | Avg hits | Ngẫu nhiên | Lift | p-value | Có ý nghĩa 5%? |
|---|---|---:|---:|---:|---:|:---:|
| Mega 6/45 | BiLSTM | 0.824 | 0.800 | 1.029 | 0.621 | Không |
| Mega 6/45 | Markov | 0.801 | 0.800 | 1.002 | 0.975 | Không |
| Mega 6/45 | Logistic | 0.798 | 0.800 | 0.997 | 0.963 | Không |
| Mega 6/45 | Random Forest | 0.765 | 0.800 | 0.956 | 0.458 | Không |
| Mega 6/45 | XGBoost | 0.761 | 0.800 | 0.951 | 0.412 | Không |
| Mega 6/45 | MLP | 0.754 | 0.800 | 0.942 | 0.330 | Không |
| Mega 6/45 | LSTM | 0.746 | 0.800 | 0.933 | 0.259 | Không |
| Power 6/55 | MLP | 0.742 | 0.655 | 1.134 | 0.045 | Danh nghĩa |
| Power 6/55 | BiLSTM | 0.703 | 0.655 | 1.073 | 0.271 | Không |
| Power 6/55 | LSTM | 0.634 | 0.655 | 0.969 | 0.644 | Không |
| Power 6/55 | Random Forest | 0.627 | 0.655 | 0.958 | 0.531 | Không |
| Power 6/55 | Markov | 0.627 | 0.655 | 0.958 | 0.531 | Không |
| Power 6/55 | XGBoost | 0.620 | 0.655 | 0.947 | 0.429 | Không |
| Power 6/55 | Logistic | 0.602 | 0.655 | 0.920 | 0.229 | Không |

Lưu ý về MLP của Power 6/55 (`p = 0.045`): đây là **p-value danh nghĩa** khi xét
riêng lẻ. Vì có 7 mô hình × 2 lottery = 14 phép so sánh, khả năng xuất hiện ít
nhất một kết quả `p < 0.05` do may mắn là đáng kể. **Chưa thể coi là tín hiệu
thật**; cần kiểm chứng bằng walk-forward trên dữ liệu mới.

Không mô hình nào vượt trội một cách thuyết phục so với ngẫu nhiên. Đây là kết quả
**đúng với bản chất ngẫu nhiên của xổ số** và là một phần quan trọng của dự án.

## 13. Dự đoán

```bash
python -m vietlott_ml.cli predict --lottery power655 --model lstm
python -m vietlott_ml.cli predict --lottery mega645 --model markov --seed 42
```

Kết quả là JSON, ví dụ:

```json
{
  "lottery": "power655",
  "lottery_name": "Power 6/55",
  "model": "lstm",
  "model_name": "LSTM",
  "numbers": [3, 12, 21, 28, 40, 51],
  "ranked": [
    {"number": 28, "probability": 0.1421},
    {"number": 3, "probability": 0.1388}
  ],
  "latest_draw": {
    "draw_id": "01422",
    "date": "2026-11-10",
    "numbers": [1, 2, 21, 38, 39, 51],
    "special": 12
  },
  "next_draw_estimate": "2026-11-12",
  "n_draws_used": 1421,
  "window": 10,
  "special": 7,
  "special_probability": 0.031,
  "special_top3": [
    {"number": 7, "probability": 0.031},
    {"number": 12, "probability": 0.029},
    {"number": 23, "probability": 0.028}
  ],
  "disclaimer": "Chỉ phục vụ học tập/nghiên cứu..."
}
```

- `ranked` chỉ gồm `k` số được chọn, sắp theo xác suất giảm dần.
- Với Power 6/55, `special` được chọn từ phân phối số đặc biệt (làm mượt Laplace
  trên 200 kỳ gần nhất), **không** dùng mô hình chính và không trùng 6 số chính.
- `next_draw_estimate` ước lượng ngày quay kế tiếp theo lịch cố định.

## 14. Web demo và API

```bash
python -m vietlott_ml.cli serve --host 0.0.0.0 --port 8000
```

Mở http://localhost:8000.

| Phương thức | Endpoint | Mô tả |
|---|---|---|
| `GET` | `/` | Giao diện web |
| `GET` | `/api/health` | Kiểm tra tình trạng |
| `GET` | `/api/lotteries` | Thông tin lottery, kỳ gần nhất, mô hình khả dụng |
| `GET` | `/api/latest/{lottery}` | Kỳ gần nhất |
| `POST` | `/api/predict` | Tạo bộ số gợi ý |
| `GET` | `/api/metrics/{lottery}` | Báo cáo đánh giá (JSON) |
| `GET` | `/docs` | Swagger UI |

Ví dụ `curl`:

```bash
curl http://localhost:8000/api/lotteries

curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"lottery": "power655", "model": "lstm"}'

curl http://localhost:8000/api/metrics/mega645
```

Body của `POST /api/predict`:

```json
{"lottery": "power655", "model": "lstm", "seed": 42}
```

`seed` là tuỳ chọn; dùng khi muốn tái lập kết quả của mô hình `random`.

## 15. Notebook phân tích

Hai notebook Jupyter (đã chạy sẵn output) trong `notebooks/`:

- `01_data_analysis.ipynb` — tổng quan dữ liệu, ma trận multi-hot, tần suất từng số,
  gap, tần suất gần đây, ma trận đồng xuất hiện, phân bố theo thời gian, phân bố
  đặc trưng.
- `02_model_confusion_matrices.ipynb` — ma trận nhầm lẫn của từng mô hình trên tập
  test (theo ngưỡng 0.5 và theo top-k), precision/recall theo từng số, quét ngưỡng,
  so sánh với baseline ngẫu nhiên lý thuyết.

```bash
pip install -e ".[notebooks]"
jupyter notebook notebooks/
```

Notebook tự dùng mô hình đã lưu trong `models/`; nếu chưa có sẽ huấn luyện lại
(chậm hơn). Chạy `train --lottery all` trước để dùng mô hình sẵn có.

Ghi chú: ở ngưỡng 0.5, vì xác suất nền rất thấp (`k/F ≈ 0.11–0.13`), hầu hết mô
hình dự đoán toàn âm → `accuracy` cao nhưng `recall = 0`. Notebook dùng thêm
ma trận nhầm lẫn **theo top-k** để phản ánh cách dùng thực tế.

## 16. Chạy trên Windows với run.bat

`run.bat` là tiện ích (ASCII + CRLF) để chạy nhanh CLI:

```bat
run.bat                              :: web demo (mặc định)
run.bat bootstrap
run.bat collect
run.bat validate
run.bat tune --lottery power655
run.bat train --models lstm,mlp
run.bat evaluate
run.bat predict mega645 markov
run.bat pipeline
run.bat serve --host 0.0.0.0 --port 8000
run.bat test
run.bat notebook
run.bat help
```

Nếu `python` không có trong `PATH`:

```bat
set PYTHON=C:\path\to\python.exe
run.bat train
```

## 17. Docker

Build và chạy web demo:

```bash
docker build -t vietlott-ml .
docker run --rm -p 8000:8000 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/reports:/app/reports" \
  vietlott-ml
```

Docker Compose:

```bash
docker compose up --build
```

Lưu ý:

- Image **không** chứa dữ liệu/mô hình (do git-ignore). Hãy mount như trên, hoặc
  chạy `pipeline`/`train` bên trong container.
- Có thể bật GPU bằng cách bỏ comment `runtime: nvidia` trong `docker-compose.yml`
  (cần NVIDIA Container Toolkit trên host).

## 18. Kiểm thử và CI

```bash
python -m pytest -q      # unit + API + training + tuning tests
ruff check .             # lint (cấu hình trong pyproject.toml)
```

Bộ test bao gồm:

- Validation dữ liệu và chuẩn hoá.
- Dataset: shape, tính đúng đắn của cửa sổ trượt, chống rò rỉ tương lai.
- Smoke test từng mô hình.
- Metrics: hit rate, lift, kiểm định thống kê.
- API (FastAPI) và training.
- Tuning: sinh tổ hợp, gộp tham số, grid search nhỏ trên dữ liệu tổng hợp.

CI (`.github/workflows/ci.yml`) chạy `ruff check .` và `pytest -q` trên Python 3.12
mỗi khi push lên `main`/`master` hoặc mở pull request.

## 19. Tái lập kết quả

- `RANDOM_SEED = 42` cho NumPy/PyTorch/sklearn.
- Chia train/test theo thời gian, không xáo trộn.
- Baseline ngẫu nhiên lặp 50 seed cố định.
- Mô hình PyTorch có thể vẫn lệch nhẹ giữa các lần chạy do tính không tất định của
  một số phép toán trên GPU; để tái lập chính xác, nên chạy trên CPU.
- Tuning lưu tham số vào `models/<lottery>/hyperparams.json`; giữ file này để
  `train` cho ra cùng cấu hình.

## 20. Xử lý sự cố

| Hiện tượng | Nguyên nhân / Cách xử lý |
|---|---|
| `cần tối thiểu ... kỳ, hiện có ...` | Chưa có dữ liệu. Chạy `bootstrap --lottery all`. |
| `Chưa có mô hình ... Hãy chạy train` | Chưa train. Chạy `train --lottery <key>`. |
| `Thư viện xgboost chưa được cài đặt` | `pip install xgboost`. |
| Lỗi import `torch` | Cài PyTorch đúng bản CPU/CUDA; kiểm tra Python 3.12. |
| `ModuleNotFoundError: fastapi` | `pip install -r requirements.txt`. |
| API live lỗi/timeout | Bỏ qua `collect`; dữ liệu lịch sử từ `bootstrap` vẫn dùng được. |
| `validate` báo lỗi bản ghi | Xem dòng lỗi; chạy lại `bootstrap` hoặc `collect` để làm mới. |
| `run.bat` hiển thị sai ký tự | Đảm bảo file giữ CRLF và ASCII (đã cấu hình sẵn trong repo). |
| Notebook thiếu `matplotlib`/`seaborn` | `pip install -e ".[notebooks]"`. |
| Tuning Random Forest chậm | Đã đặt `n_jobs=1`; có thể giảm `SEARCH_SPACES` nếu cần. |

## 21. Giới hạn và miễn trừ

- Dữ liệu lịch sử phụ thuộc bên thứ ba; có thể thiếu kỳ hoặc trễ.
- Số đặc biệt của Power 6/55 được gợi ý bằng phân phối tần suất, không dùng mô hình chính.
- Kết quả gần mức ngẫu nhiên là **kỳ vọng đúng**, không phải lỗi của pipeline.
- Kết quả tuning chỉ tối ưu trên validation nội bộ phần train; không đảm bảo cải thiện
  trên test hoặc dữ liệu tương lai.
- **Không có mô hình nào có thể dự đoán xổ số.**

> **Cảnh báo:** Dự án chỉ phục vụ học tập/nghiên cứu. Xổ số là quá trình ngẫu nhiên
> thuần túy; các bộ số gợi ý không có khả năng dự đoán kết quả thực tế.
