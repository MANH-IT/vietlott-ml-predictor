# Vietlott ML Predictor

Pipeline học máy thực hành trên dữ liệu xổ số Vietlott (**Mega 6/45** và **Power 6/55**).

> **Tuyên bố miễn trừ trách nhiệm:** Dự án **chỉ phục vụ học tập/nghiên cứu**. Xổ số là quá trình ngẫu nhiên thuần túy. Các bộ số gợi ý **không có khả năng dự đoán** kết quả thực tế. Giá trị của dự án nằm ở pipeline học máy hoàn chỉnh và đánh giá trung thực giới hạn của mô hình.

## Tính năng

- Thu thập dữ liệu real-time từ API `api.383.im` (kỳ mới nhất), lịch sử từ `vietlott-data`, và crawler dự phòng.
- Chuẩn hoá + kiểm tra dữ liệu (khoảng số, trùng lặp, đủ số, số đặc biệt).
- 08 mô hình: **LSTM, BiLSTM, Logistic Regression, Random Forest, Markov, MLP, XGBoost** (+ baseline ngẫu nhiên).
- Chia train/test theo thời gian, so sánh với baseline ngẫu nhiên bằng kiểm định z.
- Web demo FastAPI + giao diện tạo bộ số gợi ý.
- CLI đầy đủ: bootstrap, collect, validate, train, evaluate, predict, serve.

## Cấu trúc

```text
Vietlott_VN/
├── vietlott_ml/
│   ├── config.py             # cấu hình chung
│   ├── data_collection.py    # API live + lịch sử + fallback
│   ├── data_validation.py    # validate & normalize
│   ├── dataset.py            # multi-hot, cửa sổ trượt, feature engineering
│   ├── metrics.py            # hit rate, lift, kiểm định thống kê
│   ├── training.py           # train/evaluate/report
│   ├── predict.py            # sinh bộ số gợi ý
│   ├── api.py                # FastAPI web demo
│   ├── cli.py                # giao diện dòng lệnh
│   ├── models/               # LSTM, BiLSTM, Logistic, RF, Markov, MLP
│   └── templates/index.html
├── scripts/test_api_383.py
├── tests/
├── data/raw/                 # dữ liệu JSONL (sinh ra khi chạy)
├── models/                   # mô hình đã train (sinh ra khi chạy)
└── reports/                  # metrics + báo cáo markdown
```

## Cài đặt

Khuyến nghị dùng conda env có Python 3.12 + PyTorch (ví dụ env `ai`):

```bash
conda create -n vietlott python=3.12 -y
conda activate vietlott
pip install -r requirements.txt
```

Nếu dùng GPU, cài PyTorch bản CUDA phù hợp trước:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

## Sử dụng nhanh

```bash
# 1. Kiểm tra API 383.im
python scripts/test_api_383.py

# 2. Tải dữ liệu lịch sử (GitHub vietlott-data, fallback crawler)
python -m vietlott_ml.cli bootstrap --lottery all

# 3. Lấy kết quả mới nhất từ API live, append vào data/raw/vietlott_live.jsonl
python -m vietlott_ml.cli collect --lottery all

# 4. Kiểm tra dữ liệu
python -m vietlott_ml.cli validate --lottery all

# 5. Huấn luyện + đánh giá tất cả mô hình
python -m vietlott_ml.cli train --lottery all

# 6. Tạo bộ số gợi ý
python -m vietlott_ml.cli predict --lottery power655 --model lstm
python -m vietlott_ml.cli predict --lottery mega645 --model markov

# 7. Web demo
python -m vietlott_ml.cli serve --host 0.0.0.0 --port 8000
# mở http://localhost:8000
```

Chạy toàn bộ pipeline trong một lệnh:

```bash
python -m vietlott_ml.cli pipeline --lottery all
```

Chỉ train một vài mô hình:

```bash
python -m vietlott_ml.cli train --lottery mega645 --models lstm,markov,logistic
```

## Thu thập dữ liệu định kỳ

Chạy vòng lặp kiểm tra mỗi 5 phút:

```bash
python -m vietlott_ml.cli collect --lottery all --watch --interval 300
```

Hoặc dùng cron (ví dụ 18:05 các ngày có quay):

```cron
5 18 * * 2,4,5,6,7 * cd /path/to/Vietlott_VN && python -m vietlott_ml.cli collect --lottery all >> data/raw/cron.log 2>&1
```

- Mega 6/45: Thứ 4, 6, Chủ nhật.
- Power 6/55: Thứ 3, 5, 7.

## Nguồn dữ liệu

| Nguồn | Vai trò |
|---|---|
| `https://api.383.im/lottery/live.json` | Kỳ mới nhất (real-time/gần real-time) |
| `vietvudanh/vietlott-data` (NDJSON trên GitHub) | Dữ liệu lịch sử để train |
| `vietlott-crawl` (package `vietlott-data`) | Fallback khi API/history lỗi |
| `vietlott.vn` | Nguồn gốc cuối cùng của dữ liệu |

Dữ liệu chuẩn lưu tại `data/raw/{mega645,power655}.jsonl`. Mỗi lần lấy kết quả live còn append vào `data/raw/vietlott_live.jsonl` kèm `source` và `collected_at`. Log nằm ở `data/raw/collection.log`.

## Các mô hình

Tất cả mô hình đều dự đoán xác suất xuất hiện của từng số, sau đó chọn **top 6**.

| Mô hình | Cách tiếp cận |
|---|---|
| **LSTM** | Input cửa sổ 10 kỳ multi-hot; 2 lớp LSTM + Dropout + Dense sigmoid; BCE loss, early stopping. |
| **BiLSTM** | Như LSTM nhưng hai lớp bidirectional. |
| **Logistic Regression** | Một bộ phân loại nhị phân cho mỗi số; features tần suất, gap, xu hướng. |
| **Random Forest** | Một rừng nhị phân cho mỗi số, cùng bộ features. |
| **Markov** | Ma trận chuyển đổi số → số; điểm của số j là tổng xác suất chuyển từ các số trong kỳ trước. |
| **MLP** | Multi-hot kỳ trước → 2 lớp ẩn → sigmoid cho kỳ kế tiếp. |
| **XGBoost** | Bộ phân loại nhị phân gradient boosting cho mỗi số, cùng bộ features dạng bảng. |

**Features dạng bảng:** tần suất trong 5/10/20/50 kỳ, gap từ lần xuất hiện cuối, số lần xuất hiện, tần suất toàn bộ, average gap, xuất hiện ở kỳ trước.

## Đánh giá trung thực

- Chia **theo thời gian** (80% train, 20% test) — không xáo trộn để tránh rò rỉ tương lai.
- Metric chính: số số trúng trung bình/kỳ (`avg_hits`), khoảng tin cậy 95%, `lift` so với ngẫu nhiên, `precision@6`, `p-value` (kiểm định z với phương sai siêu bội).
- Baseline ngẫu nhiên: lặp **50 seed** rồi gộp phân bố để giảm nhiễu của một lần rút.
- Kỳ vọng ngẫu nhiên: `6 × 6 / F` → **0.800** cho Mega (F=45), **0.655** cho Power (F=55).

Kết quả trên dữ liệu hiện tại (xem `reports/{lottery}_report.md`):

| Lottery | Mô hình tốt nhất | Avg hits | Ngẫu nhiên | Có ý nghĩa? |
|---|---|---:|---:|:---:|
| Mega 6/45 | BiLSTM / MLP | 0.801 | 0.800 | Không |
| Power 6/55 | BiLSTM | 0.712 | 0.655 | Không |

Không mô hình nào vượt trội có ý nghĩa thống kê so với ngẫu nhiên. Đây là kết quả **đúng với bản chất ngẫu nhiên của xổ số** và là một phần quan trọng của dự án.

## Web demo

```bash
python -m vietlott_ml.cli serve
```

- `GET /` — giao diện.
- `GET /api/lotteries` — thông tin loại xổ số + kỳ gần nhất + mô hình khả dụng.
- `GET /api/latest/{lottery}` — kỳ gần nhất.
- `POST /api/predict` — body `{"lottery": "power655", "model": "lstm"}`.
- `GET /api/metrics/{lottery}` — báo cáo đánh giá.
- `GET /docs` — Swagger UI.

## Kiểm thử

```bash
python -m pytest -q      # unit + API + training tests
ruff check .             # lint (cấu hình trong pyproject.toml)
```

Bao gồm test validation, dataset (shape + chống rò rỉ tương lai), model smoke tests, metrics, API và training. CI chạy lint + test trên Python 3.12 qua `.github/workflows/ci.yml`.

## Giới hạn

- Dữ liệu lịch sử phụ thuộc bên thứ ba; có thể thiếu kỳ hoặc trễ.
- Số đặc biệt của Power 6/55 được gợi ý bằng phân phối tần suất (không dùng mô hình chính).
- Không có mô hình nào có thể dự đoán xổ số. Kết quả gần random là kỳ vọng đúng.
