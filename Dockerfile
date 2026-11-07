FROM python:3.12-slim

# ---- Cài đặt hệ điều chỉnh / thư viện ---- #
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---- Cài đặt Python packages ---- #
# Copy requirements trước để khai báo cache layer
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy mã nguồn ---- #
COPY . .

# ---- Tạo thư mục cần thiết ---- #
RUN mkdir -p data/raw models reports scripts tests vietlott_ml/templates

# ---- Expose port ---- #
EXPOSE 8000

# ---- Mặc định chạy web demo khi container start ---- #
CMD ["python", "-m", "vietlott_ml.cli", "serve"]