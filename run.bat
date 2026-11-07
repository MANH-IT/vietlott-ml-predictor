@echo off
:: ============================================================================
:: Vietlott LSTM Predictor - Run batch file
:: Giúp chạy nhanh các lệnh CLI thông thường không cần gõ python -m mỗi lần.
:: --------------------------------------------
:: Cách dùng:
::   double-click để chạy mặc định (chạy web demo)
::   hoặc truyền tham số: bootstrap, collect, validate, train, evaluate, predict, pipeline
::
::   Ví dụ: run.bat train
::        run.bat predict mega645 lstm
::        run.bat pipeline
:: ============================================================================

:: Đảm bảo chạy từ thư mục chứa project (tự đổi đường dẫn nếu cần)
cd /d "%~dp0"

:: Chạy lệnh python -m vietlott_ml.cli với tham số đã truyền
if "%1"=="" (
    :: Mặc định: chạy web demo FastAPI
    echo ==== Chạy web demo FastAPI (http://127.0.0.1:8000) ====
    python -m vietlott_ml.cli serve
    exit /b
)

:: Hỗ trợ các lệnh CLI
set CMD=%1
shift

if "%CMD%"=="bootstrap" (
    echo ==== Tải dữ liệu lịch sử (bootstrap) ====
    python -m vietlott_ml.cli bootstrap --lottery all
    exit /b
)

if "%CMD%"=="collect" (
    echo ==== Thu thập kết quả live từ API ====
    python -m vietlott_ml.cli collect --lottery all
    exit /b
)

if "%CMD%"=="validate" (
    echo ==== Kiểm tra dữ liệu đã lưu ====
    python -m vietlott_ml.cli validate --lottery all
    exit /b
)

if "%CMD%"=="train" (
    echo ==== Huấn luyện và đánh giá mô hình ====
    python -m vietlott_ml.cli train --lottery all
    exit /b
)

if "%CMD%"=="evaluate" (
    echo ==== Đánh giá lại mô hình đã lưu ====
    python -m vietlott_ml.cli evaluate --lottery all
    exit /b
)

if "%CMD%"=="predict" (
    :: Tham số thứ 2: lottery, thứ 3: model
    if "%2"=="" (
        echo Cần cung cấp lottery và model.
        echo Ví dụ: run.bat predict power655 lstm
        exit /b 1
    )
    echo ==== Tạo bộ số gợi ý ====
    python -m vietlott_ml.cli predict --lottery %2 --model %3
    exit /b
)

if "%CMD%"=="pipeline" (
    echo ==== Chạy bootstrap + collect + train ====
    python -m vietlott_ml.cli pipeline --lottery all
    exit /b
)

echo.
echo Thuong:%1 khong hop le!
echo.
echo Huong dan su dung:
echo   run.bat               -> chay web demo mac dinh
echo   run.bat bootstrap     -> tai lieu su kien
echo   run.bat collect       -> lay ket qua live
echo   run.bat validate      -> kiem tra du lieu
echo   run.bat train         -> train & tao bao cao
echo   run.bat evaluate      -> danh gia lai mo hinh
echo   run.bat predict <lo> <model> -> tao bo so
echo   run.bat pipeline      -> chay day du bo water
echo.
exit /b 1