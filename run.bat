@echo off
:: ============================================================================
:: Vietlott LSTM Predictor - Run batch file
:: Giup chay nhanh cac lenh CLI thuong dung khong can go python -m moi lan.
:: --------------------------------------------
:: Cach dung:
::   double-click de chay mac dinh (chay web demo)
::   hoac truyen tham so: bootstrap, collect, validate, train, evaluate, predict, pipeline
::
::   Vi du: run.bat train
::          run.bat predict mega645 lstm
::          run.bat pipeline
:: ============================================================================

:: Dam bao chay tu thu muc chua project (tu doi duong dan neu can)
cd /d "%~dp0"

:: Chay lenh python -m vietlott_ml.cli voi tham so da truyen
if "%1"=="" (
    :: Mac dinh: chay web demo FastAPI tai http://127.0.0.1:8000
    echo ==== Chay web demo FastAPI - http://127.0.0.1:8000 ====
    python -m vietlott_ml.cli serve
    exit /b
)

:: Ho tro cac lenh CLI (giu nguyen %1..%3, khong shift de khong mat tham so)
set CMD=%1

if "%CMD%"=="bootstrap" (
    echo ==== Tai du lieu lich su - bootstrap ====
    python -m vietlott_ml.cli bootstrap --lottery all
    exit /b
)

if "%CMD%"=="collect" (
    echo ==== Thu thap ket qua live tu API ====
    python -m vietlott_ml.cli collect --lottery all
    exit /b
)

if "%CMD%"=="validate" (
    echo ==== Kiem tra du lieu da luu ====
    python -m vietlott_ml.cli validate --lottery all
    exit /b
)

if "%CMD%"=="train" (
    echo ==== Huan luyen va danh gia mo hinh ====
    python -m vietlott_ml.cli train --lottery all
    exit /b
)

if "%CMD%"=="evaluate" (
    echo ==== Danh gia lai mo hinh da luu ====
    python -m vietlott_ml.cli evaluate --lottery all
    exit /b
)

if "%CMD%"=="predict" goto predict

if "%CMD%"=="pipeline" (
    echo ==== Chay bootstrap + collect + train ====
    python -m vietlott_ml.cli pipeline --lottery all
    exit /b
)

echo.
echo Lenh "%CMD%" khong hop le!
echo.
echo Huong dan su dung:
echo   run.bat                           - chay web demo - mac dinh
echo   run.bat bootstrap                 - tai du lieu lich su
echo   run.bat collect                   - lay ket qua live tu API
echo   run.bat validate                  - kiem tra du lieu
echo   run.bat train                     - huan luyen va tao bao cao
echo   run.bat evaluate                  - danh gia lai mo hinh da luu
echo   run.bat predict ^<lottery^> [model] - tao bo so goi y - model mac dinh: lstm
echo   run.bat pipeline                  - chay day du pipeline
echo.
exit /b 1

:: ---- predict: tham so thu 2 la lottery, thu 3 - tuy chon - la model ----
:predict
if "%2"=="" (
    echo Can cung cap lottery.
    echo Vi du: run.bat predict power655 lstm
    exit /b 1
)
echo ==== Tao bo so goi y ====
if "%3"=="" (
    python -m vietlott_ml.cli predict --lottery %2
) else (
    python -m vietlott_ml.cli predict --lottery %2 --model %3
)
exit /b 0
