@echo off
rem ============================================================================
rem Vietlott LSTM Predictor - Run batch file
rem Chay nhanh cac lenh CLI thuong dung.
rem
rem Cach dung:
rem   run.bat                                  chay web demo (mac dinh)
rem   run.bat bootstrap                        tai du lieu lich su
rem   run.bat collect                          lay ket qua live tu API
rem   run.bat validate                         kiem tra du lieu
rem   run.bat tune                             tuning hyperparameters
rem   run.bat walk-forward                     danh gia walk-forward
rem   run.bat train                            huan luyen va tao bao cao
rem   run.bat evaluate                         danh gia lai mo hinh da luu
rem   run.bat predict <lottery> [model]        tao bo so goi y
rem   run.bat pipeline                         bootstrap + collect + train
rem   run.bat serve [--host H] [--port P]      chay web demo
rem   run.bat test                             chay unit test
rem   run.bat notebook                         mo Jupyter notebook
rem
rem Vi du:
rem   run.bat train --models lstm,mlp
rem   run.bat tune --lottery power655
rem   run.bat predict mega645 markov
rem
rem Neu python khong nam trong PATH, dat bien moi truong PYTHON tro toi
rem python.exe truoc khi chay, vi du:
rem   set PYTHON=C:\path\to\python.exe
rem ============================================================================

setlocal EnableExtensions
cd /d "%~dp0"
if not defined PYTHON set "PYTHON=python"

if "%~1"=="" goto serve
set "CMD=%~1"

if /i "%CMD%"=="serve" goto serve
if /i "%CMD%"=="bootstrap" goto bootstrap
if /i "%CMD%"=="collect" goto collect
if /i "%CMD%"=="validate" goto validate
if /i "%CMD%"=="tune" goto tune
if /i "%CMD%"=="walk-forward" goto walkforward
if /i "%CMD%"=="train" goto train
if /i "%CMD%"=="evaluate" goto evaluate
if /i "%CMD%"=="predict" goto predict
if /i "%CMD%"=="pipeline" goto pipeline
if /i "%CMD%"=="test" goto test
if /i "%CMD%"=="notebook" goto notebook
if /i "%CMD%"=="help" goto usage
if /i "%CMD%"=="--help" goto usage
if /i "%CMD%"=="-h" goto usage

echo.
echo Lenh "%CMD%" khong hop le!
goto usage

:bootstrap
echo ==== Tai du lieu lich su - bootstrap ====
"%PYTHON%" -m vietlott_ml.cli bootstrap --lottery all %2 %3 %4 %5
goto done

:collect
echo ==== Thu thap ket qua live tu API ====
"%PYTHON%" -m vietlott_ml.cli collect --lottery all %2 %3 %4 %5
goto done

:validate
echo ==== Kiem tra du lieu da luu ====
"%PYTHON%" -m vietlott_ml.cli validate --lottery all %2 %3 %4 %5
goto done

:tune
echo ==== Tuning hyperparameters tren validation ====
"%PYTHON%" -m vietlott_ml.cli tune --lottery all %2 %3 %4 %5 %6
goto done

:walkforward
echo ==== Danh gia walk-forward ====
"%PYTHON%" -m vietlott_ml.cli walk-forward --lottery all %2 %3 %4 %5 %6
goto done

:train
echo ==== Huan luyen va danh gia mo hinh ====
"%PYTHON%" -m vietlott_ml.cli train --lottery all %2 %3 %4 %5 %6
goto done

:evaluate
echo ==== Danh gia lai mo hinh da luu ====
"%PYTHON%" -m vietlott_ml.cli evaluate --lottery all %2 %3 %4 %5
goto done

:pipeline
echo ==== Chay bootstrap + collect + train ====
"%PYTHON%" -m vietlott_ml.cli pipeline --lottery all %2 %3 %4 %5
goto done

:serve
echo ==== Chay web demo FastAPI - http://127.0.0.1:8000 ====
"%PYTHON%" -m vietlott_ml.cli serve %2 %3 %4 %5 %6
goto done

:predict
if "%~2"=="" (
    echo Can cung cap lottery.
    echo Vi du: run.bat predict power655 lstm
    goto fail
)
set "LOT=%~2"
set "MODEL=%~3"
echo ==== Tao bo so goi y ====
if "%MODEL%"=="" (
    "%PYTHON%" -m vietlott_ml.cli predict --lottery %LOT% %4 %5
) else (
    "%PYTHON%" -m vietlott_ml.cli predict --lottery %LOT% --model %MODEL% %4 %5
)
goto done

:test
echo ==== Chay unit test ====
"%PYTHON%" -m pytest -q
goto done

:notebook
echo ==== Mo Jupyter notebook ====
"%PYTHON%" -m jupyter notebook notebooks
goto done

:usage
echo.
echo Huong dan su dung:
echo   run.bat                                  - chay web demo - mac dinh
echo   run.bat bootstrap                        - tai du lieu lich su
echo   run.bat collect                          - lay ket qua live tu API
echo   run.bat validate                         - kiem tra du lieu
echo   run.bat tune [--lottery X] [--models A,B] - tuning hyperparameters
echo   run.bat walk-forward [--lottery X] [--splits N] - danh gia walk-forward
echo   run.bat train [--lottery X] [--models A,B] - huan luyen va tao bao cao
echo   run.bat evaluate                         - danh gia lai mo hinh da luu
echo   run.bat predict ^<lottery^> [model]       - tao bo so goi y - model mac dinh: lstm
echo   run.bat pipeline                         - chay day du pipeline
echo   run.bat serve [--host H] [--port P]      - chay web demo
echo   run.bat test                             - chay unit test
echo   run.bat notebook                         - mo Jupyter notebook
echo.
goto fail

:fail
endlocal
exit /b 1

:done
endlocal
exit /b 0
