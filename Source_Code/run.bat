@echo off
setlocal
title CyberGuard AI - SOC Log Anomaly Detection Platform
cd /d "%~dp0"

echo ======================================================================
echo    CyberGuard AI: Autonomous SOC Log Anomaly Detection Platform
echo    Drain3 + Positional Pad-Masked Bi-LSTM + Attention + Adaptive KDE
echo ======================================================================
echo.
echo [1/3] Checking and freeing port 8000...
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"
echo Port 8000 is ready.
echo.

set "PYTHON_BIN="
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_BIN=%~dp0.venv\Scripts\python.exe"
) else (
    where python >nul 2>nul && set "PYTHON_BIN=python"
)

if "%PYTHON_BIN%"=="" (
    where uv >nul 2>nul
    if %errorlevel%==0 (
        echo [*] Creating .venv and syncing dependencies via uv...
        uv venv "%~dp0.venv" --python 3.12
        uv pip install --python "%~dp0.venv\Scripts\python.exe" -r "%~dp0requirements.txt"
        set "PYTHON_BIN=%~dp0.venv\Scripts\python.exe"
    ) else (
        echo [ERROR] Python executable not found. Activate your venv or install uv.
        pause
        exit /b 1
    )
)

echo [2/3] Opening browser to http://127.0.0.1:8000 ...
start "" http://127.0.0.1:8000

echo.
echo [3/3] Starting backend FastAPI service on port 8000...
echo Using Python: %PYTHON_BIN%
echo Press CTRL+C anytime in this window to stop the server
echo ----------------------------------------------------------------------
"%PYTHON_BIN%" -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
pause
