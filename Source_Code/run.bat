@echo off
title CyberGuard AI - SOC Log Anomaly Detection Platform
echo ======================================================================
echo    CyberGuard AI: Autonomous SOC Log Anomaly Detection Platform
echo    Bi-LSTM + Multi-Head Attention + Adaptive Gaussian KDE
echo ======================================================================
echo.
echo [1/2] Opening browser to http://127.0.0.1:8000 ...
start http://127.0.0.1:8000
echo.
echo [2/2] Starting backend FastAPI service on port 8000...
echo (Press CTRL+C anytime in this window to stop the server)
echo ----------------------------------------------------------------------
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
pause
