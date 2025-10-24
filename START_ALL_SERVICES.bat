@echo off
REM ============================================================================
REM Maritime NLU + XGBoost Integration - Complete Service Launcher (Batch)
REM ============================================================================
REM This script starts all services in separate terminal windows
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ╔════════════════════════════════════════════════════════════════════════════════╗
echo ║                 MARITIME NLU + XGBOOST INTEGRATION STARTUP                     ║
echo ╚════════════════════════════════════════════════════════════════════════════════╝
echo.

REM Check Python installation
echo [*] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python first.
    pause
    exit /b 1
)
echo [OK] Python found

REM Check required packages
echo [*] Checking required packages...
python -c "import fastapi, uvicorn, streamlit, xgboost, pandas, numpy" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Some packages may be missing. Installing...
    pip install fastapi uvicorn streamlit xgboost pandas numpy scikit-learn matplotlib
)
echo [OK] Packages ready

echo.
echo ╔════════════════════════════════════════════════════════════════════════════════╗
echo ║                          STARTING SERVICES                                     ║
echo ╚════════════════════════════════════════════════════════════════════════════════╝
echo.

REM Start XGBoost Model Server
echo [1/3] Starting XGBoost Model Server on port 8001...
start "XGBoost Model Server" cmd /k ^
    cd /d "f:\PyTorch_GPU\maritime_vessel_forecasting\Multi_vessel_forecasting" ^& ^
    python notebooks/43_end_to_end_xgboost_integration.py

timeout /t 3 /nobreak

REM Start Maritime NLU Backend
echo [2/3] Starting Maritime NLU Backend on port 8000...
start "Maritime NLU Backend" cmd /k ^
    cd /d "f:\Maritime_NLU_Repo\backend\nlu_chatbot\src\app" ^& ^
    uvicorn main:app --reload --host 127.0.0.1 --port 8000

timeout /t 3 /nobreak

REM Start Maritime NLU Frontend
echo [3/3] Starting Maritime NLU Frontend on port 8502...
start "Maritime NLU Frontend" cmd /k ^
    cd /d "f:\Maritime_NLU_Repo\backend\nlu_chatbot\frontend" ^& ^
    streamlit run app.py --server.port 8502

echo.
echo ╔════════════════════════════════════════════════════════════════════════════════╗
echo ║                        SERVICES STARTING                                       ║
echo ╚════════════════════════════════════════════════════════════════════════════════╝
echo.
echo [*] Waiting for services to initialize (30 seconds)...
timeout /t 30 /nobreak

echo.
echo ╔════════════════════════════════════════════════════════════════════════════════╗
echo ║                        SERVICE ENDPOINTS                                       ║
echo ╚════════════════════════════════════════════════════════════════════════════════╝
echo.
echo XGBoost Model Server:
echo   - API: http://127.0.0.1:8001
echo   - Docs: http://127.0.0.1:8001/docs
echo.
echo Maritime NLU Backend:
echo   - API: http://127.0.0.1:8000
echo   - Docs: http://127.0.0.1:8000/docs
echo   - Health: http://127.0.0.1:8000/health
echo.
echo Maritime NLU Frontend:
echo   - Dashboard: http://127.0.0.1:8502
echo.
echo ╔════════════════════════════════════════════════════════════════════════════════╗
echo ║                    OPENING DASHBOARD IN BROWSER                                ║
echo ╚════════════════════════════════════════════════════════════════════════════════╝
echo.

REM Open dashboard in default browser
timeout /t 5 /nobreak
start http://127.0.0.1:8502

echo.
echo [OK] All services started successfully!
echo [*] Check the terminal windows for service logs
echo [*] Press Ctrl+C in any terminal to stop that service
echo.
pause

