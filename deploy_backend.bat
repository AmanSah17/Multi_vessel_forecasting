@echo off
REM Deploy XGBoost Integration to Maritime NLU Backend

setlocal enabledelayedexpansion

set "BACKEND_PATH=f:\Maritime_NLU_Repo\backend\nlu_chatbot\src\app"
set "MODEL_PATH=f:\Maritime_NLU_Repo\backend\nlu_chatbot\results\xgboost_advanced_50_vessels"
set "SOURCE_PATH=."

echo.
echo ========================================================================
echo DEPLOYING XGBOOST INTEGRATION TO MARITIME NLU BACKEND
echo ========================================================================
echo.

REM Check if backend path exists
if not exist "%BACKEND_PATH%" (
    echo ERROR: Backend path not found: %BACKEND_PATH%
    exit /b 1
)

echo [OK] Backend path found
echo.

REM Create model directory if it doesn't exist
if not exist "%MODEL_PATH%" (
    echo Creating model directory: %MODEL_PATH%
    mkdir "%MODEL_PATH%"
)

REM Copy integration modules
echo Copying Integration Modules:

if exist "%SOURCE_PATH%\xgboost_backend_integration.py" (
    copy "%SOURCE_PATH%\xgboost_backend_integration.py" "%BACKEND_PATH%\" /Y >nul
    echo   [OK] Copied xgboost_backend_integration.py
) else (
    echo   [WARN] xgboost_backend_integration.py not found
)

if exist "%SOURCE_PATH%\intent_executor_enhanced.py" (
    copy "%SOURCE_PATH%\intent_executor_enhanced.py" "%BACKEND_PATH%\" /Y >nul
    echo   [OK] Copied intent_executor_enhanced.py
) else (
    echo   [WARN] intent_executor_enhanced.py not found
)

if exist "%SOURCE_PATH%\map_prediction_visualizer.py" (
    copy "%SOURCE_PATH%\map_prediction_visualizer.py" "%BACKEND_PATH%\" /Y >nul
    echo   [OK] Copied map_prediction_visualizer.py
) else (
    echo   [WARN] map_prediction_visualizer.py not found
)

echo.
echo Copying Model Files:

set "SOURCE_MODEL_DIR=%SOURCE_PATH%\results\xgboost_advanced_50_vessels"

if exist "%SOURCE_MODEL_DIR%\xgboost_model.pkl" (
    copy "%SOURCE_MODEL_DIR%\xgboost_model.pkl" "%MODEL_PATH%\" /Y >nul
    echo   [OK] Copied xgboost_model.pkl
) else (
    echo   [WARN] xgboost_model.pkl not found
)

if exist "%SOURCE_MODEL_DIR%\scaler.pkl" (
    copy "%SOURCE_MODEL_DIR%\scaler.pkl" "%MODEL_PATH%\" /Y >nul
    echo   [OK] Copied scaler.pkl
) else (
    echo   [WARN] scaler.pkl not found
)

if exist "%SOURCE_MODEL_DIR%\pca.pkl" (
    copy "%SOURCE_MODEL_DIR%\pca.pkl" "%MODEL_PATH%\" /Y >nul
    echo   [OK] Copied pca.pkl
) else (
    echo   [WARN] pca.pkl not found
)

if exist "%SOURCE_MODEL_DIR%\model_metrics.csv" (
    copy "%SOURCE_MODEL_DIR%\model_metrics.csv" "%MODEL_PATH%\" /Y >nul
    echo   [OK] Copied model_metrics.csv
) else (
    echo   [WARN] model_metrics.csv not found
)

echo.
echo Verifying Deployment:

set "ALL_OK=1"

if exist "%BACKEND_PATH%\xgboost_backend_integration.py" (
    echo   [OK] xgboost_backend_integration.py
) else (
    echo   [MISSING] xgboost_backend_integration.py
    set "ALL_OK=0"
)

if exist "%BACKEND_PATH%\intent_executor_enhanced.py" (
    echo   [OK] intent_executor_enhanced.py
) else (
    echo   [MISSING] intent_executor_enhanced.py
    set "ALL_OK=0"
)

if exist "%BACKEND_PATH%\map_prediction_visualizer.py" (
    echo   [OK] map_prediction_visualizer.py
) else (
    echo   [MISSING] map_prediction_visualizer.py
    set "ALL_OK=0"
)

if exist "%MODEL_PATH%\xgboost_model.pkl" (
    echo   [OK] xgboost_model.pkl
) else (
    echo   [MISSING] xgboost_model.pkl
    set "ALL_OK=0"
)

if exist "%MODEL_PATH%\scaler.pkl" (
    echo   [OK] scaler.pkl
) else (
    echo   [MISSING] scaler.pkl
    set "ALL_OK=0"
)

if exist "%MODEL_PATH%\pca.pkl" (
    echo   [OK] pca.pkl
) else (
    echo   [MISSING] pca.pkl
    set "ALL_OK=0"
)

echo.
echo ========================================================================

if "%ALL_OK%"=="1" (
    echo DEPLOYMENT SUCCESSFUL
    echo.
    echo All files have been copied to the backend successfully!
    echo.
    echo Next Steps:
    echo   1. Restart the backend service
    echo   2. Test with: curl http://127.0.0.1:8000/health
    echo   3. Make a prediction query
    echo   4. Check logs for XGBoost initialization
) else (
    echo DEPLOYMENT INCOMPLETE
    echo.
    echo Some files were not copied. Please check the messages above.
)

echo.
echo ========================================================================
echo.
echo Documentation: Read BACKEND_INTEGRATION_GUIDE.md for detailed information
echo.

endlocal

