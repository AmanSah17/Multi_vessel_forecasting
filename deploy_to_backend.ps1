# Deploy XGBoost Integration to Maritime NLU Backend
# This script copies all necessary files to the backend

param(
    [string]$BackendPath = "f:\Maritime_NLU_Repo\backend\nlu_chatbot\src\app",
    [string]$ModelPath = "f:\Maritime_NLU_Repo\backend\nlu_chatbot\results\xgboost_advanced_50_vessels",
    [string]$SourcePath = "."
)

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "DEPLOYING XGBOOST INTEGRATION TO MARITIME NLU BACKEND" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "📋 Configuration:" -ForegroundColor Yellow
Write-Host "  Backend Path: $BackendPath"
Write-Host "  Model Path: $ModelPath"
Write-Host "  Source Path: $SourcePath"
Write-Host ""

# Check if backend path exists
if (-not (Test-Path $BackendPath)) {
    Write-Host "❌ Backend path not found: $BackendPath" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Backend path found" -ForegroundColor Green

# Create model directory if it doesn't exist
if (-not (Test-Path $ModelPath)) {
    Write-Host "📁 Creating model directory: $ModelPath" -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $ModelPath -Force | Out-Null
}

# Copy integration modules
Write-Host ""
Write-Host "📦 Copying Integration Modules:" -ForegroundColor Cyan

$files_to_copy = @(
    "xgboost_backend_integration.py",
    "intent_executor_enhanced.py",
    "map_prediction_visualizer.py"
)

foreach ($file in $files_to_copy) {
    $source = Join-Path $SourcePath $file
    $dest = Join-Path $BackendPath $file

    if (Test-Path $source) {
        Copy-Item -Path $source -Destination $dest -Force
        Write-Host "  [OK] Copied $file" -ForegroundColor Green
    } else {
        Write-Host "  [WARN] Source file not found: $file" -ForegroundColor Yellow
    }
}

# Copy model files
Write-Host ""
Write-Host "🤖 Copying Model Files:" -ForegroundColor Cyan

$model_files = @(
    "xgboost_model.pkl",
    "scaler.pkl",
    "pca.pkl",
    "model_metrics.csv"
)

$source_model_dir = Join-Path $SourcePath "results\xgboost_advanced_50_vessels"

foreach ($file in $model_files) {
    $source = Join-Path $source_model_dir $file
    $dest = Join-Path $ModelPath $file

    if (Test-Path $source) {
        Copy-Item -Path $source -Destination $dest -Force
        $size = (Get-Item $source).Length / 1MB
        Write-Host "  [OK] Copied $file ($([math]::Round($size, 2)) MB)" -ForegroundColor Green
    } else {
        Write-Host "  [WARN] Model file not found: $file" -ForegroundColor Yellow
    }
}

# Verify deployment
Write-Host ""
Write-Host "🔍 Verifying Deployment:" -ForegroundColor Cyan

$verification_files = @(
    (Join-Path $BackendPath "xgboost_backend_integration.py"),
    (Join-Path $BackendPath "intent_executor_enhanced.py"),
    (Join-Path $BackendPath "map_prediction_visualizer.py"),
    (Join-Path $ModelPath "xgboost_model.pkl"),
    (Join-Path $ModelPath "scaler.pkl"),
    (Join-Path $ModelPath "pca.pkl")
)

$all_verified = $true
foreach ($file in $verification_files) {
    if (Test-Path $file) {
        $size = (Get-Item $file).Length / 1KB
        Write-Host "  [OK] $file ($([math]::Round($size, 2)) KB)" -ForegroundColor Green
    } else {
        Write-Host "  [MISSING] $file" -ForegroundColor Red
        $all_verified = $false
    }
}

# Update main.py
Write-Host ""
Write-Host "📝 Updating Backend Configuration:" -ForegroundColor Cyan

$main_py = Join-Path $BackendPath "main.py"
if (Test-Path $main_py) {
    $content = Get-Content $main_py -Raw

    # Check if already updated
    if ($content -match "intent_executor_enhanced") {
        Write-Host "  [INFO] main.py already uses enhanced executor" -ForegroundColor Blue
    } else {
        # Create backup
        $backup = "$main_py.backup"
        Copy-Item -Path $main_py -Destination $backup
        Write-Host "  [OK] Created backup: $backup" -ForegroundColor Green

        # Update import
        $content = $content -replace "from intent_executor import IntentExecutor", "from intent_executor_enhanced import IntentExecutor"
        Set-Content -Path $main_py -Value $content
        Write-Host "  [OK] Updated main.py to use enhanced executor" -ForegroundColor Green
    }
} else {
    Write-Host "  [WARN] main.py not found" -ForegroundColor Yellow
}

# Summary
Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan

if ($all_verified) {
    Write-Host "DEPLOYMENT SUCCESSFUL" -ForegroundColor Green
    Write-Host ""
    Write-Host "All files have been copied to the backend successfully!" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next Steps:" -ForegroundColor Cyan
    Write-Host "  1. Restart the backend service" -ForegroundColor Cyan
    Write-Host "  2. Test with: curl http://127.0.0.1:8000/health" -ForegroundColor Cyan
    Write-Host "  3. Make a prediction query" -ForegroundColor Cyan
    Write-Host "  4. Check logs for XGBoost initialization" -ForegroundColor Cyan
} else {
    Write-Host "DEPLOYMENT INCOMPLETE" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Some files were not copied. Please check the messages above." -ForegroundColor Cyan
}

Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "Documentation:" -ForegroundColor Yellow
Write-Host "  Read BACKEND_INTEGRATION_GUIDE.md for detailed information"
Write-Host ""

