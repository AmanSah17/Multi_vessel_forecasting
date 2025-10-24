# ============================================================================
# Maritime NLU + XGBoost Integration - Complete Service Launcher
# ============================================================================
# This script starts all services:
# 1. XGBoost Model Server (FastAPI)
# 2. Maritime NLU Backend (FastAPI)
# 3. Maritime NLU Frontend (Streamlit)
# 4. Real-time Health Monitor
# ============================================================================

param(
    [switch]$SkipHealthCheck = $false,
    [int]$HealthCheckInterval = 5
)

# Color scheme
$colors = @{
    'Success' = 'Green'
    'Error' = 'Red'
    'Warning' = 'Yellow'
    'Info' = 'Cyan'
    'Header' = 'Magenta'
}

function Write-Header {
    param([string]$Text)
    Write-Host ""
    Write-Host "╔════════════════════════════════════════════════════════════════════════════════╗" -ForegroundColor $colors['Header']
    Write-Host "║ $($Text.PadRight(78)) ║" -ForegroundColor $colors['Header']
    Write-Host "╚════════════════════════════════════════════════════════════════════════════════╝" -ForegroundColor $colors['Header']
    Write-Host ""
}

function Write-Status {
    param([string]$Service, [string]$Status, [string]$Message = "")
    $statusColor = if ($Status -eq "✅") { $colors['Success'] } elseif ($Status -eq "❌") { $colors['Error'] } else { $colors['Warning'] }
    Write-Host "$Status $Service" -ForegroundColor $statusColor -NoNewline
    if ($Message) { Write-Host " - $Message" -ForegroundColor $colors['Info'] }
    else { Write-Host "" }
}

function Test-Port {
    param([int]$Port)
    try {
        $connection = New-Object System.Net.Sockets.TcpClient
        $connection.Connect("127.0.0.1", $Port)
        $connection.Close()
        return $true
    } catch {
        return $false
    }
}

function Test-Service {
    param([string]$ServiceName, [int]$Port, [string]$Endpoint = "/health")
    try {
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:$Port$Endpoint" -TimeoutSec 2 -ErrorAction SilentlyContinue
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

# ============================================================================
# MAIN STARTUP SEQUENCE
# ============================================================================

Write-Header "🚀 MARITIME NLU + XGBOOST INTEGRATION - STARTUP"

# Define services
$services = @(
    @{
        Name = "XGBoost Model Server"
        Port = 8001
        Path = "f:\PyTorch_GPU\maritime_vessel_forecasting\Multi_vessel_forecasting"
        Command = "python notebooks/43_end_to_end_xgboost_integration.py"
        Env = "torch_gpu"
        Type = "Background"
    },
    @{
        Name = "Maritime NLU Backend"
        Port = 8000
        Path = "f:\Maritime_NLU_Repo\backend\nlu_chatbot\src\app"
        Command = "uvicorn main:app --reload --host 127.0.0.1 --port 8000"
        Env = "base"
        Type = "Background"
    },
    @{
        Name = "Maritime NLU Frontend"
        Port = 8502
        Path = "f:\Maritime_NLU_Repo\backend\nlu_chatbot\frontend"
        Command = "streamlit run app.py --server.port 8502"
        Env = "base"
        Type = "Background"
    }
)

Write-Host "📋 Services to start:" -ForegroundColor $colors['Info']
$services | ForEach-Object { Write-Host "   • $($_.Name) (Port: $($_.Port))" -ForegroundColor $colors['Info'] }
Write-Host ""

# Check prerequisites
Write-Header "🔍 CHECKING PREREQUISITES"

$prereqsOK = $true

# Check Python
Write-Host "Checking Python..." -NoNewline
$pythonCheck = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Status "Python" "✅" $pythonCheck
} else {
    Write-Status "Python" "❌" "Not found"
    $prereqsOK = $false
}

# Check required packages
$packages = @("fastapi", "uvicorn", "streamlit", "xgboost", "pandas", "numpy")
Write-Host "Checking Python packages..." -NoNewline
$missingPackages = @()
foreach ($pkg in $packages) {
    $check = python -c "import $pkg" 2>&1
    if ($LASTEXITCODE -ne 0) {
        $missingPackages += $pkg
    }
}

if ($missingPackages.Count -eq 0) {
    Write-Status "Packages" "✅" "All required packages found"
} else {
    Write-Status "Packages" "⚠️" "Missing: $($missingPackages -join ', ')"
}

# Check ports availability
Write-Host "Checking port availability..." -NoNewline
$portsOK = $true
foreach ($service in $services) {
    if (Test-Port $service.Port) {
        Write-Status "Port $($service.Port)" "⚠️" "Already in use"
        $portsOK = $false
    }
}
if ($portsOK) {
    Write-Status "Ports" "✅" "All ports available"
}

Write-Host ""

# ============================================================================
# START SERVICES
# ============================================================================

Write-Header "🚀 STARTING SERVICES"

$processes = @()

foreach ($service in $services) {
    Write-Host "Starting $($service.Name)..." -ForegroundColor $colors['Info']
    
    try {
        # Create process info
        $pinfo = New-Object System.Diagnostics.ProcessStartInfo
        $pinfo.FileName = "powershell.exe"
        $pinfo.RedirectStandardOutput = $true
        $pinfo.RedirectStandardError = $true
        $pinfo.UseShellExecute = $false
        $pinfo.CreateNoWindow = $true
        $pinfo.WorkingDirectory = $service.Path
        
        # Build command with environment activation
        $cmdLine = "cd '$($service.Path)'; $($service.Command)"
        $pinfo.Arguments = "-NoProfile -Command `"$cmdLine`""
        
        # Start process
        $process = [System.Diagnostics.Process]::Start($pinfo)
        $processes += @{
            Name = $service.Name
            Process = $process
            Port = $service.Port
            StartTime = Get-Date
        }
        
        Write-Status $service.Name "✅" "Started (PID: $($process.Id))"
    } catch {
        Write-Status $service.Name "❌" $_.Exception.Message
    }
}

Write-Host ""

# ============================================================================
# WAIT FOR SERVICES TO BE READY
# ============================================================================

Write-Header "⏳ WAITING FOR SERVICES TO BE READY"

$maxWaitTime = 60
$startTime = Get-Date
$allReady = $false

while ((Get-Date) - $startTime -lt [timespan]::FromSeconds($maxWaitTime)) {
    $readyCount = 0
    
    foreach ($proc in $processes) {
        $isReady = Test-Service $proc.Name $proc.Port
        $status = if ($isReady) { "✅" } else { "⏳" }
        Write-Status $proc.Name $status
        
        if ($isReady) { $readyCount++ }
    }
    
    if ($readyCount -eq $processes.Count) {
        $allReady = $true
        break
    }
    
    Start-Sleep -Seconds 2
}

Write-Host ""

if ($allReady) {
    Write-Status "All Services" "✅" "Ready and responding"
} else {
    Write-Status "All Services" "⚠️" "Some services may still be starting"
}

# ============================================================================
# DISPLAY SERVICE ENDPOINTS
# ============================================================================

Write-Header "📡 SERVICE ENDPOINTS"

Write-Host "XGBoost Model Server:" -ForegroundColor $colors['Info']
Write-Host "  • API: http://127.0.0.1:8001" -ForegroundColor $colors['Success']
Write-Host "  • Docs: http://127.0.0.1:8001/docs" -ForegroundColor $colors['Success']
Write-Host ""

Write-Host "Maritime NLU Backend:" -ForegroundColor $colors['Info']
Write-Host "  • API: http://127.0.0.1:8000" -ForegroundColor $colors['Success']
Write-Host "  • Docs: http://127.0.0.1:8000/docs" -ForegroundColor $colors['Success']
Write-Host "  • Health: http://127.0.0.1:8000/health" -ForegroundColor $colors['Success']
Write-Host ""

Write-Host "Maritime NLU Frontend:" -ForegroundColor $colors['Info']
Write-Host "  • Dashboard: http://127.0.0.1:8502" -ForegroundColor $colors['Success']
Write-Host ""

# ============================================================================
# REAL-TIME HEALTH MONITOR
# ============================================================================

if (-not $SkipHealthCheck) {
    Write-Header "🔄 REAL-TIME HEALTH MONITOR"
    Write-Host "Monitoring services every $HealthCheckInterval seconds (Press Ctrl+C to stop)" -ForegroundColor $colors['Warning']
    Write-Host ""
    
    $monitorStartTime = Get-Date
    
    while ($true) {
        $timestamp = Get-Date -Format "HH:mm:ss"
        Write-Host "[$timestamp] Health Check:" -ForegroundColor $colors['Header']
        
        $allHealthy = $true
        foreach ($proc in $processes) {
            $isHealthy = Test-Service $proc.Name $proc.Port
            $status = if ($isHealthy) { "✅ HEALTHY" } else { "❌ DOWN" }
            $statusColor = if ($isHealthy) { $colors['Success'] } else { $colors['Error'] }
            
            $uptime = (Get-Date) - $proc.StartTime
            $uptimeStr = "{0:hh\:mm\:ss}" -f $uptime
            
            Write-Host "  $($proc.Name.PadRight(30)) $status (Uptime: $uptimeStr)" -ForegroundColor $statusColor
            
            if (-not $isHealthy) { $allHealthy = $false }
        }
        
        if ($allHealthy) {
            Write-Host "  ✅ All services operational" -ForegroundColor $colors['Success']
        } else {
            Write-Host "  ⚠️  Some services may need attention" -ForegroundColor $colors['Warning']
        }
        
        Write-Host ""
        Start-Sleep -Seconds $HealthCheckInterval
    }
}

# ============================================================================
# CLEANUP ON EXIT
# ============================================================================

trap {
    Write-Header "🛑 SHUTTING DOWN SERVICES"
    
    foreach ($proc in $processes) {
        try {
            Write-Host "Stopping $($proc.Name)..." -ForegroundColor $colors['Warning']
            $proc.Process | Stop-Process -Force -ErrorAction SilentlyContinue
            Write-Status $proc.Name "✅" "Stopped"
        } catch {
            Write-Status $proc.Name "❌" $_.Exception.Message
        }
    }
    
    Write-Host ""
    Write-Host "All services stopped. Goodbye!" -ForegroundColor $colors['Success']
    exit
}

