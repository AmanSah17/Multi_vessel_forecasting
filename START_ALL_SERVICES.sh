#!/bin/bash

# ============================================================================
# Maritime NLU + XGBoost Integration - Complete Service Launcher (Linux/Mac)
# ============================================================================

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
XGBOOST_PORT=8001
BACKEND_PORT=8000
FRONTEND_PORT=8502
HEALTH_CHECK_INTERVAL=5
MAX_WAIT_TIME=60

# Paths
XGBOOST_PATH="f:/PyTorch_GPU/maritime_vessel_forecasting/Multi_vessel_forecasting"
BACKEND_PATH="f:/Maritime_NLU_Repo/backend/nlu_chatbot/src/app"
FRONTEND_PATH="f:/Maritime_NLU_Repo/backend/nlu_chatbot/frontend"

# Process IDs
declare -a PIDS=()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

print_header() {
    echo ""
    echo -e "${MAGENTA}╔════════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${MAGENTA}║ $1${NC}"
    echo -e "${MAGENTA}╚════════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_status() {
    local status=$1
    local service=$2
    local message=$3
    
    if [ "$status" = "✅" ]; then
        echo -e "${GREEN}${status} ${service}${NC}"
    elif [ "$status" = "❌" ]; then
        echo -e "${RED}${status} ${service}${NC}"
    else
        echo -e "${YELLOW}${status} ${service}${NC}"
    fi
    
    if [ -n "$message" ]; then
        echo -e "${CYAN}     ${message}${NC}"
    fi
}

test_port() {
    local port=$1
    nc -z 127.0.0.1 $port 2>/dev/null
    return $?
}

test_service() {
    local service=$1
    local port=$2
    local endpoint=${3:-"/health"}
    
    if curl -s -f "http://127.0.0.1:${port}${endpoint}" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

cleanup() {
    print_header "🛑 SHUTTING DOWN SERVICES"
    
    for pid in "${PIDS[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            echo "Stopping process $pid..."
            kill $pid 2>/dev/null || true
        fi
    done
    
    echo -e "${GREEN}All services stopped. Goodbye!${NC}"
    exit 0
}

# ============================================================================
# MAIN STARTUP SEQUENCE
# ============================================================================

print_header "🚀 MARITIME NLU + XGBOOST INTEGRATION - STARTUP"

# Check prerequisites
print_header "🔍 CHECKING PREREQUISITES"

# Check Python
echo -n "Checking Python... "
if python3 --version > /dev/null 2>&1; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    print_status "✅" "Python" "$PYTHON_VERSION"
else
    print_status "❌" "Python" "Not found"
    exit 1
fi

# Check required packages
echo "Checking Python packages..."
MISSING_PACKAGES=()
for pkg in fastapi uvicorn streamlit xgboost pandas numpy; do
    if ! python3 -c "import $pkg" 2>/dev/null; then
        MISSING_PACKAGES+=($pkg)
    fi
done

if [ ${#MISSING_PACKAGES[@]} -eq 0 ]; then
    print_status "✅" "Packages" "All required packages found"
else
    print_status "⚠️" "Packages" "Missing: ${MISSING_PACKAGES[*]}"
    echo "Installing missing packages..."
    pip3 install "${MISSING_PACKAGES[@]}"
fi

# Check ports availability
echo "Checking port availability..."
PORTS_OK=true
for port in $XGBOOST_PORT $BACKEND_PORT $FRONTEND_PORT; do
    if test_port $port; then
        print_status "⚠️" "Port $port" "Already in use"
        PORTS_OK=false
    fi
done

if [ "$PORTS_OK" = true ]; then
    print_status "✅" "Ports" "All ports available"
fi

echo ""

# ============================================================================
# START SERVICES
# ============================================================================

print_header "🚀 STARTING SERVICES"

# Start XGBoost Model Server
echo "Starting XGBoost Model Server..."
(
    cd "$XGBOOST_PATH"
    python3 notebooks/43_end_to_end_xgboost_integration.py
) > /tmp/xgboost.log 2>&1 &
XGBOOST_PID=$!
PIDS+=($XGBOOST_PID)
print_status "✅" "XGBoost Model Server" "Started (PID: $XGBOOST_PID)"

sleep 2

# Start Maritime NLU Backend
echo "Starting Maritime NLU Backend..."
(
    cd "$BACKEND_PATH"
    uvicorn main:app --reload --host 127.0.0.1 --port $BACKEND_PORT
) > /tmp/backend.log 2>&1 &
BACKEND_PID=$!
PIDS+=($BACKEND_PID)
print_status "✅" "Maritime NLU Backend" "Started (PID: $BACKEND_PID)"

sleep 2

# Start Maritime NLU Frontend
echo "Starting Maritime NLU Frontend..."
(
    cd "$FRONTEND_PATH"
    streamlit run app.py --server.port $FRONTEND_PORT
) > /tmp/frontend.log 2>&1 &
FRONTEND_PID=$!
PIDS+=($FRONTEND_PID)
print_status "✅" "Maritime NLU Frontend" "Started (PID: $FRONTEND_PID)"

echo ""

# ============================================================================
# WAIT FOR SERVICES TO BE READY
# ============================================================================

print_header "⏳ WAITING FOR SERVICES TO BE READY"

START_TIME=$(date +%s)
MAX_WAIT=$((START_TIME + MAX_WAIT_TIME))

while [ $(date +%s) -lt $MAX_WAIT ]; do
    READY_COUNT=0
    
    if test_service "XGBoost" $XGBOOST_PORT; then
        print_status "✅" "XGBoost Model Server"
        ((READY_COUNT++))
    else
        print_status "⏳" "XGBoost Model Server"
    fi
    
    if test_service "Backend" $BACKEND_PORT; then
        print_status "✅" "Maritime NLU Backend"
        ((READY_COUNT++))
    else
        print_status "⏳" "Maritime NLU Backend"
    fi
    
    if test_service "Frontend" $FRONTEND_PORT; then
        print_status "✅" "Maritime NLU Frontend"
        ((READY_COUNT++))
    else
        print_status "⏳" "Maritime NLU Frontend"
    fi
    
    if [ $READY_COUNT -eq 3 ]; then
        break
    fi
    
    sleep 2
done

echo ""

# ============================================================================
# DISPLAY SERVICE ENDPOINTS
# ============================================================================

print_header "📡 SERVICE ENDPOINTS"

echo -e "${CYAN}XGBoost Model Server:${NC}"
echo -e "${GREEN}  • API: http://127.0.0.1:$XGBOOST_PORT${NC}"
echo -e "${GREEN}  • Docs: http://127.0.0.1:$XGBOOST_PORT/docs${NC}"
echo ""

echo -e "${CYAN}Maritime NLU Backend:${NC}"
echo -e "${GREEN}  • API: http://127.0.0.1:$BACKEND_PORT${NC}"
echo -e "${GREEN}  • Docs: http://127.0.0.1:$BACKEND_PORT/docs${NC}"
echo -e "${GREEN}  • Health: http://127.0.0.1:$BACKEND_PORT/health${NC}"
echo ""

echo -e "${CYAN}Maritime NLU Frontend:${NC}"
echo -e "${GREEN}  • Dashboard: http://127.0.0.1:$FRONTEND_PORT${NC}"
echo ""

# ============================================================================
# REAL-TIME HEALTH MONITOR
# ============================================================================

print_header "🔄 REAL-TIME HEALTH MONITOR"
echo -e "${YELLOW}Monitoring services every $HEALTH_CHECK_INTERVAL seconds (Press Ctrl+C to stop)${NC}"
echo ""

# Set up trap for cleanup
trap cleanup SIGINT SIGTERM

while true; do
    TIMESTAMP=$(date +"%H:%M:%S")
    echo -e "${MAGENTA}[$TIMESTAMP] Health Check:${NC}"
    
    ALL_HEALTHY=true
    
    if test_service "XGBoost" $XGBOOST_PORT; then
        print_status "✅" "XGBoost Model Server" "HEALTHY"
    else
        print_status "❌" "XGBoost Model Server" "DOWN"
        ALL_HEALTHY=false
    fi
    
    if test_service "Backend" $BACKEND_PORT; then
        print_status "✅" "Maritime NLU Backend" "HEALTHY"
    else
        print_status "❌" "Maritime NLU Backend" "DOWN"
        ALL_HEALTHY=false
    fi
    
    if test_service "Frontend" $FRONTEND_PORT; then
        print_status "✅" "Maritime NLU Frontend" "HEALTHY"
    else
        print_status "❌" "Maritime NLU Frontend" "DOWN"
        ALL_HEALTHY=false
    fi
    
    if [ "$ALL_HEALTHY" = true ]; then
        echo -e "${GREEN}  ✅ All services operational${NC}"
    else
        echo -e "${YELLOW}  ⚠️  Some services may need attention${NC}"
    fi
    
    echo ""
    sleep $HEALTH_CHECK_INTERVAL
done

