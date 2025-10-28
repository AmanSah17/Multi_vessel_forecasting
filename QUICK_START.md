# 🚀 Quick Start Guide - Maritime NLU + XGBoost

## One-Command Startup

### Windows (PowerShell)
```powershell
.\START_ALL_SERVICES.ps1
```

### Windows (Batch)
```batch
START_ALL_SERVICES.bat
```

### Linux/Mac
```bash
bash START_ALL_SERVICES.sh
```

---

## 📡 Access Services

| Service | URL | Purpose |
|---------|-----|---------|
| **Frontend Dashboard** | http://127.0.0.1:8502 | Main UI for predictions & verification |
| **Backend API** | http://127.0.0.1:8000 | NLU & prediction API |
| **Backend Docs** | http://127.0.0.1:8000/docs | Interactive API documentation |
| **XGBoost Server** | http://127.0.0.1:8001 | Model server (internal) |

---

## ✅ Verify Services Running

### Option 1: Health Monitor
```bash
python health_monitor.py
```

### Option 2: Run Tests
```bash
python test_services.py
```

### Option 3: Manual Check
```bash
# Check each service
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8502
```

---

## 🧪 Test Predictions

### Via Dashboard
1. Open http://127.0.0.1:8502
2. Select vessel from dropdown
3. Click "Predict Position"
4. View results and visualization

### Via API
```bash
# Predict vessel position
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"text": "Predict CHAMPAGNE CHER position after 30 minutes"}'

# Verify vessel course
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"text": "Verify CHAMPAGNE CHER course"}'

# Show vessel position
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"text": "Show CHAMPAGNE CHER"}'
```

---

## 🔍 Monitor Services

### Real-Time Health Monitor
```bash
# Check every 5 seconds
python health_monitor.py

# Check every 10 seconds
python health_monitor.py --interval 10

# Check once and exit
python health_monitor.py --once
```

### View Logs
```bash
# Backend logs
Get-Content -Path "logs/backend.log" -Wait

# Frontend logs
Get-Content -Path "logs/frontend.log" -Wait
```

---

## 🛑 Stop Services

### PowerShell Script
```powershell
# Press Ctrl+C in the PowerShell window
# Services will automatically shut down
```

### Manual Cleanup
```bash
# Kill all Python processes
taskkill /F /IM python.exe

# Or kill specific services
taskkill /F /IM uvicorn.exe
taskkill /F /IM streamlit.exe
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Latitude MAE** | 0.3056° |
| **Longitude MAE** | 1.1040° |
| **Overall MAE** | 8.18 |
| **Latitude R²** | 0.9973 |
| **Longitude R²** | 0.9971 |
| **Overall R²** | 0.9351 |

---

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Find process using port
netstat -ano | findstr :8000

# Kill process
taskkill /PID <PID> /F
```

### Module Not Found
```bash
# Reinstall dependencies
pip install --upgrade fastapi uvicorn streamlit xgboost pandas numpy scikit-learn matplotlib
```

### Database Connection Error
```bash
# Check database exists
ls f:\Maritime_NLU_Repo\backend\nlu_chatbot\maritime_*.db

# Update database path in main.py if needed
```

### XGBoost Model Not Loading
```bash
# Check model files
ls results/xgboost_advanced_50_vessels/

# Verify model integrity
python -c "import pickle; pickle.load(open('results/xgboost_advanced_50_vessels/xgboost_model.pkl', 'rb'))"
```

---

## 📚 Documentation

- **Complete Guide**: `COMPLETE_STARTUP_GUIDE.md`
- **Integration Guide**: `XGBOOST_MARITIME_NLU_INTEGRATION.md`
- **API Documentation**: http://127.0.0.1:8000/docs

---

## 🎯 Next Steps

1. ✅ Start services using `START_ALL_SERVICES.ps1`
2. ✅ Open dashboard at http://127.0.0.1:8502
3. ✅ Select a vessel and make predictions
4. ✅ Monitor health using `health_monitor.py`
5. ✅ Run tests using `test_services.py`

---

## 📞 Support

For issues or questions:
1. Check logs in terminal windows
2. Run `test_services.py` to diagnose problems
3. Review `COMPLETE_STARTUP_GUIDE.md` for detailed troubleshooting
4. Check API docs at http://127.0.0.1:8000/docs

---

**Status**: ✅ Ready for Production  
**Last Updated**: 2025-10-25  
**Version**: 1.0

