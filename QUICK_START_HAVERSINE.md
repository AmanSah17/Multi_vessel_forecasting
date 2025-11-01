# Quick Start: Enhanced LSTM with Haversine Distance

## 🚀 Run in 3 Steps

### 1. Start MLflow UI
```bash
cd Multi_vessel_forecasting
python -m mlflow ui --backend-store-uri file:./mlruns --port 5000
```
**Access**: http://127.0.0.1:5000

### 2. Run Pipeline
```bash
python 49_main_enhanced_lstm_pipeline.py
```

### 3. View Results
- Open MLflow UI in browser
- Check `results/enhanced_lstm_haversine/` for outputs

## 📊 What You Get

### Metrics
- **Haversine Distance Errors** (meters): Mean, Median, P95, P99
- **Standard Metrics**: MAE, RMSE, R²
- **Per-Output Metrics**: LAT, LON, SOG, COG errors

### Visualizations
1. Feature distributions
2. Correlation heatmap
3. PCA variance
4. Cluster map
5. Training curves with haversine errors
6. Predictions vs actual

### Artifacts
- Trained model (`best_lstm_model_haversine.pt`)
- Scaler (`scaler.pkl`)
- Configuration (`config.json`)
- All plots (PNG files)

## ⚙️ Quick Configuration

Edit in `49_main_enhanced_lstm_pipeline.py`:

```python
CONFIG = {
    'start_date': 3,           # Jan 3
    'end_date': 8,             # Jan 8
    'sample_per_day': 50000,   # Samples per day
    'n_trials': 20,            # Optuna trials
    'final_epochs': 200,       # Training epochs
}
```

## 🎯 Expected Results

| Metric | Target |
|--------|--------|
| Haversine Mean Error | < 500m |
| Haversine P95 Error | < 1500m |
| R² Score | > 0.90 |

## 🔧 Common Adjustments

### Faster Training
```python
CONFIG['n_trials'] = 10
CONFIG['final_epochs'] = 100
CONFIG['sample_per_day'] = 20000
```

### Better Accuracy
```python
CONFIG['n_trials'] = 50
CONFIG['final_epochs'] = 300
CONFIG['seq_length'] = 50
```

### Memory Issues
```python
CONFIG['sample_per_day'] = 20000
# Also reduce batch_size in hyperparameters
```

## 📁 Output Structure

```
results/enhanced_lstm_haversine/
├── 01_eda_distributions.png
├── 02_eda_correlation.png
├── 03_pca_variance.png
├── 04_clusters_map.png
├── 05_training_curves_haversine.png
├── 06_predictions_vs_actual.png
├── best_lstm_model_haversine.pt
├── final_model.pt
├── scaler.pkl
└── config.json
```

## 🐛 Troubleshooting

### MLflow UI not showing logs?
```bash
# Restart MLflow UI
python -m mlflow ui --backend-store-uri file:./mlruns --port 5000
```

### Out of memory?
Reduce `sample_per_day` and `batch_size`

### Slow training?
Reduce `n_trials` and `final_epochs`

## 📚 Full Documentation

See `ENHANCED_LSTM_HAVERSINE_README.md` for complete details.

## ✅ Checklist

- [ ] MLflow UI running
- [ ] Data path correct (`D:\Maritime_Vessel_monitoring\csv_extracted_data`)
- [ ] GPU available (optional, but faster)
- [ ] Sufficient disk space (~5GB)
- [ ] Python packages installed (torch, optuna, mlflow, sklearn)

## 🎉 That's It!

Run the pipeline and check MLflow UI for results. The model will automatically find the best hyperparameters and train with them.

---

**Time to Complete**: ~2-4 hours (depending on hardware)
**GPU Recommended**: Yes (10x faster)
**Disk Space**: ~5GB

