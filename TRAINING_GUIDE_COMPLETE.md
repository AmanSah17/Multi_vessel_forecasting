# Complete Training Guide

## Overview

Three training scripts are available with different trade-offs:

| Script | Sample Size | Time | Memory | Best For |
|--------|-------------|------|--------|----------|
| **Quick Demo** | 50K | 5-10 min | 2-3 GB | Testing, demos |
| **Optimized** | 500K | 30-60 min | 4-6 GB | Production |
| **Full** | 7.1M | 2-3 hours | 8-10 GB | Maximum accuracy |

---

## Option 1: Quick Training Demo (Recommended for Testing)

**Best for**: Quick results, testing, demonstrations

**Time**: 5-10 minutes

**Memory**: 2-3 GB

### Run Quick Training
```bash
python notebooks/07_quick_training_demo.py
```

### What It Does
1. Loads 7.1M records from AIS data
2. Samples to 50K records (stratified by vessel)
3. Preprocesses and engineers features
4. Trains all models
5. Evaluates on test set
6. Saves results and models

### Output
```
training_logs_quick/
├── training_results.json
└── training_quick_demo.log

models/
├── prediction_kalman.pkl
├── prediction_arima.pkl
├── prediction_ensemble.pkl
├── anomaly_isolation_forest.pkl
├── anomaly_rule_based.pkl
└── anomaly_ensemble.pkl
```

### Expected Results
```
Raw data: 7,118,203 records, 14,417 vessels
Sampled to: 50,000 records
Preprocessed to: ~150,000 records (after resampling)
Train: ~90,000 | Val: ~30,000 | Test: ~30,000
Trajectory consistency: ~0.88
```

---

## Option 2: Optimized Training (Recommended for Production)

**Best for**: Production use, balanced accuracy/speed

**Time**: 30-60 minutes

**Memory**: 4-6 GB

### Run Optimized Training
```bash
python notebooks/06_training_optimized_large_data.py
```

### What It Does
1. Loads 7.1M records
2. Samples to 500K records (stratified)
3. Preprocesses and engineers features
4. Trains all models
5. Evaluates on test set
6. Saves results and models

### Output
```
training_logs_optimized/
├── training_results.json
└── training_optimized.log

models/
├── prediction_kalman.pkl
├── prediction_arima.pkl
├── prediction_ensemble.pkl
├── anomaly_isolation_forest.pkl
├── anomaly_rule_based.pkl
└── anomaly_ensemble.pkl
```

### Expected Results
```
Raw data: 7,118,203 records, 14,417 vessels
Sampled to: 494,028 records
Preprocessed to: ~15,189,225 records (after resampling)
Train: ~9.1M | Val: ~3.0M | Test: ~3.0M
Trajectory consistency: ~0.90
```

---

## Option 3: Full Training (Maximum Accuracy)

**Best for**: Maximum accuracy, research

**Time**: 2-3 hours

**Memory**: 8-10 GB

### Run Full Training
```bash
python notebooks/05_training_with_logging.py
```

### What It Does
1. Loads 7.1M records
2. Uses all data (no sampling)
3. Preprocesses and engineers features
4. Trains all models
5. Evaluates on test set
6. Saves results and models

### Output
```
training_logs/
├── training_results.json
├── training_report.txt
└── training_with_logging.log

models/
├── prediction_kalman.pkl
├── prediction_arima.pkl
├── prediction_ensemble.pkl
├── anomaly_isolation_forest.pkl
├── anomaly_rule_based.pkl
└── anomaly_ensemble.pkl
```

### Expected Results
```
Raw data: 7,118,203 records, 14,417 vessels
Preprocessed to: ~16M+ records (after resampling)
Train: ~9.6M | Val: ~3.2M | Test: ~3.2M
Trajectory consistency: ~0.92
```

---

## Using Trained Models

### Load Models
```python
from src.training_pipeline import TrainingPipeline

pipeline = TrainingPipeline(output_dir='models')
pipeline.load_models()

# Access models
kalman = pipeline.prediction_models['kalman']
arima = pipeline.prediction_models['arima']
ensemble = pipeline.prediction_models['ensemble']

isolation_forest = pipeline.anomaly_detectors['isolation_forest']
rule_based = pipeline.anomaly_detectors['rule_based']
anomaly_ensemble = pipeline.anomaly_detectors['ensemble']
```

### Make Predictions
```python
# Prepare test data
X_test = test_df[['LAT', 'LON', 'SOG', 'COG', 'hour', 'day_of_week', 
                   'is_weekend', 'speed_change', 'heading_change']]

# Predict next position
predictions = ensemble.predict(X_test)

# Detect anomalies
anomalies = anomaly_ensemble.predict(X_test)
```

---

## Monitoring Training Progress

### Option 1: Watch Log File
```bash
# PowerShell
Get-Content training_quick_demo.log -Wait

# Or check last 50 lines
Get-Content training_quick_demo.log -Tail 50
```

### Option 2: Check Results JSON
```bash
# View results
Get-Content training_logs_quick/training_results.json | ConvertFrom-Json | Format-Table

# Or pretty print
python -m json.tool training_logs_quick/training_results.json
```

### Option 3: Monitor System Resources
```bash
# Check memory usage
Get-Process python | Select-Object Name, @{Name="Memory(MB)";Expression={$_.WorkingSet/1MB}}

# Check CPU usage
Get-Process python | Select-Object Name, CPU, Handles
```

---

## Troubleshooting

### Issue: "Unable to allocate memory"
**Solution**: Use smaller sample size
```python
# In script, change:
sample_size=50000  # Instead of 500000
```

### Issue: "Training takes too long"
**Solution**: Use quick demo instead
```bash
python notebooks/07_quick_training_demo.py
```

### Issue: "ModuleNotFoundError"
**Solution**: Ensure you're in correct directory
```bash
cd f:\PyTorch_GPU\maritime_vessel_forecasting\Multi_vessel_forecasting
python notebooks/07_quick_training_demo.py
```

### Issue: "File not found"
**Solution**: Check data path in script
```python
# Verify this path exists:
data_path = r"D:\Maritime_Vessel_monitoring\csv_extracted_data\AIS_2020_01_03\AIS_2020_01_03.csv"
```

---

## Performance Comparison

### Quick Demo (50K sample)
```
Data Loading: 30 sec
Sampling: 1 min
Preprocessing: 2 min
Feature Engineering: 1 min
Train/Val/Test Split: 30 sec
Model Training: 2 min
Evaluation: 1 min
Total: ~8 minutes
```

### Optimized (500K sample)
```
Data Loading: 30 sec
Sampling: 2 min
Preprocessing: 5 min
Feature Engineering: 10 min
Train/Val/Test Split: 2 min
Model Training: 10 min
Evaluation: 3 min
Total: ~32 minutes
```

### Full (7.1M all data)
```
Data Loading: 30 sec
Preprocessing: 10 min
Feature Engineering: 30 min
Train/Val/Test Split: 5 min
Model Training: 30 min
Evaluation: 10 min
Total: ~85 minutes
```

---

## Recommended Workflow

### Step 1: Quick Test
```bash
# Test with quick demo (5-10 min)
python notebooks/07_quick_training_demo.py
```

### Step 2: Review Results
```bash
# Check results
Get-Content training_logs_quick/training_results.json | ConvertFrom-Json
```

### Step 3: Production Training
```bash
# Run optimized training (30-60 min)
python notebooks/06_training_optimized_large_data.py
```

### Step 4: Use Models
```python
# Load and use trained models
from src.training_pipeline import TrainingPipeline
pipeline = TrainingPipeline(output_dir='models')
pipeline.load_models()
predictions = pipeline.prediction_models['ensemble'].predict(X_test)
```

---

## Key Metrics Explained

### Trajectory Consistency
- **Range**: 0.0 - 1.0
- **Meaning**: How smooth vessel movements are
- **Target**: > 0.85
- **Interpretation**: 0.88 = 88% of movements are smooth and realistic

### Prediction Accuracy
- **MAE**: Mean Absolute Error (position error in km)
- **RMSE**: Root Mean Squared Error
- **Target**: < 5 km for 10-minute prediction

### Anomaly Detection
- **Precision**: % of detected anomalies that are real
- **Recall**: % of real anomalies that are detected
- **F1-Score**: Harmonic mean of precision and recall

---

## Data Statistics

### Raw Data (7.1M records)
- Vessels: 14,417
- Time: 24 hours (2020-01-03)
- Columns: 17
- Memory: 2.3 GB

### After Preprocessing
- Records: 15.2M (after 1-min resampling)
- Vessels: 14,368
- Duplicates removed: 4
- Outliers removed: 29,833
- Memory: 3.6 GB

### After Feature Engineering
- Features: 13 total
- Temporal: 3 (hour, day_of_week, is_weekend)
- Kinematic: 6 (speed_change, heading_change, etc.)
- Statistical: 4 (rolling means/stds)

---

## Next Steps

1. ✅ Choose training option (Quick/Optimized/Full)
2. ✅ Run training script
3. ✅ Monitor progress via logs
4. ✅ Review results in JSON
5. ✅ Load models for inference
6. ✅ Make predictions on new data

---

**Status**: ✅ READY TO TRAIN

Choose your training option and run the script!

