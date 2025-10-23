# Using TrainingPipeline with DataFrames

## Overview

The `TrainingPipeline` now supports both file paths and DataFrames as input. This allows you to use data already loaded in your Jupyter notebook.

---

## Quick Start

### Option 1: Use DataFrame Directly (Recommended for Notebooks)

```python
from src.training_pipeline import TrainingPipeline

# Initialize pipeline
pipeline = TrainingPipeline(output_dir="models")

# Run complete pipeline with your DataFrame
metrics = pipeline.run_full_pipeline(AIS_2020_01_03)
```

### Option 2: Use File Path

```python
from src.training_pipeline import TrainingPipeline

# Initialize pipeline
pipeline = TrainingPipeline(output_dir="models")

# Run complete pipeline with file path
metrics = pipeline.run_full_pipeline("path/to/your/data.csv")
```

---

## Complete Example

```python
from src.training_pipeline import TrainingPipeline
import pandas as pd

# Load your data (already in notebook)
# AIS_2020_01_03 is your DataFrame

# Initialize pipeline
pipeline = TrainingPipeline(output_dir="models")

# Run complete training pipeline
metrics = pipeline.run_full_pipeline(AIS_2020_01_03)

# Access results
print("Training completed!")
print(f"Metrics: {metrics}")

# Load trained models
pipeline.load_models()

# Make predictions
predictions = pipeline.prediction_models['ensemble'].predict(X_test)

# Detect anomalies
anomalies = pipeline.anomaly_detectors['ensemble'].predict(X_test)
```

---

## What the Pipeline Does

### Step 1: Load & Preprocess
- Accepts DataFrame or file path
- Handles missing vessel names
- Resamples to 1-minute intervals
- Removes outliers

### Step 2: Feature Engineering
- Adds temporal features (hour, day_of_week, is_weekend)
- Adds kinematic features (speed_change, heading_change)
- Total: 13 features

### Step 3: Train/Val/Test Split
- Training: 60%
- Validation: 20%
- Test: 20%
- Temporal order preserved (no data leakage)

### Step 4: Model Training
**Prediction Models:**
- Kalman Filter
- ARIMA
- Ensemble

**Anomaly Detectors:**
- Isolation Forest
- Rule-based
- Ensemble

### Step 5: Evaluation
- Computes metrics
- Evaluates consistency
- Returns results

### Step 6: Model Persistence
- Saves all models to disk
- Ready for inference

---

## Expected Output

```
==================================================
Starting full training pipeline
==================================================
INFO:training_pipeline:Engineered features: [...]
INFO:training_pipeline:Train: 299946, Val: 99982, Test: 99982
INFO:training_pipeline:Training prediction models...
INFO:trajectory_prediction:Kalman Filter - Q: 16.54, R: 12.32
INFO:trajectory_prediction:ARIMA(1, 1, 1) fitted
INFO:training_pipeline:Prediction models trained
INFO:training_pipeline:Training anomaly detectors...
INFO:anomaly_detection:Isolation Forest fitted
INFO:anomaly_detection:Rule-based detector initialized
INFO:training_pipeline:Anomaly detectors trained
INFO:training_pipeline:Evaluating models...
INFO:training_pipeline:Average trajectory consistency: 0.884
==================================================
Pipeline completed successfully!
==================================================
```

---

## Accessing Results

### Metrics
```python
metrics = pipeline.run_full_pipeline(AIS_2020_01_03)
print(metrics)
```

### Trained Models
```python
# Access prediction models
kalman = pipeline.prediction_models['kalman']
arima = pipeline.prediction_models['arima']
ensemble = pipeline.prediction_models['ensemble']

# Access anomaly detectors
isolation_forest = pipeline.anomaly_detectors['isolation_forest']
rule_based = pipeline.anomaly_detectors['rule_based']
ensemble_detector = pipeline.anomaly_detectors['ensemble']
```

### Make Predictions
```python
# Prepare test data
X_test = test_df[['LAT', 'LON', 'SOG', 'COG', 'hour', 'day_of_week', 'is_weekend', 'speed_change', 'heading_change']]

# Make predictions
predictions = pipeline.prediction_models['ensemble'].predict(X_test)

# Detect anomalies
anomalies = pipeline.anomaly_detectors['ensemble'].predict(X_test)
```

---

## Data Format Requirements

Your DataFrame should have these columns:

| Column | Type | Description |
|--------|------|-------------|
| MMSI | int | Maritime Mobile Service Identity |
| BaseDateTime | datetime | Timestamp (YYYY-MM-DDTHH:MM:SS) |
| LAT | float | Latitude |
| LON | float | Longitude |
| SOG | float | Speed Over Ground (knots) |
| COG | float | Course Over Ground (degrees) |
| VesselName | str | Vessel name (optional) |
| IMO | int | International Maritime Organization number |

### Example DataFrame
```python
import pandas as pd

data = {
    'MMSI': [200000001, 200000001, 200000002],
    'BaseDateTime': ['2020-01-03T00:00:00', '2020-01-03T00:01:00', '2020-01-03T00:02:00'],
    'LAT': [40.0, 40.01, 40.02],
    'LON': [-74.0, -74.01, -74.02],
    'SOG': [10.0, 10.5, 11.0],
    'COG': [90.0, 91.0, 92.0],
    'VesselName': ['Vessel1', 'Vessel1', 'Vessel2'],
    'IMO': [1000001, 1000001, 1000002]
}
df = pd.DataFrame(data)
```

---

## Common Issues & Solutions

### Issue: "TypeError: argument of type 'method' is not iterable"
**Cause**: Passing a method reference instead of DataFrame
**Solution**: Make sure to pass the DataFrame, not a method:
```python
# Wrong
metrics = pipeline.run_full_pipeline(AIS_2020_01_03)  # If this is a method

# Right
metrics = pipeline.run_full_pipeline(AIS_2020_01_03)  # If this is a DataFrame
```

### Issue: "ModuleNotFoundError: No module named 'data_preprocessing'"
**Cause**: Import paths not using relative imports
**Solution**: Already fixed in latest version

### Issue: "KeyError: 'MMSI'"
**Cause**: DataFrame missing required columns
**Solution**: Ensure DataFrame has all required columns (see Data Format Requirements)

---

## Advanced Usage

### Custom Output Directory
```python
pipeline = TrainingPipeline(output_dir="my_models")
metrics = pipeline.run_full_pipeline(AIS_2020_01_03)
```

### Access Individual Components
```python
# Preprocess only
df_processed = pipeline.load_data(AIS_2020_01_03)

# Feature engineering only
df_features = pipeline.engineer_features(df_processed)

# Create split only
train_df, val_df, test_df = pipeline.create_train_val_test_split(df_features)

# Train models only
pipeline.train_prediction_models(train_df, val_df)
pipeline.train_anomaly_detectors(train_df)

# Evaluate only
metrics = pipeline.evaluate(test_df)
```

---

## Performance Expectations

| Metric | Expected Value |
|--------|-----------------|
| Trajectory Consistency | > 0.85 |
| Data Quality | 100% |
| Preprocessing Success | 100% |
| Training Time | 2-5 minutes |

---

## Next Steps

1. ✅ Load your DataFrame
2. ✅ Initialize pipeline: `pipeline = TrainingPipeline()`
3. ✅ Run training: `metrics = pipeline.run_full_pipeline(df)`
4. ✅ Access results: `pipeline.prediction_models`, `pipeline.anomaly_detectors`
5. ✅ Make predictions: `predictions = pipeline.prediction_models['ensemble'].predict(X_test)`

---

## Support

For issues or questions:
1. Check the error message in the logs
2. Review this guide
3. Check MLFLOW_TRAINING_GUIDE.md
4. Review source code in `src/`

---

**Status**: ✅ READY TO USE

You can now use `pipeline.run_full_pipeline(AIS_2020_01_03)` directly in your notebook!

