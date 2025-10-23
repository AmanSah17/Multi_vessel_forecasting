# Training and Validation Visualization Guide

## Overview

This guide explains how to run the complete training pipeline and visualize all aspects of training and validation.

---

## Quick Start

### 1. Run Training with Visualization

```bash
python notebooks/02_training_visualization.py
```

This will:
- Generate sample data (or load your own)
- Preprocess the data
- Create train/val/test split
- Train all models
- Generate predictions
- Evaluate models
- Create comprehensive visualizations
- Save results to `training_results/` directory

### 2. View Results

All visualizations are saved as PNG files in `training_results/`:
- `data_split.png` - Data distribution across splits
- `prediction_performance.png` - Model performance metrics
- `consistency_scores.png` - Trajectory consistency
- `training_report.txt` - Summary report

---

## Detailed Usage

### Using Your Own Data

```python
from notebooks.training_visualization import run_training_with_visualization

# Run with your data
pipeline, visualizer, metrics = run_training_with_visualization(
    data_filepath='path/to/your/ais_data.csv',
    output_dir='my_training_results'
)
```

### Data Format Requirements

Your CSV file should have these columns:
```
MMSI,BaseDateTime,LAT,LON,SOG,COG,VesselName,IMO
200000001,2024-01-01T00:00:00,40.7128,-74.0060,12.5,45.0,Vessel_1,1000001
200000002,2024-01-01T00:01:00,40.7130,-74.0062,11.8,46.5,Vessel_2,1000002
```

**Required Columns:**
- `MMSI`: Maritime Mobile Service Identity (9 digits)
- `BaseDateTime`: Timestamp (ISO format)
- `LAT`: Latitude (-90 to +90)
- `LON`: Longitude (-180 to +180)
- `SOG`: Speed Over Ground (knots)
- `COG`: Course Over Ground (0-360°)

**Optional Columns:**
- `VesselName`: Name of vessel
- `IMO`: International Maritime Organization number

---

## Visualization Components

### 1. Data Split Visualization

Shows how data is distributed across train/val/test sets:

**Plots:**
- **Pie Chart**: Percentage distribution
- **Bar Chart**: Absolute record counts
- **Time Series**: Records per day by split
- **Vessel Distribution**: Vessels per split

**What to Look For:**
- ✅ Balanced distribution across splits
- ✅ Temporal continuity (no gaps)
- ✅ Sufficient vessels in each split
- ⚠️ Imbalanced splits may affect validation

### 2. Prediction Performance

Compares prediction accuracy across models:

**Metrics:**
- **MAE** (Mean Absolute Error): Average prediction error
- **RMSE** (Root Mean Squared Error): Penalizes large errors
- **MAPE** (Mean Absolute Percentage Error): Percentage error

**Plots:**
- **Model Comparison**: MAE, RMSE, MAPE by model
- **Scatter Plot**: Actual vs Predicted values

**Interpretation:**
- Lower MAE/RMSE/MAPE = Better predictions
- Scatter plot should follow diagonal line
- Ensemble typically has best performance

### 3. Consistency Scores

Evaluates trajectory smoothness and physical validity:

**Plots:**
- **Bar Chart**: Consistency score per vessel
- **Histogram**: Distribution of scores

**Score Interpretation:**
- **0.85-1.0**: Excellent (green)
- **0.60-0.85**: Good (yellow)
- **< 0.60**: Poor (red)

**What Affects Scores:**
- Smoothness of movement (last 3 points)
- Speed consistency
- Turn rate validity
- Acceleration limits

### 4. Anomaly Detection Metrics

Shows effectiveness of anomaly detectors:

**Metrics:**
- **Precision**: % of detected anomalies that are real
- **Recall**: % of real anomalies that are detected
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Overall discrimination ability

**Targets:**
- Precision > 90%
- Recall > 85%
- F1-Score > 87%
- ROC-AUC > 0.90

---

## Advanced Usage

### Custom Visualization

```python
from src.training_visualization import TrainingVisualizer
import pandas as pd

# Create visualizer
visualizer = TrainingVisualizer(figsize=(16, 12))

# Load your data
train_df = pd.read_csv('train_data.csv')
val_df = pd.read_csv('val_data.csv')
test_df = pd.read_csv('test_data.csv')

# Create specific visualizations
visualizer.plot_data_split(train_df, val_df, test_df, 
                          output_path='my_split.png')

visualizer.plot_consistency_scores(consistency_dict,
                                  output_path='my_consistency.png')
```

### Programmatic Access to Metrics

```python
from src.training_pipeline import TrainingPipeline

pipeline = TrainingPipeline()
pipeline.run_full_pipeline('data.csv')

# Access metrics
metrics = pipeline.metrics
print(f"Consistency: {metrics['trajectory_verification']}")
print(f"Anomalies: {metrics['anomaly_detection']}")
```

### Batch Processing Multiple Datasets

```python
from pathlib import Path
from notebooks.training_visualization import run_training_with_visualization

data_dir = Path('datasets')
for data_file in data_dir.glob('*.csv'):
    print(f"Processing {data_file.name}")
    pipeline, visualizer, metrics = run_training_with_visualization(
        data_filepath=str(data_file),
        output_dir=f'results/{data_file.stem}'
    )
```

---

## Interpreting Results

### Good Training Indicators

✅ **Data Split:**
- Balanced distribution (60/20/20)
- Temporal continuity
- Multiple vessels in each split

✅ **Prediction Performance:**
- MAE < 1 km
- RMSE < 2 km
- MAPE < 5%
- Ensemble better than individual models

✅ **Consistency Scores:**
- Average > 0.85
- Few vessels with score < 0.60
- Smooth distribution

✅ **Anomaly Detection:**
- Precision > 90%
- Recall > 85%
- F1-Score > 87%

### Warning Signs

⚠️ **Data Issues:**
- Imbalanced splits
- Temporal gaps
- Too few vessels
- Missing values

⚠️ **Poor Performance:**
- MAE > 5 km
- RMSE > 10 km
- MAPE > 20%
- Ensemble not better than baseline

⚠️ **Consistency Issues:**
- Average score < 0.70
- Many vessels with score < 0.60
- Bimodal distribution

⚠️ **Anomaly Detection:**
- Precision < 80%
- Recall < 70%
- F1-Score < 75%

---

## Troubleshooting

### Issue: "No data generated"

**Solution:**
```python
# Ensure data has required columns
required = ['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG']
assert all(col in df.columns for col in required)
```

### Issue: "Models not training"

**Solution:**
```python
# Check data quality
print(df.info())
print(df.describe())
print(df.isnull().sum())
```

### Issue: "Visualizations not showing"

**Solution:**
```python
import matplotlib.pyplot as plt
plt.show()  # Add this at the end
```

### Issue: "Memory error with large datasets"

**Solution:**
```python
# Process in batches
batch_size = 100000
for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]
    # Process batch
```

---

## Performance Optimization

### Faster Training

```python
# Use fewer vessels for testing
df_sample = df[df['MMSI'].isin(df['MMSI'].unique()[:5])]

# Use smaller time window
df_sample = df[df['BaseDateTime'] >= '2024-01-01']
df_sample = df_sample[df_sample['BaseDateTime'] < '2024-01-08']
```

### Reduce Memory Usage

```python
# Use float32 instead of float64
df = df.astype({'LAT': 'float32', 'LON': 'float32'})

# Drop unnecessary columns
df = df[['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG']]
```

### Parallel Processing

```python
from multiprocessing import Pool

def process_vessel(mmsi):
    vessel_data = df[df['MMSI'] == mmsi]
    return train_model(vessel_data)

with Pool(4) as p:
    results = p.map(process_vessel, df['MMSI'].unique())
```

---

## Output Files

### Generated Files

```
training_results/
├── data_split.png              # Data distribution visualization
├── prediction_performance.png  # Model performance metrics
├── consistency_scores.png      # Trajectory consistency
├── training_report.txt         # Summary report
├── sample_data.csv            # Generated sample data
└── models/
    ├── prediction_kalman.pkl
    ├── prediction_arima.pkl
    ├── prediction_ensemble.pkl
    ├── anomaly_isolation_forest.pkl
    ├── anomaly_rule_based.pkl
    └── anomaly_ensemble.pkl
```

### Report Contents

The `training_report.txt` includes:
- Data statistics
- Data split information
- Models trained
- Prediction performance metrics
- Trajectory verification results
- Output file locations

---

## Next Steps

1. **Review Results**: Check all PNG visualizations
2. **Analyze Metrics**: Compare against performance targets
3. **Tune Hyperparameters**: If performance is below targets
4. **Deploy Models**: Save and integrate into production
5. **Monitor Performance**: Track metrics over time

---

## References

- **Data Visualization**: Matplotlib, Seaborn
- **Training Pipeline**: scikit-learn, TensorFlow
- **Metrics**: Standard ML evaluation metrics
- **Best Practices**: Hyndman & Athanasopoulos (2021)

---

## Support

For issues or questions:
1. Check IMPLEMENTATION_GUIDE.md
2. Review source code comments
3. Check REFERENCES.md for research backing
4. Review example notebooks

---

## Summary

This visualization system provides comprehensive insights into:
- ✅ Data quality and distribution
- ✅ Model performance and comparison
- ✅ Prediction accuracy
- ✅ Trajectory consistency
- ✅ Anomaly detection effectiveness

Use these visualizations to:
- Validate data preprocessing
- Compare model performance
- Identify issues early
- Make informed deployment decisions
- Monitor production performance

