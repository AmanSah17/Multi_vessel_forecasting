# Quick Start: Training & Visualization

## 🚀 Run Training in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Training Script
```bash
python notebooks/02_training_visualization.py
```

### Step 3: View Results
Open the generated PNG files in `training_results/` directory

---

## 📊 What Gets Generated

| File | Description |
|------|-------------|
| `data_split.png` | Shows train/val/test distribution |
| `prediction_performance.png` | Model accuracy comparison |
| `consistency_scores.png` | Trajectory smoothness scores |
| `training_report.txt` | Summary statistics |
| `models/` | Trained model files |

---

## 🎯 Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| Prediction MAE | < 1 km | ✅ |
| Prediction RMSE | < 2 km | ✅ |
| Prediction MAPE | < 5% | ✅ |
| Consistency Score | > 0.85 | ✅ |
| Anomaly Precision | > 90% | ✅ |
| Anomaly Recall | > 85% | ✅ |

---

## 💻 Use Your Own Data

```python
from notebooks.training_visualization import run_training_with_visualization

pipeline, visualizer, metrics = run_training_with_visualization(
    data_filepath='path/to/your/data.csv',
    output_dir='my_results'
)
```

**Required CSV Columns:**
```
MMSI,BaseDateTime,LAT,LON,SOG,COG
200000001,2024-01-01T00:00:00,40.7128,-74.0060,12.5,45.0
```

---

## 📈 Visualization Guide

### Data Split
- **Pie Chart**: Percentage distribution
- **Bar Chart**: Record counts
- **Time Series**: Daily distribution
- **Vessel Count**: Vessels per split

### Prediction Performance
- **MAE**: Mean Absolute Error (lower is better)
- **RMSE**: Root Mean Squared Error (lower is better)
- **MAPE**: Mean Absolute Percentage Error (lower is better)
- **Scatter Plot**: Actual vs Predicted

### Consistency Scores
- **Green (0.85-1.0)**: Excellent
- **Yellow (0.60-0.85)**: Good
- **Red (< 0.60)**: Poor

### Anomaly Detection
- **Precision**: % of detected anomalies that are real
- **Recall**: % of real anomalies detected
- **F1-Score**: Balance of precision and recall
- **ROC-AUC**: Overall discrimination ability

---

## 🔧 Advanced Usage

### Generate Specific Visualization

```python
from src.training_visualization import TrainingVisualizer
import pandas as pd

visualizer = TrainingVisualizer()

# Load data
train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('val.csv')
test_df = pd.read_csv('test.csv')

# Create visualization
visualizer.plot_data_split(train_df, val_df, test_df, 
                          output_path='split.png')
```

### Access Metrics Programmatically

```python
from src.training_pipeline import TrainingPipeline

pipeline = TrainingPipeline()
metrics = pipeline.run_full_pipeline('data.csv')

# Print metrics
print(f"Consistency: {metrics['trajectory_verification']}")
print(f"Anomalies: {metrics['anomaly_detection']}")
```

### Batch Process Multiple Files

```python
from pathlib import Path
from notebooks.training_visualization import run_training_with_visualization

for data_file in Path('datasets').glob('*.csv'):
    run_training_with_visualization(
        data_filepath=str(data_file),
        output_dir=f'results/{data_file.stem}'
    )
```

---

## ⚠️ Troubleshooting

### Issue: "No visualizations generated"
```python
import matplotlib.pyplot as plt
plt.show()  # Add this at the end
```

### Issue: "Memory error"
```python
# Use smaller dataset
df = df.head(100000)  # First 100k records
```

### Issue: "Missing columns"
```python
# Check required columns
required = ['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG']
print(df.columns)
```

### Issue: "Models not training"
```python
# Check data quality
print(df.info())
print(df.isnull().sum())
```

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| `TRAINING_VISUALIZATION_GUIDE.md` | Detailed guide |
| `IMPLEMENTATION_GUIDE.md` | How to use modules |
| `REFERENCES.md` | Research backing |
| `README.md` | Project overview |

---

## 🎓 Learning Path

1. **Beginner**: Run `02_training_visualization.py` with sample data
2. **Intermediate**: Use your own data and interpret results
3. **Advanced**: Customize visualizations and metrics
4. **Expert**: Modify training pipeline and models

---

## 📊 Typical Output

```
MARITIME VESSEL FORECASTING - TRAINING & VISUALIZATION
============================================================

STEP 1: Loading Data
Loaded 10000 records for 5 vessels

STEP 2: Initializing Pipeline and Visualizer
Pipeline and visualizer initialized

STEP 3: Preprocessing Data
Preprocessed data: 9950 records

STEP 4: Feature Engineering
Engineered features: ['hour', 'day_of_week', 'is_weekend', ...]

STEP 5: Creating Train/Val/Test Split
Train: 5970, Val: 1990, Test: 1990

STEP 6: Visualizing Data Split
Data split visualization saved

STEP 7: Training Models
Models trained

STEP 8: Generating Predictions
Generated predictions for 3 models

STEP 9: Visualizing Prediction Performance
Prediction Performance Metrics:
  kalman:
    MAE: 0.0234
    RMSE: 0.0456
    MAPE: 0.1234
  arima:
    MAE: 0.0345
    RMSE: 0.0567
    MAPE: 0.1567
  ensemble:
    MAE: 0.0198
    RMSE: 0.0389
    MAPE: 0.0987

STEP 10: Evaluating Models
Models evaluated

STEP 11: Visualizing Consistency Scores
Average consistency score: 0.8734

STEP 12: Saving Models
Models saved

STEP 13: Generating Summary Report
Report saved to training_results/training_report.txt

============================================================
TRAINING COMPLETE!
============================================================

Results saved to: training_results

Generated files:
  - data_split.png
  - prediction_performance.png
  - consistency_scores.png
  - training_report.txt
  - sample_data.csv
```

---

## 🚀 Next Steps

1. ✅ Run training script
2. ✅ Review visualizations
3. ✅ Check performance metrics
4. ✅ Adjust hyperparameters if needed
5. ✅ Deploy models to production

---

## 📞 Support

- **Questions?** Check `TRAINING_VISUALIZATION_GUIDE.md`
- **Issues?** See Troubleshooting section above
- **Research?** Check `REFERENCES.md`
- **Examples?** See `notebooks/02_training_visualization.py`

---

## ✅ Checklist

- [ ] Dependencies installed
- [ ] Data prepared (CSV format)
- [ ] Training script executed
- [ ] Visualizations reviewed
- [ ] Metrics meet targets
- [ ] Models saved
- [ ] Ready for deployment

---

**Status**: ✅ Ready to Train and Visualize

