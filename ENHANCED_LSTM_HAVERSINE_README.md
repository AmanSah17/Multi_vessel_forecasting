# Enhanced LSTM Pipeline with Haversine Distance & Hyperparameter Tuning

## 🎯 Overview

This enhanced pipeline provides a complete solution for vessel trajectory prediction with:

- **Haversine Distance Metrics**: Accurate geodesic distance measurements in meters
- **Hyperparameter Tuning**: Automated optimization using Optuna
- **Fine-tuning Capabilities**: Easy model refinement with best parameters
- **Comprehensive Evaluation**: Multiple metrics including position errors in meters
- **MLflow Integration**: Full experiment tracking and model versioning

## 📁 Files Structure

```
Multi_vessel_forecasting/
├── 49_enhanced_lstm_haversine_tuning.py          # Part 1: Core utilities, data loading, EDA
├── 49_enhanced_lstm_haversine_tuning_part2.py    # Part 2: Clustering, PCA, sequences, tuning
├── 49_enhanced_lstm_haversine_tuning_part3.py    # Part 3: Training, evaluation, visualization
├── 49_main_enhanced_lstm_pipeline.py             # Main execution script
└── ENHANCED_LSTM_HAVERSINE_README.md             # This file
```

## 🚀 Quick Start

### 1. Run the Complete Pipeline

```bash
python 49_main_enhanced_lstm_pipeline.py
```

This will:
1. Load AIS data (Jan 3-8, 2020)
2. Perform EDA and feature engineering
3. Apply clustering and PCA
4. Create sequences (70/20/10 split)
5. Run hyperparameter tuning (20 trials)
6. Train final model with best parameters
7. Evaluate with haversine metrics
8. Save all artifacts

### 2. View Results in MLflow

```bash
mlflow ui
```

Then open: http://localhost:5000

## 📊 Key Features

### 1. Haversine Distance Metrics

The pipeline calculates accurate geodesic distances between predicted and actual positions:

```python
from enhanced_lstm_haversine_tuning import calculate_haversine_errors

# Returns metrics in meters:
# - haversine_mean_m
# - haversine_median_m
# - haversine_std_m
# - haversine_p95_m
# - haversine_p99_m
```

**Why Haversine?**
- Accounts for Earth's curvature
- More accurate than simple lat/lon differences
- Industry-standard for maritime applications
- Results in meters (easy to interpret)

### 2. Hyperparameter Tuning with Optuna

Automatically optimizes:
- `hidden_size`: LSTM hidden layer size (64-256)
- `num_layers`: Number of LSTM layers (1-3)
- `dropout`: Dropout rate (0.1-0.5)
- `lr`: Learning rate (1e-4 to 1e-2)
- `batch_size`: Batch size (16, 32, 64)
- `bidirectional`: Use bidirectional LSTM (True/False)

### 3. Enhanced LSTM Architecture

```python
EnhancedLSTMModel(
    input_size=12,           # Number of features
    hidden_size=128,         # Configurable
    num_layers=2,            # Configurable
    dropout=0.3,             # Configurable
    bidirectional=False      # Configurable
)
```

Features:
- Bidirectional LSTM support
- Multi-layer architecture
- Dropout regularization
- Fully connected output layers

## 📈 Evaluation Metrics

### Standard Metrics
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (R-squared score)

### Haversine Metrics (in meters)
- **Mean Distance Error**: Average position error
- **Median Distance Error**: Robust central tendency
- **P95/P99**: 95th and 99th percentile errors
- **Std**: Standard deviation of errors

### Per-Output Metrics
- Latitude error (degrees)
- Longitude error (degrees)
- Speed Over Ground error (knots)
- Course Over Ground error (degrees)

## 🎨 Visualizations Generated

1. **01_eda_distributions.png**: Feature distributions
2. **02_eda_correlation.png**: Correlation heatmap
3. **03_pca_variance.png**: PCA explained variance
4. **04_clusters_map.png**: K-means clusters on map
5. **05_training_curves_haversine.png**: Training history with haversine errors
6. **06_predictions_vs_actual.png**: Scatter plots of predictions

## ⚙️ Configuration

Edit `CONFIG` in `49_main_enhanced_lstm_pipeline.py`:

```python
CONFIG = {
    'start_date': 3,              # Start day (Jan 3)
    'end_date': 8,                # End day (Jan 8)
    'sample_per_day': 50000,      # Samples per day (None = all)
    'n_clusters': 5,              # K-means clusters
    'n_components': 10,           # PCA components
    'seq_length': 30,             # Sequence length
    'n_trials': 20,               # Optuna trials
    'final_epochs': 200,          # Final training epochs
    'output_dir': 'results/enhanced_lstm_haversine'
}
```

## 🔧 Fine-Tuning an Existing Model

To fine-tune a pre-trained model:

```python
import torch
from enhanced_lstm_haversine_tuning import EnhancedLSTMModel

# Load existing model
model = EnhancedLSTMModel(input_size=12, hidden_size=128, num_layers=2)
model.load_state_dict(torch.load('results/enhanced_lstm_haversine/best_lstm_model_haversine.pt'))

# Fine-tune with lower learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Continue training...
```

## 📊 Expected Results

Based on similar maritime datasets:

| Metric | Expected Range |
|--------|---------------|
| Haversine Mean Error | 50-500 meters |
| Haversine Median Error | 30-300 meters |
| Haversine P95 Error | 200-1500 meters |
| MAE (Lat/Lon) | 0.001-0.01 degrees |
| R² Score | 0.85-0.99 |

## 🔍 Interpreting Haversine Errors

**Good Performance:**
- Mean < 200m: Excellent for short-term prediction
- Mean < 500m: Good for medium-term prediction
- P95 < 1000m: Reliable for most vessels

**Factors Affecting Accuracy:**
- Prediction horizon (longer = less accurate)
- Vessel type (cargo vs. passenger)
- Weather conditions
- Port proximity (more maneuvering)

## 🐛 Troubleshooting

### Out of Memory Error
Reduce `sample_per_day` or `batch_size`:
```python
CONFIG['sample_per_day'] = 20000
best_params['batch_size'] = 16
```

### Slow Training
- Reduce `n_trials` for faster tuning
- Reduce `final_epochs`
- Use GPU if available

### Poor Performance
- Increase `seq_length` for more context
- Increase `n_trials` for better hyperparameters
- Check data quality (missing values, outliers)

## 📚 Advanced Usage

### Custom Hyperparameter Search Space

Edit `objective()` in `49_enhanced_lstm_haversine_tuning_part2.py`:

```python
config = {
    'hidden_size': trial.suggest_int('hidden_size', 128, 512, step=128),
    'num_layers': trial.suggest_int('num_layers', 2, 4),
    # ... add more parameters
}
```

### Custom Loss Function

Replace MSE with custom loss in training:

```python
def haversine_loss(y_pred, y_true):
    """Custom loss based on haversine distance."""
    lat_pred, lon_pred = y_pred[:, 0], y_pred[:, 1]
    lat_true, lon_true = y_true[:, 0], y_true[:, 1]
    
    # Calculate haversine distance
    dist = haversine_distance(lat_true, lon_true, lat_pred, lon_pred)
    return torch.mean(dist)
```

### Ensemble Models

Train multiple models and average predictions:

```python
models = []
for i in range(5):
    model, _, _, _, _, _, _ = train_final_model(...)
    models.append(model)

# Average predictions
predictions = [model(X_test) for model in models]
ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
```

## 📝 Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{enhanced_lstm_haversine,
  title={Enhanced LSTM Pipeline with Haversine Distance Metrics},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/maritime_vessel_forecasting}
}
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional distance metrics (Vincenty, Great Circle)
- Multi-step prediction support
- Attention mechanisms
- Transformer architectures
- Real-time inference optimization

## 📄 License

MIT License - See LICENSE file for details

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section
2. Review MLflow logs
3. Check `logs/enhanced_lstm_haversine_main.log`
4. Open an issue on GitHub

---

**Last Updated**: 2025-01-27
**Version**: 1.0.0
**Status**: Production Ready ✅

