# LSTM Advanced Pipeline - Vessel Trajectory Forecasting

## 🚀 Overview

This document describes the advanced LSTM neural network pipeline for vessel trajectory prediction with specialized loss functions, MLflow tracking, and GPU optimization.

## 📋 Pipeline Components

### 1. **Data Preprocessing**
- **Input**: Preprocessed sequences from XGBoost pipeline (12 timesteps, 28 features)
- **Geodesic Features**: Distance calculations between consecutive points
- **Haversine Features**: Distance from first point in sequence
- **PCA Reduction**: 336 features → 48 components
- **Normalization**: StandardScaler for feature scaling

### 2. **LSTM Model Architecture**
```
Input Layer: 48 features (PCA-reduced)
    ↓
LSTM Layers: 4 layers, 128 hidden units each
    ↓
Dense Layer 1: 256 units + SiLU activation
    ↓
Dropout: 0.2
    ↓
Dense Layer 2: 128 units + GELU activation
    ↓
Dropout: 0.2
    ↓
Output Layer: 4 units (Lat, Lon, SOG, COG)
```

### 3. **Specialized Loss Functions**
- **Primary Loss**: 70% MSE + 30% MAE (combined)
- **Geodesic Loss**: Haversine distance for spatial accuracy
- **Gradient Clipping**: max_norm=1.0 for stability

### 4. **Training Configuration**
- **Optimizer**: AdamW (lr=0.001, weight_decay=1e-5)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Batch Size**: 256
- **Epochs**: 50 (with early stopping, patience=10)
- **Device**: GPU (CUDA)

### 5. **MLflow Tracking**
Tracks the following metrics per epoch:
- `train_loss`: Training loss
- `val_loss`: Validation loss
- `lat_mae`: Latitude MAE
- `lon_mae`: Longitude MAE
- `sog_mae`: Speed Over Ground MAE
- `cog_mae`: Course Over Ground MAE
- `lat_r2`: Latitude R² score
- `lon_r2`: Longitude R² score

Plus test metrics:
- `test_lat_mae`, `test_lon_mae`, `test_sog_mae`, `test_cog_mae`
- `test_lat_r2`, `test_lon_r2`

## 📊 Expected Performance

Based on XGBoost baseline:
- **Latitude**: MAE ≈ 0.22°, R² ≈ 0.9966
- **Longitude**: MAE ≈ 0.52°, R² ≈ 0.9980
- **SOG**: MAE ≈ 0.36 knots, R² ≈ 0.9739
- **COG**: MAE ≈ 41.12°, R² ≈ 0.6584

## 🔧 Key Features

### GPU Optimization
- CUDA device detection and logging
- GPU memory monitoring
- Batch processing for efficiency
- Gradient clipping for stability

### Progress Tracking
- tqdm progress bars for training/validation
- Epoch-level logging
- Best model checkpointing
- Early stopping mechanism

### Advanced Activations
- **SiLU (Swish)**: First dense layer
- **GELU**: Second dense layer
- Better gradient flow than ReLU

## 📁 Output Files

### Model Files
```
results/lstm_advanced_model/
├── lstm_model.pth          # Model weights
├── lstm_model.pkl          # Serialized model
└── best_model.pth          # Best checkpoint
```

### MLflow Tracking
```
mlruns/
└── LSTM_Advanced_Vessel_Forecasting/
    └── [run_id]/
        ├── metrics/        # Tracked metrics
        ├── params/         # Hyperparameters
        └── artifacts/      # Model artifacts
```

## 🎯 Prediction Pipeline

### 48_lstm_predictions_50_vessels.py
- Loads trained LSTM model
- Makes predictions on test set
- Groups by vessel ID
- Selects 50 random vessels
- Generates per-vessel metrics
- Creates visualization plots

### Output Files
```
results/lstm_predictions_50_vessels/
├── lstm_50_vessels_metrics.csv
├── lstm_vessel_[ID]_performance.png (top 10)
└── lstm_all_vessels_r2_comparison.png
```

## 📈 Metrics Explained

### MAE (Mean Absolute Error)
- Average absolute difference between predicted and actual
- Lower is better
- Units: degrees for Lat/Lon, knots for SOG, degrees for COG

### RMSE (Root Mean Squared Error)
- Square root of average squared errors
- Penalizes larger errors more
- Same units as MAE

### R² Score
- Coefficient of determination
- Range: -∞ to 1.0
- 1.0 = perfect prediction
- 0.0 = as good as mean baseline
- < 0 = worse than baseline

## 🚀 Usage

### Training
```bash
python 47_lstm_advanced_pipeline.py
```

### Predictions
```bash
python 48_lstm_predictions_50_vessels.py
```

### View MLflow Results
```bash
mlflow ui
```

## ⚙️ Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Hidden Units | 128 | Per LSTM layer |
| Num Layers | 4 | Depth of LSTM |
| Dropout | 0.2 | Regularization |
| Learning Rate | 0.001 | AdamW optimizer |
| Batch Size | 256 | Training batch |
| Max Epochs | 50 | With early stopping |
| Patience | 10 | Early stopping patience |
| Gradient Clip | 1.0 | Max norm |

## 📊 Data Split

- **Training**: 70% (860,830 samples)
- **Validation**: 20% (245,951 samples)
- **Test**: 10% (122,977 samples)

## 🔍 Troubleshooting

### Out of Memory
- Reduce batch size (256 → 128)
- Reduce hidden units (128 → 64)
- Use gradient accumulation

### Poor Performance
- Check data preprocessing
- Verify feature scaling
- Increase training epochs
- Adjust learning rate

### Slow Training
- Verify GPU usage
- Check batch size
- Monitor memory usage
- Use mixed precision training

## 📝 Notes

- Model uses LSTM for sequential dependencies
- Specialized loss functions for spatial accuracy
- MLflow for comprehensive experiment tracking
- GPU optimization for fast training
- Early stopping prevents overfitting
- Per-vessel analysis for detailed insights

## ✅ Status

- ✓ Pipeline implemented
- ✓ GPU optimization enabled
- ✓ MLflow tracking configured
- ✓ Prediction script ready
- ⏳ Training in progress...

