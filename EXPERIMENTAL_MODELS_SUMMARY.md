# Experimental Models Pipeline - Full Dataset Training

## Overview
Training experimental models (Small LSTM, Tiny LSTM) on the **FULL maritime AIS dataset** with GPU acceleration for robust vessel trajectory forecasting.

## Key Configuration

### Dataset
- **Total Records**: 41,582,721 AIS records (NO SAMPLING)
- **Date Range**: January 3-8, 2020
- **Vessels**: 19,267 unique vessels
- **Sampling Strategy**: FULL DATASET (sample_per_day=None)

### Sequence Configuration
- **Sequence Length**: 24 timesteps (120 minutes at 5-minute intervals)
- **Forecasting Window**: 5 minutes (next timestep prediction)
- **Total Sequences Generated**: ~300K+ sequences
- **Train/Val/Test Split**: 70/20/10

### Features
- **Total Features**: 34 comprehensive features
  - Temporal: hour, day_of_week, minute
  - Cyclical: hour_sin, hour_cos, dow_sin, dow_cos
  - Kinematic: lat_diff, lon_diff, sog_diff, cog_diff
  - Lag Features: 1, 2, 3 step lags for LAT, LON, SOG, COG
  - Polynomial: lat_sq, lon_sq, sog_sq
  - Interaction: speed_heading_int, lat_lon_int

### Target Variables
- LAT (Latitude)
- LON (Longitude)
- SOG (Speed Over Ground)
- COG (Course Over Ground)

## Model Architectures

### Small LSTM (hidden_size=128, num_layers=6)
- **Input Size**: 28 features
- **Hidden Size**: 128 units
- **Layers**: 6 LSTM layers
- **Dropout**: 0.2
- **Output**: 4 targets (LAT, LON, SOG, COG)
- **Purpose**: Balanced model for feature extraction with long-range dependencies

### Tiny LSTM (hidden_size=32, num_layers=4)
- **Input Size**: 28 features
- **Hidden Size**: 32 units
- **Layers**: 4 LSTM layers
- **Dropout**: 0.15
- **Output**: 4 targets (LAT, LON, SOG, COG)
- **Purpose**: Lightweight model for faster training and inference

## GPU Optimization

### Hardware
- **GPU**: NVIDIA CUDA-enabled GPU
- **Batch Size**: 256 (optimized for GPU memory)
- **Mixed Precision**: Enabled (AMP - Automatic Mixed Precision)
- **cuDNN Benchmark**: Enabled for faster convolutions
- **TF32 Operations**: Enabled for faster matrix multiplications

### Memory Optimization
- **Data Type**: float32 for features and targets
- **Pin Memory**: True (faster CPU-to-GPU transfer)
- **Num Workers**: 0 (Windows-safe, no multiprocessing)
- **Feature Encoding**: int8 for temporal features, float32 for computed features

## Training Configuration

### Hyperparameters
- **Epochs**: 100 (with early stopping)
- **Learning Rate**: 0.001 (Adam optimizer)
- **Batch Size**: 256
- **Early Stopping Patience**: 20 epochs
- **LR Scheduler Patience**: 10 epochs (ReduceLROnPlateau)
- **Gradient Clipping**: 1.0 (norm)

### Loss Function
- **MSE Loss** (Mean Squared Error)

### Metrics
- **MAE** (Mean Absolute Error) - overall and per-target
- **RMSE** (Root Mean Squared Error)
- **R²** (Coefficient of Determination)

## MLflow Logging

### Experiment
- **Name**: "Experimental_Models_5min_Forecasting"
- **Run Name**: "Small_LSTM_Experiments"

### Logged Parameters
- sequence_length: 24
- forecasting_window_minutes: 120
- num_features: 28
- train_size: 211,157
- val_size: 60,330
- test_size: 30,166

### Logged Metrics (per epoch)
- small_lstm_train_loss
- small_lstm_val_loss
- small_lstm_best_val_loss
- small_lstm_lr
- tiny_lstm_train_loss
- tiny_lstm_val_loss
- tiny_lstm_best_val_loss
- tiny_lstm_lr

## Output Artifacts

### Model Checkpoints
- `results/models/best_small_lstm.pt`
- `results/models/best_tiny_lstm.pt`

### Cached Sequences
- `results/cache/seq_cache_len24_full_dataset.npz`
  - X: (N, 24, 28) - sequences
  - y: (N, 4) - targets
  - features: list of feature names
  - mmsi_list: vessel identifiers

### Evaluation Results
- **Test Set Metrics**: MAE, RMSE, R² for each model
- **Per-Target Metrics**: MAE for LAT, LON, SOG, COG

### Visualizations (Post-Training)
- 300-vessel prediction plots (LAT, LON, SOG, COG)
- Actual vs Predicted comparisons
- Per-vessel error analysis

## Data Pipeline

### Step 1: Load Full Dataset
- Load all 6 days of AIS data (no sampling)
- Total: 41.5M records, 19.3K vessels

### Step 2: Feature Engineering
- Add 34 comprehensive features
- Memory-efficient: use float32 and int8 where appropriate
- Process per-vessel for kinematic features

### Step 3: Sequence Creation
- Create 24-timestep sequences
- Generate ~300K+ training sequences
- Cache to NPZ for fast reloading

### Step 4: Data Normalization
- MinMaxScaler fit on training data
- Apply to validation and test sets
- Memory-efficient: compute min/max per axis

### Step 5: DataLoader Creation
- Batch size: 256 (GPU-optimized)
- Pin memory: True
- Shuffle: True for training

### Step 6: Model Training
- Train with AMP (mixed precision)
- Early stopping with patience
- Learning rate scheduling
- Save best checkpoints

### Step 7: Evaluation
- Evaluate on test set
- Compute metrics per model
- Log to MLflow

### Step 8: Visualization
- Generate 300-vessel prediction plots
- Save to results/images/vessels_300
- Log to MLflow

## Expected Performance

### Training Time
- **Small LSTM**: ~2-3 hours (100 epochs)
- **Tiny LSTM**: ~1-2 hours (100 epochs)
- **Total Pipeline**: ~4-5 hours

### Expected Metrics (Baseline)
- **MAE**: 0.01-0.05 (normalized scale)
- **RMSE**: 0.02-0.08
- **R²**: 0.85-0.95

## Future Enhancements

1. **ARIMA Models**: Per-vessel ARIMA for comparison
2. **Kalman Filter**: State estimation for trajectory smoothing
3. **Ensemble Methods**: Combine LSTM, ARIMA, Kalman predictions
4. **Hyperparameter Tuning**: Optimize hidden_size, num_layers, dropout
5. **Attention Mechanisms**: Add attention layers for better feature focus
6. **Multi-Step Forecasting**: Predict multiple timesteps ahead

## References

- **Dataset**: Maritime AIS data (January 2020)
- **Framework**: PyTorch with CUDA
- **Logging**: MLflow
- **Optimization**: Mixed Precision Training (AMP)

