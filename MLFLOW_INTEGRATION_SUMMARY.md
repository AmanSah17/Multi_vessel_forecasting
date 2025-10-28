# MLflow Integration Summary

## Overview
Successfully integrated MLflow monitoring and logging into the corrected XGBoost pipeline for vessel trajectory forecasting.

## Files Created/Modified

### 1. **41_corrected_xgboost_pipeline.py** (MODIFIED)
Enhanced with comprehensive MLflow logging:

#### Key Additions:
- **MLflow Setup**
  ```python
  mlflow.set_experiment("XGBoost_Vessel_Trajectory_Forecasting")
  mlflow.set_tracking_uri("file:./mlruns")
  ```

- **Objective Function Logging** (Hyperparameter Tuning)
  - Logs hyperparameters for each trial
  - Logs train/val MAE, RMSE, R² for each trial
  - Nested runs for each trial under hyperparameter_tuning run

- **Hyperparameter Tuning Run**
  - Logs n_trials, train_size, val_size
  - Logs best_params and best_val_mae
  - Logs best_trial_number

- **Final Model Training Run**
  - Logs training configuration
  - Logs final_train_mae and final_train_r2
  - Logs training data size

- **Test Evaluation Metrics**
  - Logs all test metrics to MLflow:
    - Geodesic_Error_Mean_m
    - Geodesic_Error_Median_m
    - Geodesic_Error_Std_m
    - LAT_MAE_degrees, LON_MAE_degrees
    - SOG_MAE_knots, COG_MAE_degrees
    - LAT_RMSE_degrees, LON_RMSE_degrees
    - LAT_R2, LON_R2

- **Complete Pipeline Run**
  - Wraps entire pipeline in main MLflow run
  - Logs PCA components and variance explained
  - Logs pipeline version and split configuration
  - Logs training artifacts

### 2. **44_mlflow_monitoring_dashboard.py** (NEW)
Comprehensive monitoring and visualization dashboard:

#### Features:
- **MLflowMonitor Class**
  - Fetches all runs from experiment
  - Extracts metrics and parameters
  - Generates comprehensive reports

- **Visualization Functions**
  - `plot_training_metrics()`: Train vs Val MAE, RMSE, R² over trials
  - `plot_test_metrics()`: Position errors, navigation errors, R² scores, geodesic errors
  - `plot_hyperparameter_importance()`: Correlation analysis, learning rate vs MAE, depth vs MAE, n_estimators vs MAE

- **Report Generation**
  - Saves all plots as PNG files
  - Generates markdown summary report
  - Outputs to `mlflow_reports/` directory

#### Output Files:
- `training_metrics.png` - Training progress visualization
- `test_metrics.png` - Test evaluation results
- `hyperparameter_importance.png` - Hyperparameter analysis
- `report.md` - Summary statistics and runs overview

## Training Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│ COMPLETE_PIPELINE (Main MLflow Run)                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ [1/7] Load & Split Data (BEFORE preprocessing)              │
│       - Train: 860,830 samples                              │
│       - Val: 245,951 samples                                │
│       - Test: 122,977 samples                               │
│                                                              │
│ [2/7] Extract Features & Fit Preprocessing                  │
│       - Extract 483 features per sample                     │
│       - Fit StandardScaler on TRAIN only                    │
│       - Fit PCA on TRAIN only (95% variance)                │
│       - Transform VAL and TEST with fitted objects          │
│                                                              │
│ [3/7] Hyperparameter Tuning (Nested Run)                    │
│       ├─ Trial 1: Log params, train_mae, val_mae, etc.      │
│       ├─ Trial 2: Log params, train_mae, val_mae, etc.      │
│       └─ Trial N: Log params, train_mae, val_mae, etc.      │
│       └─ Log best_params and best_val_mae                   │
│                                                              │
│ [4/7] Train Final Model (Nested Run)                        │
│       - Train on TRAIN + VAL combined                       │
│       - Log final_train_mae and final_train_r2              │
│                                                              │
│ [5/7] Evaluate on TEST (Never touched before)               │
│       - Log all test metrics to MLflow                      │
│       - Geodesic errors in meters                           │
│       - Per-variable errors (LAT, LON, SOG, COG)            │
│                                                              │
│ [6/7] Save Pipeline                                         │
│       - Save model with joblib                              │
│       - Save scaler with joblib                             │
│       - Save PCA with joblib                                │
│       - Log artifacts to MLflow                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Metrics Logged

### Hyperparameter Tuning Metrics (Per Trial)
- `train_mae` - Training Mean Absolute Error
- `val_mae` - Validation Mean Absolute Error
- `val_rmse` - Validation Root Mean Squared Error
- `val_r2` - Validation R² Score

### Final Model Metrics
- `final_train_mae` - Final training MAE
- `final_train_r2` - Final training R²

### Test Evaluation Metrics
- **Position Errors (degrees)**
  - `test_LAT_MAE_degrees`
  - `test_LON_MAE_degrees`
  - `test_LAT_RMSE_degrees`
  - `test_LON_RMSE_degrees`

- **Navigation Errors**
  - `test_SOG_MAE_knots` - Speed Over Ground error
  - `test_COG_MAE_degrees` - Course Over Ground error

- **Geodesic Errors (meters)**
  - `test_Geodesic_Error_Mean_m` - Mean error in meters
  - `test_Geodesic_Error_Median_m` - Median error in meters
  - `test_Geodesic_Error_Std_m` - Std dev of errors

- **Model Performance**
  - `test_LAT_R2` - Latitude R² score
  - `test_LON_R2` - Longitude R² score

## Data Leakage Fixes

✅ **Proper Train/Val/Test Split**
- Split BEFORE feature extraction
- No information leakage from val/test to preprocessing

✅ **Scaler/PCA Fitting**
- Fit StandardScaler on TRAIN data only
- Fit PCA on TRAIN data only
- Transform VAL and TEST with fitted objects

✅ **Temporal Awareness**
- 70% train, 20% val, 10% test split
- Maintains temporal ordering

✅ **Geodesic Error Metrics**
- Calculates Haversine distance in meters
- More intuitive than degrees for spatial errors

## Training Status

**Current Status**: RUNNING
- Feature extraction in progress (1/28 dimensions completed)
- Estimated time: ~2-3 hours for complete pipeline
- Hyperparameter tuning: 100 trials
- Total dataset: 1,229,758 sequences

## How to Monitor Training

### Option 1: View MLflow UI
```bash
mlflow ui --backend-store-uri file:./mlruns
```
Then open: http://localhost:5000

### Option 2: Generate Reports
```bash
python 44_mlflow_monitoring_dashboard.py
```
Outputs to `mlflow_reports/` directory

### Option 3: Check Logs
```bash
tail -f logs/xgboost_corrected_pipeline.log
tail -f training_output.log
```

## Next Steps

1. **Wait for training to complete** (~2-3 hours)
2. **Generate monitoring dashboard** with `44_mlflow_monitoring_dashboard.py`
3. **Review metrics** in MLflow UI or generated reports
4. **Use saved model** with `42_predict_vessel_trajectories.py`
5. **Deploy predictions** to Maritime NLU backend

## Key Improvements Over Previous Version

| Aspect | Previous | Current |
|--------|----------|---------|
| Data Leakage | ❌ Fitted on test set | ✅ Fitted on train only |
| Train/Val/Test | ❌ After preprocessing | ✅ Before preprocessing |
| Monitoring | ❌ Manual logging | ✅ MLflow automated |
| Metrics | ❌ Degrees only | ✅ Degrees + meters |
| Model Saving | ❌ Pickle | ✅ Joblib + MLflow |
| Reproducibility | ❌ Limited | ✅ Full MLflow tracking |

## Files Structure

```
.
├── 41_corrected_xgboost_pipeline.py      # Main training pipeline
├── 42_predict_vessel_trajectories.py     # Prediction utilities
├── 43_advanced_prediction_patterns.py    # Advanced patterns
├── 44_mlflow_monitoring_dashboard.py     # Monitoring dashboard
├── logs/
│   └── xgboost_corrected_pipeline.log    # Training logs
├── mlruns/                               # MLflow tracking directory
├── mlflow_reports/                       # Generated reports
│   ├── training_metrics.png
│   ├── test_metrics.png
│   ├── hyperparameter_importance.png
│   └── report.md
└── results/
    └── xgboost_corrected_50_vessels/     # Saved model artifacts
        ├── xgboost_model.joblib
        ├── scaler.joblib
        ├── pca.joblib
        └── metadata.json
```

