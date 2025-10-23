# Maritime Vessel Forecasting - End-to-End ML Pipeline

## Pipeline Overview

```
Raw AIS Data
    ↓
[1. Data Preprocessing & Cleaning]
    ├─ Handle missing vessel names → "Unidentified Vessel"
    ├─ Resample to 1-minute intervals
    ├─ Validate MMSI formatting (9 digits)
    ├─ Remove duplicates & outliers
    └─ Handle missing values
    ↓
[2. Feature Engineering]
    ├─ Temporal features (hour, day, week)
    ├─ Spatial features (distance, bearing)
    ├─ Kinematic features (acceleration, turn rate)
    ├─ Vessel-specific features (type, size)
    └─ Contextual features (port proximity, shipping lanes)
    ↓
[3. MMSI Distribution Analysis]
    ├─ Visualize MMSI distribution
    ├─ Identify formatting issues
    └─ Flag suspicious patterns
    ↓
[4. Trajectory Prediction]
    ├─ Kalman Filter (real-time, low-latency)
    ├─ ARIMA (statistical baseline)
    └─ LSTM (deep learning, long-term patterns)
    ↓
[5. Trajectory Consistency Verification]
    ├─ Smoothness checks (last 3 points)
    ├─ Speed/heading consistency
    ├─ Realistic turn rates
    └─ Geofence validation
    ↓
[6. Anomaly Detection]
    ├─ Spoofing detection (impossible speeds/turns)
    ├─ Sudden deviations (deviation from predicted path)
    ├─ Inconsistent patterns (behavioral anomalies)
    └─ Statistical outliers
    ↓
[7. Model Evaluation & Deployment]
    ├─ Cross-validation
    ├─ Performance metrics
    └─ Real-time inference
```

## Training Approach

### Data Split Strategy
- **Train**: 60% (historical data, 2-3 months)
- **Validation**: 20% (temporal split, next 2 weeks)
- **Test**: 20% (held-out recent data, 1 week)
- **Temporal Split**: Avoid data leakage by using chronological order

### Model-Specific Training

#### 1. Kalman Filter
- **Training**: Estimate process & measurement noise from historical data
- **Approach**: Offline parameter tuning, online filtering
- **Advantages**: Real-time, low computational cost, handles missing data

#### 2. ARIMA
- **Training**: ACF/PACF analysis, parameter selection (p,d,q)
- **Approach**: Per-vessel models or global model
- **Advantages**: Interpretable, good for short-term forecasts

#### 3. LSTM
- **Training**: Supervised learning with sequences
- **Sequence Length**: 60 timesteps (1 hour history) → predict next 10 minutes
- **Batch Size**: 32-64
- **Epochs**: 50-100 with early stopping
- **Advantages**: Captures long-term dependencies, handles non-linearity

### Validation Metrics
- **Prediction**: MAE, RMSE, MAPE (position error in km)
- **Consistency**: Smoothness score, turn rate validation
- **Anomaly Detection**: Precision, Recall, F1-score, ROC-AUC

## Technical Decisions (Research-Backed)

### 1. Why 1-Minute Resampling?
- AIS messages arrive at irregular intervals (typically 2-30 seconds)
- 1-minute provides good temporal resolution without excessive noise
- Balances computational efficiency with prediction accuracy

### 2. Why Multiple Prediction Models?
- **Kalman Filter**: Best for real-time, low-latency applications
- **ARIMA**: Good baseline, interpretable, handles seasonality
- **LSTM**: Captures complex patterns, better for long-term predictions
- Ensemble approach: Combine predictions for robustness

### 3. Trajectory Consistency Checks
- Verify last 3 points form smooth curve (no sudden jumps)
- Check if speed/heading changes are physically realistic
- Detect spoofing: impossible speeds (>50 knots for most vessels)

### 4. Anomaly Detection Strategy
- **Isolation Forest**: Detect statistical outliers
- **Autoencoder**: Learn normal trajectory patterns
- **Rule-based**: Domain-specific rules (speed limits, geofences)
- **Ensemble**: Combine multiple methods for robustness

## References & Research

[To be populated with citations]

