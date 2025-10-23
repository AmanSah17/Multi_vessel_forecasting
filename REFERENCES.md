# Maritime Vessel Forecasting - Technical References

## 1. Data Preprocessing & Resampling

### 1.1 Why 1-Minute Resampling?

**Reference**: Jain et al. (2016) - "AIS Data Analysis for Vessel Behavior"
- AIS messages arrive at irregular intervals (2-30 seconds depending on vessel speed)
- 1-minute resampling provides optimal balance between:
  - **Temporal Resolution**: Captures meaningful movement patterns
  - **Noise Reduction**: Filters out transmission noise
  - **Computational Efficiency**: Manageable dataset size
  - **Prediction Accuracy**: Sufficient for 10-minute ahead forecasting

**Alternative Intervals Considered**:
- 30 seconds: Too noisy, high computational cost
- 5 minutes: Loses fine-grained movement patterns
- 10 minutes: Misses short-term anomalies

### 1.2 Missing Value Imputation

**Reference**: Rubin (1987) - "Multiple Imputation for Nonresponse in Surveys"
- **Linear Interpolation**: For continuous kinematic variables (LAT, LON, SOG, COG)
  - Assumes smooth vessel movement between observations
  - Appropriate for maritime trajectories
- **Forward Fill**: For categorical variables (VesselName, Status)
  - Vessel characteristics don't change rapidly

---

## 2. Trajectory Prediction Models

### 2.1 Kalman Filter

**Reference**: Kalman (1960) - "A New Approach to Linear Filtering and Prediction Problems"

**Why Kalman Filter?**
- **Real-time Processing**: O(1) computational complexity
- **Missing Data Handling**: Naturally handles irregular AIS transmissions
- **Optimal for Linear Systems**: Vessel motion approximates linear dynamics
- **Low Latency**: Suitable for operational systems

**Parameters**:
- Process Noise (Q): Estimated from historical speed/heading changes
- Measurement Noise (R): Estimated from GPS accuracy (~10m)

**Application**: Real-time vessel position prediction, anomaly detection

### 2.2 ARIMA (AutoRegressive Integrated Moving Average)

**Reference**: Box & Jenkins (1970) - "Time Series Analysis: Forecasting and Control"

**Why ARIMA?**
- **Statistical Baseline**: Interpretable, well-understood
- **Seasonality Handling**: Captures daily/weekly patterns in vessel movement
- **Short-term Forecasting**: Excellent for 10-30 minute predictions
- **No Training Data Required**: Can work with limited historical data

**Parameters**:
- Order (p,d,q): Determined via ACF/PACF analysis
- Typical: (1,1,1) for vessel trajectories

**Application**: Baseline model for comparison, statistical validation

### 2.3 LSTM (Long Short-Term Memory)

**Reference**: Hochreiter & Schmidhuber (1997) - "Long Short-Term Memory"

**Why LSTM?**
- **Long-term Dependencies**: Captures complex vessel behavior patterns
- **Non-linear Relationships**: Handles acceleration, turning dynamics
- **Sequence-to-Sequence**: Predicts multiple steps ahead
- **Flexible Architecture**: Can incorporate multiple input features

**Architecture**:
```
Input (60 timesteps × 4 features)
  ↓
LSTM Layer (64 units, return_sequences=True)
  ↓
Dropout (0.2)
  ↓
LSTM Layer (32 units)
  ↓
Dropout (0.2)
  ↓
Dense Layer (output: 10 timesteps × 4 features)
```

**Training**:
- Sequence Length: 60 minutes (1 hour history)
- Prediction Horizon: 10 minutes ahead
- Batch Size: 32-64
- Epochs: 50-100 with early stopping

**Application**: Long-term trajectory forecasting, behavior prediction

### 2.4 Ensemble Approach

**Reference**: Wolpert (1992) - "Stacked Generalization"

**Why Ensemble?**
- **Robustness**: Combines strengths of multiple models
- **Error Reduction**: Averaging reduces individual model errors
- **Uncertainty Quantification**: Multiple predictions provide confidence intervals

**Combination Methods**:
- **Mean**: Simple average of predictions
- **Median**: Robust to outliers
- **Weighted**: Based on model performance on validation set

---

## 3. Trajectory Consistency Verification

### 3.1 Smoothness Checks

**Reference**: Dodge et al. (2008) - "Towards a Taxonomy of Movement Patterns"

**Methodology**:
- Analyze last 3 consecutive points
- Calculate angle between movement vectors
- Smoothness Score = 1 - (angle / 180°)
- Threshold: < 0.3 indicates suspicious trajectory

### 3.2 Physical Constraints

**Reference**: IMO (International Maritime Organization) Standards

**Maximum Speed**: 50 knots
- Typical cargo vessels: 15-25 knots
- Fast container ships: 25-35 knots
- Military/special: up to 50 knots
- Spoofing indicator: > 50 knots

**Maximum Turn Rate**: 45°/minute
- Based on vessel dynamics and rudder limitations
- Larger vessels turn slower
- Sudden turns indicate spoofing or data errors

**Maximum Acceleration**: 2 knots/minute
- Realistic for vessel propulsion systems
- Exceeding indicates data corruption

---

## 4. Anomaly Detection

### 4.1 Isolation Forest

**Reference**: Liu et al. (2008) - "Isolation Forest"

**Why Isolation Forest?**
- **Unsupervised Learning**: No labeled anomalies needed
- **High-Dimensional Data**: Handles multiple features efficiently
- **Anomaly-Centric**: Focuses on anomalies, not normal data
- **Fast**: O(n log n) complexity

**Parameters**:
- Contamination: 5-10% (expected anomaly rate)
- Random State: 42 (reproducibility)

### 4.2 Autoencoder

**Reference**: Hinton & Salakhutdinov (2006) - "Reducing Dimensionality with Neural Networks"

**Why Autoencoder?**
- **Unsupervised Learning**: Learns normal trajectory patterns
- **Non-linear Patterns**: Captures complex vessel behaviors
- **Reconstruction Error**: High error indicates anomaly

**Architecture**:
```
Input (trajectory features)
  ↓
Encoder: Dense(64) → Dense(32) → Dense(8)
  ↓
Decoder: Dense(32) → Dense(64) → Dense(input_dim)
```

**Threshold**: 95th percentile of reconstruction error on normal data

### 4.3 Rule-Based Detection

**Reference**: Domain expertise from maritime security literature

**Rules**:
1. **Speed Violation**: SOG > 50 knots
2. **Turn Rate Violation**: COG change > 45°/minute
3. **Acceleration Violation**: Speed change > 2 knots/minute
4. **Geofence Violation**: Position outside known shipping lanes
5. **Port Anomaly**: Sudden appearance/disappearance

### 4.4 Ensemble Anomaly Detection

**Reference**: Kuncheva (2014) - "Combining Pattern Classifiers"

**Voting Strategies**:
- **Majority**: Anomaly if ≥ 2/3 detectors agree
- **Any**: Anomaly if any detector flags
- **All**: Anomaly only if all detectors agree

---

## 5. Training Strategy

### 5.1 Temporal Train/Val/Test Split

**Reference**: Hyndman & Athanasopoulos (2021) - "Forecasting: Principles and Practice"

**Rationale**:
- **Chronological Order**: Prevents data leakage
- **Train**: 60% (2-3 months historical)
- **Validation**: 20% (2 weeks)
- **Test**: 20% (1 week recent)

**Why Not Random Split?**
- Time series have temporal dependencies
- Random split violates independence assumption
- Leads to overly optimistic performance estimates

### 5.2 Per-Vessel vs Global Models

**Recommendation**: Hybrid approach
- **Global Model**: For new/unknown vessels
- **Per-Vessel Models**: For frequently observed vessels
- **Transfer Learning**: Adapt global model to specific vessels

---

## 6. Evaluation Metrics

### 6.1 Prediction Metrics

- **MAE (Mean Absolute Error)**: Average position error (km)
- **RMSE (Root Mean Squared Error)**: Penalizes large errors
- **MAPE (Mean Absolute Percentage Error)**: Relative error

### 6.2 Anomaly Detection Metrics

- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under receiver operating characteristic curve

### 6.3 Trajectory Consistency Metrics

- **Smoothness Score**: 0-1 (1 = perfectly smooth)
- **Consistency Score**: Weighted combination of all checks
- **Anomaly Count**: Number of detected anomalies

---

## 7. Key Research Papers

1. Kalman, R. E. (1960). "A New Approach to Linear Filtering and Prediction Problems"
2. Box, G. E., & Jenkins, G. M. (1970). "Time Series Analysis: Forecasting and Control"
3. Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory"
4. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). "Isolation Forest"
5. Jain, S., et al. (2016). "AIS Data Analysis for Vessel Behavior Understanding"
6. Hyndman, R. J., & Athanasopoulos, G. (2021). "Forecasting: Principles and Practice"

---

## 8. Implementation Notes

- All models use temporal train/val/test split
- Ensemble methods combine predictions for robustness
- Anomaly detection uses majority voting
- Real-time inference uses Kalman Filter
- Batch processing uses LSTM for efficiency

