# Implementation Guide - Maritime Vessel Forecasting Pipeline

## Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Usage

```python
from src.training_pipeline import TrainingPipeline

# Initialize and run pipeline
pipeline = TrainingPipeline(output_dir='models')
metrics = pipeline.run_full_pipeline('path/to/ais_data.csv')
```

---

## Module Overview

### 1. Data Preprocessing (`src/data_preprocessing.py`)

**Purpose**: Clean and standardize raw AIS data

**Key Functions**:
- `VesselDataPreprocessor.preprocess()`: Main preprocessing pipeline
- `load_and_preprocess()`: Convenience function

**Features**:
- Parse datetime columns
- Handle missing vessel names → "Unidentified Vessel"
- Validate MMSI format (9 digits)
- Remove duplicates
- Resample to 1-minute intervals
- Interpolate missing values
- Remove outliers

**Example**:
```python
from src.data_preprocessing import load_and_preprocess

df = load_and_preprocess('ais_data.csv')
print(f"Processed {len(df)} records")
```

---

### 2. MMSI Analysis (`src/mmsi_analysis.py`)

**Purpose**: Analyze vessel identification patterns

**Key Functions**:
- `MMSIAnalyzer.analyze()`: Comprehensive analysis
- `MMSIAnalyzer.visualize_distribution()`: Create visualizations

**Features**:
- MMSI distribution statistics
- Country identification from MID
- Formatting issue detection
- Suspicious pattern detection

**Example**:
```python
from src.mmsi_analysis import MMSIAnalyzer

analyzer = MMSIAnalyzer()
results = analyzer.analyze(df)
fig = analyzer.visualize_distribution(df, top_n=20)
```

---

### 3. Trajectory Prediction (`src/trajectory_prediction.py`)

**Purpose**: Predict vessel positions

**Models**:

#### Kalman Filter
```python
from src.trajectory_prediction import KalmanFilterPredictor

kf = KalmanFilterPredictor(process_variance=0.01, measurement_variance=0.1)
kf.fit(X_train[:-1], X_train[1:])
prediction = kf.predict(current_position)
```

#### ARIMA
```python
from src.trajectory_prediction import ARIMAPredictor

arima = ARIMAPredictor(order=(1, 1, 1))
arima.fit(time_series_data)
predictions = arima.predict(X, steps=10)
```

#### LSTM
```python
from src.trajectory_prediction import LSTMPredictor

lstm = LSTMPredictor(sequence_length=60, prediction_horizon=10)
lstm.fit(X_train, y_train)
predictions = lstm.predict(X_test)
```

#### Ensemble
```python
from src.trajectory_prediction import EnsemblePredictor

ensemble = EnsemblePredictor({
    'kalman': kf,
    'arima': arima,
    'lstm': lstm,
})
ensemble.fit(X_train, y_train)
predictions = ensemble.predict_ensemble(X_test, method='mean')
```

---

### 4. Trajectory Verification (`src/trajectory_verification.py`)

**Purpose**: Validate vessel movements

**Key Functions**:
- `TrajectoryVerifier.verify_trajectory()`: Comprehensive verification
- `TrajectoryVerifier.get_consistency_score()`: Overall score (0-1)
- `TrajectoryVerifier.is_trajectory_valid()`: Boolean validation

**Checks**:
- Smoothness (last 3 points)
- Speed consistency (max 50 knots)
- Heading consistency (max 45°/min)
- Acceleration (max 2 knots/min)
- Turn rate (max 45°/min)

**Example**:
```python
from src.trajectory_verification import TrajectoryVerifier

verifier = TrajectoryVerifier()
vessel_trajectory = df[df['MMSI'] == '123456789']
results = verifier.verify_trajectory(vessel_trajectory)
score = verifier.get_consistency_score(vessel_trajectory)
print(f"Consistency Score: {score:.3f}")
```

---

### 5. Anomaly Detection (`src/anomaly_detection.py`)

**Purpose**: Detect suspicious vessel behavior

**Detectors**:

#### Isolation Forest
```python
from src.anomaly_detection import IsolationForestDetector

detector = IsolationForestDetector(contamination=0.05)
detector.fit(X_train)
anomalies = detector.predict(X_test)
```

#### Autoencoder
```python
from src.anomaly_detection import AutoencoderDetector

detector = AutoencoderDetector(encoding_dim=8, threshold=0.95)
detector.fit(X_train)
anomalies = detector.predict(X_test)
```

#### Rule-Based
```python
from src.anomaly_detection import create_default_rule_detector

detector = create_default_rule_detector()
detector.fit(X_train)
anomalies = detector.predict(X_test)
```

#### Ensemble
```python
from src.anomaly_detection import EnsembleAnomalyDetector

ensemble = EnsembleAnomalyDetector({
    'isolation_forest': iso_forest,
    'rule_based': rule_detector,
})
ensemble.fit(X_train)
anomalies = ensemble.predict_ensemble(X_test, method='majority')
scores = ensemble.get_anomaly_scores(X_test)
```

---

### 6. Training Pipeline (`src/training_pipeline.py`)

**Purpose**: Orchestrate complete training workflow

**Key Functions**:
- `TrainingPipeline.run_full_pipeline()`: End-to-end training
- `TrainingPipeline.load_data()`: Load and preprocess
- `TrainingPipeline.engineer_features()`: Feature engineering
- `TrainingPipeline.train_prediction_models()`: Train predictors
- `TrainingPipeline.train_anomaly_detectors()`: Train detectors
- `TrainingPipeline.evaluate()`: Evaluate on test set
- `TrainingPipeline.save_models()`: Persist models
- `TrainingPipeline.load_models()`: Load saved models

**Example**:
```python
from src.training_pipeline import TrainingPipeline

pipeline = TrainingPipeline(output_dir='models')

# Run complete pipeline
metrics = pipeline.run_full_pipeline('ais_data.csv')

# Or step-by-step
df = pipeline.load_data('ais_data.csv')
df = pipeline.engineer_features(df)
train_df, val_df, test_df = pipeline.create_train_val_test_split(df)
pipeline.train_prediction_models(train_df, val_df)
pipeline.train_anomaly_detectors(train_df)
metrics = pipeline.evaluate(test_df)
pipeline.save_models()
```

---

## Data Format Requirements

### Input CSV Format

Required columns:
- `MMSI`: 9-digit vessel identifier
- `BaseDateTime`: Timestamp (format: YYYY-MM-DDTHH:MM:SS)
- `LAT`: Latitude (-90 to +90)
- `LON`: Longitude (-180 to +180)
- `SOG`: Speed Over Ground (knots)
- `COG`: Course Over Ground (0-360°)

Optional columns:
- `VesselName`: Vessel name
- `IMO`: IMO number
- `CallSign`: Radio call sign
- `VesselType`: Vessel type code
- `Status`: Navigation status
- `Heading`: Vessel heading (0-360°)

---

## Training Strategy

### Data Split
- **Train**: 60% (2-3 months)
- **Validation**: 20% (2 weeks)
- **Test**: 20% (1 week)
- **Temporal Order**: Chronological (no random shuffle)

### Model Training

#### Kalman Filter
- Estimate Q and R from historical data
- Online filtering for real-time predictions

#### ARIMA
- ACF/PACF analysis for (p,d,q)
- Typical: (1,1,1)

#### LSTM
- Sequence length: 60 timesteps
- Prediction horizon: 10 minutes
- Batch size: 32-64
- Epochs: 50-100 with early stopping

### Validation Metrics
- **Prediction**: MAE, RMSE, MAPE
- **Consistency**: Smoothness score, anomaly count
- **Anomaly Detection**: Precision, Recall, F1-score

---

## Performance Optimization

### For Real-Time Applications
- Use Kalman Filter (O(1) complexity)
- Batch size: 1
- Update frequency: Every 1-5 minutes

### For Batch Processing
- Use LSTM (captures patterns)
- Batch size: 64-128
- Process daily/weekly

### For Accuracy
- Use Ensemble methods
- Combine multiple models
- Weighted voting based on performance

---

## Troubleshooting

### Issue: Low prediction accuracy
- **Solution**: Increase training data, tune hyperparameters, use ensemble

### Issue: High false positive rate in anomaly detection
- **Solution**: Increase contamination parameter, adjust thresholds

### Issue: Memory errors with LSTM
- **Solution**: Reduce batch size, use smaller sequence length

### Issue: Missing data after preprocessing
- **Solution**: Check data quality, adjust interpolation method

---

## Next Steps

1. **Prepare your data**: Ensure CSV format matches requirements
2. **Run preprocessing**: Test with sample data
3. **Analyze MMSI**: Visualize distribution
4. **Train models**: Run full pipeline
5. **Evaluate**: Check metrics on test set
6. **Deploy**: Save models and integrate into production

---

## Additional Resources

- `PIPELINE_ARCHITECTURE.md`: Detailed architecture
- `REFERENCES.md`: Research citations
- `notebooks/01_pipeline_example.py`: Working example
- `README.md`: Project overview

