# Testing Guide - Maritime Vessel Forecasting Pipeline

## Overview

This guide provides instructions for testing the ML pipeline components.

---

## Unit Testing

### 1. Data Preprocessing Tests

```python
import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import VesselDataPreprocessor

def test_handle_missing_vessel_names():
    """Test missing vessel name handling."""
    preprocessor = VesselDataPreprocessor()
    
    df = pd.DataFrame({
        'VesselName': ['Ship1', None, '', 'Ship2'],
        'MMSI': ['123456789', '234567890', '345678901', '456789012'],
        'BaseDateTime': pd.date_range('2024-01-01', periods=4, freq='1min'),
        'LAT': [40.0, 41.0, 42.0, 43.0],
        'LON': [-74.0, -75.0, -76.0, -77.0],
    })
    
    result = preprocessor._handle_missing_vessel_names(df)
    
    assert (result['VesselName'] == 'Unidentified Vessel').sum() == 2
    assert result['VesselName'].iloc[0] == 'Ship1'

def test_validate_mmsi():
    """Test MMSI validation."""
    preprocessor = VesselDataPreprocessor()
    
    df = pd.DataFrame({
        'MMSI': ['123456789', '12345678', 'INVALID', '234567890'],
    })
    
    result = preprocessor._validate_mmsi(df)
    
    assert len(result) == 3  # Invalid MMSI removed
    assert '123456789' in result['MMSI'].values

def test_resample_timeseries():
    """Test time series resampling."""
    preprocessor = VesselDataPreprocessor()
    
    df = pd.DataFrame({
        'MMSI': ['123456789'] * 10,
        'BaseDateTime': pd.date_range('2024-01-01', periods=10, freq='30s'),
        'LAT': np.linspace(40.0, 40.1, 10),
        'LON': np.linspace(-74.0, -74.1, 10),
        'SOG': np.ones(10) * 15,
    })
    
    result = preprocessor._resample_timeseries(df)
    
    # Should have fewer records after resampling
    assert len(result) < len(df)
```

### 2. MMSI Analysis Tests

```python
from src.mmsi_analysis import MMSIAnalyzer

def test_mmsi_distribution():
    """Test MMSI distribution analysis."""
    analyzer = MMSIAnalyzer()
    
    df = pd.DataFrame({
        'MMSI': ['123456789'] * 50 + ['234567890'] * 30,
    })
    
    results = analyzer.analyze(df)
    
    assert results['total_mmsi'] == 2
    assert results['mmsi_distribution']['mean_records_per_mmsi'] == 40

def test_country_mapping():
    """Test MID to country mapping."""
    analyzer = MMSIAnalyzer()
    
    df = pd.DataFrame({
        'MMSI': ['201123456', '210123456', '303123456'],
    })
    
    country_dist = analyzer._get_country_distribution(df)
    
    assert 'Albania' in country_dist or 'Unknown' in country_dist
```

### 3. Trajectory Verification Tests

```python
from src.trajectory_verification import TrajectoryVerifier

def test_smoothness_check():
    """Test trajectory smoothness calculation."""
    verifier = TrajectoryVerifier()
    
    df = pd.DataFrame({
        'LAT': [40.0, 40.01, 40.02],
        'LON': [-74.0, -74.01, -74.02],
        'BaseDateTime': pd.date_range('2024-01-01', periods=3, freq='1min'),
        'SOG': [15, 15, 15],
        'COG': [90, 90, 90],
    })
    
    smoothness = verifier._check_smoothness(df)
    
    assert 0 <= smoothness <= 1
    assert smoothness > 0.8  # Straight line should be smooth

def test_speed_consistency():
    """Test speed validation."""
    verifier = TrajectoryVerifier()
    
    df = pd.DataFrame({
        'SOG': [15, 20, 60],  # 60 exceeds max
        'LAT': [40.0, 40.01, 40.02],
        'LON': [-74.0, -74.01, -74.02],
        'BaseDateTime': pd.date_range('2024-01-01', periods=3, freq='1min'),
        'COG': [90, 90, 90],
    })
    
    violations = verifier._check_speed_consistency(df)
    
    assert violations['exceeds_max_speed'] == 1

def test_trajectory_validity():
    """Test trajectory validity check."""
    verifier = TrajectoryVerifier()
    
    # Valid trajectory
    valid_df = pd.DataFrame({
        'LAT': np.linspace(40.0, 40.1, 10),
        'LON': np.linspace(-74.0, -74.1, 10),
        'SOG': np.ones(10) * 15,
        'COG': np.ones(10) * 90,
        'BaseDateTime': pd.date_range('2024-01-01', periods=10, freq='1min'),
    })
    
    assert verifier.is_trajectory_valid(valid_df)
    
    # Invalid trajectory (speed violation)
    invalid_df = valid_df.copy()
    invalid_df['SOG'].iloc[5] = 60
    
    assert not verifier.is_trajectory_valid(invalid_df)
```

### 4. Anomaly Detection Tests

```python
from src.anomaly_detection import IsolationForestDetector, RuleBasedDetector

def test_isolation_forest():
    """Test Isolation Forest detector."""
    detector = IsolationForestDetector(contamination=0.1)
    
    # Normal data
    X_train = np.random.randn(100, 4)
    detector.fit(X_train)
    
    # Test data with anomalies
    X_test = np.vstack([
        np.random.randn(90, 4),
        np.random.randn(10, 4) * 5  # Outliers
    ])
    
    predictions = detector.predict(X_test)
    
    assert predictions.shape == (100,)
    assert predictions.sum() > 0  # Some anomalies detected

def test_rule_based_detector():
    """Test rule-based detector."""
    detector = RuleBasedDetector()
    
    # Add speed rule
    detector.add_rule(lambda X: X[:, 0] > 50, "speed_anomaly")
    detector.fit(None)
    
    X_test = np.array([
        [15, 90, 0, 0],  # Normal
        [60, 90, 0, 0],  # Speed anomaly
        [20, 90, 0, 0],  # Normal
    ])
    
    predictions = detector.predict(X_test)
    
    assert predictions[0] == 0
    assert predictions[1] == 1
    assert predictions[2] == 0
```

### 5. Trajectory Prediction Tests

```python
from src.trajectory_prediction import KalmanFilterPredictor, ARIMAPredictor

def test_kalman_filter():
    """Test Kalman Filter predictor."""
    kf = KalmanFilterPredictor()
    
    # Generate synthetic data
    X_train = np.random.randn(100, 4)
    y_train = X_train[1:] + np.random.randn(99, 4) * 0.1
    
    kf.fit(X_train[:-1], y_train)
    
    # Predict
    prediction = kf.predict(X_train[-1])
    
    assert prediction.shape == (4,)

def test_arima():
    """Test ARIMA predictor."""
    arima = ARIMAPredictor(order=(1, 1, 1))
    
    # Generate synthetic time series
    X_train = np.cumsum(np.random.randn(100))
    
    arima.fit(X_train)
    
    # Predict
    predictions = arima.predict(X_train, steps=10)
    
    assert predictions.shape == (10,)
```

---

## Integration Testing

### End-to-End Pipeline Test

```python
def test_full_pipeline():
    """Test complete pipeline."""
    from src.training_pipeline import TrainingPipeline
    
    # Create sample data
    df = create_sample_data(n_records=1000, n_vessels=5)
    df.to_csv('test_data.csv', index=False)
    
    # Run pipeline
    pipeline = TrainingPipeline(output_dir='test_models')
    metrics = pipeline.run_full_pipeline('test_data.csv')
    
    # Verify outputs
    assert 'prediction' in metrics
    assert 'anomaly_detection' in metrics
    assert 'trajectory_verification' in metrics
    
    # Verify models saved
    assert (Path('test_models') / 'prediction_kalman.pkl').exists()
    assert (Path('test_models') / 'anomaly_rule_based.pkl').exists()
```

---

## Performance Testing

### Benchmark Tests

```python
import time

def test_kalman_filter_performance():
    """Test Kalman Filter speed."""
    kf = KalmanFilterPredictor()
    X = np.random.randn(1000, 4)
    
    start = time.time()
    for x in X:
        kf.predict(x)
    elapsed = time.time() - start
    
    # Should be very fast (< 1 second for 1000 predictions)
    assert elapsed < 1.0

def test_lstm_performance():
    """Test LSTM speed."""
    lstm = LSTMPredictor()
    X_train = np.random.randn(100, 60, 4)
    y_train = np.random.randn(100, 10, 4)
    
    start = time.time()
    lstm.fit(X_train, y_train)
    elapsed = time.time() - start
    
    # Should complete in reasonable time
    assert elapsed < 60.0  # 1 minute
```

---

## Running Tests

### Using pytest

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_preprocessing.py

# Run with coverage
pytest --cov=src tests/

# Run with verbose output
pytest -v tests/
```

### Using unittest

```bash
# Run all tests
python -m unittest discover

# Run specific test
python -m unittest tests.test_preprocessing.TestPreprocessing
```

---

## Test Data

### Create Sample Data

```python
def create_sample_data(n_records=1000, n_vessels=5):
    """Create sample AIS data for testing."""
    np.random.seed(42)
    data = []
    base_time = pd.Timestamp('2024-01-01')
    
    for vessel_id in range(n_vessels):
        mmsi = str(200000000 + vessel_id)
        lat = 40.0 + np.random.randn() * 0.1
        lon = -74.0 + np.random.randn() * 0.1
        
        for i in range(n_records // n_vessels):
            lat += np.random.randn() * 0.001
            lon += np.random.randn() * 0.001
            
            data.append({
                'MMSI': mmsi,
                'BaseDateTime': base_time + pd.Timedelta(minutes=i),
                'LAT': lat,
                'LON': lon,
                'SOG': max(0, 15 + np.random.randn() * 3),
                'COG': np.random.uniform(0, 360),
                'VesselName': f'Vessel_{vessel_id}',
                'IMO': str(1000000 + vessel_id),
            })
    
    return pd.DataFrame(data)
```

---

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    - name: Run tests
      run: pytest --cov=src tests/
```

---

## Expected Test Results

- **Data Preprocessing**: 100% pass rate
- **MMSI Analysis**: 100% pass rate
- **Trajectory Verification**: 100% pass rate
- **Anomaly Detection**: 100% pass rate
- **Prediction Models**: 100% pass rate
- **Integration Tests**: 100% pass rate
- **Performance Tests**: All within time limits

---

## Debugging Tips

1. **Enable logging**: Set `logging.basicConfig(level=logging.DEBUG)`
2. **Print intermediate values**: Add print statements in test functions
3. **Use pytest fixtures**: For reusable test data
4. **Mock external dependencies**: Use `unittest.mock`
5. **Check data shapes**: Verify array dimensions match expectations

