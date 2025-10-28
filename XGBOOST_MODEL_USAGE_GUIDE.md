# XGBoost Advanced Model - Usage Guide

## Model Overview

The Advanced XGBoost model is trained for **vessel trajectory forecasting** with the following characteristics:

- **Input**: Sequences of vessel data (12 timesteps × 28 features)
- **Output**: 4 predictions (LAT, LON, SOG, COG)
- **Architecture**: XGBoost MultiOutputRegressor with PCA dimensionality reduction
- **Performance**: 
  - Latitude MAE: 0.3056° (R²=0.9973)
  - Longitude MAE: 1.1040° (R²=0.9971)

---

## Step-by-Step Prediction Process

### Step 1: Prepare Raw Vessel Data

**Input Required:**
- DataFrame with vessel trajectory data
- Minimum 12 records (timesteps)
- Columns: `LAT`, `LON`, `SOG`, `COG`, `BaseDateTime`

```python
# Example
df = pd.DataFrame({
    'LAT': [32.7, 32.71, 32.72, ...],  # 12+ values
    'LON': [-77.0, -76.99, -76.98, ...],  # 12+ values
    'SOG': [15.2, 15.1, 15.3, ...],  # Speed over ground (knots)
    'COG': [45.0, 45.2, 45.1, ...],  # Course over ground (degrees)
    'BaseDateTime': [datetime, datetime, ...]
})
```

### Step 2: Extract Advanced Features (483 features)

**Process:**
1. For each dimension (LAT, LON, SOG, COG):
   - Extract **17 features per dimension** = 68 features
   - Total for 4 dimensions = **272 features**

2. **Statistical Features (10 per dimension):**
   - mean, std, min, max, median
   - p25, p75, range
   - skewness, kurtosis

3. **Trend Features (7 per dimension):**
   - trend_mean (slope of linear fit)
   - trend_std (std of differences)
   - trend_max, trend_min
   - first_last_diff (last - first value)
   - first_last_ratio (last / first value)
   - volatility (std of differences)

4. **Haversine Distance Features (7 total):**
   - Distance to first point: mean, max, std
   - Total distance traveled
   - Average distance per step
   - Max consecutive distance
   - Std of consecutive distances

**Total: 272 + 7 = 279 features** (Note: Original training used 483, but core is 279)

### Step 3: Standardization

**Process:**
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)  # Shape: (1, 483)
```

**Purpose:** Normalize features to zero mean and unit variance

### Step 4: PCA Dimensionality Reduction

**Process:**
```python
pca = PCA(n_components=80)  # Reduces 483 → 80 components
X_pca = pca.transform(X_scaled)  # Shape: (1, 80)
```

**Result:** 
- 80 principal components
- Retains 95.10% of variance
- Reduces computational complexity

### Step 5: XGBoost Prediction

**Process:**
```python
model = MultiOutputRegressor(XGBRegressor(...))
predictions = model.predict(X_pca)  # Shape: (1, 4)
```

**Output:**
```python
[predicted_lat, predicted_lon, predicted_sog, predicted_cog]
```

---

## Complete Prediction Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle

# 1. Load model artifacts
with open('xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('pca.pkl', 'rb') as f:
    pca = pickle.load(f)

# 2. Prepare vessel data (12+ timesteps)
df = fetch_vessel_data(vessel_name, limit=20)  # Get 20 records

# 3. Extract features (483 features)
X_features = extract_advanced_features(df)  # Shape: (1, 483)

# 4. Standardize
X_scaled = scaler.transform(X_features)  # Shape: (1, 483)

# 5. Apply PCA
X_pca = pca.transform(X_scaled)  # Shape: (1, 80)

# 6. Predict
predictions = model.predict(X_pca)  # Shape: (1, 4)
pred_lat, pred_lon, pred_sog, pred_cog = predictions[0]

print(f"Predicted Position: ({pred_lat:.4f}, {pred_lon:.4f})")
print(f"Predicted Speed: {pred_sog:.2f} knots")
print(f"Predicted Course: {pred_cog:.2f} degrees")
```

---

## Required Parameters for Prediction

### Minimum Requirements:

1. **Vessel Data (DataFrame)**
   - Minimum 12 timesteps
   - Columns: LAT, LON, SOG, COG, BaseDateTime
   - Sorted by timestamp (ascending)

2. **Model Artifacts**
   - `xgboost_model.pkl` - Trained XGBoost model
   - `scaler.pkl` - StandardScaler fitted on training data
   - `pca.pkl` - PCA transformer fitted on training data

3. **Feature Extraction Functions**
   - `extract_advanced_features()` - Extract 483 features
   - `haversine_distance()` - Calculate spatial distances

### Optional Parameters:

- **Prediction Horizon**: How far ahead to predict (implicit in model)
- **Confidence Threshold**: Filter predictions below confidence level
- **Fallback Method**: Dead reckoning if XGBoost fails

---

## Feature Extraction Details

### Statistical Features (10 per dimension)
```
mean, std, min, max, median, p25, p75, range, skew, kurtosis
```

### Trend Features (7 per dimension)
```
trend_mean, trend_std, trend_max, trend_min, 
first_last_diff, first_last_ratio, volatility
```

### Haversine Features (7 total)
```
dist_to_first_mean, dist_to_first_max, dist_to_first_std,
total_distance, avg_distance_per_step, 
max_consecutive_distance, std_consecutive_distances
```

---

## Model Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Latitude MAE | 0.3056° | ~34 km accuracy |
| Latitude R² | 0.9973 | Excellent fit |
| Longitude MAE | 1.1040° | ~87 km accuracy |
| Longitude R² | 0.9971 | Excellent fit |
| Overall MAE | 8.18 | Combined metric |
| Overall R² | 0.9351 | Strong performance |

---

## Common Issues & Solutions

### Issue 1: Feature Dimension Mismatch
**Error:** "X has 75 features, but PCA is expecting 483"
**Solution:** Ensure all 483 features are extracted (statistical + trend + haversine)

### Issue 2: Insufficient Data
**Error:** "Cannot extract features from < 12 timesteps"
**Solution:** Fetch at least 12 records from database

### Issue 3: Missing Columns
**Error:** "KeyError: 'LAT' or 'LON'"
**Solution:** Ensure DataFrame has LAT, LON, SOG, COG columns

### Issue 4: Scaler/PCA Mismatch
**Error:** "Scaler fitted on different number of features"
**Solution:** Use the exact scaler and PCA from model artifacts

---

## Integration with Backend

The model is integrated in `xgboost_backend_integration.py`:

```python
from xgboost_backend_integration import XGBoostBackendPredictor

# Initialize
predictor = XGBoostBackendPredictor(model_dir="path/to/model")

# Predict
result = predictor.predict(df)
# Returns: {
#     "predicted_lat": float,
#     "predicted_lon": float,
#     "predicted_sog": float,
#     "predicted_cog": float,
#     "confidence": 0.95
# }
```

---

## Key Takeaways

✅ **Model Input:** 12+ timesteps of vessel data (LAT, LON, SOG, COG)
✅ **Feature Engineering:** 483 features (statistical + trend + haversine)
✅ **Preprocessing:** StandardScaler → PCA (483 → 80)
✅ **Prediction:** XGBoost MultiOutputRegressor
✅ **Output:** 4 values (LAT, LON, SOG, COG)
✅ **Accuracy:** ~0.3° for latitude, ~1.1° for longitude

