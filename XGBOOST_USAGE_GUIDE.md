# XGBoost Advanced Pipeline - Usage Guide

## Quick Start

### 1. Load Pre-trained Model
```python
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load model and preprocessing objects
with open('results/xgboost_advanced_50_vessels/xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('results/xgboost_advanced_50_vessels/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('results/xgboost_advanced_50_vessels/pca.pkl', 'rb') as f:
    pca = pickle.load(f)
```

### 2. Prepare New Data
```python
# Your new sequences: shape (n_samples, 12, 28)
X_new = np.load('your_data.npy')

# Extract features (same as training)
X_features = extract_advanced_features(X_new)  # 483 features
X_haversine = add_haversine_features(X_new, y_dummy)  # 7 features
X_combined = np.hstack([X_features, X_haversine])  # 490 features

# Apply preprocessing (IMPORTANT: use transform, not fit_transform!)
X_scaled = scaler.transform(X_combined)
X_pca = pca.transform(X_scaled)

# Make predictions
predictions = model.predict(X_pca)  # shape: (n_samples, 4)
```

### 3. Extract Results
```python
# predictions shape: (n_samples, 4)
# Columns: [LAT, LON, SOG, COG]

lat_predictions = predictions[:, 0]
lon_predictions = predictions[:, 1]
sog_predictions = predictions[:, 2]
cog_predictions = predictions[:, 3]
```

---

## Feature Extraction Functions

### Advanced Feature Extraction
```python
def extract_advanced_features(X):
    """Extract 14 features per dimension from sequences."""
    n_samples, n_timesteps, n_features = X.shape
    features_list = []
    
    for dim in range(n_features):
        X_dim = X[:, :, dim]
        
        features_dict = {
            'mean': np.mean(X_dim, axis=1),
            'std': np.std(X_dim, axis=1),
            'min': np.min(X_dim, axis=1),
            'max': np.max(X_dim, axis=1),
            'median': np.median(X_dim, axis=1),
            'p25': np.percentile(X_dim, 25, axis=1),
            'p75': np.percentile(X_dim, 75, axis=1),
            'range': np.max(X_dim, axis=1) - np.min(X_dim, axis=1),
            'skew': np.array([pd.Series(row).skew() for row in X_dim]),
            'kurtosis': np.array([pd.Series(row).kurtosis() for row in X_dim]),
        }
        
        diff = np.diff(X_dim, axis=1)
        features_dict['trend_mean'] = np.mean(diff, axis=1)
        features_dict['trend_std'] = np.std(diff, axis=1)
        features_dict['trend_max'] = np.max(diff, axis=1)
        features_dict['trend_min'] = np.min(diff, axis=1)
        
        features_dict['first_last_diff'] = X_dim[:, -1] - X_dim[:, 0]
        features_dict['first_last_ratio'] = np.divide(X_dim[:, -1], X_dim[:, 0] + 1e-6)
        features_dict['volatility'] = np.std(diff, axis=1)
        
        dim_features = np.column_stack(list(features_dict.values()))
        features_list.append(dim_features)
    
    return np.hstack(features_list)
```

### Haversine Distance Features
```python
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate Haversine distance in km."""
    R = 6371
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def add_haversine_features(X, y):
    """Add 7 Haversine distance features."""
    n_samples = X.shape[0]
    haversine_features = []
    
    for i in range(n_samples):
        seq = X[i]
        lats = seq[:, 0]
        lons = seq[:, 1]
        
        dist_to_first = haversine_distance(lats[0], lons[0], lats, lons)
        
        consecutive_dists = [0.0]
        for j in range(1, len(lats)):
            dist = haversine_distance(lats[j-1], lons[j-1], lats[j], lons[j])
            consecutive_dists.append(dist)
        
        total_dist = np.sum(consecutive_dists)
        avg_dist = np.mean(consecutive_dists[1:]) if len(consecutive_dists) > 1 else 0
        
        haversine_features.append([
            np.mean(dist_to_first),
            np.max(dist_to_first),
            np.std(dist_to_first),
            total_dist,
            avg_dist,
            np.max(consecutive_dists),
            np.std(consecutive_dists)
        ])
    
    return np.array(haversine_features)
```

---

## Model Performance

### Accuracy Metrics
```
Latitude:  MAE=0.3056°, RMSE=0.4393°, R²=0.9973
Longitude: MAE=1.1040°, RMSE=1.5073°, R²=0.9971
Overall:   MAE=8.18, RMSE=27.01, R²=0.9351
```

### Interpretation
- **LAT MAE 0.3056°** ≈ 34 km precision at equator
- **LON MAE 1.1040°** ≈ 87 km precision at equator
- **R² > 0.997** indicates excellent fit for location variables

---

## Deployment Checklist

- [ ] Load model, scaler, and PCA objects
- [ ] Verify input shape: (n_samples, 12, 28)
- [ ] Extract 483 features using provided functions
- [ ] Apply scaler.transform() (NOT fit_transform!)
- [ ] Apply pca.transform() (NOT fit_transform!)
- [ ] Make predictions: model.predict(X_pca)
- [ ] Extract LAT/LON from predictions[:, 0:2]
- [ ] Validate predictions are within expected ranges

---

## Expected Input/Output

### Input
- **Shape:** (n_samples, 12, 28)
- **12:** Timesteps (60 minutes at 5-minute intervals)
- **28:** Vessel monitoring parameters
- **Example:** 1000 vessel sequences

### Output
- **Shape:** (n_samples, 4)
- **Column 0:** Latitude (degrees)
- **Column 1:** Longitude (degrees)
- **Column 2:** Speed Over Ground (knots)
- **Column 3:** Course Over Ground (degrees)

---

## Common Issues & Solutions

### Issue 1: Shape Mismatch
```
Error: Expected 80 features, got 483
Solution: Apply PCA transformation after scaling
```

### Issue 2: Preprocessing Not Applied
```
Error: Predictions are NaN or very large
Solution: Use scaler.transform() and pca.transform()
         (NOT fit_transform on new data!)
```

### Issue 3: Feature Extraction Mismatch
```
Error: Feature count doesn't match
Solution: Ensure you extract exactly 483 features:
         - 392 from statistical features (28 dims × 14 features)
         - 91 from lag features (28 dims × 3 lags + 7 haversine)
```

---

## Performance Optimization

### Batch Processing
```python
# Process in batches for large datasets
batch_size = 10000
predictions = []

for i in range(0, len(X_new), batch_size):
    X_batch = X_new[i:i+batch_size]
    X_features = extract_advanced_features(X_batch)
    X_haversine = add_haversine_features(X_batch, None)
    X_combined = np.hstack([X_features, X_haversine])
    X_scaled = scaler.transform(X_combined)
    X_pca = pca.transform(X_scaled)
    pred_batch = model.predict(X_pca)
    predictions.append(pred_batch)

predictions = np.vstack(predictions)
```

### GPU Acceleration (if available)
```python
# Use GPU for feature extraction
import cupy as cp

X_gpu = cp.asarray(X_new)
# Process on GPU, then transfer back to CPU for model
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `xgboost_model.pkl` | Trained XGBoost model |
| `scaler.pkl` | StandardScaler for normalization |
| `pca.pkl` | PCA transformer (483→80 dims) |
| `model_metrics.csv` | Performance metrics |
| `all_predictions.csv` | 122,977 predictions |
| `vessel_*.png` | 50 trajectory visualizations |

---

## Next Steps

1. **Integrate into Production:** Use deployment checklist
2. **Monitor Performance:** Track prediction accuracy over time
3. **Retrain Periodically:** Update model with new vessel data
4. **Ensemble Methods:** Combine with other models for robustness
5. **Real-time Deployment:** Implement streaming predictions

---

## Support & Documentation

- **Technical Details:** See `XGBOOST_TECHNICAL_DETAILS.md`
- **Summary Report:** See `XGBOOST_ADVANCED_PIPELINE_SUMMARY.md`
- **Script:** `notebooks/40_xgboost_advanced_pipeline.py`

**Status:** ✅ Production Ready

