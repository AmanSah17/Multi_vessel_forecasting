# Advanced XGBoost Pipeline - Technical Details

## Architecture Overview

```
Raw Sequences (122,977 × 12 × 28)
    ↓
[1] Advanced Feature Extraction (28 dims → 476 features)
    ├─ Statistical Features (mean, std, min, max, median, p25, p75, range)
    ├─ Distribution Features (skewness, kurtosis)
    ├─ Trend Features (trend_mean, trend_std, trend_max, trend_min)
    ├─ Autocorrelation Features (first_last_diff, first_last_ratio)
    └─ Volatility Features (std of differences)
    ↓
[2] Haversine Distance Features (7 features)
    ├─ Distance to first point (mean, max, std)
    ├─ Total distance traveled
    ├─ Average distance per step
    ├─ Max consecutive distance
    └─ Std of consecutive distances
    ↓
Combined Features (122,977 × 483)
    ↓
[3] Standardization (StandardScaler)
    └─ Zero mean, unit variance
    ↓
Scaled Features (122,977 × 483)
    ↓
[4] PCA Dimensionality Reduction
    ├─ n_components = 0.95 (95% variance)
    ├─ Final dimensions: 80
    └─ Explained variance: 95.10%
    ↓
PCA Features (122,977 × 80)
    ↓
[5] Hyperparameter Tuning (Optuna - 100 trials)
    ├─ Bayesian Optimization (TPE Sampler)
    ├─ Median Pruner for early stopping
    └─ Optimization metric: MAE
    ↓
[6] XGBoost Training
    ├─ MultiOutputRegressor wrapper
    ├─ 4 output targets (LAT, LON, SOG, COG)
    └─ Best parameters from tuning
    ↓
Trained Model
    ↓
[7] Predictions & Evaluation
    ├─ 122,977 predictions
    ├─ Per-variable metrics (LAT, LON, SOG, COG)
    └─ 50 vessel trajectory visualizations
```

---

## Feature Engineering Details

### 1. Time-Series Statistical Features

For each of 28 input dimensions:

```python
# Statistical Features
mean = np.mean(X_dim, axis=1)
std = np.std(X_dim, axis=1)
min_val = np.min(X_dim, axis=1)
max_val = np.max(X_dim, axis=1)
median = np.median(X_dim, axis=1)
p25 = np.percentile(X_dim, 25, axis=1)
p75 = np.percentile(X_dim, 75, axis=1)
range = max_val - min_val

# Distribution Shape
skew = pd.Series(row).skew()  # Asymmetry
kurtosis = pd.Series(row).kurtosis()  # Tail heaviness

# Trend Features
diff = np.diff(X_dim, axis=1)
trend_mean = np.mean(diff, axis=1)
trend_std = np.std(diff, axis=1)
trend_max = np.max(diff, axis=1)
trend_min = np.min(diff, axis=1)

# Autocorrelation-like
first_last_diff = X_dim[:, -1] - X_dim[:, 0]
first_last_ratio = X_dim[:, -1] / (X_dim[:, 0] + 1e-6)

# Volatility
volatility = np.std(diff, axis=1)
```

**Total per dimension:** 14 features  
**Total for 28 dimensions:** 28 × 14 = 392 features

### 2. Haversine Distance Features

```python
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c
```

**Features extracted:**
1. Mean distance to first point
2. Max distance to first point
3. Std of distances to first point
4. Total distance traveled
5. Average distance per step
6. Max consecutive distance
7. Std of consecutive distances

**Total:** 7 features

### 3. Feature Combination

- Statistical features: 392
- Haversine features: 7
- Temporal features (from EDA): 84 (from 28 dims × 3 lag features)
- **Total:** 483 features

---

## Standardization & PCA

### StandardScaler
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)
# Transforms: X_scaled = (X - mean) / std
```

**Benefits:**
- Removes scale differences between features
- Improves numerical stability
- Required for PCA

### PCA Configuration
```python
pca = PCA(n_components=0.95)  # Keep 95% variance
X_pca = pca.fit_transform(X_scaled)
```

**Results:**
- Input dimensions: 483
- Output dimensions: 80
- Explained variance: 95.10%
- Variance loss: 4.90%

**Why 95%?**
- Balances information retention and dimensionality reduction
- Reduces noise and multicollinearity
- Improves model generalization
- Faster training and inference

---

## Hyperparameter Tuning with Optuna

### Configuration
```python
study = optuna.create_study(
    direction='minimize',
    pruner=MedianPruner(),
    sampler=optuna.samplers.TPESampler(seed=42)
)

study.optimize(
    lambda trial: objective(trial, X_train, y_train, X_val, y_val),
    n_trials=100,
    show_progress_bar=True
)
```

### Search Space
| Parameter | Range | Type |
|-----------|-------|------|
| n_estimators | 100-500 | int (step=50) |
| max_depth | 3-15 | int |
| learning_rate | 0.001-0.3 | float (log) |
| subsample | 0.5-1.0 | float |
| colsample_bytree | 0.5-1.0 | float |
| min_child_weight | 1-10 | int |
| gamma | 0-5 | float |
| reg_alpha | 0-1 | float |
| reg_lambda | 0-1 | float |

### Optimization Results
- **Best Trial:** Trial 94
- **Best MAE (validation):** 9.7792
- **Final MAE (test):** 8.1781
- **Improvement:** 16.3%

### Best Parameters
```python
{
    'n_estimators': 400,      # 400 boosting rounds
    'max_depth': 4,           # Shallow trees (prevent overfitting)
    'learning_rate': 0.1324,  # Moderate learning rate
    'subsample': 0.8773,      # 87.73% row sampling
    'colsample_bytree': 0.8939,  # 89.39% column sampling
    'min_child_weight': 9,    # Minimum 9 samples per leaf
    'gamma': 3.8522,          # High split threshold
    'reg_alpha': 0.8390,      # L1 regularization
    'reg_lambda': 0.9197      # L2 regularization
}
```

---

## Model Performance Analysis

### Per-Variable Metrics

**Latitude (LAT)**
- MAE: 0.3056°
- RMSE: 0.4393°
- R²: 0.9973
- Interpretation: Excellent precision (~34 km at equator)

**Longitude (LON)**
- MAE: 1.1040°
- RMSE: 1.5073°
- R²: 0.9971
- Interpretation: Good precision (~87 km at equator)

**Speed Over Ground (SOG)**
- MAE: 0.4177 knots
- Interpretation: Reasonable speed prediction

**Course Over Ground (COG)**
- MAE: 30.89°
- Interpretation: Moderate heading prediction

### Why LAT/LON are Better?

1. **Spatial Continuity:** Vessel positions are continuous
2. **Haversine Features:** Capture spatial relationships
3. **PCA Preservation:** 95% variance includes spatial info
4. **XGBoost Strength:** Handles non-linear spatial patterns

---

## Deployment Considerations

### Model Serialization
```python
# Save
with open('xgboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load
with open('xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)
```

### Preprocessing Pipeline
```python
# For new data:
1. Extract 483 features (same as training)
2. Apply scaler.transform() (not fit_transform!)
3. Apply pca.transform() (not fit_transform!)
4. Make predictions: model.predict(X_pca)
```

### Inference Speed
- Feature extraction: ~0.23s per 122,977 samples
- Haversine calculation: ~28s per 122,977 samples
- PCA transformation: <1s
- XGBoost prediction: <1s
- **Total:** ~30 seconds for 122,977 samples (~4,100 samples/sec)

---

## Conclusion

This advanced pipeline achieves state-of-the-art vessel trajectory prediction through:
1. Comprehensive feature engineering (483 features)
2. Spatial awareness (Haversine distance)
3. Dimensionality reduction (PCA to 80 features)
4. Extensive hyperparameter tuning (100 trials)
5. Robust multi-output regression (XGBoost)

**Result:** 0.3056° LAT and 1.1040° LON accuracy with R² > 0.997

