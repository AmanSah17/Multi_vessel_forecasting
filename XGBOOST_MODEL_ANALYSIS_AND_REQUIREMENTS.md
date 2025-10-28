# XGBoost Advanced Model - Complete Analysis & Requirements

## Executive Summary

The Advanced XGBoost model for vessel trajectory forecasting is a sophisticated machine learning system trained on **28-dimensional sequences** with extensive feature engineering. However, integrating it with the Maritime NLU backend requires careful data preparation.

---

## Model Architecture

### Input Requirements

**Sequence Shape:** (n_samples, 12, 28)
- **12 timesteps** = 60 minutes of vessel data (5-minute intervals)
- **28 features per timestep** = comprehensive vessel state representation

### Output

**Predictions:** (n_samples, 4)
- LAT (Latitude)
- LON (Longitude)
- SOG (Speed Over Ground, knots)
- COG (Course Over Ground, degrees)

### Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| Latitude MAE | 0.3056° | ~34 km accuracy |
| Latitude R² | 0.9973 | Excellent fit |
| Longitude MAE | 1.1040° | ~87 km accuracy |
| Longitude R² | 0.9971 | Excellent fit |
| Overall MAE | 8.18 | Combined metric |
| Overall R² | 0.9351 | Strong performance |

---

## The 28 Features Explained

### Group 1: Base Navigation (4 features)
```
LAT, LON, SOG, COG
```
- **LAT**: Latitude (degrees)
- **LON**: Longitude (degrees)
- **SOG**: Speed Over Ground (knots)
- **COG**: Course Over Ground (degrees)

### Group 2: Temporal Features (4 features)
```
hour, day_of_week, is_weekend, month
```
- **hour**: Hour of day (0-23)
- **day_of_week**: Day of week (0-6, Monday=0)
- **is_weekend**: Binary flag (1 if Saturday/Sunday)
- **month**: Month of year (1-12)

### Group 3: Kinematic Changes (4 features)
```
speed_change, heading_change, lat_change, lon_change
```
- **speed_change**: Δ SOG from previous timestep
- **heading_change**: Δ COG from previous timestep
- **lat_change**: Δ LAT from previous timestep
- **lon_change**: Δ LON from previous timestep

### Group 4: Velocity Components (3 features)
```
velocity_x, velocity_y, velocity_mag
```
- **velocity_x**: SOG × cos(COG)
- **velocity_y**: SOG × sin(COG)
- **velocity_mag**: √(velocity_x² + velocity_y²)

### Group 5: Polynomial Features (3 features)
```
lat_sq, lon_sq, sog_sq
```
- **lat_sq**: LAT²
- **lon_sq**: LON²
- **sog_sq**: SOG²

### Group 6: Interaction Features (2 features)
```
speed_heading_int, lat_lon_int
```
- **speed_heading_int**: SOG × COG
- **lat_lon_int**: LAT × LON

### Group 7: Additional Features (8 features)
*Typically includes lag features, cyclical encodings, or other derived features*

**Total: 4 + 4 + 4 + 3 + 3 + 2 + 8 = 28 features**

---

## Feature Extraction Pipeline

### Step 1: Prepare 28-Dimensional Sequences
```
Raw Database Data (LAT, LON, SOG, COG)
    ↓
Engineer 28 Features per Timestep
    ↓
Create 12-Timestep Sequences (12 × 28)
```

### Step 2: Extract Advanced Features (476 features)
```
For each of 28 dimensions:
    - Statistical: mean, std, min, max, median, p25, p75, range
    - Distribution: skewness, kurtosis
    - Trend: trend_mean, trend_std, trend_max, trend_min
    - Autocorrelation: first_last_diff, first_last_ratio
    - Volatility: std of differences
    
Total: 17 features × 28 dimensions = 476 features
```

### Step 3: Add Haversine Distance Features (7 features)
```
- Distance to first point: mean, max, std
- Total distance traveled
- Average distance per step
- Max consecutive distance
- Std of consecutive distances
```

### Step 4: Combine Features
```
476 + 7 = 483 total features
```

### Step 5: Standardization
```
StandardScaler: Zero mean, unit variance
Input: (n_samples, 483)
Output: (n_samples, 483)
```

### Step 6: PCA Dimensionality Reduction
```
PCA(n_components=80): Reduces 483 → 80 components
Variance retained: 95.10%
Input: (n_samples, 483)
Output: (n_samples, 80)
```

### Step 7: XGBoost Prediction
```
MultiOutputRegressor(XGBRegressor)
Input: (n_samples, 80)
Output: (n_samples, 4) → [LAT, LON, SOG, COG]
```

---

## Integration Challenges & Solutions

### Challenge 1: Database Only Has 4 Features
**Problem:** Database stores only LAT, LON, SOG, COG
**Model Expects:** 28 features per timestep
**Solution:** Engineer the missing 24 features from available data

### Challenge 2: Feature Engineering Complexity
**Problem:** Creating 28 features requires temporal context
**Model Expects:** Sequences with proper temporal relationships
**Solution:** Use 12-timestep windows with proper lag calculations

### Challenge 3: Scaler/PCA Mismatch
**Problem:** Scaler trained on 483 features, but we extract fewer
**Model Expects:** Exactly 483 features
**Solution:** Ensure all 483 features are extracted correctly

---

## Current Implementation Status

### ✅ Completed
- Model artifacts loaded (xgboost_model.pkl, scaler.pkl, pca.pkl)
- Backend integration file created
- Intent executor updated with XGBoost support
- Model files copied to backend directory

### ⚠️ In Progress
- Feature extraction pipeline (20/28 features working)
- Sequence preparation from database data
- Integration testing

### ❌ Blocked
- Full 28-feature extraction (need to identify missing 8 features)
- XGBoost predictions (feature dimension mismatch)

---

## Recommended Approach

### Option 1: Use Dead Reckoning (Recommended for Now)
**Pros:**
- Simple, fast, no feature engineering needed
- Works with 4 features (LAT, LON, SOG, COG)
- Reliable fallback method

**Cons:**
- Less accurate than XGBoost
- Assumes constant velocity

**Implementation:** Already working in intent_executor.py

### Option 2: Retrain Simpler XGBoost Model
**Pros:**
- Use only 4 features (LAT, LON, SOG, COG)
- Simpler feature extraction
- Still better than dead reckoning

**Cons:**
- Requires retraining
- May have lower accuracy

**Implementation:** Create new training pipeline with 4 features

### Option 3: Complete 28-Feature Engineering
**Pros:**
- Use existing trained model
- Maximum accuracy

**Cons:**
- Complex feature engineering
- Need to identify all 28 features
- More computational overhead

**Implementation:** Requires detailed analysis of training data

---

## Key Takeaways

### Model Requirements
✅ **Input:** 12 timesteps × 28 features per timestep
✅ **Output:** 4 predictions (LAT, LON, SOG, COG)
✅ **Preprocessing:** StandardScaler → PCA (483 → 80)
✅ **Performance:** ~0.3° latitude, ~1.1° longitude MAE

### Integration Requirements
✅ **Database:** LAT, LON, SOG, COG, BaseDateTime
✅ **Feature Engineering:** 28 features per timestep
✅ **Sequence Creation:** 12-timestep windows
✅ **Preprocessing:** Exact scaler/PCA from training

### Current Status
✅ Model loaded and ready
⚠️ Feature extraction incomplete (20/28 features)
⚠️ XGBoost predictions not working yet
✅ Dead reckoning fallback working

---

## Next Steps

1. **Identify Missing 8 Features**
   - Check training data cache file
   - Analyze feature engineering code
   - Determine if lag features or cyclical encodings are needed

2. **Complete Feature Extraction**
   - Implement all 28 features
   - Test with sample data
   - Verify shape (12, 28)

3. **Test XGBoost Predictions**
   - Run end-to-end pipeline
   - Compare with dead reckoning
   - Measure accuracy improvement

4. **Deploy to Production**
   - Update backend with working XGBoost
   - Monitor prediction quality
   - Keep dead reckoning as fallback

---

## References

- **Training Notebook:** `notebooks/40_xgboost_advanced_pipeline.py`
- **Model Location:** `results/xgboost_advanced_50_vessels/`
- **Backend Integration:** `F:\Maritime_NLU_Repo\backend\nlu_chatbot\src\app\xgboost_backend_integration.py`
- **Intent Executor:** `F:\Maritime_NLU_Repo\backend\nlu_chatbot\src\app\intent_executor.py`

