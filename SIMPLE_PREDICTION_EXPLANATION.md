# How to Predict Vessel Trajectories - Simple Explanation

## The Big Picture

We trained an XGBoost model that learned patterns from 122,977 vessel sequences. Now we want to use this trained model to predict where a vessel will be in the future.

---

## Step-by-Step Prediction Process

### Step 1: Get Vessel Data (Last 12 timesteps = 60 minutes)
```
From Database:
- Get last 12 records of a vessel (5-minute intervals)
- Each record has: LAT, LON, SOG, COG, timestamp

Example:
Time 0:   LAT=32.7, LON=-77.0, SOG=15.2, COG=45.0
Time 5:   LAT=32.71, LON=-76.99, SOG=15.1, COG=45.2
...
Time 55:  LAT=32.78, LON=-76.98, SOG=15.3, COG=44.9
```

### Step 2: Extract Advanced Features (Convert 4 values → 483 features)

**Why?** The model was trained on 483 features, not just 4 raw values.

**How?** For each of the 4 dimensions (LAT, LON, SOG, COG), extract:

**A) Statistical Features (10 per dimension):**
- Average value
- Standard deviation (how much it varies)
- Min and Max values
- Median
- 25th and 75th percentiles
- Range (max - min)
- Skewness (is it left/right skewed?)
- Kurtosis (how peaked is the distribution?)

**B) Trend Features (7 per dimension):**
- Trend slope (is it going up or down?)
- Std of changes (how much does it jump around?)
- Max and Min changes
- First to last difference (how much changed overall?)
- First to last ratio (proportional change)
- Volatility (std of changes)

**C) Haversine Distance Features (7 total):**
- How far the vessel traveled from first point (mean, max, std)
- Total distance traveled
- Average distance per step
- Max distance in one step
- Std of distances between steps

**Result:** 4 dimensions × 17 features + 7 haversine = **483 features**

### Step 3: Standardize Features (Make them comparable)
```
StandardScaler does:
- Subtract the mean from each feature
- Divide by standard deviation
- Result: All features have mean=0, std=1

Why? So features with different scales don't dominate
```

### Step 4: Apply PCA (Compress 483 → 80 features)
```
PCA (Principal Component Analysis):
- Finds the most important patterns in the 483 features
- Keeps only 80 components that explain 95% of the variation
- Removes noise and redundancy

Why? Faster predictions, less overfitting
```

### Step 5: Feed to XGBoost Model
```
Input: 80 compressed features
Model: XGBoost (trained on 122,977 sequences)
Output: 4 predictions
  - Predicted LAT
  - Predicted LON
  - Predicted SOG
  - Predicted COG
```

### Step 6: Get Results
```
Output:
{
  "predicted_lat": 32.92,
  "predicted_lon": -76.89,
  "predicted_sog": 15.2,
  "predicted_cog": 45.1
}

This is where the vessel will be in ~60 minutes
```

---

## Real-World Example

**Scenario:** Vessel "CHAMPAGNE CHER" is at position (32.78, -76.98)

**Process:**
1. Fetch last 12 records from database
2. Extract 483 features from these 12 records
3. Standardize the 483 features
4. Compress to 80 PCA components
5. Feed to XGBoost model
6. Get prediction: (32.92, -76.89)

**Result:** Vessel will move from (32.78, -76.98) to (32.92, -76.89) in 60 minutes

---

## Key Points

### What We Need
✅ Last 12 timesteps of vessel data (LAT, LON, SOG, COG)
✅ Trained XGBoost model (saved as pickle file)
✅ Scaler (to standardize features)
✅ PCA transformer (to compress features)

### What We Get
✅ Predicted position (LAT, LON)
✅ Predicted speed (SOG)
✅ Predicted course (COG)

### Accuracy
- Latitude error: ~0.3° (about 34 km)
- Longitude error: ~1.1° (about 87 km)
- R² score: 0.99 (99% of variance explained)

---

## The Pipeline in Code (Pseudo-code)

```python
# 1. Load saved model and preprocessing objects
model = load('xgboost_model.pkl')
scaler = load('scaler.pkl')
pca = load('pca.pkl')

# 2. Get vessel data
vessel_data = database.fetch_vessel(vessel_name, last_12_records)

# 3. Extract features
features_483 = extract_advanced_features(vessel_data)

# 4. Standardize
features_scaled = scaler.transform(features_483)

# 5. Compress with PCA
features_80 = pca.transform(features_scaled)

# 6. Predict
prediction = model.predict(features_80)

# 7. Extract results
predicted_lat, predicted_lon, predicted_sog, predicted_cog = prediction[0]

print(f"Next position: ({predicted_lat}, {predicted_lon})")
```

---

## Why This Works

1. **Pattern Learning:** XGBoost learned patterns from 122,977 sequences
2. **Feature Engineering:** 483 features capture vessel behavior comprehensively
3. **Dimensionality Reduction:** PCA removes noise while keeping important info
4. **Standardization:** Ensures all features contribute equally
5. **Ensemble Method:** XGBoost combines many decision trees for robust predictions

---

## Comparison: XGBoost vs Dead Reckoning

### Dead Reckoning (Simple)
```
Assumption: Vessel continues at same speed and course
Next Position = Current Position + (Speed × Course × Time)
Accuracy: ~5-10% error
Speed: Very fast
```

### XGBoost (Advanced)
```
Learns: Vessel acceleration, course changes, patterns
Considers: 483 features of vessel behavior
Accuracy: ~0.3-1.1° error (99% R²)
Speed: Slower but still fast
```

---

## Summary

**To predict vessel trajectories:**

1. Get last 12 records of vessel
2. Extract 483 features (statistical + trend + haversine)
3. Standardize features
4. Compress with PCA (483 → 80)
5. Feed to XGBoost model
6. Get 4 predictions (LAT, LON, SOG, COG)

**That's it!** The model does all the heavy lifting.

