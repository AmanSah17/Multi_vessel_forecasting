# Quick Access Guide - Vessel Trajectory Predictions

## 📁 Output Directory
```
results/test_tiny_lstm_50_vessels/
```

---

## 📊 Key Files

### 1. Trajectory Plots (50 PNG files)
**Location:** `results/test_tiny_lstm_50_vessels/vessel_*.png`

**Sample Files:**
- `vessel_369285000_trajectory.png`
- `vessel_369303000_trajectory.png`
- `vessel_369494753_trajectory.png`
- `vessel_369550000_trajectory.png`
- `vessel_369604000_trajectory.png`
- ... (45 more files)

**How to View:**
- Open any PNG file in image viewer
- Each plot shows 4 subplots: LAT, LON, SOG, COG
- Blue line = Actual values
- Red dashed line = Model predictions

### 2. Model Metrics
**File:** `results/test_tiny_lstm_50_vessels/model_metrics.csv`

**Content:**
```
,MAE,RMSE,R2
Tiny LSTM,49.934783935546875,84.11396832850505,-1.8467326164245605
```

**Metrics:**
- MAE (Mean Absolute Error): 49.93
- RMSE (Root Mean Squared Error): 84.11
- R² Score: -1.85

### 3. Complete Predictions Dataset
**File:** `results/test_tiny_lstm_50_vessels/all_predictions.csv`

**Size:** ~9.8 MB
**Rows:** 122,977 (one per sequence)

**Columns:**
- `MMSI` - Vessel identifier
- `actual_LAT` - Actual latitude
- `actual_LON` - Actual longitude
- `actual_SOG` - Actual speed over ground
- `actual_COG` - Actual course over ground
- `pred_LAT` - Predicted latitude
- `pred_LON` - Predicted longitude
- `pred_SOG` - Predicted speed over ground
- `pred_COG` - Predicted course over ground

**How to Use:**
- Open in Excel, Python, or any CSV reader
- Analyze prediction errors
- Calculate per-vessel metrics
- Identify best/worst performing vessels

---

## 🔧 Execution Script

**File:** `notebooks/38_test_tiny_lstm_50_vessels.py`

**How to Run:**
```bash
python notebooks/38_test_tiny_lstm_50_vessels.py
```

**What It Does:**
1. Loads test data (122,977 sequences)
2. Loads pre-trained Tiny LSTM model
3. Makes predictions on all test sequences
4. Generates 50 trajectory plots
5. Exports metrics and predictions

**Execution Time:** ~47 seconds

---

## 📈 Model Information

**Model Name:** Tiny LSTM
**Model File:** `results/models/best_tiny_lstm.pt`

**Architecture:**
- 4-layer LSTM with 32 hidden units
- Input: 28 features
- Output: 4 variables (LAT, LON, SOG, COG)
- Total Parameters: ~35,652

**Performance:**
- MAE: 49.93
- RMSE: 84.11
- R²: -1.85

---

## 📊 Data Summary

**Test Set:**
- Total Sequences: 122,977
- Unique Vessels: 66
- Sequence Length: 12 timesteps
- Time Interval: 5 minutes per timestep
- Total Time per Sequence: 60 minutes

**Visualized Vessels:**
- Selected: 50 random vessels
- Coverage: 75.8% of unique vessels

---

## 🎯 How to Analyze Results

### Option 1: View Trajectory Plots
1. Navigate to `results/test_tiny_lstm_50_vessels/`
2. Open any `vessel_*.png` file
3. Compare blue line (actual) vs red line (predicted)
4. Identify patterns and prediction errors

### Option 2: Analyze CSV Data
1. Open `all_predictions.csv` in Excel or Python
2. Calculate error metrics per vessel
3. Identify best/worst performing vessels
4. Analyze temporal patterns

### Option 3: Use Python for Analysis
```python
import pandas as pd
import numpy as np

# Load predictions
df = pd.read_csv('results/test_tiny_lstm_50_vessels/all_predictions.csv')

# Calculate errors
df['lat_error'] = abs(df['actual_LAT'] - df['pred_LAT'])
df['lon_error'] = abs(df['actual_LON'] - df['pred_LON'])
df['sog_error'] = abs(df['actual_SOG'] - df['pred_SOG'])
df['cog_error'] = abs(df['actual_COG'] - df['pred_COG'])

# Per-vessel analysis
vessel_errors = df.groupby('MMSI')[['lat_error', 'lon_error', 'sog_error', 'cog_error']].mean()
print(vessel_errors)
```

---

## 📝 Logs

**File:** `logs/test_tiny_lstm_50_vessels.log`

**Contains:**
- Execution timeline
- Data loading information
- Model loading status
- Prediction progress
- Metrics calculation
- Visualization generation status

---

## ✅ Verification Checklist

- [x] 50 trajectory plots generated
- [x] Model metrics calculated
- [x] Predictions exported to CSV
- [x] All files saved to output directory
- [x] Execution completed successfully
- [x] No training performed (inference only)

---

## 🚀 Next Steps

1. **View Plots** - Open trajectory PNG files to visualize predictions
2. **Analyze Data** - Use CSV files for detailed analysis
3. **Calculate Metrics** - Compute per-vessel performance metrics
4. **Identify Issues** - Find vessels with high prediction errors
5. **Model Improvement** - Consider retraining with different hyperparameters

---

## 📞 Support

**Script Location:** `notebooks/38_test_tiny_lstm_50_vessels.py`
**Output Location:** `results/test_tiny_lstm_50_vessels/`
**Model Location:** `results/models/best_tiny_lstm.pt`

**Key Functions:**
- `load_test_data()` - Load test sequences
- `load_tiny_lstm_model()` - Load pre-trained model
- `make_predictions()` - Generate predictions
- `plot_vessel_trajectory()` - Create visualization
- `calculate_metrics()` - Compute performance metrics

---

**Last Updated:** 2025-10-25
**Status:** ✅ COMPLETE

