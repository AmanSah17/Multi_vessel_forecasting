# Final Predictions Report - 50 Random Vessels Trajectory Forecasting

## Executive Summary

Successfully completed **trajectory predictions for 50 random vessels** using the pre-trained **Tiny LSTM** model without any additional training. The model was applied to the test set to generate predictions for vessel movement variables (Latitude, Longitude, Speed Over Ground, Course Over Ground) with 5-minute interval time series visualizations.

---

## Task Completion

✅ **Task:** Use the last trained model weights for testing on random 50 vessels trajectories prediction and plot them against their real values.

✅ **Status:** COMPLETE

✅ **Deliverables:**
- 50 trajectory prediction plots (PNG format)
- Model performance metrics (CSV)
- Complete predictions dataset (CSV)
- Detailed execution logs

---

## Methodology

### 1. Data Preparation
- **Source:** Cached test sequences (`results/cache/seq_cache_len12_sampled_3pct.npz`)
- **Test Set Size:** 122,977 sequences from 66 unique vessels
- **Sequence Length:** 12 timesteps (60 minutes at 5-minute intervals)
- **Features:** 28 vessel monitoring parameters

### 2. Model Selection
- **Model:** Tiny LSTM (Best Performer from Previous Training)
- **Model Path:** `results/models/best_tiny_lstm.pt`
- **Architecture:** 4-layer LSTM with 32 hidden units
- **Parameters:** ~35,652 total parameters

### 3. Inference Process
- **Batch Size:** 256 sequences
- **Device:** NVIDIA GeForce GTX 1650 (GPU)
- **Inference Speed:** 632 batches/second
- **Total Predictions:** 122,977 sequences

### 4. Visualization Generation
- **Vessels Selected:** 50 random vessels from 66 unique vessels
- **Plot Type:** 2x2 subplot grid (4 variables per vessel)
- **Variables Plotted:**
  1. Latitude (degrees)
  2. Longitude (degrees)
  3. SOG - Speed Over Ground (knots)
  4. COG - Course Over Ground (degrees)

---

## Results

### Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **MAE** | 49.93 | Average absolute error across all predictions |
| **RMSE** | 84.11 | Root mean squared error |
| **R² Score** | -1.85 | Model performs worse than baseline |

**Note:** Negative R² suggests the model may need retraining or the test set characteristics differ from training data.

### Output Statistics

| Item | Count |
|------|-------|
| Trajectory Plots Generated | 50 |
| Total Sequences Predicted | 122,977 |
| Unique Vessels in Test Set | 66 |
| Unique Vessels Visualized | 50 |
| CSV Records Exported | 122,977 |

---

## Output Files

### Directory: `results/test_tiny_lstm_50_vessels/`

**1. Trajectory Plots (50 PNG files)**
```
vessel_369285000_trajectory.png
vessel_369303000_trajectory.png
vessel_369494753_trajectory.png
vessel_369550000_trajectory.png
vessel_369604000_trajectory.png
... (45 more files)
```

**2. Model Metrics**
- File: `model_metrics.csv`
- Content: MAE, RMSE, R² scores

**3. Predictions Dataset**
- File: `all_predictions.csv`
- Rows: 122,977
- Columns: MMSI, actual_LAT, actual_LON, actual_SOG, actual_COG, pred_LAT, pred_LON, pred_SOG, pred_COG

---

## Visualization Examples

### Sample Vessel Trajectories

**Vessel 369285000:**
- Latitude: Predicted vs Actual comparison
- Longitude: Predicted vs Actual comparison
- SOG: Speed predictions over time
- COG: Course predictions over time

Each plot shows:
- **Blue Line (Actual):** Ground truth values from test set
- **Red Dashed Line (Predicted):** Tiny LSTM model predictions
- **Time Axis:** 5-minute intervals (0-60 minutes per sequence)

---

## Technical Implementation

### Script: `notebooks/38_test_tiny_lstm_50_vessels.py`

**Key Functions:**
1. `load_test_data()` - Load test sequences and MMSI identifiers
2. `load_tiny_lstm_model()` - Load pre-trained model weights
3. `make_predictions()` - Generate predictions in batches
4. `plot_vessel_trajectory()` - Create visualization for each vessel
5. `calculate_metrics()` - Compute MAE, RMSE, R²

**Dependencies:**
- PyTorch (GPU acceleration)
- NumPy (numerical operations)
- Pandas (data handling)
- Matplotlib/Seaborn (visualization)
- Scikit-learn (metrics calculation)

---

## Execution Timeline

| Step | Duration | Status |
|------|----------|--------|
| Load Test Data | 3 sec | ✓ Complete |
| Load Model | 1 sec | ✓ Complete |
| Make Predictions | 1 sec | ✓ Complete |
| Calculate Metrics | <1 sec | ✓ Complete |
| Generate 50 Plots | 41 sec | ✓ Complete |
| **Total Time** | **~47 seconds** | ✓ Complete |

---

## Key Findings

1. **Model Successfully Loaded** - Pre-trained Tiny LSTM weights loaded without errors
2. **Fast Inference** - GPU-accelerated predictions at 632 batches/second
3. **Complete Visualizations** - All 50 vessel trajectory plots generated successfully
4. **Data Export** - Full predictions dataset exported for further analysis
5. **Performance Baseline** - Metrics established for model evaluation

---

## Recommendations

### For Model Improvement:
1. **Retraining** - Consider retraining with adjusted hyperparameters
2. **Feature Engineering** - Add Haversine distance and other advanced features
3. **Ensemble Methods** - Combine multiple models (LSTM, XGBoost, Random Forest)
4. **Data Augmentation** - Increase training data diversity

### For Analysis:
1. **Error Analysis** - Investigate why R² is negative
2. **Temporal Analysis** - Analyze prediction errors over time
3. **Vessel-Specific Analysis** - Identify which vessels have better/worse predictions
4. **Feature Importance** - Determine which input features drive predictions

---

## Conclusion

Successfully completed trajectory predictions for 50 random vessels using the pre-trained Tiny LSTM model. All 50 trajectory plots have been generated showing predicted vs actual vessel movement variables at 5-minute intervals. The complete predictions dataset and metrics have been exported for further analysis and model evaluation.

**Status:** ✅ TASK COMPLETE

---

## File Locations

- **Trajectory Plots:** `results/test_tiny_lstm_50_vessels/*.png` (50 files)
- **Metrics CSV:** `results/test_tiny_lstm_50_vessels/model_metrics.csv`
- **Predictions CSV:** `results/test_tiny_lstm_50_vessels/all_predictions.csv`
- **Execution Script:** `notebooks/38_test_tiny_lstm_50_vessels.py`
- **Logs:** `logs/test_tiny_lstm_50_vessels.log`

---

**Generated:** 2025-10-25
**Model:** Tiny LSTM (Best Performer)
**Test Set:** 122,977 sequences from 66 vessels
**Visualizations:** 50 random vessels

