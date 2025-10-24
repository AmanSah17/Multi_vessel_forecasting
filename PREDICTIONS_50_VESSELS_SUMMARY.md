# Tiny LSTM Model - Predictions on 50 Random Vessels

## Overview
Successfully generated trajectory predictions for 50 random vessels from the test set using the pre-trained **Tiny LSTM** model (best performer from previous training).

**No additional training was performed** - only inference on test data.

---

## Execution Details

### Script: `notebooks/38_test_tiny_lstm_50_vessels.py`

**Process:**
1. **Load Test Data** - Loaded 122,977 test sequences from 66 unique vessels
2. **Load Pre-trained Model** - Loaded Tiny LSTM model weights from `results/models/best_tiny_lstm.pt`
3. **Make Predictions** - Generated predictions for all test sequences (batch size: 256)
4. **Generate Visualizations** - Created trajectory plots for 50 randomly selected vessels
5. **Save Results** - Exported metrics and predictions to CSV files

---

## Model Architecture

**Tiny LSTM Model:**
- Input Size: 28 features (vessel monitoring data)
- Hidden Size: 32
- Number of Layers: 4
- Dropout: 0.15
- Output Size: 4 (LAT, LON, SOG, COG)

**Architecture:**
```
LSTM(28 -> 32, 4 layers, dropout=0.15)
  ↓
Linear(32 -> 64) + ReLU + Dropout(0.15)
  ↓
Linear(64 -> 4)
```

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| **MAE** | 49.93 |
| **RMSE** | 84.11 |
| **R² Score** | -1.85 |

**Note:** The negative R² indicates the model predictions have higher error than a simple mean baseline. This suggests the model may need retraining or the test set has different characteristics than the training set.

---

## Output Files

### Location: `results/test_tiny_lstm_50_vessels/`

**Files Generated:**
1. **50 Trajectory PNG Files** - Visualization plots for each vessel
   - Format: `vessel_{MMSI}_trajectory.png`
   - Size: ~150KB - 500KB per plot
   - Total: 50 plots

2. **model_metrics.csv** - Performance metrics summary
   - MAE, RMSE, R² scores

3. **all_predictions.csv** - Detailed predictions for all test sequences
   - Columns: MMSI, actual_LAT, actual_LON, actual_SOG, actual_COG, pred_LAT, pred_LON, pred_SOG, pred_COG
   - Rows: 122,977 sequences

---

## Visualization Details

Each trajectory plot shows:
- **4 Subplots** - One for each prediction variable:
  1. Latitude (degrees)
  2. Longitude (degrees)
  3. SOG - Speed Over Ground (knots)
  4. COG - Course Over Ground (degrees)

**Plot Features:**
- **Blue Line with Circles** - Actual values from test set
- **Red Dashed Line with Squares** - Tiny LSTM model predictions
- **X-axis** - Time in minutes (5-minute intervals)
- **Y-axis** - Variable values
- **Grid** - Enabled for easy reading

**Example Vessel:** Vessel 369285000
- Multiple sequences showing trajectory over time
- Comparison between predicted and actual vessel movement

---

## Data Characteristics

**Test Set:**
- Total Sequences: 122,977
- Unique Vessels: 66
- Sequence Length: 12 timesteps
- Time Interval: 5 minutes per timestep
- Total Time per Sequence: 60 minutes (1 hour)

**Selected Vessels for Visualization:**
- Random sample of 50 vessels from 66 unique vessels
- Each vessel has multiple sequences in the test set

---

## Key Observations

1. **Model Loading** - Successfully loaded pre-trained Tiny LSTM weights
2. **Inference Speed** - Fast predictions on GPU (632 batches/sec)
3. **Visualization Quality** - All 50 plots generated successfully
4. **Data Export** - Complete predictions saved for further analysis

---

## Next Steps (Optional)

1. **Model Retraining** - Consider retraining with different hyperparameters
2. **Feature Engineering** - Add more advanced features (Haversine distance, etc.)
3. **Ensemble Methods** - Combine multiple models for better predictions
4. **Hyperparameter Tuning** - Optimize learning rate, hidden size, dropout
5. **Data Analysis** - Investigate why R² is negative (data distribution issues?)

---

## Files Location

- **Trajectory Plots:** `results/test_tiny_lstm_50_vessels/*.png`
- **Metrics:** `results/test_tiny_lstm_50_vessels/model_metrics.csv`
- **Predictions:** `results/test_tiny_lstm_50_vessels/all_predictions.csv`
- **Script:** `notebooks/38_test_tiny_lstm_50_vessels.py`

---

## Execution Time

- Feature Extraction: ~4 seconds
- Model Loading: <1 second
- Predictions: ~1 second
- Visualization Generation: ~41 seconds
- **Total Time: ~47 seconds**

---

**Status:** ✓ COMPLETE - All 50 vessel trajectory predictions generated and visualized successfully.

