# Architecture Diagram - Maritime Vessel Forecasting Pipeline

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RAW AIS DATA (CSV)                              │
│  MMSI | BaseDateTime | LAT | LON | SOG | COG | VesselName | IMO | ...  │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    DATA PREPROCESSING MODULE                             │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ • Parse datetime                                                 │  │
│  │ • Handle missing vessel names → "Unidentified Vessel"           │  │
│  │ • Validate MMSI format (9 digits)                               │  │
│  │ • Remove duplicates                                             │  │
│  │ • Resample to 1-minute intervals                                │  │
│  │ • Interpolate missing values                                    │  │
│  │ • Remove outliers (speed > 50 knots, invalid coords)            │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    CLEANED & UNIFORM DATA                                │
│  1-minute intervals, validated MMSI, no outliers                        │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                ┌────────────┼────────────┐
                │            │            │
                ▼            ▼            ▼
        ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
        │ MMSI         │ │ FEATURE      │ │ TRAIN/VAL/  │
        │ ANALYSIS     │ │ ENGINEERING  │ │ TEST SPLIT  │
        │              │ │              │ │              │
        │ • Distrib.   │ │ • Temporal   │ │ • 60% Train │
        │ • Country    │ │ • Kinematic  │ │ • 20% Val   │
        │ • Formatting │ │ • Vessel     │ │ • 20% Test  │
        │ • Patterns   │ │   features   │ │ • Temporal  │
        └──────────────┘ └──────────────┘ └──────────────┘
                │            │                    │
                └────────────┼────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    TRAJECTORY PREDICTION MODELS                          │
│                                                                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐     │
│  │  KALMAN FILTER   │  │     ARIMA        │  │      LSTM        │     │
│  │                  │  │                  │  │                  │     │
│  │ • Real-time      │  │ • Statistical    │  │ • Deep Learning  │     │
│  │ • O(1) complex   │  │ • Interpretable  │  │ • Complex        │     │
│  │ • Low latency    │  │ • Baseline       │  │   patterns       │     │
│  │ • Missing data   │  │ • Seasonality    │  │ • Long-term      │     │
│  │   handling       │  │                  │  │   dependencies   │     │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘     │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │              ENSEMBLE PREDICTOR                                │   │
│  │  Combines: Mean | Median | Weighted voting                    │   │
│  └────────────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              TRAJECTORY CONSISTENCY VERIFICATION                         │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ • Smoothness Check (last 3 points)                              │  │
│  │ • Speed Validation (max 50 knots)                               │  │
│  │ • Turn Rate Validation (max 45°/min)                            │  │
│  │ • Acceleration Validation (max 2 knots/min)                     │  │
│  │ • Consistency Score (0-1)                                       │  │
│  │ • Anomaly Detection                                             │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    ANOMALY DETECTION ENSEMBLE                            │
│                                                                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐     │
│  │ ISOLATION FOREST │  │   AUTOENCODER    │  │   RULE-BASED     │     │
│  │                  │  │                  │  │                  │     │
│  │ • Statistical    │  │ • Learns normal  │  │ • Domain rules   │     │
│  │   outliers       │  │   patterns       │  │ • Speed limits   │     │
│  │ • Unsupervised   │  │ • Reconstruction │  │ • Turn rates     │     │
│  │ • Fast           │  │   error          │  │ • Acceleration   │     │
│  │ • Scalable       │  │ • Non-linear     │  │ • Geofences      │     │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘     │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │         ENSEMBLE VOTING (Majority | Any | All)                │   │
│  │  Anomaly Scores: 0-1 (proportion of detectors flagging)       │   │
│  └────────────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    MODEL EVALUATION & METRICS                            │
│                                                                          │
│  Prediction Metrics:        Anomaly Detection:      Consistency:        │
│  • MAE (km)                 • Precision             • Smoothness        │
│  • RMSE (km)                • Recall                • Consistency       │
│  • MAPE (%)                 • F1-Score              • Anomaly Count     │
│                             • ROC-AUC                                   │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    MODEL PERSISTENCE                                     │
│                                                                          │
│  Save to disk:              Load from disk:                             │
│  • prediction_kalman.pkl    • For inference                             │
│  • prediction_arima.pkl     • For batch processing                      │
│  • prediction_lstm.pkl      • For real-time systems                     │
│  • anomaly_*.pkl            • For monitoring                            │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    INFERENCE & DEPLOYMENT                                │
│                                                                          │
│  Real-Time:                 Batch Processing:       Monitoring:         │
│  • Kalman Filter            • LSTM                  • Anomaly scores    │
│  • Low latency              • High accuracy         • Consistency       │
│  • Streaming data           • Daily/weekly          • Alerts            │
│                                                                          │
│  Output:                                                                 │
│  • Next position (lat, lon)                                             │
│  • Confidence interval                                                  │
│  • Anomaly score (0-1)                                                  │
│  • Consistency score (0-1)                                              │
│  • Alerts (if anomalies detected)                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

```
┌─────────────────┐
│   Raw AIS CSV   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  VesselDataPreprocessor.preprocess()    │
│  • Parse, clean, resample, validate     │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Cleaned DataFrame (1-min intervals)    │
└────────┬────────────────────────────────┘
         │
    ┌────┴────┬──────────┬──────────┐
    │          │          │          │
    ▼          ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│ MMSI   │ │Feature │ │Train   │ │Verify  │
│Analyzer│ │Engineer│ │/Val/   │ │Traj.   │
│        │ │        │ │Test    │ │        │
└────────┘ └────────┘ └────────┘ └────────┘
    │          │          │          │
    └────┬─────┴──────────┴──────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  TrainingPipeline.run_full_pipeline()   │
│  • Train prediction models              │
│  • Train anomaly detectors              │
│  • Evaluate on test set                 │
│  • Save models                          │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Trained Models (saved to disk)         │
│  • Kalman Filter                        │
│  • ARIMA                                │
│  • LSTM                                 │
│  • Anomaly Detectors                    │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Inference on New Data                  │
│  • Load models                          │
│  • Predict positions                    │
│  • Detect anomalies                     │
│  • Verify consistency                   │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Predictions & Alerts                   │
│  • Next position                        │
│  • Anomaly score                        │
│  • Consistency score                    │
│  • Alerts (if needed)                   │
└─────────────────────────────────────────┘
```

---

## Module Dependencies

```
training_pipeline.py
├── data_preprocessing.py
├── mmsi_analysis.py
├── trajectory_prediction.py
│   ├── KalmanFilterPredictor
│   ├── ARIMAPredictor
│   ├── LSTMPredictor
│   └── EnsemblePredictor
├── trajectory_verification.py
│   └── TrajectoryVerifier
└── anomaly_detection.py
    ├── IsolationForestDetector
    ├── AutoencoderDetector
    ├── RuleBasedDetector
    └── EnsembleAnomalyDetector
```

---

## Training Workflow

```
Raw Data
   │
   ▼
Preprocessing
   │
   ▼
Feature Engineering
   │
   ▼
Train/Val/Test Split (Temporal)
   │
   ├─────────────────────────────┐
   │                             │
   ▼                             ▼
Train Set (60%)            Val Set (20%)
   │                             │
   ├─ Kalman Filter              │
   ├─ ARIMA                       │
   ├─ LSTM                        │
   └─ Anomaly Detectors           │
                                  │
                                  ▼
                            Validation
                                  │
                                  ▼
                            Test Set (20%)
                                  │
                                  ▼
                            Evaluation
                                  │
                                  ▼
                            Save Models
```

---

## Inference Workflow

```
New Vessel Data
   │
   ▼
Load Trained Models
   │
   ├─ Kalman Filter
   ├─ ARIMA
   ├─ LSTM
   ├─ Anomaly Detectors
   └─ Trajectory Verifier
   │
   ▼
Predict Next Position
   │
   ├─ Kalman: Real-time
   ├─ ARIMA: Statistical
   └─ LSTM: Deep learning
   │
   ▼
Ensemble Prediction
   │
   ▼
Verify Consistency
   │
   ▼
Detect Anomalies
   │
   ▼
Output Results
   ├─ Position (lat, lon)
   ├─ Confidence
   ├─ Anomaly Score
   └─ Alerts
```

