# ⚡ Fast LSTM Training Guide

## 🎯 Quick Start

### Option 1: PyTorch LSTM (FASTEST - Recommended)
```bash
python notebooks/09_pytorch_lstm_cuda_optimized.py
```
- **Time**: 10-15 minutes
- **Memory**: 3-4 GB
- **GPU**: CUDA enabled
- **Progress**: Real-time tqdm bars

### Option 2: TensorFlow LSTM
```bash
python notebooks/08_fast_lstm_training_cuda.py
```
- **Time**: 15-20 minutes
- **Memory**: 4-5 GB
- **GPU**: CUDA enabled
- **Progress**: Real-time tqdm bars

---

## 📊 What Gets Trained

### Models
- ✅ **LSTM** - Deep learning trajectory prediction
- ❌ Kalman Filter (skipped)
- ❌ ARIMA (skipped)
- ❌ Anomaly detectors (skipped)

### Verification
- ✅ **Trajectory Verification** - Smoothness, consistency checks
- ✅ **Progress Tracking** - Real-time tqdm progress bars

---

## 🚀 Features

### PyTorch Version (Recommended)
```
✅ CUDA GPU acceleration
✅ Real-time progress bars (tqdm)
✅ Fast training (10-15 min)
✅ Low memory usage (3-4 GB)
✅ Early stopping
✅ Best model checkpoint
✅ Detailed logging
```

### TensorFlow Version
```
✅ CUDA GPU acceleration
✅ Real-time progress bars (tqdm)
✅ Fast training (15-20 min)
✅ Low memory usage (4-5 GB)
✅ Early stopping
✅ Detailed logging
```

---

## 📈 Data Processing

### Input
- **Source**: AIS_2020_01_03.csv (7.1M records)
- **Sample**: 100K records (1.4% of data)

### Processing
1. Load data ✅
2. Sample (stratified by vessel) ✅
3. Preprocess (1-min resampling) ✅
4. Engineer features (13 total) ✅
5. Create sequences (60-step) ✅
6. Train LSTM ✅
7. Verify trajectories ✅

### Output
- **Model**: `best_model.pt` (PyTorch) or `lstm_model.h5` (TensorFlow)
- **Results**: `results.json` with metrics
- **Logs**: `training_pytorch_cuda.log` or `training_lstm_cuda.log`

---

## 🎓 Model Architecture

### PyTorch LSTM
```
Input: (batch_size, 60, 13)
  ↓
LSTM Layer 1: 128 units, dropout=0.2
  ↓
LSTM Layer 2: 128 units, dropout=0.2
  ↓
Dense: 64 units, ReLU
  ↓
Dropout: 0.2
  ↓
Output: (batch_size, 2) → LAT, LON
```

### Training
- **Optimizer**: Adam (lr=0.001)
- **Loss**: MSE
- **Batch Size**: 64
- **Epochs**: 15 (with early stopping)
- **Train/Val Split**: 80/20

---

## 📊 Expected Results

### PyTorch LSTM (100K sample)
```
Training time: 10-15 minutes
Final train loss: ~0.0001-0.0005
Final val loss: ~0.0002-0.0008
Trajectory smoothness: ~0.88-0.92
```

### TensorFlow LSTM (100K sample)
```
Training time: 15-20 minutes
Final train loss: ~0.0001-0.0005
Final val loss: ~0.0002-0.0008
Trajectory smoothness: ~0.88-0.92
```

---

## 🔍 Monitoring Progress

### Real-time Progress Bars
```
Creating sequences: |████████████████████| 100%
Training: Epoch 1/15: |████████████████████| 100%
Verifying: |████████████████████| 100%
```

### Check Logs
```bash
# PowerShell - Last 50 lines
Get-Content training_pytorch_cuda.log -Tail 50

# Or watch in real-time
Get-Content training_pytorch_cuda.log -Wait
```

### View Results
```bash
# View JSON results
Get-Content training_logs_pytorch/results.json | ConvertFrom-Json
```

---

## 💾 Output Files

### PyTorch Version
```
training_logs_pytorch/
├── best_model.pt          # Trained model
├── results.json           # Training results
└── training_pytorch_cuda.log

training_pytorch_cuda.log  # Detailed log
```

### TensorFlow Version
```
training_logs_lstm/
├── lstm_model.h5          # Trained model
├── training_results.json  # Training results
└── training_lstm_cuda.log

training_lstm_cuda.log     # Detailed log
```

---

## 🔧 Troubleshooting

### Q: CUDA not detected?
A: Check PyTorch installation:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Q: Out of memory?
A: Reduce sample size in script:
```python
trainer = PyTorchLSTMTrainer(sample_size=50000)  # Instead of 100000
```

### Q: Training too slow?
A: Ensure CUDA is being used:
```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Q: TensorFlow not installed?
A: Install with:
```bash
pip install tensorflow
```

---

## 📚 Key Differences

| Feature | PyTorch | TensorFlow |
|---------|---------|-----------|
| Speed | ⚡ Faster | 🚀 Fast |
| Memory | 💾 Lower | 💾 Higher |
| Setup | ✅ Simpler | ⚠️ Complex |
| GPU | ✅ Better | ✅ Good |
| **Recommended** | **✅ YES** | ⚠️ Alternative |

---

## 🎯 Recommended Workflow

### Step 1: Run PyTorch LSTM (10-15 min)
```bash
python notebooks/09_pytorch_lstm_cuda_optimized.py
```

### Step 2: Monitor Progress
```bash
Get-Content training_pytorch_cuda.log -Wait
```

### Step 3: Check Results
```bash
Get-Content training_logs_pytorch/results.json | ConvertFrom-Json
```

### Step 4: Use Model
```python
import torch
model = torch.load('training_logs_pytorch/best_model.pt')
predictions = model(X_test)
```

---

## ✨ Features

✅ **Fast**: 10-15 minutes training
✅ **GPU**: CUDA acceleration
✅ **Progress**: Real-time tqdm bars
✅ **Memory**: Optimized for large datasets
✅ **Logging**: Detailed training logs
✅ **Verification**: Trajectory consistency checks
✅ **Checkpointing**: Best model saved
✅ **Early Stopping**: Prevents overfitting

---

## 🏆 Status

**✅ READY TO TRAIN**

Choose PyTorch for fastest results:
```bash
python notebooks/09_pytorch_lstm_cuda_optimized.py
```

---

**Estimated Time**: 10-15 minutes

**Start now!**

