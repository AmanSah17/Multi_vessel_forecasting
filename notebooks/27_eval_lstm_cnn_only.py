"""
Evaluate-only script to generate test metrics and 300-vessel plots for LSTM and CNN
without interrupting an ongoing training process.

- Loads cached sequences (npz) created by the main pipeline
- Recreates the same split/scaling
- Loads best LSTM/CNN checkpoints from results/models
- Evaluates on the test set and generates 300-vessel plots
- Saves outputs to results/images/vessels_300_now and results/csv/*_now.csv
"""
from __future__ import annotations
import os
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt

# Output dirs (mirror of the main pipeline)
output_dirs = {
    'logs': Path('logs'),
    'results': Path('results'),
    'images': Path('results/images'),
    'csv': Path('results/csv'),
    'models': Path('results/models'),
}
for d in output_dirs.values():
    d.mkdir(parents=True, exist_ok=True)

LOGGER = logging.getLogger("eval_lstm_cnn_only")

# ------------------------------ Models (copied from 26) ------------------------------
class EnhancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=4, output_size=4, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, output_size)
        )
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class TemporalCNNModel(nn.Module):
    def __init__(self, input_size, output_size=4, num_filters=128, num_layers=5, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Conv1d(input_size, num_filters, 1)
        blocks = []
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (3 - 1) * dilation // 2
            blocks.append(nn.Sequential(
                nn.Conv1d(num_filters, num_filters, 3, padding=padding, dilation=dilation),
                nn.BatchNorm1d(num_filters), nn.ReLU(), nn.Dropout(dropout)
            ))
        self.blocks = nn.ModuleList(blocks)
        self.fc = nn.Sequential(
            nn.Linear(num_filters, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, output_size)
        )
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=2)
        return self.fc(x)

# ------------------------------ Eval helper ------------------------------
def evaluate_model(model, test_loader, device='cuda', model_name='model'):
    LOGGER.info("Evaluating %s", model_name.upper())
    model.eval()
    y_true_all, y_pred_all = [], []
    amp_enabled = bool(torch.cuda.is_available() and (device == 'cuda' or (isinstance(device, torch.device) and device.type=='cuda')))
    autocast = torch.cuda.amp.autocast
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc=f"Evaluating {model_name}", unit="batch", dynamic_ncols=True):
            X_batch = X_batch.to(device)
            with autocast(enabled=amp_enabled):
                preds = model(X_batch).cpu().numpy()
            y_pred_all.append(preds)
            y_true_all.append(y_batch.numpy())
    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAE_LAT': float(mean_absolute_error(y_true[:,0], y_pred[:,0])),
        'MAE_LON': float(mean_absolute_error(y_true[:,1], y_pred[:,1])),
        'MAE_SOG': float(mean_absolute_error(y_true[:,2], y_pred[:,2])),
        'MAE_COG': float(mean_absolute_error(y_true[:,3], y_pred[:,3])),
    }
    LOGGER.info("%s -> MAE=%.6f RMSE=%.6f R2=%.6f", model_name.upper(), mae, rmse, r2)
    return metrics, y_true, y_pred

# ------------------------------ Plot helper ------------------------------
def save_vessel_plots(pred_dict, selected_vessels, out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {"lstm": "#1f77b4", "cnn": "#ff7f0e"}
    metrics = [("lat", "Latitude"), ("lon", "Longitude"), ("sog", "SOG"), ("cog", "COG")]
    for mmsi in tqdm(selected_vessels, desc="Plotting vessels", unit="vessel", dynamic_ncols=True):
        mmsi_int = int(mmsi)
        if mmsi_int not in pred_dict:
            continue
        d = pred_dict[mmsi_int]
        n = len(d['actual_lat'])
        if n == 0:
            continue
        t = np.arange(n)
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
        axes = axes.flatten()
        for ax, (key, title) in zip(axes, metrics):
            ax.plot(t, d[f"actual_{key}"], label="Actual", color="black", linewidth=1.6)
            for model_name in ["lstm", "cnn"]:
                pred_key = f"{model_name}_{key}"
                if pred_key in d:
                    ax.plot(t, d[pred_key], label=model_name.upper(), color=colors[model_name], linewidth=1.2, alpha=0.95)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
        axes[0].legend(loc="upper right", ncol=3, fontsize=8)
        fig.suptitle(f"Vessel {mmsi_int} - Actual vs Predictions (LSTM/CNN)", fontsize=14)
        fig.tight_layout(rect=[0, 0.02, 1, 0.96])
        out_path = out_dir / f"vessel_{mmsi_int}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)

# ------------------------------ Cache loader ------------------------------
def find_latest_cache(seq_len: int = 120) -> Path | None:
    cache_dir = output_dirs['results'] / 'cache'
    pattern = f"seq_cache_len{seq_len}_feat"
    if not cache_dir.exists():
        return None
    candidates = [p for p in cache_dir.glob(f"{pattern}*.npz")]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

# ------------------------------ Main ------------------------------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dirs['logs'] / 'eval_lstm_cnn_only.log', encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True,
    )
    LOGGER.info("Starting evaluation-only (LSTM, CNN)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LOGGER.info("Device: %s", device)

    # Load cached sequences
    cache_file = find_latest_cache(seq_len=120)
    if cache_file is None:
        LOGGER.error("No cached sequences found in %s. Please run the main pipeline to create cache first.", output_dirs['results'] / 'cache')
        return
    LOGGER.info("Loading sequences from cache: %s", cache_file)
    data = np.load(cache_file, allow_pickle=True)
    X = data['X']  # (N, T, F)
    y = data['y']  # (N, 4)
    features = data['features'].tolist()
    mmsi_list = data['mmsi_list'].tolist()

    n = len(X)
    train_idx = int(n * 0.7)
    val_idx = int(n * 0.9)

    X_train, y_train = X[:train_idx], y[:train_idx]
    X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
    X_test, y_test = X[val_idx:], y[val_idx:]

    LOGGER.info("Split sizes -> Train: %s, Val: %s, Test: %s", len(X_train), len(X_val), len(X_test))

    # Scale features for test set using train-set min/max (memory efficient)
    # Compute per-feature min/max over the training sequences (axes: sequence, time)
    mins = X_train.min(axis=(0, 1), keepdims=True)
    maxs = X_train.max(axis=(0, 1), keepdims=True)
    denom = maxs - mins
    denom[denom == 0] = 1.0
    X_test_scaled = (X_test - mins) / denom

    # Test loader
    test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test))
    num_workers = 0  # Windows-safe
    pin_memory = torch.cuda.is_available()
    test_loader = DataLoader(test_dataset, batch_size=128 if torch.cuda.is_available() else 64, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)

    input_size = len(features)

    # Instantiate and load checkpoints
    models_to_eval = {}
    lstm_ckpt = output_dirs['models'] / 'best_lstm.pt'
    cnn_ckpt = output_dirs['models'] / 'best_cnn.pt'
    if not lstm_ckpt.exists():
        LOGGER.error("Missing checkpoint: %s", lstm_ckpt)
        return
    if not cnn_ckpt.exists():
        LOGGER.error("Missing checkpoint: %s", cnn_ckpt)
        return

    lstm_model = EnhancedLSTMModel(input_size=input_size, hidden_size=256, num_layers=4, dropout=0.1).to(device)
    cnn_model = TemporalCNNModel(input_size=input_size, num_filters=128, num_layers=5, dropout=0.1).to(device)
    lstm_model.load_state_dict(torch.load(lstm_ckpt, map_location=device))
    cnn_model.load_state_dict(torch.load(cnn_ckpt, map_location=device))
    models_to_eval['lstm'] = lstm_model
    models_to_eval['cnn'] = cnn_model

    all_metrics = {}
    all_predictions = {}
    y_true_test_ref = None

    for name, model in models_to_eval.items():
        metrics, y_true_test, y_pred = evaluate_model(model, test_loader, device=device, model_name=name)
        all_metrics[name] = metrics
        all_predictions[name] = y_pred
        if y_true_test_ref is None:
            y_true_test_ref = y_true_test

    # 300-vessel predictions
    LOGGER.info("Generating 300-vessel predictions and plots (LSTM/CNN only)...")
    test_mmsi = np.array(mmsi_list)[val_idx:]
    unique_mmsi = np.unique(test_mmsi)
    np.random.seed(42)
    selected_vessels = np.random.choice(unique_mmsi, size=min(300, len(unique_mmsi)), replace=False)

    pred_dict = {}
    for mmsi in tqdm(selected_vessels, desc="Building vessel data", unit="vessel", dynamic_ncols=True):
        idxs = np.where(test_mmsi == mmsi)[0]
        if len(idxs) == 0:
            continue
        entry = {
            'actual_lat': y_true_test_ref[idxs, 0],
            'actual_lon': y_true_test_ref[idxs, 1],
            'actual_sog': y_true_test_ref[idxs, 2],
            'actual_cog': y_true_test_ref[idxs, 3],
        }
        for model_name, y_pred in all_predictions.items():
            entry[f'{model_name}_lat'] = y_pred[idxs, 0]
            entry[f'{model_name}_lon'] = y_pred[idxs, 1]
            entry[f'{model_name}_sog'] = y_pred[idxs, 2]
            entry[f'{model_name}_cog'] = y_pred[idxs, 3]
            entry[f'{model_name}_mae_lat'] = float(np.mean(np.abs(entry[f'{model_name}_lat'] - entry['actual_lat'])))
            entry[f'{model_name}_mae_lon'] = float(np.mean(np.abs(entry[f'{model_name}_lon'] - entry['actual_lon'])))
            entry[f'{model_name}_mae_sog'] = float(np.mean(np.abs(entry[f'{model_name}_sog'] - entry['actual_sog'])))
            entry[f'{model_name}_mae_cog'] = float(np.mean(np.abs(entry[f'{model_name}_cog'] - entry['actual_cog'])))
        pred_dict[int(mmsi)] = entry

    images_dir = output_dirs['images'] / 'vessels_300_now'
    save_vessel_plots(pred_dict, selected_vessels, images_dir)
    LOGGER.info("Saved vessel plots to: %s", images_dir)

    # Save CSVs
    rows = []
    for mmsi, d in pred_dict.items():
        n = len(d['actual_lat'])
        for i in range(n):
            row = {
                'MMSI': mmsi, 'idx': i,
                'actual_lat': float(d['actual_lat'][i]),
                'actual_lon': float(d['actual_lon'][i]),
                'actual_sog': float(d['actual_sog'][i]),
                'actual_cog': float(d['actual_cog'][i]),
                'lstm_lat': float(d.get('lstm_lat', [np.nan]*n)[i]) if 'lstm_lat' in d else np.nan,
                'lstm_lon': float(d.get('lstm_lon', [np.nan]*n)[i]) if 'lstm_lon' in d else np.nan,
                'lstm_sog': float(d.get('lstm_sog', [np.nan]*n)[i]) if 'lstm_sog' in d else np.nan,
                'lstm_cog': float(d.get('lstm_cog', [np.nan]*n)[i]) if 'lstm_cog' in d else np.nan,
                'cnn_lat': float(d.get('cnn_lat', [np.nan]*n)[i]) if 'cnn_lat' in d else np.nan,
                'cnn_lon': float(d.get('cnn_lon', [np.nan]*n)[i]) if 'cnn_lon' in d else np.nan,
                'cnn_sog': float(d.get('cnn_sog', [np.nan]*n)[i]) if 'cnn_sog' in d else np.nan,
                'cnn_cog': float(d.get('cnn_cog', [np.nan]*n)[i]) if 'cnn_cog' in d else np.nan,
            }
            rows.append(row)
    df_pred = pd.DataFrame(rows)
    pred_csv = output_dirs['csv'] / 'vessel_predictions_300_detailed_now.csv'
    df_pred.to_csv(pred_csv, index=False)
    LOGGER.info("Saved detailed predictions to: %s", pred_csv)

    comp_rows = [
        {'Model': 'LSTM', **all_metrics['lstm']},
        {'Model': 'CNN', **all_metrics['cnn']},
    ]
    comp_df = pd.DataFrame(comp_rows)
    comp_csv = output_dirs['csv'] / 'model_comparison_lstm_cnn_now.csv'
    comp_df.to_csv(comp_csv, index=False)
    LOGGER.info("Saved comparison to: %s", comp_csv)

    LOGGER.info("Evaluation-only complete.")

if __name__ == '__main__':
    main()

