"""
Standalone LSTM Training with PyTorch and CUDA

Completely independent - no external preprocessing pipeline
- Fast training (10-15 minutes)
- CUDA GPU acceleration
- Real-time tqdm progress bars
- Minimal dependencies
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_standalone_lstm.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """PyTorch LSTM Model."""
    
    def __init__(self, input_size, hidden_size=64, output_size=2):
        """Initialize LSTM."""
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        """Forward pass."""
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


def load_data(data_path, sample_size=50000):
    """Load and sample data."""
    logger.info("\n" + "="*70)
    logger.info("[1/5] LOADING DATA")
    logger.info("="*70)
    
    logger.info(f"Loading: {data_path}")
    df = pd.read_csv(data_path)
    df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'], format="%Y-%m-%dT%H:%M:%S", errors='coerce')
    
    logger.info(f"Loaded {len(df):,} records, {df['MMSI'].nunique():,} vessels")
    
    # Sample
    logger.info(f"Sampling to {sample_size:,}...")
    if len(df) > sample_size:
        indices = np.random.choice(len(df), sample_size, replace=False)
        df = df.iloc[indices].reset_index(drop=True)
    
    logger.info(f"Sampled to {len(df):,} records")
    
    return df


def prepare_features(df):
    """Prepare features."""
    logger.info("\n" + "="*70)
    logger.info("[2/5] PREPARING FEATURES")
    logger.info("="*70)
    
    # Sort by MMSI and time
    df = df.sort_values(['MMSI', 'BaseDateTime']).reset_index(drop=True)
    
    # Select key features
    features = ['LAT', 'LON', 'SOG', 'COG']
    
    # Add temporal features
    logger.info("Adding temporal features...")
    df['hour'] = df['BaseDateTime'].dt.hour
    df['day_of_week'] = df['BaseDateTime'].dt.dayofweek
    features.extend(['hour', 'day_of_week'])
    
    # Add kinematic features
    logger.info("Adding kinematic features...")
    df['speed_change'] = df.groupby('MMSI')['SOG'].diff().fillna(0)
    df['heading_change'] = df.groupby('MMSI')['COG'].diff().fillna(0)
    df['heading_change'] = df['heading_change'].apply(lambda x: min(abs(x), 360 - abs(x)))
    features.extend(['speed_change', 'heading_change'])
    
    logger.info(f"Features: {features}")
    
    return df, features


def create_sequences(df, features, seq_length=10):
    """Create sequences."""
    logger.info("\n" + "="*70)
    logger.info("[3/5] CREATING SEQUENCES")
    logger.info("="*70)

    X, y = [], []

    logger.info(f"Creating sequences (length={seq_length})...")

    for mmsi in tqdm(df['MMSI'].unique(), desc="Processing vessels", unit="vessel"):
        vessel_data = df[df['MMSI'] == mmsi][features].values

        if len(vessel_data) < seq_length + 1:
            continue

        for i in range(len(vessel_data) - seq_length):
            X.append(vessel_data[i:i+seq_length])
            y.append(vessel_data[i+seq_length, :2])  # Next LAT, LON

    if len(X) == 0:
        logger.warning("No sequences created! Using shorter sequence length...")
        # Fallback: use shorter sequence length
        for mmsi in df['MMSI'].unique():
            vessel_data = df[df['MMSI'] == mmsi][features].values
            if len(vessel_data) >= 5:
                for i in range(len(vessel_data) - 4):
                    X.append(vessel_data[i:i+5])
                    y.append(vessel_data[i+5, :2] if i+5 < len(vessel_data) else vessel_data[-1, :2])

    # Convert to numpy arrays with padding if needed
    if len(X) > 0:
        # Pad sequences to same length
        max_len = max(len(x) for x in X)
        X_padded = []
        for x in X:
            if len(x) < max_len:
                padding = np.zeros((max_len - len(x), x.shape[1]))
                x_padded = np.vstack([x, padding])
            else:
                x_padded = x
            X_padded.append(x_padded)

        X = np.array(X_padded, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
    else:
        logger.error("Still no sequences! Data may be too sparse.")
        raise ValueError("Cannot create sequences from data")

    logger.info(f"Created {len(X):,} sequences")

    # Normalize
    logger.info("Normalizing data...")
    scaler = MinMaxScaler()
    X_flat = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.fit_transform(X_flat).reshape(X.shape)

    return X_scaled, y, scaler


def train_model(X, y, epochs=10, batch_size=32):
    """Train LSTM model."""
    logger.info("\n" + "="*70)
    logger.info("[4/5] TRAINING LSTM")
    logger.info("="*70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Convert to tensors
    logger.info("Converting to PyTorch tensors...")
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.FloatTensor(y).to(device)
    
    # Split
    split_idx = int(0.8 * len(X))
    X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
    y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]
    
    logger.info(f"Train: {len(X_train):,}, Val: {len(X_val):,}")
    
    # DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    logger.info("Building LSTM model...")
    model = LSTMModel(input_size=X.shape[2]).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    logger.info(f"Training for {epochs} epochs...")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []

    for epoch in tqdm(range(epochs), desc="Epochs", unit="epoch"):
        # Train
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []

        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1} batches", leave=False, unit="batch"):
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_preds.append(outputs.detach().cpu().numpy())
            train_targets.append(y_batch.detach().cpu().numpy())

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Calculate train MAE
        train_preds = np.vstack(train_preds)
        train_targets = np.vstack(train_targets)
        train_mae = mean_absolute_error(train_targets, train_preds)
        train_maes.append(train_mae)

        # Validate
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            val_mae = mean_absolute_error(y_val.cpu().numpy(), val_outputs.cpu().numpy())

        val_losses.append(val_loss)
        val_maes.append(val_mae)

        logger.info(f"Epoch {epoch+1}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, Train MAE={train_mae:.6f}, Val MAE={val_mae:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_lstm_model.pt')

    logger.info("Model saved to best_lstm_model.pt")

    return model, train_losses, val_losses, train_maes, val_maes


def verify_trajectories(df):
    """Verify trajectory smoothness."""
    logger.info("\n" + "="*70)
    logger.info("[5/5] TRAJECTORY VERIFICATION")
    logger.info("="*70)
    
    logger.info("Verifying trajectories...")
    smoothness_scores = []
    
    for mmsi in tqdm(df['MMSI'].unique(), desc="Verifying vessels", unit="vessel"):
        vessel_traj = df[df['MMSI'] == mmsi].sort_values('BaseDateTime')
        
        if len(vessel_traj) >= 3:
            # Check smoothness using last 3 points
            points = vessel_traj[['LAT', 'LON']].tail(3).values
            
            if len(points) == 3:
                v1 = points[1] - points[0]
                v2 = points[2] - points[1]
                
                # Calculate angle between vectors
                dot_product = np.dot(v1, v2)
                norm_v1 = np.linalg.norm(v1)
                norm_v2 = np.linalg.norm(v2)
                
                if norm_v1 > 0 and norm_v2 > 0:
                    cos_angle = dot_product / (norm_v1 * norm_v2)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    smoothness = (cos_angle + 1) / 2  # Normalize to 0-1
                    smoothness_scores.append(smoothness)
    
    avg_smoothness = np.mean(smoothness_scores) if smoothness_scores else 0
    logger.info(f"Average smoothness: {avg_smoothness:.4f}")
    
    return avg_smoothness


def plot_training_curves(train_losses, val_losses, train_maes, val_maes):
    """Plot training and validation curves."""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('LSTM Model Training Performance - 50 Epochs', fontsize=16, fontweight='bold')

    epochs = range(1, len(train_losses) + 1)

    # Plot 1: Training vs Validation Loss
    axes[0, 0].plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', marker='o', markersize=3)
    axes[0, 0].plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=3)
    axes[0, 0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('MSE Loss', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Training vs Validation Loss', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10, loc='best')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Training vs Validation MAE
    axes[0, 1].plot(epochs, train_maes, 'g-', linewidth=2, label='Training MAE', marker='o', markersize=3)
    axes[0, 1].plot(epochs, val_maes, 'orange', linewidth=2, label='Validation MAE', marker='s', markersize=3)
    axes[0, 1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Mean Absolute Error (km)', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Training vs Validation MAE', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10, loc='best')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Loss Comparison (zoomed)
    axes[1, 0].plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', marker='o', markersize=3)
    axes[1, 0].plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=3)
    axes[1, 0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('MSE Loss', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Loss Convergence (Last 20 Epochs)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlim(max(1, len(train_losses) - 20), len(train_losses))
    axes[1, 0].legend(fontsize=10, loc='best')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Metrics Summary
    axes[1, 1].axis('off')
    summary_text = f"""
    TRAINING SUMMARY (50 Epochs)
    {'='*40}

    Final Training Loss:    {train_losses[-1]:.6f}
    Final Validation Loss:  {val_losses[-1]:.6f}
    Best Validation Loss:   {min(val_losses):.6f}

    Final Training MAE:     {train_maes[-1]:.6f} km
    Final Validation MAE:   {val_maes[-1]:.6f} km
    Best Validation MAE:    {min(val_maes):.6f} km

    Loss Improvement:       {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.2f}%
    MAE Improvement:        {((train_maes[0] - train_maes[-1]) / train_maes[0] * 100):.2f}%

    Total Epochs:           {len(train_losses)}
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('training_curves_50epochs.png', dpi=300, bbox_inches='tight')
    logger.info("✓ Training curves saved to training_curves_50epochs.png")
    plt.show()


def main():
    """Main execution."""
    logger.info("\n" + "="*70)
    logger.info("STANDALONE LSTM TRAINING WITH PYTORCH + CUDA")
    logger.info("="*70)
    
    # Load data
    data_path = r"D:\Maritime_Vessel_monitoring\csv_extracted_data\AIS_2020_01_03\AIS_2020_01_03.csv"
    df = load_data(data_path, sample_size=50000)
    
    # Prepare features
    df, features = prepare_features(df)
    
    # Create sequences
    X, y, scaler = create_sequences(df, features, seq_length=30)
    
    # Train model
    model, train_losses, val_losses, train_maes, val_maes = train_model(X, y, epochs=50, batch_size=32)

    # Verify trajectories
    avg_smoothness = verify_trajectories(df)

    # Plot training curves
    logger.info("\n" + "="*70)
    logger.info("PLOTTING TRAINING CURVES")
    logger.info("="*70)
    plot_training_curves(train_losses, val_losses, train_maes, val_maes)

    # Save results
    results = {
        'data': {'records': len(df), 'vessels': df['MMSI'].nunique()},
        'sequences': {'count': len(X)},
        'training': {
            'epochs': len(train_losses),
            'final_train_loss': float(train_losses[-1]),
            'final_val_loss': float(val_losses[-1]),
            'final_train_mae': float(train_maes[-1]),
            'final_val_mae': float(val_maes[-1]),
            'best_val_loss': float(min(val_losses)),
            'best_val_mae': float(min(val_maes))
        },
        'verification': {'avg_smoothness': float(avg_smoothness)}
    }
    
    with open('training_results_standalone.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*70)
    logger.info(f"Model saved to: best_lstm_model.pt")
    logger.info(f"Results saved to: training_results_standalone.json")


if __name__ == '__main__':
    main()

