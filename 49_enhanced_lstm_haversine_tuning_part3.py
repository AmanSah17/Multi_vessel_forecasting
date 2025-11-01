"""
Part 3: Final Training, Evaluation with Haversine Metrics, and Visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.pytorch
import joblib

logger = logging.getLogger(__name__)


def train_final_model(X_train, y_train, X_val, y_val, best_params, epochs=200, output_dir='results/enhanced_lstm_haversine'):
    """Train final model with best hyperparameters."""
    logger.info(f"\n{'='*70}\n[7/10] TRAINING FINAL MODEL WITH BEST PARAMS\n{'='*70}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    logger.info(f"Best params: {best_params}")
    
    # Prepare config
    config = {
        **best_params,
        'epochs': epochs,
        'patience': 20
    }
    
    # Import necessary functions
    from enhanced_lstm_haversine_tuning import EnhancedLSTMModel, calculate_haversine_errors
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=config['batch_size'],
        shuffle=True
    )
    
    model = EnhancedLSTMModel(
        input_size=X_train.shape[2],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        bidirectional=config.get('bidirectional', False)
    ).to(device)
    
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Training history
    train_losses, val_losses = [], []
    train_maes, val_maes = [], []
    val_haversine_means = [], []
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        # Training phase
        model.train()
        train_loss, train_preds, train_targets = 0, [], []
        
        for X_batch, y_batch in train_loader:
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
        train_mae = mean_absolute_error(np.vstack(train_targets), np.vstack(train_preds))
        train_maes.append(train_mae)
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
            val_pred_np = val_outputs.cpu().numpy()
            val_true_np = y_val_t.cpu().numpy()
            val_mae = mean_absolute_error(val_true_np, val_pred_np)
            
            # Calculate haversine errors
            hav_errors = calculate_haversine_errors(val_true_np, val_pred_np)
            val_haversine_means.append(hav_errors['haversine_mean_m'])
        
        val_losses.append(val_loss)
        val_maes.append(val_mae)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / 'best_lstm_model_haversine.pt')
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load(output_dir / 'best_lstm_model_haversine.pt'))
    logger.info(f"✓ Best model saved to {output_dir / 'best_lstm_model_haversine.pt'}")
    
    return model, train_losses, val_losses, train_maes, val_maes, val_haversine_means, device


def plot_training_curves(train_losses, val_losses, train_maes, val_maes, val_haversine_means, output_dir='results/enhanced_lstm_haversine'):
    """Plot comprehensive training curves."""
    logger.info(f"\n{'='*70}\n[8/10] PLOTTING TRAINING CURVES\n{'='*70}")
    
    output_dir = Path(output_dir)
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Curves - Enhanced LSTM with Haversine Metrics', fontsize=16, fontweight='bold')
    
    # Loss curves
    axes[0, 0].plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', marker='o', markersize=3)
    axes[0, 0].plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=3)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0, 0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE curves
    axes[0, 1].plot(epochs, train_maes, 'g-', linewidth=2, label='Training MAE', marker='^', markersize=3)
    axes[0, 1].plot(epochs, val_maes, 'orange', linewidth=2, label='Validation MAE', marker='v', markersize=3)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('MAE', fontsize=12)
    axes[0, 1].set_title('Training & Validation MAE', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Haversine distance error
    axes[1, 0].plot(epochs, val_haversine_means, 'purple', linewidth=2, marker='D', markersize=3)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Haversine Distance Error (meters)', fontsize=12)
    axes[1, 0].set_title('Validation Haversine Distance Error', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary statistics
    axes[1, 1].axis('off')
    summary_text = f"""
    TRAINING SUMMARY
    ─────────────────────────────────
    Total Epochs: {len(train_losses)}
    
    Final Training Loss: {train_losses[-1]:.6f}
    Final Validation Loss: {val_losses[-1]:.6f}
    
    Final Training MAE: {train_maes[-1]:.6f}
    Final Validation MAE: {val_maes[-1]:.6f}
    
    Final Haversine Error: {val_haversine_means[-1]:.2f} m
    
    Best Validation Loss: {min(val_losses):.6f}
    Best Haversine Error: {min(val_haversine_means):.2f} m
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                   verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / '05_training_curves_haversine.png', dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {output_dir / '05_training_curves_haversine.png'}")
    plt.close()


def evaluate_with_haversine(model, X_test, y_test, device, output_dir='results/enhanced_lstm_haversine'):
    """Evaluate model with comprehensive haversine metrics."""
    logger.info(f"\n{'='*70}\n[9/10] EVALUATING WITH HAVERSINE METRICS\n{'='*70}")
    
    output_dir = Path(output_dir)
    
    # Import haversine functions
    from enhanced_lstm_haversine_tuning import calculate_haversine_errors
    
    model.eval()
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)
    
    with torch.no_grad():
        y_pred = model(X_test_t).cpu().numpy()
    
    y_test_np = y_test_t.cpu().numpy()
    
    # Standard metrics
    mae = mean_absolute_error(y_test_np, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_np, y_pred))
    r2 = r2_score(y_test_np, y_pred)
    
    # Per-output metrics
    output_names = ['Latitude', 'Longitude', 'SOG (knots)', 'COG (degrees)']
    logger.info("\n📊 Per-Output Metrics:")
    for idx, name in enumerate(output_names):
        mae_out = mean_absolute_error(y_test_np[:, idx], y_pred[:, idx])
        rmse_out = np.sqrt(mean_squared_error(y_test_np[:, idx], y_pred[:, idx]))
        r2_out = r2_score(y_test_np[:, idx], y_pred[:, idx])
        logger.info(f"  {name}: MAE={mae_out:.6f}, RMSE={rmse_out:.6f}, R²={r2_out:.6f}")
    
    # Haversine metrics
    hav_errors = calculate_haversine_errors(y_test_np, y_pred)
    logger.info("\n🌍 Haversine Distance Errors:")
    logger.info(f"  Mean:   {hav_errors['haversine_mean_m']:.2f} meters")
    logger.info(f"  Median: {hav_errors['haversine_median_m']:.2f} meters")
    logger.info(f"  Std:    {hav_errors['haversine_std_m']:.2f} meters")
    logger.info(f"  Min:    {hav_errors['haversine_min_m']:.2f} meters")
    logger.info(f"  Max:    {hav_errors['haversine_max_m']:.2f} meters")
    logger.info(f"  P95:    {hav_errors['haversine_p95_m']:.2f} meters")
    logger.info(f"  P99:    {hav_errors['haversine_p99_m']:.2f} meters")
    
    # Log to MLflow
    mlflow.log_metrics({
        'test_mae': mae,
        'test_rmse': rmse,
        'test_r2': r2,
        **{f'test_{k}': v for k, v in hav_errors.items()}
    })
    
    # Visualization
    visualize_predictions(y_test_np, y_pred, output_names, output_dir)
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        **hav_errors
    }
    
    return metrics


def visualize_predictions(y_test, y_pred, output_names, output_dir):
    """Create comprehensive prediction visualizations."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Test Set Predictions vs Actual', fontsize=16, fontweight='bold')
    
    for idx, (ax, name) in enumerate(zip(axes.flat, output_names)):
        actual = y_test[:, idx]
        predicted = y_pred[:, idx]
        
        # Scatter plot
        ax.scatter(actual, predicted, alpha=0.5, s=20, edgecolors='black', linewidth=0.3)
        
        # Perfect prediction line
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Metrics
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)
        
        ax.set_xlabel('Actual', fontsize=11, fontweight='bold')
        ax.set_ylabel('Predicted', fontsize=11, fontweight='bold')
        ax.set_title(f'{name}\nMAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / '06_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {output_dir / '06_predictions_vs_actual.png'}")
    plt.close()


def save_model_and_artifacts(model, scaler, best_params, metrics, output_dir='results/enhanced_lstm_haversine'):
    """Save model and all artifacts."""
    logger.info(f"\n{'='*70}\n[10/10] SAVING MODEL AND ARTIFACTS\n{'='*70}")
    
    output_dir = Path(output_dir)
    
    # Save PyTorch model
    torch.save(model.state_dict(), output_dir / 'final_model.pt')
    logger.info(f"✓ Saved: {output_dir / 'final_model.pt'}")
    
    # Save scaler
    joblib.dump(scaler, output_dir / 'scaler.pkl')
    logger.info(f"✓ Saved: {output_dir / 'scaler.pkl'}")
    
    # Save config
    config = {
        'best_params': best_params,
        'metrics': metrics
    }
    import json
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"✓ Saved: {output_dir / 'config.json'}")
    
    # Log to MLflow
    mlflow.pytorch.log_model(model, "lstm_model_haversine")
    mlflow.log_artifact(str(output_dir / 'config.json'))
    
    logger.info(f"\n✓ All artifacts saved to {output_dir}")

