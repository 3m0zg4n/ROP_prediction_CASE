import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import os
import glob
import json
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_latest_training_history():
    """Load training history from the most recent metadata file"""
    models_dir = Path('models')
    if not models_dir.exists():
        return None
        
    metadata_files = list(models_dir.glob('metadata_*.json'))
    if not metadata_files:
        return None
        
    latest_meta = max(metadata_files, key=lambda p: p.stat().st_mtime)
    print(f"[INFO] Loading training history from {latest_meta.name}")
    
    try:
        with open(latest_meta, 'r') as f:
            data = json.load(f)
            return data.get('history')
    except Exception as e:
        print(f"[ERROR] Could not load history: {e}")
        return None

def create_learning_curves(history):
    """Generate learning curve plots"""
    if not history:
        print("[WARN] No training history found to plot learning curves")
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Model Training Performance (Learning Curves)', fontsize=16, fontweight='bold')
    
    # Plot Loss (MSE)
    epochs = range(1, len(history['loss']) + 1)
    ax1.plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], 'r--', label='Validation Loss', linewidth=2)
    ax1.set_title('Loss (Huber/MSE) vs Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot MAE
    if 'mae' in history:
        ax2.plot(epochs, history['mae'], 'b-', label='Training MAE', linewidth=2)
        if 'val_mae' in history:
            ax2.plot(epochs, history['val_mae'], 'r--', label='Validation MAE', linewidth=2)
        ax2.set_title('Mean Absolute Error vs Epochs')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('MAE (m/h)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    print("[OK] Saved learning_curves.png")
    plt.close()

def create_accuracy_report(data):
    """Generate Detailed Accuracy Report and Residual Plots"""
    predictions = {k: v for k, v in data.items() if 'Predictions' in k}
    
    if not predictions:
        return

    n_preds = len(predictions)
    fig = plt.figure(figsize=(15, 5 * n_preds))
    fig.suptitle('Model Evaluation & Residual Analysis', fontsize=20, y=0.98)
    
    for i, (name, df) in enumerate(predictions.items()):
        well_name = name.replace('_Predictions', '')
        row = i
        
        # Calculate Metrics
        y_true = df['ROP_Actual']
        y_pred = df['ROP_Predicted']
        residuals = y_true - y_pred
        
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # 1. Predicted vs Actual
        ax1 = plt.subplot(n_preds, 2, 2*i + 1)
        ax1.scatter(y_true, y_pred, alpha=0.5, color='blue')
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Fit')
        
        title_text = f"{well_name}\nR² = {r2:.3f} | MAE = {mae:.2f} | RMSE = {rmse:.2f}"
        ax1.set_title(title_text, fontweight='bold')
        ax1.set_xlabel('Actual ROP')
        ax1.set_ylabel('Predicted ROP')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Residuals vs Predicted (Homoscedasticity Check)
        ax2 = plt.subplot(n_preds, 2, 2*i + 2)
        ax2.scatter(y_pred, residuals, alpha=0.5, color='purple')
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_title(f'Residuals vs Predicted ({well_name})')
        ax2.set_xlabel('Predicted ROP')
        ax2.set_ylabel('Residuals (Actual - Predicted)')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_accuracy_report.png')
    print("[OK] Saved model_accuracy_report.png")
    plt.close()

def calculate_physics_features(df):
    """Calculate derived engineering features (Physics-Informed)"""
    df_feat = df.copy()
    bit_dia = 12.25
    bit_area = np.pi * (bit_dia ** 2) / 4
    
    # Check for required columns (case insensitive)
    cols = {c.upper(): c for c in df_feat.columns}
    
    wob_col = cols.get('WOB')
    torque_col = cols.get('TORQUE')
    rpm_col = cols.get('RPM')
    rop_col = cols.get('ROP')
    
    if all([wob_col, torque_col, rpm_col, rop_col]):
        # Avoid division by zero
        rop_clean = df_feat[rop_col].replace(0, 0.1)
        
        df_feat['MSE'] = (
            (df_feat[wob_col] * 1000 / bit_area) +
            (480 * df_feat[rpm_col] * df_feat[torque_col]) / 
            (bit_area * rop_clean)
        )
    return df_feat

def load_well_data():
    """Load all sample well data and predictions"""
    data = {}
    
    # 1. Load Well Data
    data_dir = Path('sample_data/time_data')
    if data_dir.exists():
        for file_path in data_dir.glob('*.csv'):
            try:
                well_name = file_path.stem
                df = pd.read_csv(file_path)
                data[well_name] = df
                print(f"[INFO] Loaded {well_name}: {len(df)} records")
            except Exception as e:
                print(f"[ERROR] Loading {file_path}: {e}")
    else:
        print(f"[WARNING] Data directory not found: {data_dir}")

    # 2. Load Predictions
    # Look for predictions matching our wells
    outputs_dir = Path('outputs')
    if outputs_dir.exists():
        for well_name in list(data.keys()):
            # Find latest prediction for this well
            pattern = f"predictions_{well_name}_*.csv"
            pred_files = list(outputs_dir.glob(pattern))
            if pred_files:
                latest_pred = max(pred_files, key=lambda p: p.stat().st_mtime)
                try:
                    pred_df = pd.read_csv(latest_pred)
                    data[f"{well_name}_Predictions"] = pred_df
                    print(f"[INFO] Loaded predictions for {well_name}")
                except Exception as e:
                    print(f"[ERROR] Loading prediction {latest_pred}: {e}")

    return data

def create_comprehensive_dashboard(data):
    """Create a summary dashboard for all wells"""
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle('ROP Prediction System Dashboard (Sample Data)', fontsize=20, fontweight='bold')
    
    # 1. ROP Distribution Comparison
    ax1 = plt.subplot(3, 3, 1)
    for name, df in data.items():
        if 'Predictions' not in name:
            sns.kdeplot(data=df, x='ROP', label=name, ax=ax1, fill=True, alpha=0.3)
    ax1.set_title('ROP Distribution by Well')
    ax1.set_xlabel('ROP (m/h)')
    ax1.legend()
    
    # 2. WOB vs ROP
    ax2 = plt.subplot(3, 3, 2)
    for name, df in data.items():
        if 'Predictions' not in name:
             # Sample for scatter plot performance
            subset = df.sample(min(len(df), 500))
            ax2.scatter(subset['WOB'], subset['ROP'], label=name, alpha=0.5, s=20)
    ax2.set_title('WOB vs ROP')
    ax2.set_xlabel('WOB (klbs)')
    ax2.set_ylabel('ROP (m/h)')
    ax2.legend()
    
    # 3. Prediction Accuracy (If available)
    ax3 = plt.subplot(3, 3, 3)
    pred_errors = []
    labels = []
    for name, df in data.items():
        if 'Predictions' in name:
            if 'Error' in df.columns:
                pred_errors.append(df['Error'])
                labels.append(name.replace('_Predictions', ''))
            elif 'ROP_Actual' in df.columns and 'ROP_Predicted' in df.columns:
                err = df['ROP_Actual'] - df['ROP_Predicted']
                pred_errors.append(err)
                labels.append(name.replace('_Predictions', ''))
                
    if pred_errors:
        ax3.boxplot(pred_errors, labels=labels)
        ax3.set_title('Prediction Error Distribution')
        ax3.set_ylabel('Error (Actual - Predicted)')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No Predictions Loaded', ha='center', va='center')
        
    # 4. MSE Analysis
    ax4 = plt.subplot(3, 3, 4)
    for name, df in data.items():
        if 'Predictions' not in name:
            df_phys = calculate_physics_features(df)
            if 'MSE' in df_phys.columns:
                subset = df_phys.sample(min(len(df), 500))
                ax4.scatter(subset['MSE'], subset['ROP'], label=name, alpha=0.5, s=20)
    ax4.set_title('MSE vs ROP (Physics Check)')
    ax4.set_xlabel('MSE (kpsi)')
    ax4.set_ylabel('ROP (m/h)')
    ax4.set_xlim(0, 200) # trim outliers
    # ax4.legend()
    
    # 5. Prediction Time Series (Example for first well)
    ax5 = plt.subplot(3, 1, 3) # Full width bottom
    found_pred = False
    for name, df in data.items():
        if 'Predictions' in name:
            # Sort by Depth or Time? Assuming Index or Depth
            if 'Depth' in df.columns:
                idx = df['Depth']
                xlabel = 'Depth (m)'
            else:
                idx = range(len(df))
                xlabel = 'Sample Index'
                
            ax5.plot(idx, df['ROP_Actual'], 'b-', label='Actual', alpha=0.7)
            ax5.plot(idx, df['ROP_Predicted'], 'r--', label='Predicted', alpha=0.7)
            ax5.set_title(f'Prediction Performance: {name.replace("_Predictions", "")}')
            ax5.set_xlabel(xlabel)
            ax5.set_ylabel('ROP (m/h)')
            ax5.legend()
            found_pred = True
            break
            
    if not found_pred:
        ax5.text(0.5, 0.5, 'No Prediction Data', ha='center')

    plt.tight_layout()
    plt.savefig('enhanced_rop_prediction_dashboard_new.png')
    print("[OK] Dashboard saved: enhanced_rop_prediction_dashboard_new.png")

def create_individual_well_plots(data):
    """Create individual plots"""
    for name, df in data.items():
        if 'Predictions' in name: continue
        
        # Check if we have predictions for this well
        pred_key = f"{name}_Predictions"
        pred_df = data.get(pred_key)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Analysis: {name}', fontsize=16)
        
        # 1. Depth Trends
        ax1 = axes[0, 0]
        ax1.plot(df['MD'], df['ROP'], 'b-', label='ROP')
        ax1.set_xlabel('Depth')
        ax1.set_ylabel('ROP')
        ax1.set_title('ROP vs Depth')
        
        # 2. Physics
        ax2 = axes[0, 1]
        df_phys = calculate_physics_features(df)
        if 'MSE' in df_phys.columns:
             sc = ax2.scatter(df_phys['MSE'], df['ROP'], c=df['MD'], cmap='viridis', s=10)
             plt.colorbar(sc, ax=ax2, label='Depth')
             ax2.set_xlabel('MSE')
             ax2.set_ylabel('ROP')
             ax2.set_title('MSE Analysis')
             ax2.set_xlim(0, 300)
        
        # 3. Parameters
        ax3 = axes[1, 0]
        ax3.scatter(df['WOB'], df['ROP'], alpha=0.3, color='g')
        ax3.set_xlabel('WOB')
        ax3.set_ylabel('ROP')
        ax3.set_title('WOB vs ROP')
        
        # 4. Predictions if available
        ax4 = axes[1, 1]
        if pred_df is not None:
            ax4.scatter(pred_df['ROP_Actual'], pred_df['ROP_Predicted'], alpha=0.5, color='purple')
            ax4.plot([0, 100], [0, 100], 'r--')
            ax4.set_xlabel('Actual')
            ax4.set_ylabel('Predicted')
            ax4.set_title('Prediction Accuracy')
            ax4.grid(True)
        else:
            ax4.text(0.5, 0.5, 'No Predictions', ha='center')
            
        plt.tight_layout()
        plt.savefig(f'{name}_analysis.png')
        print(f"[OK] Saved {name}_analysis.png")
        plt.close()

def main():
    print("Running new diagram generation script...")
    data = load_well_data()
    if not data:
        print("No data found!")
        return
        
    create_comprehensive_dashboard(data)
    
    # New Visualization Functions
    print("\nGenerating advanced analysis...")
    history = load_latest_training_history()
    create_learning_curves(history)
    create_accuracy_report(data)
    
    create_individual_well_plots(data)
    print("Done.")

if __name__ == '__main__':
    main()
