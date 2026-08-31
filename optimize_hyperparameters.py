import os
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from rop_prediction_system import (
    ROPPredictionSystem, 
    DEFAULT_CONFIG, 
    CBT_LSTMTrainer,
    create_cbt_lstm_model
)
from sklearn.model_selection import train_test_split
from datetime import datetime
import json
import warnings
import joblib
from copy import deepcopy

# Configuration
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODEL_DIR = 'models'
REPORT_FILE = 'OPTIMIZATION_REPORT.md'

def get_data_paths():
    """Discover all enhanced data files."""
    base_dir = r"enhanced_sample_data"
    wells = ["Enhanced_Well_1", "Enhanced_Well_2", "Enhanced_Well_3"]
    
    time_files = []
    log_files = []
    
    for w in wells:
        t_path = os.path.join(base_dir, "time_data", f"{w}.csv")
        l_path = os.path.join(base_dir, "log_data", f"{w}.csv")
        
        if os.path.exists(t_path):
            time_files.append(t_path)
            log_files.append(l_path if os.path.exists(l_path) else None)
            
    return time_files, log_files, wells

def load_and_preprocess_data():
    """Load, merge, and return processed sequences ready for training."""
    print("[INIT] Loading and preprocessing data for optimization...")
    time_files, log_files, wells = get_data_paths()
    
    # We use the system class just for data processing logic
    system = ROPPredictionSystem(DEFAULT_CONFIG)
    
    # 1. Process Raw Data
    if not time_files:
        raise ValueError("No time data files found in enhanced_sample_data/time_data/")

    _, combined_df = system.data_processor.load_and_process(time_files, log_files, wells)
    
    # 2. Select Features (Standard Logic)
    exclude_cols = ['WELL_NAME', 'ROP', 'timestamp', 'TIME', 'DATE', 'MD']
    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
    candidate_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Variance Thresholding
    variances = combined_df[candidate_cols].var().sort_values(ascending=False)
    feature_columns = variances.head(20).index.tolist()
    
    # 3. Create Sequences
    X, y, groups = system.data_processor.create_sequences(combined_df, feature_columns)
    
    print(f"[DATA] X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y, feature_columns

def random_grid_search(X, y, n_trials=5):
    """Perform Random Search over hyperparameter grid."""
    
    # Define Search Space
    param_grid = {
        'lstm_units': [64, 128, 256],
        'dense_units': [32, 64, 128],
        'dropout_rate': [0.1, 0.2, 0.3, 0.5],
        'learning_rate': [0.001, 0.0005, 0.0001],
        'batch_size': [16, 32, 64]
    }
    
    # Split Data (Static split for all trials)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    best_mae = float('inf')
    best_config = None
    results_log = []
    
    print(f"\n[SEARCH] Starting Grid Search ({n_trials} trials)...")
    
    for i in range(n_trials):
        # Sample parameters - Explicit casting to native types for JSON serialization
        trial_params = {
            'lstm_units': int(np.random.choice(param_grid['lstm_units'])),
            'dense_units': int(np.random.choice(param_grid['dense_units'])),
            'dropout_rate': float(np.random.choice(param_grid['dropout_rate'])),
            'learning_rate': float(np.random.choice(param_grid['learning_rate'])),
            'batch_size': int(np.random.choice(param_grid['batch_size']))
        }
        
        # Prepare Config
        config = deepcopy(DEFAULT_CONFIG)
        config['model']['cbt_params']['lstm_units'] = int(trial_params['lstm_units'])
        config['model']['cbt_params']['dense_units'] = int(trial_params['dense_units'])
        config['model']['cbt_params']['dropout_rate'] = float(trial_params['dropout_rate'])
        config['model']['cbt_params']['learning_rate'] = float(trial_params['learning_rate'])
        config['model']['training']['batch_size'] = int(trial_params['batch_size']) # Update batch size in config
        
        print(f"\n--- Trial {i+1}/{n_trials} ---")
        print(json.dumps(trial_params, indent=2))
        
        # Instantiate Trainer & Build Model
        # trainer = CBT_LSTMTrainer(config) # Not needed for simple build
        
        # Force build model manually to inject params
        model = create_cbt_lstm_model(
            input_shape=(X_train.shape[1], X_train.shape[2]), 
            config=config
        )
        
        # Short Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10, # Fast check
            batch_size=int(trial_params['batch_size']),
            verbose=0
        )
        
        val_mae = history.history['val_mae'][-1]
        print(f"Result: Val MAE (Scaled) = {val_mae:.4f}")
        
        results_log.append({
            'params': trial_params,
            'mae': val_mae
        })
        
        if val_mae < best_mae:
            best_mae = val_mae
            best_config = config
            print(">>> NEW BEST!")
            
    return best_config, results_log

def train_final_model(best_config, X, y, feature_cols):
    print("\n[FINAL] Training Final Model with Best Hyperparameters...")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    # System
    system = ROPPredictionSystem(best_config)
    system.feature_columns = feature_cols # Inject chosen features
    
    # Update epochs for final training in config directly
    system.config['model']['training']['epochs'] = 30
    
    # Train
    # Correct signature: train(self, X_train, y_train, X_val, y_val, input_shape, feature_columns)
    system.trainer.train(
        X_train, y_train, 
        X_test, y_test, 
        input_shape=(X_train.shape[1], X_train.shape[2]),
        feature_columns=feature_cols
    )
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", f"cbt_lstm_OPTIMIZED_{timestamp}.h5")
    system.trainer.model.save(model_path)
    
    # Use existing scalers logic
    # The last call to create_sequences saved scalers with current timestamp.
    # We need to manually duplicate them for this new model name so predict() works.
    
    # Hack: Find most recent scaler files
    scaler_files = sorted(glob.glob('models/scaler_*.pkl'))
    if scaler_files:
        latest_feat_scaler = [f for f in scaler_files if 'feat' in f][-1]
        latest_targ_scaler = [f for f in scaler_files if 'target' in f][-1]
        
        new_feat_path = model_path.replace('.h5', '.pkl').replace('cbt_lstm_', 'scaler_feat_')
        new_targ_path = model_path.replace('.h5', '.pkl').replace('cbt_lstm_', 'scaler_target_')
        
        try:
            import shutil
            shutil.copy2(latest_feat_scaler, new_feat_path)
            shutil.copy2(latest_targ_scaler, new_targ_path)
        except Exception as e:
            print(f"Warning: Could not copy scalers: {e}")

    # Metadata
    metadata = {
        'feature_columns': feature_cols,
        'config': best_config,
        'performance': {
            'val_mae': float(system.trainer.model.evaluate(X_test, y_test, verbose=0)[1])
        },
        'scaler_paths': {
            'feature': new_feat_path if 'new_feat_path' in locals() else "",
            'target': new_targ_path if 'new_targ_path' in locals() else ""
        }
    }
    meta_path = model_path.replace('.h5', '.json').replace('cbt_lstm_', 'metadata_')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    return model_path, metadata['performance']['val_mae']

def write_report(results_log, best_config, final_model_path, final_mae):
    with open(REPORT_FILE, 'w') as f:
        f.write("# Hyperparameter Optimization Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## 1. Optimization Results\n")
        f.write("| Trial | LSTM Units | Dense Units | Dropout | LR | Batch | Val MAE (Scaled) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        
        for i, res in enumerate(results_log):
            p = res['params']
            f.write(f"| {i+1} | {p['lstm_units']} | {p['dense_units']} | {p['dropout_rate']} | {p['learning_rate']} | {p['batch_size']} | **{res['mae']:.4f}** |\n")
            
        f.write("\n## 2. Best Configuration\n")
        f.write("```json\n")
        # Extract relevant cbt_params
        f.write(json.dumps(best_config['model']['cbt_params'], indent=4))
        f.write("\n```\n\n")
        
        f.write("## 3. Deployment\n")
        f.write(f"- **Final Model Saved:** `{final_model_path}`\n")
        f.write(f"- **Final Test MAE (Scaled):** {final_mae:.4f}\n")
        f.write("- **Recommendation:** Use this model for future predictions if Test MAE < 0.1 (good for scaled data).\n")
        
    print(f"\n[REPORT] Generated {REPORT_FILE}")

def main():
    X, y, feature_cols = load_and_preprocess_data()
    best_config, results_log = random_grid_search(X, y, n_trials=5)
    final_path, final_mae = train_final_model(best_config, X, y, feature_cols)
    write_report(results_log, best_config, final_path, final_mae)

if __name__ == "__main__":
    main()
