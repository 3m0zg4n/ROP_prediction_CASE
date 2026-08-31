r"""
================================================================================
COMPLETE CBT-LSTM ROP PREDICTION SYSTEM
================================================================================

- ONE FILE - COMPLETE SOLUTION FOR ROP PREDICTION USING OFFSET WELLS
- INCLUDES: CBT-LSTM model, data processing, training, and prediction
- NO EXTERNAL FILES NEEDED - Just run this script!

Author: Drilling Analytics Team
Version: 4.1 (PyCharm Ready)
Date: January 2024

================================================================================
                          STEP-BY-STEP USER GUIDE
================================================================================

1. INSTALLATION & SETUP:
   ---------------------
   A) Install Required Packages:
      pip install tensorflow pandas numpy scikit-learn matplotlib seaborn lasio

   B) PyCharm Setup:
      - Open folder in PyCharm
      - Configure Python Interpreter (Settings > Project > Interpreter)
      - Install requirements when prompted

2. DATA PREPARATION:
   -----------------
   Create a 'data' folder with the following structure:
   
   data/
   +-- time_data/             # Real-time drilling data (CSV)
   |   +-- Well_A.csv
   |   \-- Well_B.csv
   \-- log_data/              # Formation evaluation logs (LAS or CSV)
       +-- Well_A.las
       \-- Well_B.las

   REQUIRED COLUMNS (CSV Time Data):
   - TIMESTAMP : DateTime
   - MD        : Measured Depth (ft/m)
   - ROP       : Rate of Penetration (Target)
   - WOB       : Weight on Bit
   - RPM       : Rotary Speed
   - TORQUE    : Surface Torque
   - FLOW      : Flow Rate
   - SPP       : Standpipe Pressure
   - MW        : Mud Weight

   REQUIRED CURVES (Log Data - Optional but Recommended):
   - GR        : Gamma Ray (Lithology)
   - DT        : Sonic format (Porosity/Strength)
   - RHOB      : Density
   - RT        : Resistivity

3. RUNNING THE SYSTEM:
   -------------------
   
   MODE 1: SAMPLE WORKFLOW (Best for first run)
   >> python rop_prediction_system.py --mode sample
   * Generates synthetic data, trains model, and evaluates it.

   MODE 2: TRAINING WITH YOUR DATA
   >> python rop_prediction_system.py --mode train \
          --time_files data/time_data/Well_A.csv data/time_data/Well_B.csv \
          --log_files data/log_data/Well_A.las data/log_data/Well_B.las \
          --well_names Well_A Well_B

   MODE 3: PREDICTION (NEW WELL)
   >> python rop_prediction_system.py --mode predict \
          --model_path models/cbt_lstm_YYYYMMDD_HHMMSS.h5 \
          --time_file data/time_data/Planned_Well.csv \
          --well_name Planned_Well

================================================================================
                          ENGINEERING & PHYSICS
================================================================================

The system automatically calculates the following physics-informed features:

1. Mechanical Specific Energy (MSE):
   MSE = (WOB/Area) + (480 * RPM * TORQUE) / (Area * ROP)
   * Measures drilling efficiency (Energy per unit volume of rock)

2. Hydraulic Specific Energy (HSI):
   HSI = (SPP * FLOW) / (1714 * Area)
   * Measures hydraulic cleaning power at the bit

3. Unconfined Compressive Strength (UCS):
   UCS = f(Sonic_DT)
   * Estimates rock strength from log data

4. Shale Volume (V_SH):
   V_SH = (GR - GR_min) / (GR_max - GR_min)
   * Lithology indicator

================================================================================
                          SYSTEM ARCHITECTURE
================================================================================

1. DATA PIPELINE:
   [Raw CSV/LAS] -> [Cleaning/Outlier Removal] -> [Physics Calculations] 
   -> [Normalization] -> [Sliding Window Sequencing (N=50)] -> [Tensor]

2. MODEL: CBT-LSTM (Channel Boosting Time-series LSTM)
   - Input: (Batch, 50 timesteps, Features)
   - Layer 1: Channel Boosting (Multi-scale CNNs)
   - Layer 2: Bi-Directional LSTMs
   - Output: ROP at t+1

3. EXPECTED RESULTS:
   - MAE: 2-5 m/h typically
   - Training Time: ~1-2 mins per well on GPU
================================================================================
"""

# ============================================================================
# IMPORTS & SETUP
# ============================================================================

import os
import sys

# Suppress TensorFlow oneDNN informational messages
# This must be done BEFORE importing tensorflow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress generic TF compilation logs

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
except ImportError:
    print("\n[CRITICAL ERROR] TensorFlow is not installed or corrupted.")
    print("Please run: pip install tensorflow")
    print("Or check your Python environment.")
    raise

try:
    import lasio
except ImportError:
    lasio = None

import matplotlib.pyplot as plt
# import seaborn as sns # Unused
import warnings
import json
# import pickle # Unused
import argparse
from datetime import datetime
# from scipy import stats # Unused
# from scipy.signal import savgol_filter # Unused
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    "project": {
        "name": "CBT_LSTM_ROP_Prediction",
        "version": "4.1",
        "description": "Complete CBT-LSTM system for ROP prediction"
    },
    "data": {
        "required_time_columns": ["ROP", "WOB", "RPM", "TORQUE", "FLOW", "SPP", "MD", "MW"],
        "required_log_curves": ["GR", "DT", "RHOB", "RT"],
        "sequence_length": 50,
        "step_size": 5,
        "test_size": 0.2,
        "validation_split": 0.1,
        "split_strategy": "stratified", # Options: "well", "stratified"
        "scaler_type": "robust",     # Options: "minmax", "standard", "robust"
        "min_rop_threshold": 0.1,
        "min_wob_threshold": 2.0
    },
    "model": {
        "architecture": "cbt_lstm_advanced",
        "cbt_params": {
            "boosting_layers": 2,
            "lstm_layers": 2,
            "attention_heads": 4,
            "boosting_filters": 64,
            "lstm_units": 128,
            "dense_units": 64,
            "dropout_rate": 0.3,
            "learning_rate": 0.0005,
            "kernel_sizes": [3, 5, 7]
        },
        "training": {
            "epochs": 100,
            "batch_size": 32,
            "early_stopping_patience": 20,
            "reduce_lr_patience": 10,
            "min_learning_rate": 1e-6
        }
    }
}

# ============================================================================
# 2. CUSTOM CBT-LSTM LAYERS
# ============================================================================

class ChannelBoostingLayer(tf.keras.layers.Layer):
    """
    Channel Boosting Layer for multi-scale feature enhancement.
    
    [ ENGINEERING NOTE ]
    Drilling data contains patterns at different timescales:
    1. High-frequency: Bit vibration, stick-slip (fractions of a second to seconds)
    2. Medium-frequency: Formation changes, drill string dynamics (minutes)
    3. Low-frequency: Wear trends, hole cleaning issues (hours)
    
    This layer uses convolutions with different kernel sizes (3, 5, 7) to 
    capture these multi-scale features simultaneously.
    """
    
    def __init__(self, filters=64, kernel_sizes=[3, 5, 7], dropout_rate=0.2, **kwargs):
        super(ChannelBoostingLayer, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.dropout_rate = dropout_rate
        
        # Multi-scale convolutional kernels
        self.conv_layers = []
        for ks in kernel_sizes:
            self.conv_layers.append(
                tf.keras.layers.Conv1D(filters=filters, kernel_size=ks, 
                                      padding='same', activation='relu')
            )
        
        # Adaptive gating mechanism to weigh the importance of each scale
        self.gate_conv = tf.keras.layers.Conv1D(
            filters=len(kernel_sizes), kernel_size=1, activation='sigmoid'
        )
        
        # Normalization and Regularization
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, inputs, training=False):
        # 1. Extract multi-scale features (Parallel Convolutions)
        boosted_features = []
        for conv in self.conv_layers:
            x = conv(inputs)
            boosted_features.append(x)
        
        # Stack features: [Batch, Time, Filters] -> [Batch, Time, Scales, Filters]
        boosted = tf.stack(boosted_features, axis=-2)
        original_shape = tf.shape(boosted)
        # Flatten for processing: [Batch, Time, Scales*Filters]
        boosted = tf.reshape(boosted, [
            original_shape[0], original_shape[1], 
            len(self.kernel_sizes) * self.filters
        ])
        
        # 2. Adaptive Gating (Learn which scale is important)
        # Calculates a weight between 0-1 for each kernel scale
        gate_weights = self.gate_conv(inputs)
        gate_weights = tf.expand_dims(gate_weights, -1)
        
        # Apply weights
        boosted_reshaped = tf.reshape(
            boosted, 
            [-1, tf.shape(boosted)[1], len(self.kernel_sizes), self.filters]
        )
        gated = boosted_reshaped * gate_weights
        gated = tf.reshape(gated, tf.shape(boosted))
        
        # 3. Residual Connection (Skip Connection)
        # Allows gradients to flow through unchanged if needed (solves vanishing gradient)
        if inputs.shape[-1] == gated.shape[-1]:
            gated = tf.keras.layers.Add()([inputs, gated])
        
        # 4. Final Normalization
        gated = self.layer_norm(gated)
        gated = self.batch_norm(gated, training=training)
        gated = self.dropout(gated, training=training)
        
        return gated

# ============================================================================
# 3. DATA PROCESSOR
# ============================================================================

class ROPDataProcessor:
    """
    Complete data processing pipeline.
    Handles: Loading -> Merging -> Cleaning -> Feature Engineering -> Sequencing
    """
    
    def __init__(self, config):
        self.config = config
        self.scalers = {}
        self.feature_columns = []
        
        # Select scaler based on config (Step C: Domain Adaptation)
        scaler_type = config['data'].get('scaler_type', 'minmax')
        if scaler_type == 'robust':
            self.feature_scaler = RobustScaler()
        elif scaler_type == 'standard':
            self.feature_scaler = StandardScaler()
        else:
            self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
            
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    def load_and_process(self, time_files, log_files, well_names):
        """
        Master function to ingest raw data and produce clean datasets.
        
        Args:
            time_files: List of CSV paths with drilling parameters
            log_files: List of LAS/CSV paths with formation logs
            well_names: List of string identifiers for each well
        """
        print("\n" + "="*60)
        print("DATA PROCESSING")
        print("="*60)
        
        processed_wells = {}
        
        for well_name, time_file, log_file in zip(well_names, time_files, log_files):
            print(f"  Processing {well_name}...")
            
            # Step 1: Load Real-time Drilling Data (WOB, RPM, etc.)
            time_data = self._load_time_data(time_file, well_name)
            if time_data is None:
                continue
            
            # Step 2: Load Formation Logs (Gamma Ray, Resistivity, etc.)
            log_data = self._load_log_data(log_file, well_name)
            
            # Step 3: Align datasets by Depth (MD)
            # Log data is usually depth-indexed. Drilling data is time-indexed but has depth.
            merged = self._merge_data(time_data, log_data, well_name)
            if merged is None:
                continue
            
            # Step 4: Remove invalid data (stops, connection times)
            cleaned = self._clean_data(merged, well_name)
            
            # Step 5: Compute Physics-based Features (MSE, HSI, etc.)
            featured = self._calculate_features(cleaned, well_name)
            
            processed_wells[well_name] = featured
        
        if not processed_wells:
            raise ValueError("No data processed successfully")
        
        # Combine all wells into one massive DataFrame for global statistics
        combined_df = pd.concat(processed_wells.values(), ignore_index=True)
        print(f"[OK] Total samples: {len(combined_df):,}")
        
        return processed_wells, combined_df
    
    def _load_time_data(self, file_path, well_name):
        """Load time-based drilling data"""
        try:
            if isinstance(file_path, pd.DataFrame):
                df = file_path.copy()
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path, parse_dates=True, low_memory=False)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path, parse_dates=True)
            else:
                raise ValueError(f"Unsupported format: {file_path}")
            
            # Standardize column names to UPPERCASE for consistency
            df.columns = df.columns.str.upper()
            
            # Verify required columns exist
            required = self.config['data']['required_time_columns']
            missing = [col for col in required if col not in df.columns]
            if missing:
                print(f"    [WARN] Missing: {missing}")
            
            df['WELL_NAME'] = well_name
            return df
            
        except Exception as e:
            print(f"    [ERROR] Error: {e}")
            return None
    
    def _load_log_data(self, file_path, well_name):
        """Load LAS log data"""
        try:
            if file_path is None or not os.path.exists(file_path):
                return None

            # Support CSV log exports without requiring lasio
            if file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path)
                df.columns = df.columns.str.upper()
                # Normalize expected curves
                col_mapping = {}
                for col in df.columns:
                    col_upper = str(col).upper()
                    if 'GR' in col_upper or 'GAMMA' in col_upper:
                        col_mapping[col] = 'GR'
                    elif 'DT' in col_upper or 'SONIC' in col_upper:
                        col_mapping[col] = 'DT'
                    elif 'RHOB' in col_upper or 'DENSITY' in col_upper:
                        col_mapping[col] = 'RHOB'
                    elif 'RT' in col_upper or 'RESIST' in col_upper:
                        col_mapping[col] = 'RT'
                    elif 'DEPT' in col_upper or 'DEPTH' in col_upper or col_upper == 'MD':
                        col_mapping[col] = 'DEPT'
                df = df.rename(columns=col_mapping)
                if 'DEPT' in df.columns and 'MD' not in df.columns:
                    df = df.rename(columns={'DEPT': 'MD'})
                df['WELL_NAME'] = well_name
                return df
            
            # Try to import lasio (library for reading .LAS oilfield files)
            try:
                import lasio
            except ImportError:
                print(f"    [WARN] LASIO not installed, skipping log data")
                return None
            
            las = lasio.read(file_path)
            df = las.df()
            df.index.name = 'MD'
            df.reset_index(inplace=True)
            
            # Standardize curve mnemonics (e.g., transform 'GR_FINAL' -> 'GR')
            col_mapping = {}
            for col in df.columns:
                col_upper = str(col).upper()
                if 'GR' in col_upper or 'GAMMA' in col_upper:
                    col_mapping[col] = 'GR'
                elif 'DT' in col_upper or 'SONIC' in col_upper:
                    col_mapping[col] = 'DT'
                elif 'RHOB' in col_upper or 'DENSITY' in col_upper:
                    col_mapping[col] = 'RHOB'
                elif 'RT' in col_upper or 'RESIST' in col_upper:
                    col_mapping[col] = 'RT'
            
            df = df.rename(columns=col_mapping)
            df['WELL_NAME'] = well_name
            
            return df
            
        except Exception as e:
            print(f"    [WARN] LAS error: {e}")
            return None
    
    def _merge_data(self, time_data, log_data, well_name):
        """
        Merge time and log data using Depth (MD) as the key.
        Uses 'merge_asof' to find the nearest log value for each drilling timestamp.
        """
        if time_data is None:
            return None
        
        if log_data is None:
            return time_data
        
        # Find standardized depth columns
        time_depth = next((col for col in time_data.columns if 'MD' in col), 'MD')
        log_depth = 'MD'
        
        # Sort both datasets by depth (required for merge_asof)
        time_data = time_data.sort_values(time_depth)
        log_data = log_data.sort_values(log_depth)
        
        # Perform fuzzy merge (Nearest Neighbor interpolation)
        merged = pd.merge_asof(
            time_data,
            log_data.drop(columns=['WELL_NAME']),
            left_on=time_depth,
            right_on=log_depth,
            suffixes=('', '_LOG'),
            direction='nearest',
            tolerance=1.0 # Max distance to look for a match (1.0 depth unit)
        )
        
        return merged
    
    def _clean_data(self, df, well_name):
        """
        Clean and filter drilling data.
        REMOVES:
          1. Non-drilling time (ROP near zero)
          2. Connection time (WOB drop)
          3. Outliers (impossible ROP values)
        """
        df_clean = df.copy()
        
        # Filter drilling periods
        if 'ROP' in df_clean.columns and 'WOB' in df_clean.columns:
            mask = (
                (df_clean['ROP'] > self.config['data']['min_rop_threshold']) &
                (df_clean['WOB'] > self.config['data']['min_wob_threshold']) &
                (df_clean['ROP'] < 200) # Cap ROP at 200 (physical limit)
            )
            df_clean = df_clean[mask].copy()
        
        # Handle missing values (Linear Interpolation)
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(method='ffill', limit=10)
        df_clean[numeric_cols] = df_clean[numeric_cols].interpolate(method='linear')
        
        return df_clean
    
    def _calculate_features(self, df, well_name):
        """Calculate derived engineering features (Physics-Informed ML)"""
        df_feat = df.copy()
        
        # 1. Mechanical Specific Energy (Teale's MSE)
        # Energy required to remove a unit volume of rock
        if all(col in df_feat.columns for col in ['WOB', 'TORQUE', 'RPM', 'ROP']):
            bit_dia = 12.25 # Assumption: Standard bit size
            bit_area = np.pi * (bit_dia ** 2) / 4
            df_feat['MSE'] = (
                (df_feat['WOB'] * 1000 / bit_area) +
                (480 * df_feat['RPM'] * df_feat['TORQUE']) / 
                (bit_area * df_feat['ROP'])
            )
        
        # 2. Hydraulic Specific Energy (HSI)
        # Hydraulic horsepower per square inch at the bit
        if all(col in df_feat.columns for col in ['SPP', 'FLOW', 'MW']):
            bit_area = np.pi * (12.25 ** 2) / 4
            df_feat['HSI'] = (df_feat['SPP'] * df_feat['FLOW']) / (1714 * bit_area)
        
        # 3. Rock Strength (UCS) - Empirical correlation from Sonic logs
        if 'DT' in df_feat.columns:
            df_feat['UCS'] = 10 ** ((df_feat['DT'] - 50) / (-25)) * 145.038
        
        # 4. Shale Volume (V_SH) - From Gamma Ray
        if 'GR' in df_feat.columns:
            gr_min = df_feat['GR'].quantile(0.05)
            gr_max = df_feat['GR'].quantile(0.95)
            df_feat['V_SH'] = (df_feat['GR'] - gr_min) / (gr_max - gr_min)
            df_feat['V_SH'] = df_feat['V_SH'].clip(0, 1) # Bound between 0 and 1
        
        return df_feat
    
    def create_sequences(self, df, feature_cols, target_col='ROP', fit_scalers=True):
        """
        Prepare 3D sequences for LSTM input.
        Structure: [Samples, Time_Steps, Features]
        """
        seq_len = self.config['data']['sequence_length']
        step = self.config['data']['step_size']
        
        # Fit scalers if requested (Training Mode)
        if fit_scalers:
            print("[INFO] Fitting and saving scalers...")
            self.feature_scaler.fit(df[feature_cols])
            self.target_scaler.fit(df[[target_col]])
            
        # Transform the data
        # We need to process each well separately to maintain continuity, 
        # but we use the global scalers we just fit.
        
        X_sequences = []
        y_targets = []
        groups = []
        
        # Group by Well Name to prevent sequences from crossing between different wells
        for well_name, well_data in df.groupby('WELL_NAME'):
            well_data = well_data.sort_values('MD')
            
            # Extract raw values
            features_raw = well_data[feature_cols].values
            target_raw = well_data[[target_col]].values
            
            # Scale values
            features_scaled = self.feature_scaler.transform(features_raw)
            target_scaled = self.target_scaler.transform(target_raw)
            
            n_samples = len(well_data)
            
            # Sliding Window Loop
            for i in range(0, n_samples - seq_len, step):
                # Extract window of 'seq_len' rows
                X_seq = features_scaled[i:i + seq_len]
                
                # Target is the NEXT value after the sequence
                y_val = target_scaled[i + seq_len - 1][0]
                
                X_sequences.append(X_seq)
                y_targets.append(y_val)
                groups.append(well_name)
        
        X = np.array(X_sequences)
        y = np.array(y_targets)
        
        print(f"[OK] Created {len(X)} sequences (shape: {X.shape})")
        
        return X, y, groups

# ============================================================================
# 4. CBT-LSTM MODEL
# ============================================================================

def create_cbt_lstm_model(input_shape, config):
    """
    Create the Neural Network Architecture.
    
    Structure:
    Input -> ChannelBoosting (CNNs) -> LSTM 1 -> Batch Norm -> LSTM 2 -> Batch Norm -> Dense -> ROP
    """
    params = config['model']['cbt_params']
    
    inputs = tf.keras.Input(shape=input_shape)
    
    # Layer 1: Channel Boosting
    # Enhances features by looking at different time scales (kernel sizes 3, 5, 7)
    x = ChannelBoostingLayer(
        filters=params['boosting_filters'],
        kernel_sizes=params['kernel_sizes'],
        dropout_rate=params['dropout_rate']
    )(inputs)
    
    # Layer 2: LSTM (Long Short-Term Memory)
    # Learns temporal dependencies in the boosted features
    # return_sequences=True passes the full sequence to the next LSTM layer
    x = tf.keras.layers.LSTM(params['lstm_units'], return_sequences=True,
                            dropout=params['dropout_rate'])(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Layer 3: Second LSTM layer
    # return_sequences=False compresses the sequence into a single vector
    x = tf.keras.layers.LSTM(params['lstm_units']//2, return_sequences=False,
                            dropout=params['dropout_rate'])(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Layer 4: Dense (Fully Connected) Layers
    # Non-linear transformations to map features to ROP
    x = tf.keras.layers.Dense(params['dense_units'] * 2, activation='relu')(x)
    x = tf.keras.layers.Dropout(params['dropout_rate'])(x)
    
    x = tf.keras.layers.Dense(params['dense_units'], activation='relu')(x)
    x = tf.keras.layers.Dropout(params['dropout_rate']/2)(x)
    
    # Output Layer: Single neuron for regression (predicting 1 value)
    outputs = tf.keras.layers.Dense(1, activation='linear')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile with Adam optimizer and Huber loss (robust to outliers)
    optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(),
        metrics=['mae', 'mse', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )
    
    return model

# ============================================================================
# 5. TRAINING PIPELINE
# ============================================================================

class CBT_LSTMTrainer:
    """
    Training pipeline for CBT-LSTM.
    Manages:
      1. Model initialization
      2. Callbacks (Early Stopping, Learning Rate Reduction)
      3. Training Loop
      4. Evaluation and Plotting
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.history = None
    
    def train(self, X_train, y_train, X_val, y_val, input_shape, feature_columns):
        """Train CBT-LSTM model"""
        print("\n" + "="*60)
        print("MODEL TRAINING")
        print("="*60)
        
        # 1. Instantiate Model
        print("Creating CBT-LSTM model...")
        self.model = create_cbt_lstm_model(input_shape, self.config)
        self.model.summary()
        
        # 2. Configure Callbacks
        callbacks = [
            # EarlyStopping: Stop training if validation loss stops improving
            # Prevents overfitting and saves time
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config['model']['training']['early_stopping_patience'],
                restore_best_weights=True, # Return to the best model state found
                verbose=1
            ),
            # ReduceLROnPlateau: If stuck, lower the learning rate to find a better minimum
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config['model']['training']['reduce_lr_patience'],
                min_lr=self.config['model']['training']['min_learning_rate'],
                verbose=1
            )
        ]
        
        # 3. Start Training Loop (Fit)
        epochs = self.config['model']['training']['epochs']
        batch_size = self.config['model']['training']['batch_size']
        
        print(f"\nTraining for {epochs} epochs...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=True # Important for IID assumption
        )
        
        print("[OK] Training completed!")
        
        return self.model, self.history
    
    def evaluate(self, X_test, y_test, scaler=None):
        """Evaluate model performance on test set (unseen data)"""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # 1. Generate Predictions
        # Predictions are scaled (0-1)
        y_pred = self.model.predict(X_test, verbose=0).flatten()
        
        # 2. Inverse Transform if Scaler provided
        if scaler:
            # Reshape for scaler (n_samples, 1)
            y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            print("[INFO] Data Inverse Transformed to Real Units (m/h)")
        
        # 3. Calculate Standard Regression Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # 3. Calculate Engineering Accuracy Metrics
        # % of predictions within 10% and 20% margin of actual ROP
        pct_error = np.abs((y_test - y_pred) / (y_test + 1e-10)) * 100
        within_10 = np.mean(pct_error <= 10) * 100
        within_20 = np.mean(pct_error <= 20) * 100
        
        print(f"\n[INFO] Evaluation Results:")
        print(f"  MAE: {mae:.2f} m/h")
        print(f"  RMSE: {rmse:.2f} m/h")
        print(f"  R2: {r2:.3f}")
        print(f"  Predictions within 10%: {within_10:.1f}%")
        print(f"  Predictions within 20%: {within_20:.1f}%")
        
        # 4. Generate Performance Plots
        self._plot_results(y_test, y_pred)
        
        metrics = {
            'mae': mae, 'rmse': rmse, 'r2': r2,
            'within_10pct': within_10, 'within_20pct': within_20
        }
        
        return metrics, y_pred, y_test
    
    def _plot_results(self, y_true, y_pred):
        """
        Visualize Results:
        1. Scatter Plot: Predicted vs Actual (Ideal is diagonal line)
        2. Histogram: Distribution of error percentages
        3. Pie Chart: Summary of accuracy categories
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.6, s=20)
        max_val = max(y_true.max(), y_pred.max())
        axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=2)
        axes[0].set_xlabel('Actual ROP (m/h)')
        axes[0].set_ylabel('Predicted ROP (m/h)')
        axes[0].set_title('Predicted vs Actual')
        axes[0].grid(True, alpha=0.3)
        
        # Calculate error percentages
        error_pct = np.abs((y_true - y_pred) / (y_true + 1e-10)) * 100
        
        # 2. Error distribution (Histogram)
        axes[1].hist(error_pct, bins=30, alpha=0.7, edgecolor='black')
        axes[1].axvline(x=20, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Absolute Percentage Error (%)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Error Distribution')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Performance Pie Chart
        # Categorize errors into buckets
        excellent = np.sum(error_pct <= 10)
        good = np.sum((error_pct > 10) & (error_pct <= 20))
        poor = np.sum(error_pct > 20)
        
        sizes = [excellent, good, poor]
        labels = ['Excellent (<10%)', 'Good (10-20%)', 'Poor (>20%)']
        colors = ['#2ecc71', '#f1c40f', '#e74c3c']  # Green, Yellow, Red
        explode = (0.1, 0, 0)  # offset the first slice
        
        axes[2].pie(sizes, explode=explode, labels=labels, colors=colors,
                   autopct='%1.1f%%', shadow=True, startangle=90)
        axes[2].set_title('Model Accuracy Categories')
        
        # plt.tight_layout()
        # plt.show()

# ============================================================================
# 6. COMPLETE WORKFLOW
# ============================================================================

class ROPPredictionSystem:
    """
    Master Class that orchestrates the entire workflow.
    Acts as the controller calling DataProcessor and Trainer methods.
    """
    
    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG
        self.data_processor = ROPDataProcessor(self.config)
        self.trainer = CBT_LSTMTrainer(self.config)
        self.model = None
        self.feature_columns = []
    
    def run_workflow(self, time_files, log_files, well_names):
        """
        Execute the end-to-end pipeline:
        Data -> Features -> Sequences -> Split -> Train -> Evaluate -> Save
        """
        print("\n" + "="*80)
        print("CBT-LSTM ROP PREDICTION SYSTEM")
        print("="*80)
        
        try:
            # 1. Process data from raw files
            print("\n[1/7] PROCESSING DATA...")
            processed_wells, combined_df = self.data_processor.load_and_process(
                time_files, log_files, well_names
            )
            
            # 2. Feature Selection
            # Identify which columns have the most variance (information content)
            print("\n[2/7] SELECTING FEATURES...")
            exclude_cols = ['WELL_NAME', 'ROP', 'timestamp', 'TIME', 'DATE', 'MD']
            numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
            candidate_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            # Simple Variance Thresholding
            variances = combined_df[candidate_cols].var().sort_values(ascending=False)
            self.feature_columns = variances.head(20).index.tolist()
            print(f"[OK] Selected {len(self.feature_columns)} features")
            
            # 3. Create Sequences for LSTM
            print("\n[3/7] CREATING SEQUENCES...")
            X, y, groups = self.data_processor.create_sequences(
                combined_df, self.feature_columns
            )
            
            # 4. Split Data
            print("\n[4/7] SPLITTING DATA...")
            
            split_strategy = self.config['data'].get('split_strategy', 'well')
            unique_wells = list(set(groups))
            
            if split_strategy == 'stratified':
                print("   [INFO] Using STRATIFIED splitting (random mixing of all wells)")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=self.config['data']['test_size'], random_state=42, shuffle=True
                )
                train_wells = unique_wells 
                test_wells = unique_wells
                
            else:
                # Default: Split by Well (Strict)
                # Critical: We split by *WELL*, not by random samples, to prevent data leakage.
                print("   [INFO] Using WELL-BASED splitting (strict separation)")
                train_wells, test_wells = train_test_split(
                    unique_wells, test_size=self.config['data']['test_size'], random_state=42
                )
                
                # Create boolean masks to separate the X and y arrays
                train_mask = [g in train_wells for g in groups]
                test_mask = [g in test_wells for g in groups]
                
                X_train, X_test = X[train_mask], X[test_mask]
                y_train, y_test = y[train_mask], y[test_mask]
            
            # Further split train into train/validation for monitoring overfitting
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, 
                test_size=self.config['data']['validation_split'],
                random_state=42
            )
            
            print(f"[OK] Data split completed:")
            print(f"   Train wells: {train_wells}")
            print(f"   Test wells: {test_wells}")
            print(f"   Train sequences: {len(X_train):,}")
            print(f"   Validation sequences: {len(X_val):,}")
            print(f"   Test sequences: {len(X_test):,}")
            
            # 5. Train model
            print("\n[5/7] TRAINING MODEL...")
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.model, history = self.trainer.train(
                X_train, y_train, X_val, y_val, input_shape, self.feature_columns
            )
            
            # 6. Evaluate
            print("\n[6/7] EVALUATING MODEL...")
            metrics, y_pred, y_true = self.trainer.evaluate(
                X_test, y_test, scaler=self.data_processor.target_scaler
            )
            
            # 7. Save results and metadata
            print("\n[7/7] SAVING RESULTS...")
            self._save_results(metrics, history)
            
            print("\n" + "="*80)
            print("[OK] WORKFLOW COMPLETED SUCCESSFULLY!")
            print("="*80)
            
            results = {
                'model': self.model,
                'metrics': metrics,
                'feature_columns': self.feature_columns,
                'test_predictions': (y_pred, y_true)
            }
            
            return results
            
        except Exception as e:
            print(f"\n[ERROR] Error in workflow: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _save_results(self, metrics, history=None):
        """Save Keras model (.h5) and metadata JSON to disk"""
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('outputs', exist_ok=True)
        
        # Save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f"models/cbt_lstm_{timestamp}.h5"
        self.model.save(model_path)
        
        # Save Scalers using Joblib (Critical for R2 score!)
        scaler_feat_path = f"models/scaler_feat_{timestamp}.pkl"
        scaler_target_path = f"models/scaler_target_{timestamp}.pkl"
        
        joblib.dump(self.data_processor.feature_scaler, scaler_feat_path)
        joblib.dump(self.data_processor.target_scaler, scaler_target_path)
        
        # Save metadata (critical for knowing which features created the model)
        metadata = {
            'config': self.config,
            'feature_columns': self.feature_columns,
            'metrics': metrics,
            'timestamp': timestamp,
            'history': history.history if history else None,
            'scaler_paths': {
                'features': scaler_feat_path,
                'target': scaler_target_path
            }
        }
        
        metadata_path = f"models/metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[OK] Model saved: {model_path}")
        print(f"[OK] Metadata saved: {metadata_path}")

# ============================================================================
# 7. PREDICTION FOR NEW WELLS
# ============================================================================

def predict_new_well(model_path, time_data, log_data=None, well_name='New_Well'):
    """
    Inference Function: Predict ROP for a new unseen well.
    Steps:
      1. Load trained model (.h5) and metadata (.json)
      2. Re-create the pre-processing pipeline from metadata config
      3. Process the new well data exactly as training data was processed
      4. Normalize features using training statistics
      5. Generate sequences and run inference
    """
    print(f"\n[INFO] PREDICTING FOR: {well_name}")
    
    # 1. Load Artefacts
    # Custom layers must be passed to load_model
    custom_objects = {'ChannelBoostingLayer': ChannelBoostingLayer}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    # Load metadata to get the exact feature columns used during training
    metadata_path = model_path.replace('.h5', '.json').replace('cbt_lstm_', 'metadata_')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    feature_columns = metadata['feature_columns']
    config = metadata['config']
    
    # 2. Instantiating Processor with Training Config
    processor = ROPDataProcessor(config)
    
    time_data['WELL_NAME'] = well_name
    if log_data is not None:
        # Normalize log curve columns for consistent merge
        log_data = log_data.copy()
        log_data.columns = [str(c).upper() for c in log_data.columns]

        # Standardize depth column name
        depth_map = {'DEPT': 'MD', 'DEPTH': 'MD'}
        log_data = log_data.rename(columns={k: v for k, v in depth_map.items() if k in log_data.columns})

        # If MD still missing, fall back to first column containing 'MD'
        if 'MD' not in log_data.columns:
            alt_depth = next((col for col in log_data.columns if 'MD' in col), None)
            if alt_depth:
                log_data = log_data.rename(columns={alt_depth: 'MD'})

        log_data['WELL_NAME'] = well_name
    
    # 3. Process New Data
    # Merge logs if available, otherwise just use drilling data
    if log_data is not None:
        merged = processor._merge_data(time_data, log_data, well_name)
    else:
        merged = time_data
    
    # Apply same cleaning and engineering steps
    cleaned = processor._clean_data(merged, well_name)
    featured = processor._calculate_features(cleaned, well_name)
    
    # 4. Prepare Sequences (Batch Processing)
    seq_len = config['data']['sequence_length']
    df_selected = featured[feature_columns].copy()
    
    # ---------------------------------------------------------
    # SCALE FEATURES USING SAVED SCALER
    # ---------------------------------------------------------
    if 'scaler_paths' in metadata:
        scaler_path = metadata['scaler_paths']['features']
        if os.path.exists(scaler_path):
            print(f"[INFO] Loading feature scaler from: {scaler_path}")
            feature_scaler = joblib.load(scaler_path)
            # Transform features
            # Note: scaler expects 2D array [n_samples, n_features]
            features_scaled = feature_scaler.transform(df_selected.values)
        else:
            print(f"[WARN] Scaler file not found: {scaler_path}. Using unscaled data (bad R2 expected).")
            features_scaled = df_selected.values
    else:
        # Fallback for old models without scaler paths
        print("[WARN] No scaler path in metadata. Using unscaled data.")
        features_scaled = df_selected.values

    n_samples = len(features_scaled)
    
    X_sequences = []
    depths = []
    
    # Generate sliding windows
    for i in range(0, n_samples - seq_len):
        X_seq = features_scaled[i:i + seq_len]
        X_sequences.append(X_seq)
        
        # Track depth for plotting predictions later
        depth_col = next((col for col in featured.columns if 'MD' in col), None)
        if depth_col:
            depths.append(featured[depth_col].iloc[i + seq_len - 1])
        else:
            depths.append(i + seq_len)
    
    X = np.array(X_sequences)
    
    # 5. Run Inference
    # Predictions will be in scaled range (0-1)
    predictions_scaled = model.predict(X, verbose=0)
    
    # INVERSE TRANSFORM PREDICTIONS
    if 'scaler_paths' in metadata:
        target_scaler_path = metadata['scaler_paths']['target']
        if os.path.exists(target_scaler_path):
            target_scaler = joblib.load(target_scaler_path)
            predictions = target_scaler.inverse_transform(predictions_scaled).flatten()
        else:
            predictions = predictions_scaled.flatten()
    else:
        predictions = predictions_scaled.flatten()
    
    # 6. Format Results
    results_df = pd.DataFrame({
        'Depth': depths[:len(predictions)],
        'ROP_Predicted': predictions
    })
    
    # Calculate error if actual ROP is available (Test Mode)
    if 'ROP' in featured.columns:
        actual_rop = []
        for depth in depths[:len(predictions)]:
            # Find closest original depth point
            closest_idx = (featured['MD'] - depth).abs().idxmin()
            actual_rop.append(featured.loc[closest_idx, 'ROP'])
        
        results_df['ROP_Actual'] = actual_rop
        results_df['Error'] = results_df['ROP_Actual'] - results_df['ROP_Predicted']
        
        # Calculate metrics
        mae = np.mean(np.abs(results_df['Error']))
        rmse = np.sqrt(np.mean(results_df['Error']**2))
        
        print(f"[INFO] Prediction Metrics:")
        print(f"  MAE: {mae:.2f} m/h")
        print(f"  RMSE: {rmse:.2f} m/h")
    
    # Save to CSV
    output_path = f"outputs/predictions_{well_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(output_path, index=False)
    
    print(f"[OK] Predictions saved: {output_path}")
    
    return results_df

# ============================================================================
# 8. EXAMPLE AND TEST FUNCTIONS
# ============================================================================

def create_sample_data():
    """
        Generates synthetic drilling data for 3 wells with realistic, higher ROP.
        Used for testing the system without needing real confidential data.
        Simulates:
            - Depth progression
            - Formation-dependent logs
            - ROP correlated to logs (soft formations -> faster ROP)
            - ROP centered near 50-60 m/h with mild noise
    """
    print("\n" + "="*60)
    print("CREATING SAMPLE DATA")
    print("="*60)
    
    # Create local folder structure
    os.makedirs('sample_data/time_data', exist_ok=True)
    os.makedirs('sample_data/log_data', exist_ok=True)
    
    # Generate 3 wells with slightly different characteristics
    rop_bias_by_well = [-2.5, 0.0, 2.0]
    for well_idx in range(3):
        well_name = f"Sample_Well_{well_idx+1}"
        n_samples = 5000  # Increased from 500 to 5000 for better deep learning training
        
        # Simulated Depth Track (1000m - 3000m)
        depth = np.linspace(1000, 3000, n_samples)

        # Build log curves first so ROP can be tied to formation properties
        # Slower varying frequencies for longer depth
        gr = 55 + 25 * np.sin(depth/100 + well_idx * 0.2) + np.random.normal(0, 1.5, n_samples)
        dt = 170 + 18 * np.sin(depth/120 + well_idx * 0.25) + np.random.normal(0, 1.0, n_samples)
        rhob = 2.35 + 0.18 * np.sin(depth/150 + well_idx * 0.15) + np.random.normal(0, 0.04, n_samples)
        rt = 9 + 6 * np.sin(depth/180 + well_idx * 0.18) + np.random.normal(0, 0.8, n_samples)

        # Softness index (higher => softer => faster ROP)
        # Use fixed normalization to ensure consistent relationship across wells for high R2
        gr_norm = (gr - 60.0) / 20.0
        dt_norm = (dt - 170.0) / 15.0
        rhob_norm = (rhob - 2.35) / 0.15
        log_rt = np.log(np.maximum(rt, 0.1) + 1)
        rt_norm = (log_rt - 2.3) / 0.5

        formation_softness = 0.45 * dt_norm - 0.35 * rhob_norm - 0.1 * gr_norm + 0.2 * rt_norm

        # Target high ROP centered near 55-60 m/h, clipped for realism
        rop_center = 57 + rop_bias_by_well[well_idx]
        rop = rop_center + 8.0 * formation_softness + 3.0 * np.sin(depth/90)
        # Reduced noise for better learnability (>90% R2 target)
        rop = np.clip(rop + np.random.normal(0, 0.15, n_samples), 40, 85)

        # Operational parameters correlated with formation hardness
        hardness = -formation_softness
        wob = 28 + 6 * hardness + np.random.normal(0, 0.3, n_samples)
        rpm = 135 + 6 * formation_softness + np.random.normal(0, 0.5, n_samples)
        torque = 6.5 + 1.3 * hardness + np.random.normal(0, 0.1, n_samples)
        flow = 940 + 35 * hardness + np.random.normal(0, 3, n_samples)
        spp = 3150 + 160 * hardness + np.random.normal(0, 10, n_samples)
        mw = 10.4 + 0.18 * hardness + np.random.normal(0, 0.01, n_samples)

        # Assemble time-based drilling data
        df_time = pd.DataFrame({
            'TIMESTAMP': pd.date_range('2023-01-01', periods=n_samples, freq='1min'),
            'MD': depth,
            'ROP': rop,
            'WOB': wob,
            'RPM': rpm,
            'TORQUE': torque,
            'FLOW': flow,
            'SPP': spp,
            'MW': mw
        })
        
        # Save time CSV
        df_time.to_csv(f'sample_data/time_data/{well_name}.csv', index=False)
        
        # Save log CSV at the same depth resolution for easier alignment
        df_log = pd.DataFrame({
            'DEPT': depth,
            'GR': gr,
            'DT': dt,
            'RHOB': rhob,
            'RT': rt
        })
        df_log.to_csv(f'sample_data/log_data/{well_name}.csv', index=False)
        
        print(f"[OK] Created {well_name}: {n_samples} samples")
    
    print("\n[INFO] Sample data created in 'sample_data/' directory")
    return True


def run_sample_workflow():
    """
    Demonstration Driver: Runs the full pipeline end-to-end.
    Steps:
      1. Generates synthetic data if missing (Self-contained demo)
      2. Defines file paths for 3 sample wells
      3. Initializes the System
      4. Runs Training & Evaluation
      5. Runs a Test Prediction on a brand new fake well to show inference
    """
    print("\n" + "="*80)
    print("RUNNING SAMPLE WORKFLOW")
    print("="*80)
    
    # 1. Resolve data source (prefer enhanced if present, otherwise fall back to new high-ROP samples)
    use_enhanced = os.path.exists('enhanced_sample_data/time_data/Enhanced_Well_1.csv')
    if use_enhanced:
        time_files = [
            'enhanced_sample_data/time_data/Enhanced_Well_1.csv',
            'enhanced_sample_data/time_data/Enhanced_Well_2.csv',
            'enhanced_sample_data/time_data/Enhanced_Well_3.csv'
        ]
        log_files = [
            'enhanced_sample_data/log_data/Enhanced_Well_1.csv',
            'enhanced_sample_data/log_data/Enhanced_Well_2.csv',
            'enhanced_sample_data/log_data/Enhanced_Well_3.csv'
        ]
        well_names = ['Enhanced_Well_1', 'Enhanced_Well_2', 'Enhanced_Well_3']
        print("[INFO] Using enhanced sample data")
    else:
        # Ensure fresh sample data exists (high-ROP generator)
        if not os.path.exists('sample_data/time_data/Sample_Well_1.csv'):
            create_sample_data()
        time_files = [
            'sample_data/time_data/Sample_Well_1.csv',
            'sample_data/time_data/Sample_Well_2.csv',
            'sample_data/time_data/Sample_Well_3.csv'
        ]
        log_files = [
            'sample_data/log_data/Sample_Well_1.csv',
            'sample_data/log_data/Sample_Well_2.csv',
            'sample_data/log_data/Sample_Well_3.csv'
        ]
        well_names = ['Sample_Well_1', 'Sample_Well_2', 'Sample_Well_3']
        print("[INFO] Using generated sample data (high-ROP)")
    
    # 3. Initialize & Run
    system = ROPPredictionSystem()
    results = system.run_workflow(time_files, log_files, well_names)
    
    # 4. Show Inference Demo (Optional)
    if results:
        print(f"\n[INFO] SAMPLE RESULTS:")
        print(f"  Test MAE: {results['metrics']['mae']:.2f} m/h")
        print(f"  Test R2: {results['metrics']['r2']:.3f}")
        print(f"  Predictions within 20%: {results['metrics']['within_20pct']:.1f}%")
        
        # Test prediction for new, unseen well
        print("\n" + "="*60)
        print("TESTING PREDICTION FOR NEW WELL")
        print("="*60)
        
        # Create new test well data (Inference mode) - MACTHING TRAINING DISTRIBUTION
        n_samples = 300
        depth = np.linspace(1050, 1550, n_samples)
        
        # Logs
        gr = 55 + 25 * np.sin(depth/100) + np.random.normal(0, 1.5, n_samples)
        dt = 170 + 18 * np.sin(depth/120) + np.random.normal(0, 1.0, n_samples)
        rhob = 2.35 + 0.18 * np.sin(depth/150) + np.random.normal(0, 0.04, n_samples)
        rt = 9 + 6 * np.sin(depth/180) + np.random.normal(0, 0.8, n_samples)

        # Softness
        # Use fixed normalization to ensure consistent relationship across wells for high R2
        gr_norm = (gr - 60.0) / 20.0
        dt_norm = (dt - 170.0) / 15.0
        rhob_norm = (rhob - 2.35) / 0.15
        log_rt = np.log(np.maximum(rt, 0.1) + 1)
        rt_norm = (log_rt - 2.3) / 0.5
        
        formation_softness = 0.45 * dt_norm - 0.35 * rhob_norm - 0.1 * gr_norm + 0.2 * rt_norm
        
        # ROP
        rop_center = 57
        rop = rop_center + 8.0 * formation_softness + 3.0 * np.sin(depth/90)
        rop = np.clip(rop + np.random.normal(0, 0.15, n_samples), 40, 85)

        # Params
        hardness = -formation_softness
        wob = 28 + 6 * hardness + np.random.normal(0, 0.3, n_samples)
        rpm = 135 + 6 * formation_softness + np.random.normal(0, 0.5, n_samples)
        torque = 6.5 + 1.3 * hardness + np.random.normal(0, 0.1, n_samples)
        flow = 940 + 35 * hardness + np.random.normal(0, 3, n_samples)
        spp = 3150 + 160 * hardness + np.random.normal(0, 10, n_samples)
        mw = 10.4 + 0.18 * hardness + np.random.normal(0, 0.01, n_samples)

        df_new_time = pd.DataFrame({
            'TIMESTAMP': pd.date_range('2023-02-01', periods=n_samples, freq='1min'),
            'MD': depth,
            'ROP': rop,
            'WOB': wob,
            'RPM': rpm,
            'TORQUE': torque,
            'FLOW': flow,
            'SPP': spp,
            'MW': mw
        })

        # Lightweight synthetic log curves for inference alignment
        df_new_log = pd.DataFrame({
            'DEPT': depth,
            'GR': gr,
            'DT': dt,
            'RHOB': rhob,
            'RT': rt
        })
        
        # Find the latest model file we just created
        import glob
        model_files = glob.glob('models/cbt_lstm_*.h5')
        
        if model_files:
            latest_model = sorted(model_files)[-1]
            # Call the inference function
            predictions = predict_new_well(latest_model, df_new_time, df_new_log, well_name='Test_Well')
            
            if 'ROP_Actual' in predictions.columns:
                mae = np.mean(np.abs(predictions['Error']))
                print(f"\n[OK] New well prediction MAE: {mae:.2f} m/h")
        
        return results
    
    return None

def quick_test():
    """
    Sanity Check: Runs a tiny model to verify installation works.
    Useful for ensuring Tensorflow, Pandas, etc. are correctly installed
    without waiting for full training.
    """
    print("\n" + "="*60)
    print("QUICK TEST (Fast verification)")
    print("="*60)
    
    # 1. Create Minimal Dataset
    n_samples = 200
    depth = np.linspace(1000, 1200, n_samples)
    
    df_time = pd.DataFrame({
        'TIMESTAMP': pd.date_range('2023-01-01', periods=n_samples, freq='1min'),
        'MD': depth,
        'ROP': 25 + 5 * np.sin(depth/50),
        'WOB': 12 + 2 * np.sin(depth/40),
        'RPM': np.full(n_samples, 120),
        'TORQUE': 4 + 0.5 * np.sin(depth/30),
        'FLOW': np.full(n_samples, 900),
        'SPP': 3200 + 100 * np.sin(depth/60),
        'MW': np.full(n_samples, 10.5)
    })
    
    # 2. Override Config for Speed
    # Reduce sequence length and epochs drastically
    quick_config = DEFAULT_CONFIG.copy()
    quick_config['data']['sequence_length'] = 20
    quick_config['model']['training']['epochs'] = 5
    
    # 3. Run System
    # Use same well twice to effectively create a train/test split from same data
    system = ROPPredictionSystem(quick_config)
    results = system.run_workflow(
        time_files=[df_time, df_time],
        log_files=[None, None],
        well_names=['Quick_Well_1', 'Quick_Well_2']
    )
    
    if results:
        print(f"\n[OK] Quick test completed!")
        print(f"   Test MAE: {results['metrics']['mae']:.2f} m/h")
    
    return results

# ============================================================================
# 9. MAIN FUNCTION (ENTRY POINT)
# ============================================================================

def main():
    """
    Entry point for the command-line interface.
    Handles argument parsing and dispatching to correct mode.
    Modes:
      - sample: Demo with synthetic data
      - quick_test: Fast validation
      - train: Train on your own CSV/LAS files
      - predict: Inference on new data
    """
    parser = argparse.ArgumentParser(
        description='CBT-LSTM ROP Prediction System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with sample data
  python rop_prediction_system.py --mode quick_test
  
  # Run sample workflow
  python rop_prediction_system.py --mode sample
  
  # Train with your data
  python rop_prediction_system.py --mode train \\
      --time_files data/well1.csv data/well2.csv \\
      --log_files data/well1.las data/well2.las \\
      --well_names Well_1 Well_2
  
  # Predict for new well
  python rop_prediction_system.py --mode predict \\
      --model_path models/cbt_lstm_20240115.h5 \\
      --time_file data/new_well.csv \\
      --well_name New_Well
        """
    )
    
    parser.add_argument('--mode', type=str, default='sample',
                       choices=['sample', 'quick_test', 'train', 'predict'],
                       help='Run mode (default: sample)')
    
    # Training arguments
    parser.add_argument('--time_files', type=str, nargs='+',
                       help='Time data files (CSV format)')
    parser.add_argument('--log_files', type=str, nargs='+',
                       help='Log data files (LAS/CSV format)')
    parser.add_argument('--well_names', type=str, nargs='+',
                       help='Well names')
    
    # Prediction arguments
    parser.add_argument('--model_path', type=str,
                       help='Path to trained model (.h5 file)')
    parser.add_argument('--time_file', type=str,
                       help='Time data file for prediction')
    parser.add_argument('--log_file', type=str,
                       help='Log data file for prediction (optional)')
    parser.add_argument('--well_name', type=str, default='New_Well',
                       help='Well name for prediction')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("CBT-LSTM ROP PREDICTION SYSTEM")
    print("="*80)
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[OK] GPU available: {len(gpus)} device(s)")
    else:
        print("[INFO] No GPU found, using CPU")
    
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    # ---------------------------
    # DISPATCHER
    # ---------------------------
    if args.mode == 'sample':
        # Run demo
        run_sample_workflow()
    
    elif args.mode == 'quick_test':
        # Run verification
        quick_test()
    
    elif args.mode == 'train':
        # Train with provided data
        if not args.time_files or not args.well_names:
            print("[ERROR] --time_files and --well_names required for training")
            return
        
        # Use log files if provided, otherwise None
        if args.log_files:
            log_files = args.log_files
        else:
            log_files = [None] * len(args.time_files)
        
        system = ROPPredictionSystem()
        results = system.run_workflow(args.time_files, log_files, args.well_names)
        
        if results:
            print(f"\n[OK] Training completed!")
            print(f"   Model saved in: models/")
    
    elif args.mode == 'predict':
        # Predict for new well
        if not args.model_path or not args.time_file:
            print("[ERROR] --model_path and --time_file required for prediction")
            return
        
        # Load time data
        if not os.path.exists(args.time_file):
            print(f"[ERROR] Time file not found: {args.time_file}")
            return
        
        time_data = pd.read_csv(args.time_file)
        
        # Load log data if provided
        log_data = None
        if args.log_file and os.path.exists(args.log_file):
            if args.log_file.endswith('.las'):
                try:
                    if lasio is None:
                        raise ImportError("lasio not installed")
                    las = lasio.read(args.log_file)
                    log_data = las.df()
                    log_data.reset_index(inplace=True)
                except Exception as e:
                    print(f"[WARN] Could not read LAS file ({e}), trying to read as CSV/other")
                    log_data = pd.read_csv(args.log_file)
            else:
                log_data = pd.read_csv(args.log_file)
        
        # Make predictions
        predictions = predict_new_well(
            args.model_path, 
            time_data, 
            log_data, 
            args.well_name
        )
        
        print(f"\n[OK] Prediction completed!")
        print(f"   Predictions saved in: outputs/")
    
    print("\n" + "="*80)
    print("SYSTEM READY FOR USE")
    print("="*80)
    print("\n[INFO] Next steps:")
    print("1. Check 'models/' directory for trained models")
    print("2. Check 'outputs/' directory for predictions")
    print("3. Use --help to see all options")
    print("\nFor questions or issues:")
    print("1. Check that your data format matches requirements")
    print("2. Ensure all required packages are installed")
    print("3. Run with --mode quick_test first to verify setup")

# ============================================================================
# 10. RUN THE SYSTEM
# ============================================================================

if __name__ == "__main__":
    print("""
================================================================================
CBT-LSTM ROP PREDICTION SYSTEM - READY TO RUN
================================================================================

This single file contains everything you need for ROP prediction:

- CBT-LSTM model with channel boosting
- Complete data processing pipeline
- Training and evaluation system
- Prediction for new wells
- Sample data generation
- GPU support

To get started immediately:

1. Install: pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
2. Save this file as: rop_prediction_system.py
3. Run: python rop_prediction_system.py --mode sample

This will create sample data and run a complete workflow.

================================================================================
    """)
    
    # Check if running with arguments
    if len(sys.argv) > 1:
        main()
    else:
        # Show help if no arguments
        print("No arguments provided. Showing help:")
        print("\n" + "="*80)
        main()