# CBT-LSTM ROP Prediction System

## Overview

This is a complete **CBT-LSTM (Channel Boosting Time-series LSTM)** system for predicting Rate of Penetration (ROP) in drilling operations. The system uses machine learning to analyze offset well data and predict ROP for new wells, incorporating both time-based drilling parameters and log data.

**Key Features:**
- Multi-scale feature extraction with Channel Boosting layers
- Temporal pattern recognition using LSTM networks
- Automatic feature engineering
- Uncertainty quantification
- Real-time prediction ready
- Model ensemble support

**Expected Performance:**
- MAE: 2-5 m/h (depends on data quality)
- R²: 0.7-0.9 (higher with more offset wells)
- Training time: 30-60 minutes for 3 wells

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Data Requirements](#data-requirements)
4. [System Architecture](#system-architecture)
5. [Engineering Calculations](#engineering-calculations)
6. [Usage Guide](#usage-guide)
7. [Command Line Options](#command-line-options)
8. [Troubleshooting](#troubleshooting)
9. [API Reference](#api-reference)

## Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.10+
- CUDA-compatible GPU (optional, but recommended for training)

### Required Packages

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn lasio
```

**Note:** `lasio` is optional but recommended for reading LAS log files. If not installed, the system will skip log data processing.

## Quick Start
### Option 1: Sample Data & Visualization
Run the complete workflow with generated sample data, then visualize the results.

```bash
# 1. Run the prediction system (generates data + trains model)
python rop_prediction_system.py --mode sample

# 2. Visualize the results (creates plots in outputs/)
python visualize_results.py

# 3. Run unit tests
python -m unittest tests/test_basic.py
```

This will:
1. Generate sample drilling data for 3 wells.
2. Train a CBT-LSTM model.
3. Generate Loss Curves, Depth Plots, and Correlation Matrices.
4. Verify system integrity.

### Option 2: Your Data

```bash
# Train with your data
python rop_prediction_system.py --mode train \
    --time_files data/well1.csv data/well2.csv \
    --log_files data/well1.las data/well2.las \
    --well_names Well_1 Well_2
```

### Option 3: Quick Test

```bash
# Fast verification with minimal data
python rop_prediction_system.py --mode quick_test
```

## PyCharm Setup

This project is fully compatible with JetBrains PyCharm. Here is how to set it up:

1.  **Open Project**: Launch PyCharm and select `Open` -> Select the `ROP_prediction` folder.
2.  **Configure Interpreter**:
    *   Go to `File` > `Settings` > `Project: ROP_prediction` > `Python Interpreter`.
    *   Click the gear icon > `Add...`
    *   Select `Virtualenv Environment` > `New environment`.
    *   Click `OK` to create the virtual environment.
3.  **Install Requirements**:
    *   PyCharm detects `requirements.txt` automatically.
    *   Click the banner "Install requirements" that appears at the top of the editor.
    *   Or run manually in the PyCharm Terminal: `pip install -r requirements.txt`
4.  **Create Run Configuration**:
    *   Right-click `rop_prediction_system.py` in the project view.
    *   Select `Modify Run Configuration...`.
    *   In `Parameters`, add the arguments for the mode you want to run (e.g., `--mode sample` or `--mode quick_test`).
    *   Click `OK`.
    *   Now you can click the green **Play** button to run the system.

## Data Requirements

### Time Data (Required)

CSV format with the following columns:

| Column | Description | Units | Required |
|--------|-------------|-------|----------|
| TIMESTAMP | Date/time of measurement | ISO format | Yes |
| MD | Measured Depth | ft or m | Yes |
| ROP | Rate of Penetration | ft/h or m/h | Yes |
| WOB | Weight on Bit | klbf or kN | Yes |
| RPM | Rotary speed | rpm | Yes |
| TORQUE | Rotary torque | ft-lbf or kN-m | Yes |
| FLOW | Mud flow rate | gpm or L/min | Yes |
| SPP | Standpipe pressure | psi or kPa | Yes |
| MW | Mud weight | ppg or kg/L | Yes |

**Example:**
```csv
TIMESTAMP,MD,ROP,WOB,RPM,TORQUE,FLOW,SPP,MW
2023-01-01 08:00:00,1000.0,25.6,12.3,120,4.5,900,3200,10.5
```

### Log Data (Optional but Recommended)

Either LAS format or CSV with depth and curve data:

**Required Curves:**
- **GR**: Gamma Ray (API units)
- **DT**: Sonic travel time (μs/ft or μs/m)
- **RHOB**: Bulk density (g/cm³)
- **RT**: True resistivity (ohm-m)

**CSV Format Example:**
```csv
DEPT,GR,DT,RHOB,RT
1000.0,45.2,85.3,2.45,12.5
1000.5,46.1,84.8,2.47,11.8
```

## System Architecture

### CBT-LSTM Model

The system uses a novel **Channel Boosting Time-series LSTM** architecture:

1. **Channel Boosting Layer**: Multi-scale convolutional feature extraction
   - Multiple kernel sizes (3, 5, 7) for different temporal scales
   - Adaptive gating mechanism
   - Residual connections

2. **LSTM Layers**: Temporal pattern recognition
   - Bidirectional processing capability
   - Dropout regularization
   - Batch normalization

3. **Dense Layers**: Feature integration and prediction

### Data Processing Pipeline

1. **Data Loading**: CSV/LAS file parsing
2. **Data Cleaning**: Outlier removal, interpolation
3. **Feature Engineering**: Engineering calculations (see below)
4. **Sequence Creation**: Sliding window approach
5. **Normalization**: Feature scaling
6. **Train/Validation/Test Split**: Well-based splitting

## Data Flow and Processing

This section explains how data flows through the system, what specific data is used at each stage, and the rationale behind data processing decisions.

### Input Data Sources

The system processes two main types of data:

#### 1. Time-Series Drilling Data (Required)
**Purpose**: Captures real-time drilling operations and their immediate effects on ROP
**Why needed**: Drilling parameters change continuously and directly influence penetration rate

**Key Parameters:**
- **ROP (Rate of Penetration)**: Target variable - what we're predicting
- **WOB (Weight on Bit)**: Primary drilling force, directly affects rock crushing
- **RPM (Rotary Speed)**: Bit rotation speed, affects cutting efficiency
- **TORQUE**: Resistance to rotation, indicates formation hardness
- **FLOW**: Mud circulation rate, affects hole cleaning and cooling
- **SPP (Standpipe Pressure)**: Hydraulic pressure, affects bit hydraulics
- **MW (Mud Weight)**: Mud density, affects wellbore stability
- **MD (Measured Depth)**: Depth reference for temporal and spatial correlation

#### 2. Log Data (Optional but Recommended)
**Purpose**: Geological context and formation properties
**Why needed**: ROP varies significantly with rock type and properties

**Key Curves:**
- **GR (Gamma Ray)**: Shale content indicator, affects drillability
- **DT (Sonic)**: Formation hardness and porosity
- **RHOB (Density)**: Rock density, correlates with strength
- **RT (Resistivity)**: Fluid content and rock type

### Data Processing Flow

#### Stage 1: Data Loading and Validation
```
Input: CSV files, LAS files
Process: Parse, standardize column names, validate required columns
Output: Raw DataFrames per well
Why: Ensures consistent data format across different sources
```

**Data Used**: All available columns from input files
**Validation**: Checks for required time-series columns (ROP, WOB, RPM, etc.)

#### Stage 2: Data Cleaning and Filtering
```
Input: Raw well data
Process: Remove drilling stops, interpolate gaps, filter outliers
Output: Clean time-series data
Why: Poor quality data (stops, gaps) can mislead the model
```

**Filtering Criteria:**
- ROP > 0.1 m/h (removes drilling stops)
- WOB > 2.0 klbf (removes tripping operations)
- ROP < 200 ft/h (removes data errors)

**Data Used**: Time-series parameters for quality assessment

#### Stage 3: Feature Engineering
```
Input: Clean time-series data
Process: Calculate engineering parameters, merge with logs
Output: Enhanced feature set
Why: Raw parameters alone insufficient; engineering calculations provide physical insights
```

**Derived Features:**
- **MSE**: Energy efficiency of drilling process
- **HSI**: Hydraulic power at the bit
- **UCS**: Estimated rock strength
- **V_SH**: Shale volume fraction

**Data Used**: 
- Time-series: WOB, RPM, TORQUE, ROP, SPP, FLOW
- Logs: GR, DT, RHOB, RT (if available)

#### Stage 4: Feature Selection
```
Input: All available features (raw + derived)
Process: Variance-based selection, correlation analysis
Output: Top 20 most informative features
Why: Reduces dimensionality, removes redundant/noisy features
```

**Selection Method:**
- Calculate variance for each feature
- Select top features by variance (avoids low-information features)
- Exclude depth/timestamp columns

**Data Used**: All numeric features after engineering calculations

#### Stage 5: Sequence Creation
```
Input: Feature matrix per well
Process: Sliding window with overlap
Output: 3D sequences (samples × timesteps × features)
Why: LSTM requires temporal context; sequences capture drilling patterns
```

**Parameters:**
- **Sequence Length**: 50 timesteps (configurable)
- **Step Size**: 5 timesteps (overlap for more samples)
- **Window**: Moving window captures recent drilling history

**Data Used**: Selected features arranged chronologically by depth

#### Stage 6: Data Splitting Strategy
```
Input: All sequences with well labels
Process: Well-based train/test split
Output: Train/val/test sets
Why: Prevents data leakage; wells are independent geological entities
```

**Splitting Logic:**
- Split wells into train/test groups (not individual samples)
- Further split training wells into train/validation
- Maintains geological diversity in each set

**Data Used**: Well identifiers to ensure proper grouping

#### Stage 7: Normalization
```
Input: Raw feature values
Process: Feature-wise scaling
Output: Normalized sequences
Why: Neural networks require scaled inputs; prevents feature dominance
```

**Method**: Min-Max scaling per feature (0-1 range)
**Why not StandardScaler**: Preserves relationships in time-series data

### Why This Data Flow Matters

#### Temporal Dependencies
- **Problem**: ROP at time t depends on recent drilling history
- **Solution**: Sequences capture 50-timestep context
- **Impact**: Model learns drilling patterns, not just instantaneous relationships

#### Geological Context
- **Problem**: Same drilling parameters give different ROP in different rocks
- **Solution**: Log data provides formation properties
- **Impact**: Model adapts to geological variations

#### Engineering Physics
- **Problem**: Raw parameters don't capture drilling physics
- **Solution**: MSE, HSI calculations provide physical insights
- **Impact**: Model understands energy efficiency and hydraulic effects

#### Data Quality
- **Problem**: Drilling data has noise, gaps, and operational artifacts
- **Solution**: Multi-stage cleaning and validation
- **Impact**: Reliable training data leads to better predictions

#### Generalization
- **Problem**: Model must work on unseen wells/geologies
- **Solution**: Well-based splitting prevents overfitting to specific wells
- **Impact**: Robust predictions for new drilling scenarios

### Data Volume Requirements

- **Minimum**: 2-3 offset wells for reliable training
- **Recommended**: 3+ wells with diverse geological conditions
- **Sequence Count**: 1000+ sequences per well (depends on data length)
- **Feature Count**: 15-25 features after selection

### Memory Considerations

- **Sequence Storage**: 3D arrays (N_samples × 50 × N_features)
- **Typical Size**: ~100MB for 10,000 sequences with 20 features
- **GPU Memory**: Batch size × sequence length × features
- **Optimization**: Smaller batches if memory constrained

## Code Execution Flow

This section details how the Python code implements the data flow described above, mapping specific functions to data processing stages.

### 1. Main Execution Loop (`ROPPredictionSystem.run_workflow`)

The `run_workflow` method orchestrates the entire process:

1. **Initialization**: Instantiates `ROPDataProcessor` and `CBT_LSTMTrainer` with the config.
2. **Data Coordination**: Calls processor methods to transform raw files into training tensors.
3. **Training Management**: Passes tensors to the trainer and monitors progress.
4. **Evaluation**: Triggers testing on the held-out well set.
5. **Serialization**: Saves the trained model and metadata for future use.

### 2. Method-to-Data Mapping

| Data Stage | Class | Method | Description |
|------------|-------|--------|-------------|
| **1. Ingestion** | `ROPDataProcessor` | `load_and_process` | Entry point for all data loading |
| | | `_load_time_data` | Reads CSVs, standardizes time/drilling columns |
| | | `_load_log_data` | Reads LAS files using `lasio`, standardizes curves |
| **2. Cleaning** | `ROPDataProcessor` | `_merge_data` | Aligns log depths with drilling time data |
| | | `_clean_data` | Filters stops (ROP<0.1), outliers, and interpolates gaps |
| **3. Engineering** | `ROPDataProcessor` | `_calculate_features` | Computes MSE, HSI, UCS, V_SH physics parameters |
| **4. Sequencing** | `ROPDataProcessor` | `create_sequences` | Transforms flat DataFrame → 3D LSTM arrays `(N, 50, F)` |
| **5. Training** | `CBT_LSTMTrainer` | `train` | Builds model graph, compiles with loss, runs fit loop |
| **6. Prediction** | `Global Function` | `predict_new_well` | Standalone function for inference on new data |

### 3. Key Data Structures

- **`processed_wells` (Dict)**: Stores individual DataFrames for each well after cleaning.
  - *Key*: Well Name
  - *Value*: DataFrame with ~15-20 columns (Time + Log + Eng. Features)

- **`X` (Numpy Array)**: Input features for the model.
  - *Shape*: `(n_samples, sequence_length, n_features)`
  - *Example*: `(10000, 50, 18)` for 10k samples, 50 steps, 18 features

- **`y` (Numpy Array)**: Target separation.
  - *Shape*: `(n_samples, )` -> Scalar ROP value for the *next* timestep

## Engineering Calculations

The system automatically calculates several drilling engineering parameters:

### 1. Mechanical Specific Energy (MSE)

**Formula:**
```
MSE = (WOB × 1000 / bit_area) + (480 × RPM × TORQUE) / (bit_area × ROP)
```

**Where:**
- `bit_area = π × (bit_diameter/2)²`
- `bit_diameter` assumed as 12.25 inches (311.15 mm)

**Units:**
- MSE: psi (pounds per square inch)
- WOB: klbf (thousand pounds force)
- RPM: revolutions per minute
- TORQUE: ft-lbf (foot-pounds)
- ROP: ft/h (feet per hour)

**Physical Meaning:**
MSE represents the energy required to drill per unit volume of rock. Lower MSE indicates more efficient drilling.

### 2. Hydraulic Specific Energy (HSI)

**Formula:**
```
HSI = (SPP × FLOW) / (1714 × bit_area)
```

**Where:**
- `bit_area` same as above
- Constant 1714 converts units appropriately

**Units:**
- HSI: hp/in² (horsepower per square inch)
- SPP: psi (pounds per square inch)
- FLOW: gpm (gallons per minute)

**Physical Meaning:**
HSI quantifies hydraulic energy at the bit, important for hole cleaning and cuttings transport.

### 3. Unconfined Compressive Strength (UCS)

**Formula:**
```
UCS = 10^((DT - 50) / (-25)) × 145.038
```

**Where:**
- DT: Sonic travel time in μs/ft
- Constants derived from empirical correlations

**Units:**
- UCS: psi
- DT: μs/ft

**Physical Meaning:**
Estimated rock strength from sonic logs. Higher values indicate harder formations.

### 4. Shale Volume (V_SH)

**Formula:**
```
V_SH = (GR - GR_min) / (GR_max - GR_min)
```

**Where:**
- GR_min/max: 5th and 95th percentiles of GR in the dataset

**Units:**
- Dimensionless (0-1)

**Physical Meaning:**
Fraction of shale in the formation. Used for lithology classification and porosity estimation.

## Usage Guide

### Step 1: Prepare Your Data

Create a `data` folder and organize your files:

```
data/
├── time_data/
│   ├── well1.csv
│   ├── well2.csv
│   └── well3.csv
└── log_data/
    ├── well1.las
    ├── well2.las
    └── well3.las
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run Training

```bash
python rop_prediction_system.py --mode train \
    --time_files data/time_data/well1.csv data/time_data/well2.csv \
    --log_files data/log_data/well1.las data/log_data/well2.las \
    --well_names Well_1 Well_2
```

### Step 4: Check Results

- **Models**: Saved in `models/` directory
- **Predictions**: Saved in `outputs/` directory
- **Logs**: Training logs in `logs/` directory

### Step 5: Predict New Well

```bash
python rop_prediction_system.py --mode predict \
    --model_path models/cbt_lstm_20240115_120000.h5 \
    --time_file data/time_data/new_well.csv \
    --well_name New_Well
```

## Command Line Options

### Main Options

- `--mode`: Operation mode
  - `sample`: Run with generated sample data
  - `quick_test`: Fast verification
  - `train`: Train with your data
  - `predict`: Predict for new well

### Training Options

- `--time_files`: List of time data CSV files
- `--log_files`: List of log data LAS/CSV files
- `--well_names`: List of well names

### Prediction Options

- `--model_path`: Path to trained model (.h5 file)
- `--time_file`: Time data for prediction
- `--log_file`: Optional log data for prediction
- `--well_name`: Name for the prediction well

### Examples

```bash
# Sample workflow
python rop_prediction_system.py --mode sample

# Training with 3 wells
python rop_prediction_system.py --mode train \
    --time_files well1.csv well2.csv well3.csv \
    --log_files well1.las well2.las well3.las \
    --well_names Well_A Well_B Well_C

# Prediction
python rop_prediction_system.py --mode predict \
    --model_path models/cbt_lstm_20240115.h5 \
    --time_file new_well.csv \
    --well_name New_Well
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all packages are installed
   - Check Python version (3.8+ required)

2. **Memory Errors**
   - Reduce batch size in config
   - Use smaller sequence length
   - Add more RAM or use GPU

3. **Poor Performance**
   - Check data quality and format
   - Ensure sufficient offset wells (minimum 2-3)
   - Verify engineering calculations are reasonable

4. **LAS File Issues**
   - Install `lasio` package
   - Check LAS file format
   - Use CSV format as alternative

### Performance Tuning

- **Increase epochs**: For better convergence (but watch for overfitting)
- **Adjust learning rate**: Lower values (1e-5 to 1e-3) for stable training
- **Modify sequence length**: Longer sequences capture more context
- **Add regularization**: Increase dropout if overfitting

### Data Quality Checks

- **ROP range**: Should be 5-100 ft/h (1.5-30 m/h)
- **WOB range**: Should be 5-50 klbf (22-222 kN)
- **RPM range**: Should be 50-200 rpm
- **Missing data**: Less than 10% gaps preferred

## API Reference

### Main Classes

#### `ROPPredictionSystem`

Main system class for complete workflow.

**Methods:**
- `__init__(config)`: Initialize with configuration
- `run_workflow(time_files, log_files, well_names)`: Run complete training workflow

#### `ROPDataProcessor`

Data processing and feature engineering.

**Methods:**
- `load_and_process()`: Load and process all data
- `create_sequences()`: Create LSTM sequences
- `calculate_features()`: Engineering feature calculations

#### `CBT_LSTMTrainer`

Model training and evaluation.

**Methods:**
- `train()`: Train the CBT-LSTM model
- `evaluate()`: Evaluate model performance

### Key Functions

#### `predict_new_well()`

Predict ROP for a new well.

**Parameters:**
- `model_path`: Path to trained model
- `time_data`: Time-series data (DataFrame)
- `log_data`: Optional log data (DataFrame)
- `well_name`: Well identifier

**Returns:** DataFrame with predictions

#### `create_sample_data()`

Generate synthetic drilling data for testing.

## Configuration

The system uses a configuration dictionary (`DEFAULT_CONFIG`) with the following sections:

### Data Configuration
- `sequence_length`: LSTM input sequence length (default: 50)
- `step_size`: Sliding window step (default: 5)
- `test_size`: Test set fraction (default: 0.2)

### Model Configuration
- `architecture`: Model type (currently "cbt_lstm_advanced")
- `boosting_layers`: Number of channel boosting layers
- `lstm_units`: LSTM hidden units
- `dropout_rate`: Regularization rate

### Training Configuration
- `epochs`: Maximum training epochs
- `batch_size`: Training batch size
- `learning_rate`: Optimizer learning rate

## File Structure

```
rop_prediction_project/
├── rop_prediction_system.py    # Main system file
├── requirements.txt            # Python dependencies
├── README.md                   # This documentation
├── data/                       # Your data (create this)
│   ├── time_data/             # CSV time-series files
│   └── log_data/              # LAS/CSV log files
├── models/                     # Trained models (auto-created)
│   ├── cbt_lstm_*.h5          # Keras model files
│   └── metadata_*.json        # Model metadata
├── outputs/                    # Predictions (auto-created)
│   └── predictions_*.csv      # Prediction results
└── logs/                       # Training logs (auto-created)
```

## Contributing

For improvements or bug reports, please check the code comments and ensure data format compliance.

## License

This system is provided for educational and research purposes in drilling engineering applications.

---

**Version:** 4.0
**Date:** January 2024
**Author:** Drilling Analytics Team</content>
<parameter name="filePath">c:\Users\dusan\Desktop\ROP_prediction\README.md