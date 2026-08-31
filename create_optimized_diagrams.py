"""
Performance Diagram Generator for OPTIMIZED ROP Prediction System
=================================================================
Generates comprehensive visualization reports specifically for the OPTIMIZED model.
Now includes Correlation Matrix and Parameter Influence plots.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from rop_prediction_system import predict_new_well
import warnings

warnings.filterwarnings('ignore')

def generate_diagrams():
    # 1. Setup paths
    model_dir = 'models'
    output_dir = 'performance_reports'
    
    # Strategy: Find explicitly Optimized models
    print("Searching for OPTIMIZED models...")
    h5_files = sorted(glob.glob(os.path.join(model_dir, 'cbt_lstm_OPTIMIZED_*.h5')))
    
    if not h5_files:
        print("No optimized models found! Falling back to standard models.")
        h5_files = sorted(glob.glob(os.path.join(model_dir, 'cbt_lstm_*.h5')))
        
    if not h5_files:
        print("No models found!")
        return

    latest_model = h5_files[-1]
    
    try:
        # Expected format: cbt_lstm_OPTIMIZED_YYYYMMDD_HHMMSS.h5
        parts = latest_model.split('_')
        # parts[-2] is date, parts[-1] is time.h5
        timestamp = parts[-2] + '_' + parts[-1].replace('.h5', '')
    except:
        timestamp = "latest_run"
    
    report_folder = os.path.join(output_dir, f"report_OPTIMIZED_{timestamp}")
    os.makedirs(report_folder, exist_ok=True)
    
    print(f"Generating report for model: {latest_model}")
    print(f"Output folder: {report_folder}")
    
    # 2. Check for enhanced data
    data_dir = 'enhanced_sample_data'
    wells_map = {
        'Enhanced_Well_1': 'Well_X_1',
        'Enhanced_Well_2': 'Well_X_2',
        'Enhanced_Well_3': 'Well_X_3'
    }
    
    all_results = []
    
    for filename, display_name in wells_map.items():
        print(f"Processing {filename} (Display: {display_name})...")
        time_path = os.path.join(data_dir, 'time_data', f"{filename}.csv")
        log_path = os.path.join(data_dir, 'log_data', f"{filename}.csv")
        
        if not os.path.exists(time_path):
            print(f"Skipping {filename} (file not found)")
            continue
            
        # Load Data
        df_time = pd.read_csv(time_path)
        df_log = pd.read_csv(log_path) if os.path.exists(log_path) else None
        
        # Run Prediction
        try:
            # results contains: Depth, ROP_Predicted, ROP_Actual, Error
            results = predict_new_well(latest_model, df_time, df_log, well_name=display_name)
            results['Well'] = display_name
            
            # --- MERGE FEATURES FOR ANALYSIS ---
            results = results.sort_values('Depth')
            df_time = df_time.sort_values('MD')
            merged = pd.merge_asof(results, df_time, left_on='Depth', right_on='MD', direction='nearest')
            
            all_results.append(merged)
            
            # 3. Individual Well Depth Plot
            create_depth_plot(merged, display_name, report_folder)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
        
    if not all_results:
        print("No results generated.")
        return

    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # 4. Parity Plot (All Wells)
    create_parity_plot(combined_df, report_folder)
    
    # 5. Residual Distribution
    create_residual_plot(combined_df, report_folder)
    
    # 6. Global Correlation Matrix (Drilling Parameters vs ROP)
    create_correlation_matrix(combined_df, report_folder)
    
    # 7. Parameter Influence Scatter Plots
    create_parameter_scatter(combined_df, report_folder)
    
    # 8. Comparison Table
    create_comparison_table(combined_df, report_folder)
    
    print(f"\n[SUCCESS] All diagrams created in {report_folder}")

def create_depth_plot(df, well_name, output_folder):
    """Creates a Depth vs ROP plot comparing Actual vs Predicted"""
    plt.figure(figsize=(10, 15))
    df = df.sort_values('Depth')
    
    plt.plot(df['ROP_Actual'], df['Depth'], 'k-', label='Actual ROP', linewidth=1.5, alpha=0.7)
    plt.plot(df['ROP_Predicted'], df['Depth'], 'b--', label='Predicted ROP', linewidth=1.5, alpha=0.9)
    
    plt.gca().invert_yaxis()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    
    r2 = r2_score(df['ROP_Actual'], df['ROP_Predicted'])
    mae = mean_absolute_error(df['ROP_Actual'], df['ROP_Predicted'])
    
    plt.title(f"ROP Prediction Profile: {well_name}\nR2: {r2:.3f} | MAE: {mae:.2f} m/h")
    plt.xlabel("Rate of Penetration (m/h)")
    plt.ylabel("Measured Depth (m)")
    
    # Annotations
    try:
        max_idx = df['ROP_Actual'].idxmax()
        max_rop = df.loc[max_idx, 'ROP_Actual']
        max_depth = df.loc[max_idx, 'Depth']
        plt.annotate(f'Max Speed: {max_rop:.1f} m/h\n@ {max_depth:.0f}m', 
                     xy=(max_rop, max_depth), xytext=(max_rop*0.7, max_depth-100),
                     arrowprops=dict(facecolor='red', shrink=0.05))
    except: pass

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"Depth_Plot_{well_name}.png"), dpi=150)
    plt.close()

def create_parity_plot(df, output_folder):
    plt.figure(figsize=(10, 10))
    plt.scatter(df['ROP_Actual'], df['ROP_Predicted'], alpha=0.5, c=df['Depth'], cmap='viridis', s=20)
    plt.colorbar(label='Depth (m)')
    
    min_val = 0
    max_val = max(df['ROP_Actual'].max(), df['ROP_Predicted'].max()) + 5
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    r2 = r2_score(df['ROP_Actual'], df['ROP_Predicted'])
    plt.title(f"Parity Plot (Global R2: {r2:.3f})")
    plt.xlabel("Actual ROP (m/h)")
    plt.ylabel("Predicted ROP (m/h)")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "Parity_Plot_Global.png"), dpi=150)
    plt.close()

def create_residual_plot(df, output_folder):
    plt.figure(figsize=(10, 6))
    residuals = df['ROP_Actual'] - df['ROP_Predicted']
    plt.hist(residuals, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--')
    plt.title("Residual Distribution (Error)")
    plt.xlabel("Error (m/h)")
    plt.ylabel("Count")
    plt.savefig(os.path.join(output_folder, "Residual_Distribution.png"), dpi=150)
    plt.close()

def create_correlation_matrix(df, output_folder):
    """Generates a Correlation Matrix Heatmap for Drilling Parameters vs ROP"""
    potential_cols = ['ROP_Actual', 'ROP_Predicted', 'WOB', 'RPM', 'TORQUE', 'FLOW', 'SPP', 'MW', 'GR', 'RES']
    cols_to_use = [c for c in potential_cols if c in df.columns]
    
    if len(cols_to_use) < 3:
        print("Not enough columns for correlation matrix.")
        return

    corr = df[cols_to_use].corr()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap='coolwarm', interpolation='nearest', aspect='auto')
    plt.colorbar(label='Correlation Coefficient')
    
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='right')
    plt.yticks(range(len(corr.columns)), corr.columns)
    
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            text = plt.text(j, i, f"{corr.iloc[i, j]:.2f}",
                           ha="center", va="center", color="black" if abs(corr.iloc[i, j]) < 0.5 else "white")
            
    plt.title("Correlation Matrix: Drilling Parameters vs ROP")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "Correlation_Matrix.png"), dpi=150)
    plt.close()

def create_parameter_scatter(df, output_folder):
    """Creates scatter plots for Key Parameters vs Actual/Predicted ROP"""
    params = ['WOB', 'RPM', 'TORQUE', 'FLOW']
    valid_params = [p for p in params if p in df.columns]
    
    if not valid_params:
        return
        
    fig, axes = plt.subplots(len(valid_params), 2, figsize=(12, 4 * len(valid_params)))
    if len(valid_params) == 1: axes = [axes] # Handle single case
    
    if hasattr(axes, 'shape') and len(axes.shape) == 1:
        axes = np.array([axes])
        
    for i, param in enumerate(valid_params):
        ax1 = axes[i][0]
        ax1.scatter(df[param], df['ROP_Actual'], alpha=0.3, c='black', s=10)
        ax1.set_xlabel(param)
        ax1.set_ylabel("Actual ROP (m/h)")
        ax1.set_title(f"{param} vs Actual ROP")
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[i][1]
        ax2.scatter(df[param], df['ROP_Predicted'], alpha=0.3, c='blue', s=10)
        ax2.set_xlabel(param)
        ax2.set_ylabel("Predicted ROP (m/h)")
        ax2.set_title(f"{param} vs Predicted ROP")
        ax2.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "Parameter_Influence_Scatter.png"), dpi=150)
    plt.close()

def create_comparison_table(df, output_folder):
    """Creates a detailed performance table with MAE, RMSE, R2"""
    metrics_list = []
    
    for well, group in df.groupby('Well'):
        y_true = group['ROP_Actual']
        y_pred = group['ROP_Predicted']
        error = y_true - y_pred
        
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        bias = np.mean(error)
        
        metrics_list.append({
            'Well': well,
            'R2 Score': r2,
            'MAE (m/h)': mae,
            'RMSE (m/h)': rmse,
            'Bias (m/h)': bias,
            'Mean ROP (m/h)': y_true.mean(),
            'Max ROP (m/h)': y_true.max()
        })
    
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df = metrics_df.round(3)
    metrics_df.set_index('Well', inplace=True)
    
    print("\nPerformance Metrics:")
    print(metrics_df)
    
    metrics_df.to_csv(os.path.join(output_folder, "Performance_Summary_Table.csv"))

if __name__ == "__main__":
    generate_diagrams()