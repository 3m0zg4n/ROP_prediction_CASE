"""
Performance Diagram Generator for ROP Prediction System
=======================================================
Generates comprehensive visualization reports for the latest trained model.
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
    
    # Simple strategy to get the absolute text latest model
    h5_files = sorted(glob.glob(os.path.join(model_dir, 'cbt_lstm_*.h5')))
    if not h5_files:
        print("No models found!")
        return

    latest_model = h5_files[-1]
    # Extract timestamp properly depending on filename format
    # expected: cbt_lstm_YYYYMMDD_HHMMSS.h5
    try:
        parts = latest_model.split('_')
        timestamp = parts[-2] + '_' + parts[-1].replace('.h5', '')
    except:
        timestamp = "latest_run"
    
    report_folder = os.path.join(output_dir, f"report_{timestamp}")
    os.makedirs(report_folder, exist_ok=True)
    
    print(f"Generating report for model: {latest_model}")
    print(f"Output folder: {report_folder}")
    
    # 2. Check for enhanced data
    data_dir = 'enhanced_sample_data'
    wells = ['Enhanced_Well_1', 'Enhanced_Well_2', 'Enhanced_Well_3']
    
    all_results = []
    
    for well in wells:
        print(f"Processing {well}...")
        time_path = os.path.join(data_dir, 'time_data', f"{well}.csv")
        log_path = os.path.join(data_dir, 'log_data', f"{well}.csv")
        
        if not os.path.exists(time_path):
            print(f"Skipping {well} (file not found)")
            continue
            
        # Load Data
        df_time = pd.read_csv(time_path)
        df_log = pd.read_csv(log_path) if os.path.exists(log_path) else None
        
        # Run Prediction
        # Note: predict_new_well returns a DataFrame with 'Depth', 'ROP_Predicted', 'ROP_Actual', 'Error'
        try:
            results = predict_new_well(latest_model, df_time, df_log, well_name=well)
            results['Well'] = well
            all_results.append(results)
            
            # 3. Individual Well Depth Plot
            create_depth_plot(results, well, report_folder)
        except Exception as e:
            print(f"Error processing {well}: {e}")
        
    if not all_results:
        print("No results generated.")
        return

    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # 4. Parity Plot (All Wells)
    create_parity_plot(combined_df, report_folder)
    
    # 5. Residual Distribution
    create_residual_plot(combined_df, report_folder)
    
    # 6. Time/Depth Series Zoom (Optional - helps see 1-5m/h vs faster)
    create_comparison_table(combined_df, report_folder)
    
    print(f"\n[SUCCESS] All diagrams created in {report_folder}")

def create_depth_plot(df, well_name, output_folder):
    """Creates a Depth vs ROP plot comparing Actual vs Predicted"""
    plt.figure(figsize=(10, 15))
    
    # Sort by depth just in case
    df = df.sort_values('Depth')
    
    plt.plot(df['ROP_Actual'], df['Depth'], 'k-', label='Actual ROP', linewidth=1.5, alpha=0.7)
    plt.plot(df['ROP_Predicted'], df['Depth'], 'b--', label='Predicted ROP', linewidth=1.5, alpha=0.9)
    
    plt.gca().invert_yaxis()  # Depth increases downwards
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    
    # Metrics for this well
    r2 = r2_score(df['ROP_Actual'], df['ROP_Predicted'])
    mae = mean_absolute_error(df['ROP_Actual'], df['ROP_Predicted'])
    
    plt.title(f"ROP Prediction Profile: {well_name}\nR2: {r2:.3f} | MAE: {mae:.2f} m/h")
    plt.xlabel("Rate of Penetration (m/h)")
    plt.ylabel("Measured Depth (m)")
    
    # Add coordinate annotations for peaks
    try:
        max_idx = df['ROP_Actual'].idxmax()
        max_rop = df.loc[max_idx, 'ROP_Actual']
        max_depth = df.loc[max_idx, 'Depth']
        plt.annotate(f'Max Speed: {max_rop:.1f} m/h\n@ {max_depth:.0f}m', 
                     xy=(max_rop, max_depth), xytext=(max_rop*0.7, max_depth-100),
                     arrowprops=dict(facecolor='red', shrink=0.05))
                     
        # Find hardest rock (slowest ROP)
        min_idx = df['ROP_Actual'].idxmin()
        min_rop = df.loc[min_idx, 'ROP_Actual']
        min_depth = df.loc[min_idx, 'Depth']
        plt.annotate(f'Hard Rock: {min_rop:.1f} m/h\n@ {min_depth:.0f}m', 
                     xy=(min_rop, min_depth), xytext=(min_rop+5, min_depth+100),
                     arrowprops=dict(facecolor='black', shrink=0.05))
    except:
        pass

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"Depth_Plot_{well_name}.png"), dpi=150)
    plt.close()

def create_parity_plot(df, output_folder):
    """Creates a Predicted vs Actual scatter plot"""
    plt.figure(figsize=(12, 10))
    
    # Scatter plot
    plt.scatter(df['ROP_Actual'], df['ROP_Predicted'], 
                alpha=0.6, c=df['Depth'], cmap='viridis', label='ROP Samples', s=30)
    cbar = plt.colorbar(label='Depth (m)')
    
    # Ideal 1:1 line
    min_val = 0
    max_val = max(df['ROP_Actual'].max(), df['ROP_Predicted'].max()) + 5
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Add confidence bands (approximate +/- 5 m/h)
    plt.fill_between([min_val, max_val], 
                     [min_val-5, max_val-5], 
                     [min_val+5, max_val+5], color='gray', alpha=0.1, label='+/- 5 m/h Margin')
    
    # Metrics
    r2 = r2_score(df['ROP_Actual'], df['ROP_Predicted'])
    mae = mean_absolute_error(df['ROP_Actual'], df['ROP_Predicted'])
    mse = mean_squared_error(df['ROP_Actual'], df['ROP_Predicted'])
    
    # Add text box with metrics
    textstr = '\n'.join((
        f'R2 Score = {r2:.3f}',
        f'MAE = {mae:.2f} m/h',
        f'RMSE = {np.sqrt(mse):.2f} m/h',
        f'N Sample = {len(df):,}'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
                   verticalalignment='top', bbox=props)
    
    plt.title(f"Parity Plot: Actual vs Predicted ROP\nGlobal Performance Across All Wells")
    plt.xlabel("Actual ROP (m/h)")
    plt.ylabel("Predicted ROP (m/h)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "Parity_Plot_Global.png"), dpi=150)
    plt.close()

def create_residual_plot(df, output_folder):
    """Creates a distribution plot of errors"""
    plt.figure(figsize=(10, 6))
    
    # Calculate residuals
    residuals = df['ROP_Actual'] - df['ROP_Predicted']
    
    # Histogram
    plt.hist(residuals, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    
    # Add stats
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    
    plt.title(f"Error Distribution (Residuals)\nMean Bias: {mean_res:.2f} m/h | StdDev: {std_res:.2f} m/h")
    plt.xlabel("Prediction Error (Actual - Predicted) [m/h]")
    plt.ylabel("Frequency (Count)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "Residual_Distribution.png"), dpi=150)
    plt.close()

def create_comparison_table(df, output_folder):
    """Saves a summary CSV"""
    summary = df.groupby('Well').agg({
        'Error': ['mean', 'std', lambda x: np.sqrt(np.mean(x**2))],
        'ROP_Actual': ['mean', 'max']
    }).round(2)
    
    summary.to_csv(os.path.join(output_folder, "Performance_Summary_Table.csv"))

if __name__ == "__main__":
    generate_diagrams()