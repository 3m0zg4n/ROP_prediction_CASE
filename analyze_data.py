
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def analyze_correlations(well_name):
    print(f"\n{'='*50}")
    print(f"ANALYZING: {well_name}")
    print(f"{'='*50}")
    
    time_path = f'sample_data/time_data/{well_name}.csv'
    log_path = f'sample_data/log_data/{well_name}.csv'
    
    if not os.path.exists(time_path) or not os.path.exists(log_path):
        print(f"Files not found for {well_name}")
        return

    # Load Data
    df_time = pd.read_csv(time_path)
    df_log = pd.read_csv(log_path)
    
    print(f"Time Data Columns: {list(df_time.columns)}")
    print(f"Log Data Columns: {list(df_log.columns)}")
    
    # Merge on Depth
    # Time data usually has MD, Log data has DEPT
    # Log data is usually sparser or different resolution, so we use merge_asof
    
    df_time = df_time.sort_values('MD')
    df_log = df_log.sort_values('DEPT')
    
    merged = pd.merge_asof(
        df_time, 
        df_log, 
        left_on='MD', 
        right_on='DEPT', 
        direction='nearest',
        tolerance=1.0 # 1 meter tolerance
    )
    
    # distinct features only
    cols_to_corr = [c for c in merged.columns if c not in ['TIMESTAMP', 'WELL_NAME']]
    correlation_matrix = merged[cols_to_corr].corr()
    
    # 1. Correlations with ROP
    print("\n--- CORRELATIONS WITH ROP (Target) ---")
    rop_corr = correlation_matrix['ROP'].sort_values(ascending=False)
    print(rop_corr)
    
    # 2. Strongest Cross-Correlations
    print("\n--- STRONGEST CROSS-CORRELATIONS (|r| > 0.5) ---")
    # Mask diagonal and lower triangle to avoid duplicates
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            col1 = correlation_matrix.columns[i]
            col2 = correlation_matrix.columns[j]
            val = correlation_matrix.iloc[i, j]
            
            if abs(val) > 0.5:
                print(f"{col1} <--> {col2}: {val:.3f}")

if __name__ == "__main__":
    analyze_correlations('Sample_Well_1')
    analyze_correlations('Sample_Well_2')
