import pandas as pd
from sklearn.metrics import r2_score
import glob
import os

outputs_dir = 'outputs'
# Find the latest predictions for Sample wells
files = glob.glob(os.path.join(outputs_dir, 'predictions_Sample_Well_*.csv'))

print("Checking R2 Scores:")
for file_path in files:
    try:
        df = pd.read_csv(file_path)
        if 'ROP_Actual' in df.columns and 'ROP_Predicted' in df.columns:
            r2 = r2_score(df['ROP_Actual'], df['ROP_Predicted'])
            print(f"{os.path.basename(file_path)}: R2 = {r2:.4f}")
        else:
            print(f"{os.path.basename(file_path)}: Missing columns")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
