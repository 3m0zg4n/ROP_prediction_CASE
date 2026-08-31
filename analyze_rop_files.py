import pandas as pd
import os
import glob

data_dirs = [
    r'c:\Users\dusan\Desktop\ROP_prediction\enhanced_sample_data\time_data',
    r'c:\Users\dusan\Desktop\ROP_prediction\sample_data\time_data',
    r'c:\Users\dusan\Desktop\ROP_prediction'
]
files = []
for d in data_dirs:
    if os.path.exists(d):
        if os.path.isdir(d):
            files.extend(glob.glob(os.path.join(d, '*.csv')))
        else: # it might be a file path if I added one directly, but here I only added dirs. 
            pass # simplified for now

# Check specific file in root
root_file = r'c:\Users\dusan\Desktop\ROP_prediction\timedatawell.csv'
if os.path.exists(root_file) and root_file not in files:
    files.append(root_file)

print(f"Found {len(files)} files.")

for file_path in files:
    try:
        df = pd.read_csv(file_path)
        if 'ROP' in df.columns:
            rop = df['ROP']
            mean_rop = rop.mean()
            min_rop = rop.min()
            max_rop = rop.max()
            
            # Count values in 50-60 range
            count_50_60 = ((rop >= 50) & (rop <= 60)).sum()
            total_rows = len(df)
            percentage = (count_50_60 / total_rows) * 100
            
            print(f"\nFile: {os.path.basename(file_path)}")
            print(f"  Mean ROP: {mean_rop:.2f}")
            print(f"  Range: {min_rop:.2f} - {max_rop:.2f}")
            print(f"  Count 50-60: {count_50_60} ({percentage:.2f}%)")
            
            # Check high ROP segments
            if count_50_60 > 0:
                print("  HAS DATA in 50-60 m/h range")
        else:
            print(f"\nFile: {os.path.basename(file_path)} (No ROP column)")
            
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
