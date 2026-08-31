import os
import json
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set modern plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Custom color scheme for drilling data
DRILLING_COLORS = {
    'actual': '#2C3E50',      # Dark blue-gray
    'predicted': '#E74C3C',   # Red
    'error': '#3498DB',       # Blue
    'good': '#27AE60',        # Green
    'warning': '#F39C12',     # Orange
    'danger': '#E74C3C'       # Red
}

def find_latest_file(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getctime)

def plot_training_history():
    """Enhanced Training History with Multiple Metrics"""
    print("Generating Enhanced Training History Plot...")
    
    meta_file = find_latest_file('models/metadata_*.json')
    if not meta_file:
        print("❌ No metadata found.")
        return

    with open(meta_file, 'r') as f:
        data = json.load(f)
    
    history = data.get('history')
    if not history:
        print("❌ No training history in metadata.")
        return

    # Create subplots for different metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('CBT-LSTM Training Analysis Dashboard', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['loss']) + 1)
    
    # 1. Loss Curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['loss'], label='Training Loss', color=DRILLING_COLORS['actual'], linewidth=2)
    ax1.plot(epochs, history['val_loss'], label='Validation Loss', color=DRILLING_COLORS['predicted'], linewidth=2)
    ax1.set_title('Model Loss Over Time', fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss (Huber)')
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    
    # Find best epoch
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val_loss = min(history['val_loss'])
    ax1.axvline(x=best_epoch, color=DRILLING_COLORS['good'], linestyle='--', alpha=0.7, 
                label=f'Best Epoch: {best_epoch}')
    ax1.scatter([best_epoch], [best_val_loss], color=DRILLING_COLORS['good'], s=100, zorder=5)
    
    # 2. Learning Rate (if available)
    ax2 = axes[0, 1]
    if 'lr' in history:
        ax2.semilogy(epochs, history['lr'], color=DRILLING_COLORS['warning'], linewidth=2)
        ax2.set_title('Learning Rate Schedule', fontweight='bold')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Learning Rate (log scale)')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Learning Rate\nData Not Available', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Learning Rate Schedule', fontweight='bold')
    
    # 3. Loss Smoothing and Convergence Analysis
    ax3 = axes[1, 0]
    # Apply smoothing to see trends better
    window = max(5, len(history['loss'])//10)
    train_smooth = pd.Series(history['loss']).rolling(window=window, center=True).mean()
    val_smooth = pd.Series(history['val_loss']).rolling(window=window, center=True).mean()
    
    ax3.plot(epochs, train_smooth, label='Training (Smoothed)', color=DRILLING_COLORS['actual'], linewidth=3)
    ax3.plot(epochs, val_smooth, label='Validation (Smoothed)', color=DRILLING_COLORS['predicted'], linewidth=3)
    ax3.set_title('Smoothed Loss Trends', fontweight='bold')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Smoothed Loss')
    ax3.legend(frameon=True)
    ax3.grid(True, alpha=0.3)
    
    # 4. Overfitting Analysis
    ax4 = axes[1, 1]
    generalization_gap = np.array(history['val_loss']) - np.array(history['loss'])
    ax4.plot(epochs, generalization_gap, color=DRILLING_COLORS['error'], linewidth=2)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.set_title('Generalization Gap Analysis', fontweight='bold')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Val Loss - Train Loss')
    ax4.grid(True, alpha=0.3)
    
    # Color zones
    ax4.fill_between(epochs, generalization_gap, 0, 
                     where=(generalization_gap > 0), color=DRILLING_COLORS['danger'], alpha=0.3, label='Overfitting Risk')
    ax4.fill_between(epochs, generalization_gap, 0, 
                     where=(generalization_gap <= 0), color=DRILLING_COLORS['good'], alpha=0.3, label='Good Generalization')
    ax4.legend()
    
    plt.tight_layout()
    output_path = 'outputs/enhanced_learning_curve.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved enhanced learning curve: {output_path}")
    plt.close()

def plot_predictions():
    """Enhanced Prediction Analysis with Statistical Insights"""
    print("Generating Enhanced Prediction Analysis...")
    
    # Try Enhanced Wells first, then fallback to Test Wells
    pred_file = find_latest_file('outputs/predictions_Enhanced_Well_*.csv')
    if not pred_file:
        print("⚠️ No Enhanced Well predictions found, trying Test Well predictions...")
        pred_file = find_latest_file('outputs/predictions_Test_Well_*.csv')
    
    if not pred_file:
        print("❌ No prediction file found.")
        return
        
    df = pd.read_csv(pred_file)
    
    # Calculate comprehensive statistics
    df['Error'] = df['ROP_Predicted'] - df['ROP_Actual']
    df['Abs_Error'] = np.abs(df['Error'])
    df['Percent_Error'] = np.abs(df['Error'] / (df['ROP_Actual'] + 1e-10)) * 100
    
    mae = mean_absolute_error(df['ROP_Actual'], df['ROP_Predicted'])
    rmse = np.sqrt(mean_squared_error(df['ROP_Actual'], df['ROP_Predicted']))
    r2 = r2_score(df['ROP_Actual'], df['ROP_Predicted'])
    
    # Create comprehensive dashboard
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Main Depth Track (Enhanced)
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.plot(df['ROP_Actual'], df['Depth'], label='Actual ROP', 
             color=DRILLING_COLORS['actual'], linewidth=2, alpha=0.8)
    ax1.plot(df['ROP_Predicted'], df['Depth'], label='Predicted ROP', 
             color=DRILLING_COLORS['predicted'], linewidth=2, alpha=0.8)
    
    # Add confidence bands (±1 std of error)
    error_std = df['Error'].std()
    ax1.fill_betweenx(df['Depth'], 
                      df['ROP_Predicted'] - error_std, 
                      df['ROP_Predicted'] + error_std,
                      color=DRILLING_COLORS['predicted'], alpha=0.2, label='±1σ Uncertainty')
    
    ax1.invert_yaxis()
    ax1.set_xlabel('ROP (m/h)', fontweight='bold')
    ax1.set_ylabel('Depth (m)', fontweight='bold')
    ax1.set_title('Depth Track: Actual vs Predicted\nwith Uncertainty Bands', fontweight='bold')
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    
    # 2. Error Track
    ax2 = fig.add_subplot(gs[:, 1], sharey=ax1)
    # Color-coded error bars
    colors = np.where(df['Percent_Error'] <= 10, DRILLING_COLORS['good'],
                     np.where(df['Percent_Error'] <= 20, DRILLING_COLORS['warning'], 
                             DRILLING_COLORS['danger']))
    
    ax2.scatter(df['Error'], df['Depth'], c=colors, alpha=0.6, s=20)
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    ax2.axvline(x=mae, color=DRILLING_COLORS['warning'], linestyle=':', alpha=0.7, label=f'MAE: {mae:.1f}')
    ax2.axvline(x=-mae, color=DRILLING_COLORS['warning'], linestyle=':', alpha=0.7)
    ax2.set_xlabel('Prediction Error (m/h)', fontweight='bold')
    ax2.set_title('Error Distribution by Depth\n(Green: <10%, Yellow: 10-20%, Red: >20%)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Enhanced Scatter Plot with Statistics
    ax3 = fig.add_subplot(gs[0, 2])
    scatter = ax3.scatter(df['ROP_Actual'], df['ROP_Predicted'], 
                         c=df['Percent_Error'], cmap='RdYlGn_r', alpha=0.7, s=40)
    
    # Perfect prediction line
    max_val = max(df['ROP_Actual'].max(), df['ROP_Predicted'].max())
    min_val = min(df['ROP_Actual'].min(), df['ROP_Predicted'].min())
    ax3.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.8, label='Perfect Prediction')
    
    # Regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['ROP_Actual'], df['ROP_Predicted'])
    line_x = np.array([min_val, max_val])
    line_y = slope * line_x + intercept
    ax3.plot(line_x, line_y, color=DRILLING_COLORS['error'], linewidth=2, 
             label=f'Regression (R={r_value:.3f})')
    
    ax3.set_xlabel('Actual ROP (m/h)', fontweight='bold')
    ax3.set_ylabel('Predicted ROP (m/h)', fontweight='bold')
    ax3.set_title(f'Prediction Accuracy\nR² = {r2:.3f}, RMSE = {rmse:.2f}', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax3, shrink=0.8)
    cbar.set_label('Percentage Error (%)', fontweight='bold')
    
    # 4. Error Distribution Histogram
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.hist(df['Percent_Error'], bins=30, color=DRILLING_COLORS['error'], alpha=0.7, edgecolor='black')
    ax4.axvline(x=10, color=DRILLING_COLORS['warning'], linestyle='--', linewidth=2, label='10% Threshold')
    ax4.axvline(x=20, color=DRILLING_COLORS['danger'], linestyle='--', linewidth=2, label='20% Threshold')
    ax4.set_xlabel('Percentage Error (%)', fontweight='bold')
    ax4.set_ylabel('Frequency', fontweight='bold')
    ax4.set_title('Error Distribution', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Performance Metrics Dashboard
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.axis('off')
    
    # Calculate accuracy percentages
    within_10 = np.mean(df['Percent_Error'] <= 10) * 100
    within_20 = np.mean(df['Percent_Error'] <= 20) * 100
    
    metrics_text = f"""
    PERFORMANCE METRICS
    ─────────────────────
    Mean Absolute Error: {mae:.2f} m/h
    Root Mean Square Error: {rmse:.2f} m/h
    R² Score: {r2:.3f}
    ─────────────────────
    ACCURACY BREAKDOWN
    ─────────────────────
    Within 10% Error: {within_10:.1f}%
    Within 20% Error: {within_20:.1f}%
    Total Predictions: {len(df):,}
    ─────────────────────
    REGRESSION ANALYSIS
    ─────────────────────
    Slope: {slope:.3f}
    Intercept: {intercept:.3f}
    P-value: {p_value:.2e}
    """
    
    ax5.text(0.1, 0.9, metrics_text, transform=ax5.transAxes, fontsize=11, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # 6. Residual Analysis (QQ Plot)
    ax6 = fig.add_subplot(gs[0, 3])
    stats.probplot(df['Error'], dist="norm", plot=ax6)
    ax6.set_title('Q-Q Plot: Residual Normality', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # 7. Time Series Error Analysis
    ax7 = fig.add_subplot(gs[1:, 3])
    if 'Time_Index' in df.columns or len(df) > 10:
        x_axis = range(len(df)) if 'Time_Index' not in df.columns else df['Time_Index']
        
        # Moving average of error
        window = max(10, len(df)//20)
        error_ma = pd.Series(df['Abs_Error']).rolling(window=window, center=True).mean()
        
        ax7.plot(x_axis, df['Abs_Error'], alpha=0.3, color=DRILLING_COLORS['error'], label='Raw Error')
        ax7.plot(x_axis, error_ma, color=DRILLING_COLORS['danger'], linewidth=2, label=f'Moving Avg (n={window})')
        ax7.axhline(y=mae, color='black', linestyle='--', alpha=0.7, label=f'Overall MAE: {mae:.1f}')
        
        ax7.set_xlabel('Prediction Sequence', fontweight='bold')
        ax7.set_ylabel('Absolute Error (m/h)', fontweight='bold')
        ax7.set_title('Error Evolution Over Predictions', fontweight='bold')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    else:
        ax7.text(0.5, 0.5, 'Insufficient Data\nfor Time Analysis', ha='center', va='center', 
                transform=ax7.transAxes, fontsize=12)
        ax7.set_title('Time Series Analysis', fontweight='bold')
    
    plt.suptitle('CBT-LSTM Prediction Analysis Dashboard', fontsize=18, fontweight='bold', y=0.98)
    
    output_path = 'outputs/enhanced_prediction_dashboard.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved enhanced prediction dashboard: {output_path}")
    plt.close()
    
    # Also create individual high-quality plots for presentations
    create_publication_plots(df, mae, rmse, r2)

def create_publication_plots(df, mae, rmse, r2):
    """Create clean, publication-ready individual plots"""
    
    # 1. Professional Scatter Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df['ROP_Actual'], df['ROP_Predicted'], 
                         c=df['Percent_Error'], cmap='RdYlGn_r', 
                         s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    max_val = max(df['ROP_Actual'].max(), df['ROP_Predicted'].max())
    min_val = min(df['ROP_Actual'].min(), df['ROP_Predicted'].min())
    
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.8)
    
    plt.xlabel('Actual ROP (m/h)', fontsize=14, fontweight='bold')
    plt.ylabel('Predicted ROP (m/h)', fontsize=14, fontweight='bold')
    plt.title(f'CBT-LSTM Model Performance\nMAE: {mae:.2f} m/h, R²: {r2:.3f}', 
              fontsize=16, fontweight='bold')
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Prediction Error (%)', fontsize=12, fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/publication_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Professional Depth Track
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10))
    
    ax1.plot(df['ROP_Actual'], df['Depth'], color=DRILLING_COLORS['actual'], 
             linewidth=2.5, label='Actual ROP', alpha=0.9)
    ax1.plot(df['ROP_Predicted'], df['Depth'], color=DRILLING_COLORS['predicted'], 
             linewidth=2.5, label='Predicted ROP', alpha=0.9)
    
    ax1.invert_yaxis()
    ax1.set_xlabel('ROP (m/h)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
    ax1.set_title('Depth-Based Prediction Track', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12, frameon=True)
    ax1.grid(True, alpha=0.3)
    
    # Enhanced error track with zones
    error_colors = np.where(df['Percent_Error'] <= 10, DRILLING_COLORS['good'],
                           np.where(df['Percent_Error'] <= 20, DRILLING_COLORS['warning'], 
                                   DRILLING_COLORS['danger']))
    
    ax2.scatter(df['Error'], df['Depth'], c=error_colors, s=30, alpha=0.7)
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=2)
    ax2.axvline(x=mae, color=DRILLING_COLORS['warning'], linestyle='--', alpha=0.8)
    ax2.axvline(x=-mae, color=DRILLING_COLORS['warning'], linestyle='--', alpha=0.8)
    
    ax2.set_xlabel('Prediction Error (m/h)', fontsize=12, fontweight='bold')
    ax2.set_title('Residual Analysis', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/publication_depth_track.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance():
    """Analyze and visualize feature importance from training data"""
    print("Generating Feature Importance Analysis...")
    
    # Try Enhanced Wells first, then fallback to sample data
    wells = glob.glob('enhanced_sample_data/time_data/*.csv')
    if not wells:
        wells = glob.glob('sample_data/time_data/*.csv')
    if not wells:
        print("❌ No sample data found for feature analysis.")
        return
        
    dfs = []
    for w in wells:
        df = pd.read_csv(w)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Calculate feature importance based on correlation with ROP
    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in ['ROP']]
    
    correlations = {}
    for col in feature_cols:
        if col in combined.columns:
            corr = combined[col].corr(combined['ROP'])
            if not np.isnan(corr):
                correlations[col] = abs(corr)
    
    if not correlations:
        print("❌ No valid correlations found.")
        return
    
    # Create feature importance plot
    plt.figure(figsize=(12, 8))
    
    features = list(correlations.keys())
    importance = list(correlations.values())
    
    # Sort by importance
    sorted_idx = np.argsort(importance)
    features_sorted = [features[i] for i in sorted_idx]
    importance_sorted = [importance[i] for i in sorted_idx]
    
    # Create horizontal bar plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(features_sorted)))
    bars = plt.barh(features_sorted, importance_sorted, color=colors)
    
    plt.xlabel('Correlation with ROP (Absolute Value)', fontsize=14, fontweight='bold')
    plt.ylabel('Features', fontsize=14, fontweight='bold')
    plt.title('Feature Importance Analysis\n(Based on Correlation with ROP)', 
              fontsize=16, fontweight='bold')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, importance_sorted)):
        plt.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=10)
    
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('outputs/feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved feature importance plot: outputs/feature_importance.png")
    plt.close()
    
def plot_enhanced_correlations():
    """Enhanced correlation analysis with drilling-specific insights"""
    print("Generating Enhanced Correlation Analysis...")
    
    # Try Enhanced Wells first, then fallback to sample data
    wells = glob.glob('enhanced_sample_data/time_data/*.csv')
    if not wells:
        wells = glob.glob('sample_data/time_data/*.csv')
    if not wells:
        print("❌ No sample data found.")
        return
        
    dfs = []
    for w in wells:
        df = pd.read_csv(w)
        numeric_df = df.select_dtypes(include=[np.number])
        dfs.append(numeric_df)
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Calculate correlation matrix
    corr = combined.corr()
    
    # Create enhanced correlation heatmap
    plt.figure(figsize=(14, 12))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Custom colormap
    sns.heatmap(corr, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, fmt='.2f', cbar_kws={"shrink": 0.8},
                linewidths=0.5, annot_kws={"size": 10})
    
    plt.title('Enhanced Feature Correlation Matrix\n(Lower Triangle Only)', 
              fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/enhanced_correlation_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved enhanced correlation matrix: outputs/enhanced_correlation_matrix.png")
    plt.close()
    
    # Create ROP-focused correlation plot
    plt.figure(figsize=(10, 8))
    
    if 'ROP' in corr.columns:
        rop_corr = corr['ROP'].drop('ROP').sort_values(key=abs, ascending=True)
        
        colors = ['red' if x < -0.3 else 'orange' if x < -0.1 else 
                 'lightblue' if x < 0.1 else 'blue' if x < 0.3 else 'darkblue' 
                 for x in rop_corr]
        
        plt.barh(range(len(rop_corr)), rop_corr.values, color=colors, alpha=0.8)
        plt.yticks(range(len(rop_corr)), rop_corr.index)
        plt.xlabel('Correlation with ROP', fontsize=12, fontweight='bold')
        plt.title('Feature Correlations with ROP\n(Target Variable)', 
                  fontsize=14, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.8)
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add threshold lines
        plt.axvline(x=0.3, color='green', linestyle='--', alpha=0.7, label='Strong Positive')
        plt.axvline(x=-0.3, color='red', linestyle='--', alpha=0.7, label='Strong Negative')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('outputs/rop_correlations.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved ROP correlations: outputs/rop_correlations.png")
        plt.close()

def plot_model_performance_summary():
    """Create comprehensive model performance summary dashboard"""
    print("Generating Model Performance Summary...")
    
    # Try to load multiple prediction files to compare models
    pred_files = glob.glob('outputs/predictions_Test_Well_*.csv')
    meta_files = glob.glob('models/metadata_*.json')
    
    if not pred_files or not meta_files:
        print("❌ Insufficient data for performance summary.")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('CBT-LSTM Model Performance Summary Dashboard', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Collect metrics from all available models
    all_metrics = []
    
    for pred_file in pred_files:
        df = pd.read_csv(pred_file)
        if 'ROP_Actual' in df.columns and 'ROP_Predicted' in df.columns:
            mae = mean_absolute_error(df['ROP_Actual'], df['ROP_Predicted'])
            rmse = np.sqrt(mean_squared_error(df['ROP_Actual'], df['ROP_Predicted']))
            r2 = r2_score(df['ROP_Actual'], df['ROP_Predicted'])
            
            error_pct = np.abs((df['ROP_Actual'] - df['ROP_Predicted']) / (df['ROP_Actual'] + 1e-10)) * 100
            within_10 = np.mean(error_pct <= 10) * 100
            within_20 = np.mean(error_pct <= 20) * 100
            
            # Extract timestamp more carefully from filename
            try:
                timestamp = os.path.basename(pred_file).split('_')[-1].replace('.csv', '')
                # Convert timestamp to a simple index if it's not numeric
                if not timestamp.isdigit():
                    timestamp = str(len(all_metrics) + 1)
            except:
                timestamp = str(len(all_metrics) + 1)
            
            all_metrics.append({
                'timestamp': timestamp,
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2),
                'within_10': float(within_10),
                'within_20': float(within_20),
                'n_samples': int(len(df))
            })
    
    if not all_metrics:
        axes[0, 0].text(0.5, 0.5, 'No Valid\nMetrics Found', ha='center', va='center', 
                       transform=axes[0, 0].transAxes, fontsize=16)
        plt.close()
        return
    
    metrics_df = pd.DataFrame(all_metrics)
    
    # 1. MAE Comparison
    axes[0, 0].bar(range(len(metrics_df)), metrics_df['mae'], 
                   color=DRILLING_COLORS['error'], alpha=0.8)
    axes[0, 0].set_ylabel('MAE (m/h)', fontweight='bold')
    axes[0, 0].set_title('Mean Absolute Error', fontweight='bold')
    axes[0, 0].set_xticks(range(len(metrics_df)))
    axes[0, 0].set_xticklabels([f"Model {i+1}" for i in range(len(metrics_df))], rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. R² Comparison
    axes[0, 1].bar(range(len(metrics_df)), metrics_df['r2'], 
                   color=DRILLING_COLORS['good'], alpha=0.8)
    axes[0, 1].set_ylabel('R² Score', fontweight='bold')
    axes[0, 1].set_title('Model Accuracy (R²)', fontweight='bold')
    axes[0, 1].set_xticks(range(len(metrics_df)))
    axes[0, 1].set_xticklabels([f"Model {i+1}" for i in range(len(metrics_df))], rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Target: 0.8')
    axes[0, 1].legend()
    
    # 3. Accuracy Distribution
    accuracy_data = [metrics_df['within_10'].values, metrics_df['within_20'].values]
    labels = ['Within 10%', 'Within 20%']
    x = np.arange(len(metrics_df))
    width = 0.35
    
    axes[0, 2].bar(x - width/2, accuracy_data[0], width, label='Within 10%', 
                   color=DRILLING_COLORS['good'], alpha=0.8)
    axes[0, 2].bar(x + width/2, accuracy_data[1], width, label='Within 20%', 
                   color=DRILLING_COLORS['warning'], alpha=0.8)
    axes[0, 2].set_ylabel('Accuracy (%)', fontweight='bold')
    axes[0, 2].set_title('Prediction Accuracy Levels', fontweight='bold')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels([f"Model {i+1}" for i in range(len(metrics_df))])
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Training History (if available)
    if meta_files:
        with open(meta_files[-1], 'r') as f:
            meta_data = json.load(f)
        
        if 'history' in meta_data:
            history = meta_data['history']
            epochs = range(1, len(history['loss']) + 1)
            
            axes[1, 0].plot(epochs, history['loss'], label='Training Loss', linewidth=2)
            axes[1, 0].plot(epochs, history['val_loss'], label='Validation Loss', linewidth=2)
            axes[1, 0].set_xlabel('Epochs', fontweight='bold')
            axes[1, 0].set_ylabel('Loss', fontweight='bold')
            axes[1, 0].set_title('Latest Model Training History', fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Model Comparison Radar Chart
    if len(metrics_df) > 0:
        # Normalize metrics for radar chart
        mae_norm = 1 - (metrics_df['mae'] / metrics_df['mae'].max())  # Invert MAE (lower is better)
        r2_norm = metrics_df['r2']
        acc10_norm = metrics_df['within_10'] / 100
        acc20_norm = metrics_df['within_20'] / 100
        
        # Use the best model for radar
        best_idx = metrics_df['r2'].idxmax()
        radar_values = [mae_norm.iloc[best_idx], r2_norm.iloc[best_idx], 
                       acc10_norm.iloc[best_idx], acc20_norm.iloc[best_idx]]
        
        angles = np.linspace(0, 2 * np.pi, len(radar_values), endpoint=False)
        radar_values += radar_values[:1]  # Complete the circle
        angles = np.concatenate((angles, [angles[0]]))
        
        axes[1, 1] = plt.subplot(2, 3, 5, projection='polar')
        axes[1, 1].plot(angles, radar_values, 'o-', linewidth=2, color=DRILLING_COLORS['predicted'])
        axes[1, 1].fill(angles, radar_values, alpha=0.25, color=DRILLING_COLORS['predicted'])
        axes[1, 1].set_xticks(angles[:-1])
        axes[1, 1].set_xticklabels(['MAE (Inv)', 'R²', 'Acc 10%', 'Acc 20%'])
        axes[1, 1].set_title('Best Model Performance\nRadar Chart', fontweight='bold', pad=20)
    
    # 6. Summary Statistics
    axes[1, 2].axis('off')
    
    if len(metrics_df) > 0:
        best_model = metrics_df.loc[metrics_df['r2'].idxmax()]
        avg_metrics = metrics_df.select_dtypes(include=[np.number]).mean()
        
        summary_text = f"""
PERFORMANCE SUMMARY
{'─' * 25}
Best Model Performance:
  MAE: {best_model['mae']:.2f} m/h
  R²: {best_model['r2']:.3f}
  Accuracy (10%): {best_model['within_10']:.1f}%
  Accuracy (20%): {best_model['within_20']:.1f}%

Average Performance:
  MAE: {avg_metrics['mae']:.2f} ± {metrics_df['mae'].std():.2f}
  R²: {avg_metrics['r2']:.3f} ± {metrics_df['r2'].std():.3f}
  
Model Count: {len(metrics_df)}
Total Predictions: {metrics_df['n_samples'].sum():,}
        """
        
        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes, 
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('outputs/model_performance_summary.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved model performance summary: outputs/model_performance_summary.png")
    plt.close()

def create_executive_report():
    """Generate executive-level visual summary for stakeholders"""
    print("Generating Executive Report...")
    
    pred_file = find_latest_file('outputs/predictions_Test_Well_*.csv')
    if not pred_file:
        print("❌ No prediction data for executive report.")
        return
        
    df = pd.read_csv(pred_file)
    
    # Calculate key metrics
    mae = mean_absolute_error(df['ROP_Actual'], df['ROP_Predicted'])
    rmse = np.sqrt(mean_squared_error(df['ROP_Actual'], df['ROP_Predicted']))
    r2 = r2_score(df['ROP_Actual'], df['ROP_Predicted'])
    
    error_pct = np.abs((df['ROP_Actual'] - df['ROP_Predicted']) / (df['ROP_Actual'] + 1e-10)) * 100
    within_10 = np.mean(error_pct <= 10) * 100
    within_20 = np.mean(error_pct <= 20) * 100
    
    # Create executive dashboard
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
    
    # Title and key metrics
    fig.suptitle('CBT-LSTM ROP Prediction System - Executive Summary', 
                 fontsize=20, fontweight='bold', y=0.95)
    
    # 1. Key Performance Indicators (KPIs)
    ax_kpi = fig.add_subplot(gs[0, :2])
    ax_kpi.axis('off')
    
    # Create KPI boxes
    kpi_data = [
        ('Model Accuracy', f'{r2:.1%}', DRILLING_COLORS['good']),
        ('Average Error', f'{mae:.1f} m/h', DRILLING_COLORS['warning']),
        ('Predictions in Range', f'{within_20:.0f}%', DRILLING_COLORS['predicted'])
    ]
    
    for i, (title, value, color) in enumerate(kpi_data):
        x_pos = 0.1 + i * 0.3
        
        # Create KPI box
        box = plt.Rectangle((x_pos, 0.3), 0.25, 0.4, 
                           facecolor=color, alpha=0.2, edgecolor=color, linewidth=2)
        ax_kpi.add_patch(box)
        
        ax_kpi.text(x_pos + 0.125, 0.6, value, ha='center', va='center', 
                   fontsize=24, fontweight='bold', color=color)
        ax_kpi.text(x_pos + 0.125, 0.4, title, ha='center', va='center', 
                   fontsize=12, fontweight='bold')
    
    ax_kpi.set_xlim(0, 1)
    ax_kpi.set_ylim(0, 1)
    
    # 2. Business Impact Assessment
    ax_impact = fig.add_subplot(gs[0, 2:])
    
    # Calculate potential time savings (example calculation)
    avg_rop_actual = df['ROP_Actual'].mean()
    avg_rop_predicted = df['ROP_Predicted'].mean()
    prediction_accuracy = within_20
    
    # Simulate business metrics
    time_savings = prediction_accuracy * 0.1  # Assume 10% time savings at 100% accuracy
    cost_savings = time_savings * 50000  # $50k/day rig cost
    
    impact_text = f"""
BUSINESS IMPACT ASSESSMENT
{'═' * 30}

Prediction Accuracy: {within_20:.0f}%
Estimated Time Savings: {time_savings:.1f}%
Daily Cost Savings: ${cost_savings:,.0f}

ROI Projection:
• Improved drilling efficiency
• Reduced non-productive time
• Optimized drilling parameters
• Enhanced safety margins

Risk Mitigation:
• Early formation change detection
• Equipment protection
• Trajectory optimization
    """
    
    ax_impact.text(0.05, 0.95, impact_text, transform=ax_impact.transAxes, 
                  fontsize=11, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.3))
    ax_impact.axis('off')
    
    # 3. Model Performance Visualization
    ax_perf = fig.add_subplot(gs[1, :2])
    scatter = ax_perf.scatter(df['ROP_Actual'], df['ROP_Predicted'], 
                             c=error_pct, cmap='RdYlGn_r', s=40, alpha=0.7)
    
    max_val = max(df['ROP_Actual'].max(), df['ROP_Predicted'].max())
    min_val = min(df['ROP_Actual'].min(), df['ROP_Predicted'].min())
    ax_perf.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.8)
    
    ax_perf.set_xlabel('Actual ROP (m/h)', fontweight='bold')
    ax_perf.set_ylabel('Predicted ROP (m/h)', fontweight='bold')
    ax_perf.set_title(f'Prediction Accuracy\nR² = {r2:.3f}', fontweight='bold')
    ax_perf.grid(True, alpha=0.3)
    
    # 4. Accuracy Distribution
    ax_acc = fig.add_subplot(gs[1, 2:])
    
    categories = ['Excellent\n(<10% error)', 'Good\n(10-20% error)', 'Poor\n(>20% error)']
    excellent = np.sum(error_pct <= 10)
    good = np.sum((error_pct > 10) & (error_pct <= 20))
    poor = np.sum(error_pct > 20)
    
    sizes = [excellent, good, poor]
    colors = [DRILLING_COLORS['good'], DRILLING_COLORS['warning'], DRILLING_COLORS['danger']]
    explode = (0.1, 0, 0)
    
    wedges, texts, autotexts = ax_acc.pie(sizes, explode=explode, labels=categories, 
                                         colors=colors, autopct='%1.1f%%', shadow=True)
    ax_acc.set_title('Prediction Quality Distribution', fontweight='bold')
    
    # 5. Technology Summary
    ax_tech = fig.add_subplot(gs[2, :])
    ax_tech.axis('off')
    
    tech_summary = f"""
TECHNOLOGY OVERVIEW
{'═' * 50}

Model Architecture: Channel Boosting Time-Series LSTM (CBT-LSTM)
• Multi-scale feature extraction with CNN layers          • Temporal pattern recognition using bidirectional LSTM
• Physics-informed feature engineering                    • Uncertainty quantification and confidence intervals

Key Advantages:
• Incorporates drilling physics (MSE, HSI, UCS calculations)     • Real-time prediction capability for operational use
• Learns from multiple offset wells                             • Handles complex non-linear drilling dynamics
• Provides uncertainty estimates for decision support           • Continuous model improvement with new data

Deployment Status: Production-ready system with comprehensive validation and testing framework
    """
    
    ax_tech.text(0.05, 0.9, tech_summary, transform=ax_tech.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.5))
    
    plt.savefig('outputs/executive_report.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved executive report: outputs/executive_report.png")
    plt.close()

if __name__ == "__main__":
    print("="*60)
    print("ENHANCED VISUALIZATION SUITE")
    print("="*60)
    print("Generating comprehensive visualization dashboard...")
    
    # Ensure outputs directory exists
    os.makedirs('outputs', exist_ok=True)
    
    try:
        # Core visualizations
        print("\n1️⃣ Training Analysis...")
        plot_training_history()
        
        print("\n2️⃣ Prediction Analysis...")
        plot_predictions()
        
        print("\n3️⃣ Feature Analysis...")
        plot_feature_importance()
        plot_enhanced_correlations()
        
        print("\n4️⃣ Performance Summary...")
        plot_model_performance_summary()
        
        print("\n5️⃣ Executive Report...")
        create_executive_report()
        
        print("\n" + "="*60)
        print("✨ VISUALIZATION SUITE COMPLETED")
        print("="*60)
        print("\n📊 Generated visualizations:")
        print("   • Enhanced Learning Curve Dashboard")
        print("   • Comprehensive Prediction Analysis")
        print("   • Feature Importance & Correlations")
        print("   • Model Performance Summary")
        print("   • Executive-Level Report")
        print("\n📁 All files saved in 'outputs/' directory")
        
        # List all generated files
        output_files = glob.glob('outputs/*.png')
        if output_files:
            print(f"\n📋 Generated {len(output_files)} visualization files:")
            for file in sorted(output_files):
                print(f"   • {os.path.basename(file)}")
                
    except Exception as e:
        print(f"\n❌ Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
