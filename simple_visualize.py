#!/usr/bin/env python3
"""
Simple Visualization Script for ROP Prediction System
================================================================================
This is a simplified version that checks for dependencies and provides
fallback options if advanced packages are not available.
"""

import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    missing_packages = []
    
    try:
        import pandas
        print("✓ pandas found")
    except ImportError:
        missing_packages.append("pandas")
        print("✗ pandas missing")
    
    try:
        import numpy
        print("✓ numpy found") 
    except ImportError:
        missing_packages.append("numpy")
        print("✗ numpy missing")
        
    try:
        import matplotlib
        print("✓ matplotlib found")
    except ImportError:
        missing_packages.append("matplotlib")
        print("✗ matplotlib missing")
        
    try:
        import seaborn
        print("✓ seaborn found")
    except ImportError:
        missing_packages.append("seaborn")
        print("✗ seaborn missing")
    
    return missing_packages

def install_packages(packages):
    """Attempt to install missing packages"""
    print(f"\\nAttempting to install: {', '.join(packages)}")
    
    for package in packages:
        try:
            import subprocess
            result = subprocess.run([sys.executable, "-m", "pip", "install", package], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ Successfully installed {package}")
            else:
                print(f"✗ Failed to install {package}: {result.stderr}")
        except Exception as e:
            print(f"✗ Error installing {package}: {e}")

def simple_analysis():
    """Perform basic analysis without advanced plotting"""
    print("\\n" + "="*50)
    print("SIMPLE DATA ANALYSIS")
    print("="*50)
    
    # Check for prediction files
    import glob
    pred_files = glob.glob('outputs/predictions_Test_Well_*.csv')
    
    if not pred_files:
        print("❌ No prediction files found in outputs/ directory")
        return
    
    print(f"Found {len(pred_files)} prediction files:")
    for f in pred_files:
        print(f"  • {os.path.basename(f)}")
    
    # Try to load and analyze the latest file
    latest_file = max(pred_files, key=os.path.getctime) if pred_files else None
    
    if latest_file:
        try:
            # Basic file analysis without pandas
            with open(latest_file, 'r') as f:
                lines = f.readlines()
            
            print(f"\\n📊 Analysis of {os.path.basename(latest_file)}:")
            print(f"   Total rows: {len(lines) - 1}")  # -1 for header
            print(f"   Header: {lines[0].strip()}")
            
            if len(lines) > 1:
                print(f"   Sample data: {lines[1].strip()}")
                
        except Exception as e:
            print(f"❌ Error reading file: {e}")

def main():
    """Main execution function"""
    print("ROP Prediction Visualization System")
    print("="*50)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Check dependencies
    print("\\nChecking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"\\n⚠️  Missing packages: {', '.join(missing)}")
        response = input("\\nAttempt to install missing packages? (y/n): ")
        
        if response.lower().startswith('y'):
            install_packages(missing)
            # Re-check after installation
            missing = check_dependencies()
    
    if missing:
        print(f"\\n❌ Still missing packages: {', '.join(missing)}")
        print("\\nPlease install Python packages manually:")
        print("pip install pandas numpy matplotlib seaborn")
        print("\\nRunning basic analysis instead...")
        simple_analysis()
    else:
        print("\\n✅ All dependencies found! Running full visualization...")
        try:
            # Import and run the main visualization functions
            import visualize_results
            
            # Run the enhanced visualization suite
            visualize_results.plot_training_history()
            visualize_results.plot_predictions()
            visualize_results.plot_feature_importance()
            visualize_results.plot_enhanced_correlations()
            visualize_results.plot_model_performance_summary()
            visualize_results.create_executive_report()
            
            print("\\n✨ Full visualization suite completed!")
            
        except Exception as e:
            print(f"❌ Error running visualizations: {e}")
            print("\\nFalling back to simple analysis...")
            simple_analysis()

if __name__ == "__main__":
    main()