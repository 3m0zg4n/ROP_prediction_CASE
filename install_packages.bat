@echo off
echo Installing Required Python Packages for ROP Prediction System
echo =============================================================

echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing core packages...
python -m pip install tensorflow pandas numpy scikit-learn matplotlib seaborn

echo.
echo Installing additional packages...
python -m pip install scipy lasio

echo.
echo Verifying installations...
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import pandas as pd; print(f'Pandas: {pd.__version__}')"
python -c "import numpy as np; print(f'NumPy: {np.__version__}')"
python -c "import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')"
python -c "import seaborn as sns; print(f'Seaborn: {sns.__version__}')"

echo.
echo Installation completed! You can now run:
echo python visualize_results.py
echo.
pause