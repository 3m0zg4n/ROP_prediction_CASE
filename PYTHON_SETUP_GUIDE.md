# Python Installation Guide for ROP Prediction System

## Current Issue
Python is not properly installed or configured on your system. The system shows evidence of Python 3.12 packages (in AppData\\Roaming) but no working Python executable.

## Solution Options

### Option 1: Install Python via Microsoft Store (Recommended)
1. Open Microsoft Store
2. Search for "Python 3.12"
3. Install "Python 3.12" (Publisher: Python Software Foundation)
4. After installation, restart your command prompt
5. Test with: `python --version`

### Option 2: Install Python via winget (Command Line)
```powershell
winget install Python.Python.3.12
```
After installation, restart your terminal and test with `python --version`

### Option 3: Download from python.org
1. Go to https://www.python.org/downloads/
2. Download Python 3.12.x for Windows
3. Run installer and **check "Add Python to PATH"**
4. Install for all users (recommended)
5. Test installation: `python --version`

### Option 4: Install Anaconda (Data Science Stack)
```powershell
winget install Anaconda.Anaconda3
```
This installs Python + all required packages automatically.

## Verify Installation
After installing Python, run these commands:
```powershell
python --version
python -m pip --version
```

## Install Required Packages
Once Python is working, install the required packages:
```powershell
python -m pip install tensorflow pandas numpy scikit-learn matplotlib seaborn scipy
```

## Quick Test
To test if everything works:
```powershell
python simple_visualize.py
```

## Troubleshooting

### Issue: "Python was not found"
- Make sure you restarted your command prompt after installation
- Check if Python is in your PATH environment variable
- Try using `py` instead of `python`

### Issue: "Module not found" errors
- Install missing packages: `python -m pip install <package_name>`
- Use the requirements.txt file: `python -m pip install -r requirements.txt`

### Issue: Permission errors
- Run command prompt as Administrator
- Use `--user` flag: `python -m pip install --user <package_name>`

## Alternative: Use Jupyter Notebook
If you have Jupyter installed elsewhere:
1. Copy the visualize_results.py content into a Jupyter notebook
2. Run cells individually
3. This bypasses command-line Python issues

## Contact Support
If issues persist:
- Check Windows PATH environment variable includes Python
- Verify Windows Execution Policy allows scripts
- Consider using Windows Subsystem for Linux (WSL) with Python