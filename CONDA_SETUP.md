# Conda Environment Setup - Soccer Analytics Backend

## Your Current Setup ✅

- **Environment**: `soccerpy311` (conda)
- **Python Version**: 3.11.14 ✅ **Perfect!**
- **ML Packages**: Already installed (torch, ultralytics, opencv)
- **Missing**: FastAPI and web dependencies

## Quick Setup

### 1. Activate Your Conda Environment

```powershell
# PowerShell
conda activate soccerpy311

# Or if that doesn't work:
C:\Users\snell\anaconda3\Scripts\activate.bat soccerpy311
```

### 2. Install Missing Web Dependencies

```bash
# Install FastAPI and web server
pip install fastapi uvicorn[standard] python-multipart

# Install data processing (if not already installed)
pip install pandas

# Or install everything from requirements.txt
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
# Check Python version
python --version
# Should show: Python 3.11.14

# Check installed packages
pip list

# Test imports
python -c "import torch; import ultralytics; import cv2; import fastapi; print('✓ All packages working!')"
```

## Using Conda Environment

### Activate Environment

```powershell
conda activate soccerpy311
```

### Run Your Scripts

```bash
# Run prediction script
python basic_predict.py

# Run FastAPI server
uvicorn main:app --reload
```

### Deactivate Environment

```bash
conda deactivate
```

## Updating Your Environment

### Option 1: Using pip (Recommended for this project)

```bash
conda activate soccerpy311
pip install -r requirements.txt
```

### Option 2: Using environment.yml

```bash
# Update environment from file
conda env update -f environment.yml --prune
```

## Current Package Status

### ✅ Already Installed (in soccerpy311)
- Python 3.11.14
- torch 2.9.0
- torchvision 0.24.0
- ultralytics 8.3.225
- opencv-python 4.12.0.88
- numpy 2.2.6
- matplotlib 3.10.7
- scipy 1.16.3

### ❌ Missing (need to install)
- fastapi
- uvicorn
- python-multipart
- pandas (optional, for data_extraction.py)

## Recommended Workflow

1. **Always activate conda environment first**:
   ```bash
   conda activate soccerpy311
   ```

2. **Install missing packages**:
   ```bash
   pip install fastapi uvicorn[standard] python-multipart pandas
   ```

3. **Run your scripts**:
   ```bash
   python basic_predict.py
   python main.py  # or uvicorn main:app --reload
   ```

## Why Conda is Good for This Project

✅ **Python 3.11.14** - Perfect version for ML/AI
✅ **Pre-installed ML packages** - torch, ultralytics, opencv
✅ **Easy environment management** - Isolated from system Python
✅ **Package management** - Handles dependencies well

## Troubleshooting

### Issue: "conda: command not found"
**Solution**: Make sure Anaconda/Miniconda is installed and in PATH
- Add to PATH: `C:\Users\snell\anaconda3\Scripts`
- Or use full path: `C:\Users\snell\anaconda3\Scripts\conda.exe`

### Issue: "Module not found" errors
**Solution**: Make sure you're in the correct conda environment
```bash
conda activate soccerpy311
pip install <missing-package>
```

### Issue: "FastAPI not found"
**Solution**: Install web dependencies
```bash
conda activate soccerpy311
pip install fastapi uvicorn[standard] python-multipart
```

## Next Steps

1. ✅ Activate conda environment: `conda activate soccerpy311`
2. ✅ Install web dependencies: `pip install fastapi uvicorn[standard] python-multipart`
3. ✅ Test your scripts: `python basic_predict.py`
4. ✅ Run API server: `uvicorn main:app --reload`

## Summary

You're all set with Python 3.11.14 in your conda environment! Just need to install the FastAPI/web dependencies and you're good to go.

**Your environment is perfect for this project - no need to change anything!**


