# Setup Guide - Soccer Analytics Backend

## Python Version Recommendation

**Use Python 3.10 or Python 3.11** for this project.

### Why?
- Your current Python 3.14.0 is too new and may have compatibility issues
- PyTorch and Ultralytics work best with Python 3.8-3.11
- Python 3.10 is the most stable for ML/AI projects

## Step-by-Step Setup

### 1. Install Python 3.10 or 3.11

Download from: https://www.python.org/downloads/
- Choose Python 3.10.11 or Python 3.11.7 (latest stable versions)
- **Important**: Check "Add Python to PATH" during installation

### 2. Create New Virtual Environment

**Option A: Using Python 3.10 (Recommended)**
```powershell
# Windows PowerShell
C:\Python310\python.exe -m venv venv
# OR if Python 3.10 is in PATH:
python3.10 -m venv venv
```

**Option B: Using Python 3.11**
```powershell
C:\Python311\python.exe -m venv venv
# OR
python3.11 -m venv venv
```

### 3. Activate Virtual Environment

```powershell
# Windows
venv\Scripts\Activate.ps1
# If you get execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Windows CMD
venv\Scripts\activate.bat

# Linux/Mac
source venv/bin/activate
```

### 4. Upgrade Pip

```bash
python -m pip install --upgrade pip
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Verify Installation

```bash
# Check Python version
python --version
# Should show: Python 3.10.x or Python 3.11.x

# Check installed packages
pip list

# Test imports
python -c "import torch; import ultralytics; import cv2; import fastapi; print('✓ All packages working!')"
```

## Current Environment Status

Your current setup:
- **System Python**: 3.14.0 (too new)
- **Venv Python**: 3.14.0 (needs to be recreated)
- **Installed packages**: Only FastAPI and web dependencies (missing ML packages)

## Troubleshooting

### Issue: "No module named 'torch'"
**Solution**: Install PyTorch first, then other packages
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Issue: "Python 3.14 not supported"
**Solution**: Use Python 3.10 or 3.11 (see steps above)

### Issue: "Execution policy error" (Windows)
**Solution**: 
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: "Package installation fails"
**Solution**: 
1. Make sure you're using Python 3.10 or 3.11
2. Upgrade pip: `python -m pip install --upgrade pip`
3. Try installing packages one by one to identify the problematic package

## Managing Multiple Python Versions

### Using pyenv-win (Windows)

1. Install pyenv-win: https://github.com/pyenv-win/pyenv-win
2. Install Python 3.10:
   ```bash
   pyenv install 3.10.11
   pyenv local 3.10.11
   ```
3. Create venv:
   ```bash
   python -m venv venv
   ```

### Using Multiple Python Installations

If you have multiple Python versions installed:
- Python 3.10: `C:\Python310\python.exe`
- Python 3.11: `C:\Python311\python.exe`
- Python 3.14: `C:\Python314\python.exe` (current, not recommended for ML)

Use the specific Python executable when creating venv:
```powershell
C:\Python310\python.exe -m venv venv
```

## Recommended Workflow

1. **Keep Python 3.10 or 3.11** for ML/AI projects (this project)
2. **Use Python 3.14** for other projects that don't need ML libraries
3. **Always use virtual environments** for isolation
4. **Pin versions** in requirements.txt for reproducibility

## Next Steps

After setting up the environment:

1. Test `basic_predict.py`:
   ```bash
   python basic_predict.py
   ```

2. Test FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

3. Verify models are accessible:
   - Check that `models/player.pt`, `models/ball.pt`, `models/field.pt` exist

## Quick Reference

```bash
# Activate environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run prediction script
python basic_predict.py

# Run API server
uvicorn main:app --reload

# Deactivate environment
deactivate
```


