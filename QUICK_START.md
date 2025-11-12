# Quick Start Guide - Soccer Analytics Backend

## ✅ Your Setup is Complete!

**Environment**: `soccerpy311` (conda)  
**Python**: 3.11.14  
**Status**: All packages installed and ready to use!

## How to Use

### 1. Activate Conda Environment

```powershell
conda activate soccerpy311
```

### 2. Run Your Scripts

```bash
# Run video prediction (players, ball, field detection)
python basic_predict.py

# Run FastAPI server
uvicorn main:app --reload

# Run data extraction
python data_extraction.py
```

### 3. Access FastAPI Server

Once running, open your browser:
- API docs: http://127.0.0.1:8000/docs
- Upload endpoint: http://127.0.0.1:8000/upload

## Installed Packages

✅ **ML/Computer Vision**
- torch 2.9.0
- torchvision 0.24.0
- ultralytics 8.3.225
- opencv-python 4.12.0.88

✅ **Web Framework**
- fastapi 0.121.1
- uvicorn 0.38.0
- python-multipart 0.0.20

✅ **Data Processing**
- pandas 2.3.3
- numpy 2.2.6

## File Structure

```
soccer_analytics_backend/
├── basic_predict.py      # Video prediction script
├── data_extraction.py    # Data extraction script
├── main.py               # FastAPI server
├── requirements.txt      # Python dependencies
├── environment.yml       # Conda environment file
└── models/               # YOLO models
    ├── player.pt
    ├── ball.pt
    └── field.pt
```

## Next Steps

1. **Test video prediction**:
   ```bash
   conda activate soccerpy311
   python basic_predict.py
   ```

2. **Start API server**:
   ```bash
   conda activate soccerpy311
   uvicorn main:app --reload
   ```

3. **Upload a video** via the API and process it

## Troubleshooting

### If conda activate doesn't work:
```powershell
# Use full path
C:\Users\snell\anaconda3\Scripts\activate.bat soccerpy311
```

### If scripts don't run:
```bash
# Make sure you're in the conda environment
conda activate soccerpy311
python --version  # Should show 3.11.14
```

### If models are not found:
- Check that `models/player.pt`, `models/ball.pt`, `models/field.pt` exist
- Update paths in `basic_predict.py` if needed

## Summary

✅ Python 3.11.14 - Perfect for ML/AI  
✅ All packages installed  
✅ Ready to use!  

Just activate your conda environment and run your scripts!


