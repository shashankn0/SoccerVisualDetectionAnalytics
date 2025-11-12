from fastapi import FastAPI, File, UploadFile, HTTPException
import shutil
import os

app = FastAPI()
UPLOAD_DIR = "uploaded_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename.endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(status_code=400, detail="Invalid file type")
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # Here you can later add your analytics processing
    return {"status": "success", "message": "Video uploaded", "filename": file.filename}

@app.get("/")
def root():
    return {"message": "Hello World"}
