# unified soccer analytics api - combines video detection with event interpretation
import asyncio
import threading
import os
import shutil
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Depends, HTTPException, File, UploadFile
from sqlalchemy.orm import Session
from sqlalchemy import and_
import cv2
import pandas as pd
from ultralytics import YOLO
import torch
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel

# import our modules
from database import get_db, init_db, engine
from models import Event
from api_client import SoccerAPIClient
from classifier import EventClassifier
from pydantic import BaseModel, ConfigDict

# configure safe globals for model loading
add_safe_globals([DetectionModel])

# video detection config
VIDEO_UPLOAD_DIR = "uploaded_videos"
MODEL_PATHS = {
    "players": "models/player.pt",
    "ball": "models/ball.pt", 
    "field": "models/field.pt"
}
CONFIDENCE_THRESHOLD = 0.4
OUTPUT_DIR = "runs/detect/predict"

# colors for different detections (bgr format for opencv)
COLORS = {
    "players": (0, 255, 0),    # green for players
    "ball": (0, 0, 255),       # red for ball
    "field": (255, 0, 0)       # blue for field
}

# display labels
LABELS = {
    "players": "Player",
    "ball": "Ball",
    "field": "Field"
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """initialize database and models on startup."""
    # ensure upload directory exists
    os.makedirs(VIDEO_UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # initialize database
    init_db()
    print("database initialized")
    
    yield
    # cleanup if needed

# initialize fastapi app
app = FastAPI(
    title="unified soccer analytics api",
    description="combines video detection with live event interpretation",
    version="1.0.0",
    lifespan=lifespan,
)

# global variables for video models and event processing
video_models = {}
api_client = SoccerAPIClient()
classifier = EventClassifier()
polling_active = False
polling_thread = None

# pydantic models
class EventResponse(BaseModel):
    """event response model."""
    model_config = ConfigDict(from_attributes=True)
    id: int
    match_id: int
    event_id: str
    event_type: str
    player_name: Optional[str]
    team_name: Optional[str]
    minute: Optional[int]
    description: Optional[str]
    created_at: datetime

class PollingStatus(BaseModel):
    """polling status response model."""
    active: bool
    message: str

class VideoAnalysisResult(BaseModel):
    """video analysis result model."""
    filename: str
    total_frames: int
    detections: Dict[str, int]
    output_path: str

class MatchEventRequest(BaseModel):
    """request for adding match events from video analysis."""
    match_id: int
    events: List[Dict[str, Any]]

def load_video_models():
    """load all yolo models for video detection."""
    global video_models
    print("loading video models...")
    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            try:
                video_models[name] = YOLO(path)
                print(f"✓ loaded {name} model")
            except Exception as e:
                print(f"✗ failed to load {name} model: {e}")
        else:
            print(f"✗ model file not found: {path}")

def process_video_file(video_path: str) -> Dict[str, Any]:
    """process uploaded video and return detection results."""
    if not video_models:
        load_video_models()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # setup output video
    output_path = os.path.join(OUTPUT_DIR, f"processed_{os.path.basename(video_path)}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    detection_counts = {name: 0 for name in MODEL_PATHS.keys()}
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_with_detections = frame.copy()
            
            # run all models on the frame
            for model_name, model in video_models.items():
                if model is None:
                    continue
                    
                results = model.predict(
                    frame,
                    conf=CONFIDENCE_THRESHOLD,
                    verbose=False,
                    device='cpu'
                )[0]
                
                if results.boxes is not None:
                    detection_counts[model_name] += len(results.boxes)
                    # draw detections on frame
                    draw_detections_on_frame(frame_with_detections, results, model_name)
            
            out.write(frame_with_detections)
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"processed {frame_count}/{total_frames} frames")
    
    finally:
        cap.release()
        out.release()
    
    return {
        "total_frames": frame_count,
        "detections": detection_counts,
        "output_path": output_path
    }

def draw_detections_on_frame(frame, results, model_name):
    """draw detections from a single model on the frame."""
    if results.boxes is None or len(results.boxes) == 0:
        return frame
    
    color = COLORS[model_name]
    label = LABELS[model_name]
    
    boxes = results.boxes.xyxy.cpu().numpy()
    confidences = results.boxes.conf.cpu().numpy()
    
    for box, conf in zip(boxes, confidences):
        x1, y1, x2, y2 = map(int, box)
        
        # draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # prepare label text
        label_text = f"{label} {conf:.2f}"
        
        # get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # draw label background
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # draw label text
        cv2.putText(
            frame,
            label_text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

# api endpoints
@app.get("/")
async def root():
    """root endpoint showing available features."""
    return {
        "message": "unified soccer analytics api",
        "features": {
            "video_upload": "/upload_video",
            "video_analysis": "/analyze_video/{filename}",
            "event_api": "/matches/{match_id}/events",
            "event_polling": "/polling/start",
            "add_events": "/add_events"
        }
    }

@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """upload video file for analysis."""
    if not file.filename.endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(status_code=400, detail="invalid file type")
    
    file_location = os.path.join(VIDEO_UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {
        "status": "success", 
        "message": "video uploaded", 
        "filename": file.filename,
        "next_step": f"use /analyze_video/{file.filename} to process"
    }

@app.post("/analyze_video/{filename}", response_model=VideoAnalysisResult)
async def analyze_video(filename: str):
    """analyze uploaded video and detect objects."""
    video_path = os.path.join(VIDEO_UPLOAD_DIR, filename)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="video file not found")
    
    try:
        results = process_video_file(video_path)
        return VideoAnalysisResult(
            filename=filename,
            total_frames=results["total_frames"],
            detections=results["detections"],
            output_path=results["output_path"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"video analysis failed: {str(e)}")

@app.get("/matches/{match_id}/events", response_model=List[EventResponse])
async def get_match_events(match_id: int, db: Session = Depends(get_db)):
    """get all classified events for a specific match."""
    events = db.query(Event).filter(Event.match_id == match_id).order_by(Event.minute, Event.created_at).all()
    
    if not events:
        raise HTTPException(status_code=404, detail=f"no events found for match {match_id}")
    
    return events

@app.post("/add_events")
async def add_events_from_video(request: MatchEventRequest, db: Session = Depends(get_db)):
    """add events detected from video analysis to the database."""
    added_events = []
    
    for event_data in request.events:
        # classify the event
        event_type = classifier.classify(event_data)
        if event_type is None:
            continue  # skip unclassified events
        
        # check if event already exists
        existing = db.query(Event).filter(
            and_(Event.match_id == request.match_id, Event.event_id == event_data.get("event_id", ""))
        ).first()
        
        if existing:
            continue  # skip duplicates
        
        # extract event details
        details = classifier.extract_event_details(event_data)
        
        new_event = Event(
            match_id=request.match_id,
            event_id=event_data.get("event_id", f"video_{datetime.now().timestamp()}"),
            event_type=event_type,
            player_name=details.get("player_name"),
            team_name=details.get("team_name"),
            minute=details.get("minute"),
            description=details.get("description"),
            raw_data=details.get("raw_data")
        )
        
        db.add(new_event)
        added_events.append(new_event)
    
    db.commit()
    
    return {
        "message": f"added {len(added_events)} events to match {request.match_id}",
        "events_added": len(added_events)
    }

@app.post("/polling/start")
async def start_polling(match_id: Optional[int] = None):
    """start polling for live events from external api."""
    global polling_active, polling_thread
    
    if polling_active:
        return {"message": "polling is already active", "active": True}
    
    polling_active = True
    
    def poll_loop():
        """background polling loop."""
        while polling_active:
            try:
                if match_id:
                    process_match_events(match_id)
                else:
                    live_matches = api_client.get_live_matches()
                    for match in live_matches:
                        match_id_to_poll = match.get("id") or match.get("matchId")
                        if match_id_to_poll:
                            process_match_events(match_id_to_poll)
                
                # wait 10 seconds
                for _ in range(10):
                    if not polling_active:
                        break
                    threading.Event().wait(1)
                    
            except Exception as e:
                print(f"error in polling loop: {e}")
                threading.Event().wait(10)
    
    polling_thread = threading.Thread(target=poll_loop, daemon=True)
    polling_thread.start()
    
    return {
        "message": f"polling started for {'match ' + str(match_id) if match_id else 'all live matches'}",
        "active": True
    }

@app.post("/polling/stop")
async def stop_polling():
    """stop polling for live events."""
    global polling_active
    polling_active = False
    return {"message": "polling stopped", "active": False}

@app.get("/polling/status", response_model=PollingStatus)
async def get_polling_status():
    """get current polling status."""
    return {
        "active": polling_active,
        "message": "polling is active" if polling_active else "polling is stopped"
    }

def process_match_events(match_id: int):
    """process events for a match: fetch, classify, and store."""
    from database import SessionLocal
    
    db = SessionLocal()
    try:
        events = api_client.get_match_events(match_id)
        
        if not events:
            return
        
        for event_data in events:
            event_id = str(event_data.get("id", event_data.get("eventId", "")))
            
            if not event_id:
                continue
            
            existing_event = db.query(Event).filter(
                and_(Event.match_id == match_id, Event.event_id == event_id)
            ).first()
            
            if existing_event:
                continue
            
            event_type = classifier.classify(event_data)
            
            if event_type:
                details = classifier.extract_event_details(event_data)
                
                new_event = Event(
                    match_id=match_id,
                    event_id=event_id,
                    event_type=event_type,
                    player_name=details.get("player_name"),
                    team_name=details.get("team_name"),
                    minute=details.get("minute"),
                    description=details.get("description"),
                    raw_data=details.get("raw_data")
                )
                
                db.add(new_event)
                db.commit()
                
                print(f"stored event: {event_type} at minute {details.get('minute')} for match {match_id}")
    
    except Exception as e:
        print(f"error processing match {match_id}: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    # load models on startup
    load_video_models()
    uvicorn.run(app, host="0.0.0.0", port=8000)
