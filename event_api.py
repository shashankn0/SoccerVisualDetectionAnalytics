"""FastAPI application for live soccer event interpreter."""
import asyncio
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import and_

from database import get_db, init_db, engine
from models import Event
from api_client import SoccerAPIClient
from classifier import EventClassifier
from pydantic import BaseModel, ConfigDict

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup, cleanup on shutdown."""
    init_db()
    print("Database initialized")
    yield
    # shutdown cleanup if needed


# Initialize FastAPI app
app = FastAPI(
    title="Soccer Event Interpreter API",
    description="Live soccer event interpreter with classification",
    version="1.0.0",
    lifespan=lifespan,
)

# Initialize API client and classifier
api_client = SoccerAPIClient()
classifier = EventClassifier()

# Track polling status
polling_active = False
polling_thread = None


# Pydantic models for API responses
class EventResponse(BaseModel):
    """Event response model."""
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
    """Polling status response model."""
    active: bool
    message: str


class TestEventRequest(BaseModel):
    """Request body for POST /events/test."""
    match_id: int
    event_id: str
    description: Optional[str] = None
    player_name: Optional[str] = None
    team_name: Optional[str] = None
    minute: Optional[int] = None


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Soccer Event Interpreter API",
        "version": "1.0.0",
        "endpoints": {
            "get_events": "/matches/{match_id}/events",
            "start_polling": "/polling/start",
            "stop_polling": "/polling/stop",
            "polling_status": "/polling/status"
        }
    }


@app.get("/matches/{match_id}/events", response_model=List[EventResponse])
async def get_match_events(match_id: int, db: Session = Depends(get_db)):
    """
    Get all classified events for a specific match.
    
    Args:
        match_id: The ID of the match
        db: Database session
        
    Returns:
        List of classified events
    """
    events = db.query(Event).filter(Event.match_id == match_id).order_by(Event.minute, Event.created_at).all()
    
    if not events:
        raise HTTPException(status_code=404, detail=f"No events found for match {match_id}")
    
    return events


@app.post("/events/test", response_model=EventResponse)
async def test_event(body: TestEventRequest, db: Session = Depends(get_db)):
    """
    Accept raw soccer event JSON, run it through the classifier, and store if classified.
    Returns the stored event or an error if not classified or duplicate.
    """
    event_data = {
        "id": body.event_id,
        "eventId": body.event_id,
        "comment": body.description or "",
        "description": body.description or "",
        "player": {"name": body.player_name} if body.player_name else {},
        "playerName": body.player_name,
        "team": {"name": body.team_name} if body.team_name else {},
        "teamName": body.team_name,
        "minute": body.minute,
    }
    event_type = classifier.classify(event_data)
    if event_type is None:
        raise HTTPException(
            status_code=422,
            detail="Event could not be classified as goal, shot_on_target, yellow_card, or substitution",
        )
    existing = (
        db.query(Event)
        .filter(and_(Event.match_id == body.match_id, Event.event_id == body.event_id))
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Event {body.event_id} for match {body.match_id} already exists",
        )
    details = classifier.extract_event_details(event_data)
    new_event = Event(
        match_id=body.match_id,
        event_id=body.event_id,
        event_type=event_type,
        player_name=details.get("player_name"),
        team_name=details.get("team_name"),
        minute=details.get("minute"),
        description=details.get("description"),
        raw_data=details.get("raw_data"),
    )
    db.add(new_event)
    db.commit()
    db.refresh(new_event)
    return new_event


@app.post("/polling/start")
async def start_polling(match_id: Optional[int] = None):
    """
    Start polling for live events.
    
    Args:
        match_id: Optional specific match ID to poll. If not provided, polls all live matches.
    """
    global polling_active, polling_thread
    
    if polling_active:
        return {"message": "Polling is already active", "active": True}
    
    polling_active = True
    
    def poll_loop():
        """Background polling loop."""
        while polling_active:
            try:
                if match_id:
                    # Poll specific match
                    process_match_events(match_id)
                else:
                    # Poll all live matches
                    live_matches = api_client.get_live_matches()
                    for match in live_matches:
                        match_id_to_poll = match.get("id") or match.get("matchId")
                        if match_id_to_poll:
                            process_match_events(match_id_to_poll)
                
                # Wait 10 seconds before next poll
                for _ in range(10):
                    if not polling_active:
                        break
                    threading.Event().wait(1)
                    
            except Exception as e:
                print(f"Error in polling loop: {e}")
                threading.Event().wait(10)
    
    polling_thread = threading.Thread(target=poll_loop, daemon=True)
    polling_thread.start()
    
    return {
        "message": f"Polling started for {'match ' + str(match_id) if match_id else 'all live matches'}",
        "active": True
    }


@app.post("/polling/stop")
async def stop_polling():
    """Stop polling for live events."""
    global polling_active
    
    polling_active = False
    return {"message": "Polling stopped", "active": False}


@app.get("/polling/status", response_model=PollingStatus)
async def get_polling_status():
    """Get current polling status."""
    return {
        "active": polling_active,
        "message": "Polling is active" if polling_active else "Polling is stopped"
    }


def process_match_events(match_id: int):
    """
    Process events for a match: fetch, classify, and store.
    
    Args:
        match_id: The ID of the match to process
    """
    from database import SessionLocal
    
    db = SessionLocal()
    try:
        # Fetch events from API
        events = api_client.get_match_events(match_id)
        
        if not events:
            return
        
        for event_data in events:
            # Extract event ID to check if already stored
            event_id = str(event_data.get("id", event_data.get("eventId", "")))
            
            if not event_id:
                continue
            
            # Check if event already exists
            existing_event = db.query(Event).filter(
                and_(Event.match_id == match_id, Event.event_id == event_id)
            ).first()
            
            if existing_event:
                continue  # Skip already processed events
            
            # Classify the event
            event_type = classifier.classify(event_data)
            
            # Only store classified events
            if event_type:
                # Extract event details
                details = classifier.extract_event_details(event_data)
                
                # Create new event record
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
                
                print(f"Stored event: {event_type} at minute {details.get('minute')} for match {match_id}")
    
    except Exception as e:
        print(f"Error processing match {match_id}: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
