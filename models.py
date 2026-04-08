# database models for soccer events
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON
from sqlalchemy.sql import func
from database import Base


class Event(Base):
    """event model for storing soccer match events."""
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, nullable=False, index=True)
    event_id = Column(String(255), nullable=False, unique=True, index=True)
    event_type = Column(String(64), nullable=False, index=True)  # goal, shot_on_target, yellow_card, substitution
    player_name = Column(String(255))
    team_name = Column(String(255))
    minute = Column(Integer)
    description = Column(Text)
    raw_data = Column(JSON)  # Store original API response data
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<Event(id={self.id}, match_id={self.match_id}, event_type={self.event_type}, minute={self.minute})>"
