# database configuration and session management
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

# database url from environment variable or default (mysql)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://root:password@localhost:3306/soccer_events"
)

# create engine
engine = create_engine(DATABASE_URL, echo=True)

# create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# base class for models
Base = declarative_base()


def get_db():
    """dependency for fastapi to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """initialize database tables."""
    Base.metadata.create_all(bind=engine)
