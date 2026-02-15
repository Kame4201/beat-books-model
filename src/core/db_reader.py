"""
Read-only database connection to the shared BeatTheBooks PostgreSQL database.

CRITICAL: This repo NEVER creates, alters, or drops tables.
All schema changes go through beat-books-data via Alembic migrations.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.core.config import settings

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    echo=False,
)

SessionLocal = sessionmaker(bind=engine)


def get_read_session():
    """Get a read-only database session."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
