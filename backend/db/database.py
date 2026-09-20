"""
DB engine/session. Defaults to local SQLite for zero-setup Phase 1 dev.
Set ASTRA_DATABASE_URL to point at PostgreSQL for multi-writer / production use,
e.g. postgresql+psycopg2://user:pass@localhost:5432/astra
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.environ.get(
    "ASTRA_DATABASE_URL",
    "sqlite:///./astra.db",
)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from . import models  # noqa: F401  (ensure models are registered)
    Base.metadata.create_all(bind=engine)
