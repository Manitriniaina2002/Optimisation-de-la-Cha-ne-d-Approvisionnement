from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.engine.url import URL
from typing import Optional
from src.config.settings import settings

# Convert regular psycopg2 URL to asyncpg URL used by SQLAlchemy async
DATABASE_URL: str = settings.DATABASE_URL
if DATABASE_URL.startswith("postgresql://") and "+asyncpg" not in DATABASE_URL:
    ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
else:
    ASYNC_DATABASE_URL = DATABASE_URL

engine = create_async_engine(ASYNC_DATABASE_URL, future=True, echo=False, pool_pre_ping=True)
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

# Declarative base for models
Base = declarative_base()

async def init_db():
    """Create database tables (development convenience).

    For production use Alembic migrations instead of create_all.
    """
    # Import models so they are registered on Base.metadata
    try:
        # dynamic import to avoid circular imports when models import Base
        import importlib
        # if you add models under src/db/models, import that package here
        try:
            importlib.import_module("src.db.models")
        except Exception:
            # no models package yet
            pass

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception as e:
        # Let caller handle logging
        raise
