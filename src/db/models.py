from sqlalchemy import Column, Integer, String, Float, DateTime, func
from .postgres import Base


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    sku = Column(String(64), unique=True, nullable=False, index=True)
    name = Column(String(256), nullable=False)
    category = Column(String(128), nullable=True)
    price = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# Note:
# - This file is a small example to show how models are defined and registered
#   with SQLAlchemy's declarative `Base` used in `src/db/postgres.py`.
# - For production schema changes, add Alembic and create versioned migrations
#   instead of relying on `Base.metadata.create_all`.
