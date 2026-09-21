from datetime import datetime

from sqlalchemy import DateTime, Integer, String, func, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class Knowledge(Base):
    __tablename__ = "knowledge"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    session_id: Mapped[int] = mapped_column(Integer, ForeignKey("sessions.id"), nullable=False, index=True)

    url: Mapped[str] = mapped_column(String(2048), nullable=False)

    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending", index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
