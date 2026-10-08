from sqlalchemy import ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base
from app.models.knowledge import Knowledge  # noqa: F401
from app.models.session import Session  # noqa: F401


class OriginalContent(Base):
    __tablename__ = "original_content"
    __table_args__ = (
        Index("ix_original_content_session_id", "session_id"),
        Index("ix_original_content_knowledge_id", "knowledge_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("sessions.id"),
        nullable=False,
    )
    knowledge_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("knowledge.id"),
        nullable=False,
    )
    entity_type: Mapped[str] = mapped_column(String(20), nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
