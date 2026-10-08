from collections.abc import Iterable

from sqlalchemy.orm import Session

from app.models.original_content import OriginalContent


class OriginalContentRepository:
    def create_many(
        self,
        db: Session,
        session_id: int,
        knowledge_id: int,
        entries: Iterable[tuple[str, str]],
    ) -> list[OriginalContent]:
        rows = [
            OriginalContent(
                session_id=session_id,
                knowledge_id=knowledge_id,
                entity_type=entity_type,
                value=value,
            )
            for entity_type, value in entries
        ]
        db.add_all(rows)
        db.flush()
        return rows
