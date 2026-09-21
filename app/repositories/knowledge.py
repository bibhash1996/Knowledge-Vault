from typing import Optional
from sqlalchemy.orm import Session

from app.models.knowledge import Knowledge


class KnowledgeRepository:
    def create(self, db: Session, session_id: int, url: str) -> Knowledge:
        obj = Knowledge(session_id=session_id, url=url, status="pending")
        db.add(obj)
        db.commit()
        db.refresh(obj)
        return obj

    def get(self, db: Session, id: int) -> Optional[Knowledge]:
        return db.get(Knowledge, id)

    def update_status(self, db: Session, id: int, status: str) -> Optional[Knowledge]:
        obj = db.get(Knowledge, id)
        if not obj:
            return None
        obj.status = status
        db.add(obj)
        db.commit()
        db.refresh(obj)
        return obj
