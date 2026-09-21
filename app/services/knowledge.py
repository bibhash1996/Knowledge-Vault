from sqlalchemy.orm import Session

from app.repositories.knowledge import KnowledgeRepository


class KnowledgeService:
    def __init__(self):
        self.repo = KnowledgeRepository()

    def create_for_session(self, db: Session, session_id: int, url: str):
        return self.repo.create(db, session_id=session_id, url=url)

    def mark_status(self, db: Session, id: int, status: str):
        return self.repo.update_status(db, id=id, status=status)
