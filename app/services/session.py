from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.repositories.session import SessionRepository
from app.schemas.session import SessionCreate


class SessionService:

    def __init__(self):
        self.repository = SessionRepository()

    def create_session(
        self,
        db: Session,
        session_data: SessionCreate,
    ):
        existing_session = self.repository.get_by_email(
            db,
            session_data.email,
        )

        if existing_session:
            raise HTTPException(
                status_code=409,
                detail="Session already exists",
            )

        return self.repository.create(
            db,
            session_data,
        )

    def get_or_create(
        self,
        db: Session,
        session_data: SessionCreate,
    ):
        """Return existing session for email or create a new one."""
        existing = self.repository.get_by_email(db, session_data.email)
        if existing:
            return existing

        return self.repository.create(db, session_data)