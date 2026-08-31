from sqlalchemy import select
from sqlalchemy.orm import Session

from typing import Optional

from app.models.session import Session as SessionModel
from app.schemas.session import SessionCreate


class SessionRepository:

    def create(
        self,
        db: Session,
        session_data: SessionCreate,
    ) -> SessionModel:

        session = SessionModel(
            email=session_data.email,
        )

        db.add(session)
        db.commit()
        db.refresh(session)

        return session

    def get_by_email(
        self,
        db: Session,
        email: str,
    ) -> Optional[SessionModel]:

        return db.scalar(
            select(SessionModel).where(SessionModel.email == email)
        )