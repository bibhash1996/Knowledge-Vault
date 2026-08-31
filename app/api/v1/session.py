from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.session import SessionService
from app.schemas.session import SessionCreate, SessionResponse

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("/", response_model=SessionResponse)
def create_or_get_session(
    session_data: SessionCreate,
    db: Session = Depends(get_db),
):
    service = SessionService()
    return service.get_or_create(db, session_data)
