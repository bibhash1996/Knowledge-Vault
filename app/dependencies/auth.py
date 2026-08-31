from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.session import Session as SessionModel


def get_authenticated_session(
    request: Request,
    db: Session = Depends(get_db),
) -> SessionModel:
    session_id_header = request.headers.get("x-session-id")

    if not session_id_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing x-session-id header",
        )

    try:
        session_id = int(session_id_header)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid x-session-id header",
        ) from exc

    session = db.get(SessionModel, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session not found",
        )

    return session
