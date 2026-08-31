from fastapi import APIRouter, Depends

from app.dependencies.auth import get_authenticated_session
from app.models.session import Session as SessionModel
from app.schemas.knowledge_import import KnowledgeImportRequest

router = APIRouter(prefix="/import", tags=["import"])


@router.post("/knowledge")
async def import_knowledge(
    payload: KnowledgeImportRequest,
    session: SessionModel = Depends(get_authenticated_session),
):
    return {
        "message": "Session validated successfully",
        "session_id": session.id,
        "email": session.email,
        "url": payload.url,
    }
