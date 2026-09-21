from fastapi import APIRouter, Depends
import json
import logging

from app.dependencies.auth import get_authenticated_session
from app.models.session import Session as SessionModel
from app.schemas.knowledge_import import KnowledgeImportRequest
from app.core.config import get_settings
from app.kafka.producer import send_message
from app.services.knowledge import KnowledgeService
from app.schemas.knowledge import KnowledgeResponse
from sqlalchemy.orm import Session
from app.db.session import get_db

router = APIRouter(prefix="/import", tags=["import"])


@router.post("/knowledge")
async def import_knowledge(
    payload: KnowledgeImportRequest,
    session: SessionModel = Depends(get_authenticated_session),
    db: Session = Depends(get_db),
):
    """Validate session then publish a Kafka message of type `doc_import` with the URL."""
    settings = get_settings()

    # create a knowledge row in pending state
    ks = KnowledgeService()
    kb = ks.create_for_session(db, session.id, payload.url)

    message = {
        "type": "doc_import",
        "knowledge_id": kb.id,
        "url": payload.url,
        "session_id": session.id,
    }

    topic = settings.kafka_producer_topic or "file-events"
    published = False
    try:
        await send_message(topic, json.dumps(message).encode("utf-8"))
        published = True
    except Exception:
        logging.exception("Failed to publish Kafka message for doc_import")

    return {
        "message": "Session validated successfully",
        "knowledge_id": kb.id,
        "session_id": session.id,
        "email": session.email,
        "url": payload.url,
        "published": published,
    }
