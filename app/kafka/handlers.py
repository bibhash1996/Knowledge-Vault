import logging
import json
from typing import Any

from app.services.knowledge import KnowledgeService
from app.db.session import SessionLocal


async def handle_doc_import(payload: dict) -> None:
    """Handle a `doc_import` event payload.

    Expected payload: {"knowledge_id": int, "url": str, ...}
    """
    knowledge_id = payload.get("knowledge_id")
    if not knowledge_id:
        logging.warning("doc_import event missing knowledge_id: %s", payload)
        return

    db = SessionLocal()
    try:
        ks = KnowledgeService()
        ks.mark_status(db, knowledge_id, "processing")
        logging.info("Marked knowledge %s as processing", knowledge_id)
        # TODO: add actual import processing here and mark completed/failed
        # For now, mark completed immediately for demo purposes
        ks.mark_status(db, knowledge_id, "completed")
        logging.info("Marked knowledge %s as completed", knowledge_id)
    except Exception:
        logging.exception("Failed to handle doc_import event for %s", knowledge_id)
    finally:
        db.close()


async def default_handler(msg: Any) -> None:
    """Dispatch incoming Kafka messages to event-specific handlers."""
    try:
        value = msg.value
        # try decode if bytes
        if isinstance(value, (bytes, bytearray)):
            try:
                decoded = value.decode("utf-8")
            except Exception:
                decoded = repr(value)
        else:
            decoded = str(value)

        logging.info("Received Kafka message on %s: %s", msg.topic, decoded)

        try:
            payload = json.loads(decoded)
        except Exception:
            logging.exception("Failed to parse Kafka message as JSON")
            return

        if not isinstance(payload, dict):
            logging.warning("Unexpected payload type: %s", type(payload))
            return

        event_type = payload.get("type")
        if event_type == "doc_import":
            await handle_doc_import(payload)
        else:
            logging.info("Unhandled event type: %s", event_type)
    except Exception:
        logging.exception("Error in default_handler")
