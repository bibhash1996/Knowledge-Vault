from fastapi import APIRouter

from app.api.v1.health import router as health_router
from app.api.v1.knowledge_import import router as import_router
from app.api.v1.session import router as session_router

router = APIRouter()

router.include_router(health_router)
router.include_router(import_router)
router.include_router(session_router)