from pydantic import BaseModel, HttpUrl
from typing import Optional
from datetime import datetime


class KnowledgeCreate(BaseModel):
    url: HttpUrl


class KnowledgeResponse(BaseModel):
    id: int
    session_id: int
    url: HttpUrl
    status: str
    created_at: Optional[datetime]
    updated_at: Optional[datetime]

    class Config:
        orm_mode = True
