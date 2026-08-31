from pydantic import BaseModel, field_validator


class KnowledgeImportRequest(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("url cannot be empty")
        return value
