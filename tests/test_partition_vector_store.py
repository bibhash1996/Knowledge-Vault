import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.session import Base
from app.models.knowledge import Knowledge
from app.models.original_content import OriginalContent
from app.models.session import Session as SessionModel
from app.services.partition import summarise_chunks, save_chunks_to_vector_store


def test_settings_accepts_openai_api_key_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    settings = Settings()

    assert settings.openai_api_key == "test-key"


def test_summarise_chunks_tags_session_id_in_metadata():
    chunks = [SimpleNamespace(text="hello world", metadata=SimpleNamespace(orig_elements=[]))]

    docs = summarise_chunks(chunks, "https://example.com/doc.pdf", session_id=42)

    assert len(docs) == 1
    assert docs[0].metadata["session_id"] == 42
    assert docs[0].metadata["file_url"] == "https://example.com/doc.pdf"


@patch("app.services.partition.create_ai_enhanced_summary", return_value="summary")
def test_summarise_chunks_persists_large_content_and_stores_references(_mock_summary):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)

    table = type("Table", (), {})()
    table.text = "<table>data</table>"
    table.metadata = SimpleNamespace(text_as_html=table.text)
    image = type("Image", (), {})()
    image.metadata = SimpleNamespace(image_base64="aW1hZ2U=")
    chunk = SimpleNamespace(
        text="table and image",
        metadata=SimpleNamespace(orig_elements=[table, image]),
    )

    with Session(engine) as db:
        db.add(SessionModel(id=42, email="test@example.com"))
        db.flush()
        knowledge = Knowledge(
            session_id=42,
            url="https://example.com/doc.pdf",
            status="processing",
        )
        db.add(knowledge)
        db.flush()

        docs = summarise_chunks(
            [chunk],
            "https://example.com/doc.pdf",
            session_id=42,
            knowledge_id=knowledge.id,
            db=db,
        )

        stored_content = db.scalars(select(OriginalContent)).all()
        original_content = json.loads(docs[0].metadata["original_content"])

    assert [(row.entity_type, row.value) for row in stored_content] == [
        ("table", "<table>data</table>"),
        ("image", "aW1hZ2U="),
    ]
    assert all(row.session_id == 42 for row in stored_content)
    assert all(row.knowledge_id == knowledge.id for row in stored_content)
    assert original_content["content_references"] == [
        {"entity_type": row.entity_type, "id": row.id}
        for row in stored_content
    ]
    assert "tables_html" not in original_content
    assert "images_base64" not in original_content


@patch("app.services.partition.get_vector_store")
def test_save_chunks_to_vector_store_adds_documents(mock_get_vector_store):
    mock_store = Mock()
    mock_get_vector_store.return_value = mock_store
    documents = [Document(page_content="chunk text", metadata={"session_id": 42, "file_url": "https://example.com/doc.pdf"})]

    save_chunks_to_vector_store(documents, session_id=42)

    mock_store.add_documents.assert_called_once_with(documents)


@patch("app.services.partition.get_vector_store")
def test_save_chunks_to_vector_store_skips_missing_index(mock_get_vector_store):
    mock_get_vector_store.side_effect = RuntimeError("Resource doc-rag-index not found")
    documents = [Document(page_content="chunk text", metadata={"session_id": 42, "file_url": "https://example.com/doc.pdf"})]

    save_chunks_to_vector_store(documents, session_id=42)
