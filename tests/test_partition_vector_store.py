from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from app.core.config import Settings
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
