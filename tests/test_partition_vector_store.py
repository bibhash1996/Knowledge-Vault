import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.session import Base
from app.db.elasticsearch_store import search_chunks
from app.models.knowledge import Knowledge
from app.models.original_content import OriginalContent
from app.models.session import Session as SessionModel
from app.services.partition import (
    summarise_chunks,
    save_chunks_to_keyword_search,
    save_chunks_to_vector_store,
)
from app.services.retrieval import retrieve_chunks


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
def test_summarise_chunks_persists_large_content_and_stores_references(mock_summary):
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
    mock_summary.assert_called_once()


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


@patch("app.services.partition.helpers.bulk")
@patch("app.services.partition.ensure_chunks_index")
@patch("app.services.partition.get_elasticsearch_client")
@patch("app.services.partition.get_settings")
def test_save_chunks_to_keyword_search_indexes_session_scoped_content(
    mock_get_settings,
    mock_get_client,
    mock_ensure_index,
    mock_bulk,
):
    mock_get_settings.return_value.elasticsearch_index_name = "knowledge-chunks"
    mock_bulk.return_value = (1, [])
    documents = [
        Document(
            page_content="A searchable chunk summary",
            metadata={
                "chunk_id": "chunk-1",
                "knowledge_id": 24,
                "file_url": "https://example.com/doc.pdf",
                "original_content": '{"raw_text": "original words"}',
            },
        )
    ]

    save_chunks_to_keyword_search(documents, session_id=42)

    action = mock_bulk.call_args.args[1][0]
    assert action["_id"] == "chunk-1"
    assert action["_index"] == "knowledge-chunks"
    assert action["_source"] == {
        "chunk_id": "chunk-1",
        "session_id": 42,
        "knowledge_id": 24,
        "file_url": "https://example.com/doc.pdf",
        "content": "A searchable chunk summary",
        "original_content": '{"raw_text": "original words"}',
    }
    mock_get_client.assert_called_once_with()
    mock_ensure_index.assert_called_once_with()


@patch("app.db.elasticsearch_store.get_elasticsearch_client")
@patch("app.db.elasticsearch_store.get_settings")
def test_search_chunks_filters_by_session_and_returns_documents(
    mock_get_settings,
    mock_get_client,
):
    mock_get_settings.return_value.elasticsearch_index_name = "knowledge-chunks"
    mock_client = Mock()
    mock_client.indices.exists.return_value = True
    mock_client.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_id": "chunk-1",
                    "_score": 2.5,
                    "_source": {
                        "chunk_id": "chunk-1",
                        "session_id": 42,
                        "knowledge_id": 24,
                        "file_url": "https://example.com/doc.pdf",
                        "content": "Matching chunk",
                        "original_content": '{"raw_text": "raw"}',
                    },
                }
            ]
        }
    }
    mock_get_client.return_value = mock_client

    results = search_chunks("matching", session_id=42, top_k=3, knowledge_id=24)

    assert len(results) == 1
    assert results[0].page_content == "Matching chunk"
    assert results[0].metadata["chunk_id"] == "chunk-1"
    assert results[0].metadata["bm25_score"] == 2.5
    query = mock_client.search.call_args.kwargs["query"]
    assert query["bool"]["filter"] == [
        {"term": {"session_id": 42}},
        {"term": {"knowledge_id": 24}},
    ]


@patch("app.services.retrieval.search_chunks")
@patch("app.services.retrieval.get_vector_store")
def test_retrieve_chunks_searches_both_stores_with_session_filter(
    mock_get_vector_store,
    mock_search_chunks,
):
    vector_chunk = Document(page_content="vector result")
    keyword_chunk = Document(page_content="keyword result")
    vector_store = Mock()
    vector_store.similarity_search.return_value = [vector_chunk]
    mock_get_vector_store.return_value = vector_store
    mock_search_chunks.return_value = [keyword_chunk]

    results = retrieve_chunks("query", session_id=42, top_k=5)

    vector_store.similarity_search.assert_called_once_with(
        "query",
        k=5,
        filter={"session_id": 42},
    )
    mock_search_chunks.assert_called_once_with("query", session_id=42, top_k=5)
    assert results == {
        "vector": [vector_chunk],
        "keyword": [keyword_chunk],
    }


def test_retrieve_chunks_returns_empty_results_for_blank_query():
    assert retrieve_chunks("  ", session_id=42) == {"vector": [], "keyword": []}
