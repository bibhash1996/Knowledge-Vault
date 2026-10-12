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
from app.services.retrieval import (
    reciprocal_rank_fusion,
    rerank_chunks,
    retrieve_chunks,
    retrieve_reranked_chunks,
)


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


def test_reciprocal_rank_fusion_combines_rankings_and_deduplicates_chunks():
    vector_first = Document(
        page_content="first from vector",
        metadata={"chunk_id": "chunk-1", "origin": "vector"},
    )
    vector_second = Document(
        page_content="second from vector",
        metadata={"chunk_id": "chunk-2"},
    )
    keyword_first = Document(
        page_content="first from keyword",
        metadata={"chunk_id": "chunk-2", "origin": "keyword"},
    )
    keyword_second = Document(
        page_content="second from keyword",
        metadata={"chunk_id": "chunk-1"},
    )

    fused = reciprocal_rank_fusion(
        {
            "vector": [vector_first, vector_second],
            "keyword": [keyword_first, keyword_second],
        },
        top_k=2,
    )

    assert [document.metadata["chunk_id"] for document in fused] == [
        "chunk-1",
        "chunk-2",
    ]
    assert fused[0].page_content == "first from vector"
    assert fused[0].metadata["rrf_score"] == 1 / 61 + 1 / 62
    assert fused[0].metadata["retrieval_sources"] == ["vector", "keyword"]
    assert fused[1].metadata["rrf_score"] == 1 / 62 + 1 / 61


def test_reciprocal_rank_fusion_keeps_documents_without_chunk_ids_separate():
    fused = reciprocal_rank_fusion(
        {
            "vector": [Document(page_content="vector result")],
            "keyword": [Document(page_content="keyword result")],
        },
        top_k=5,
    )

    assert {document.page_content for document in fused} == {
        "vector result",
        "keyword result",
    }


@patch("app.services.retrieval.cohere.ClientV2")
@patch("app.services.retrieval.get_settings")
def test_rerank_chunks_uses_cohere_order_and_preserves_metadata(
    mock_get_settings,
    mock_client_class,
):
    mock_get_settings.return_value.cohere_api_key = "test-key"
    mock_get_settings.return_value.cohere_rerank_model = "rerank-v3.5"
    mock_client = Mock()
    mock_client.rerank.return_value.results = [
        SimpleNamespace(index=1, relevance_score=0.98),
        SimpleNamespace(index=0, relevance_score=0.72),
    ]
    mock_client_class.return_value = mock_client
    candidates = [
        Document(page_content="first candidate", metadata={"chunk_id": "first", "rrf_score": 0.04}),
        Document(page_content="second candidate", metadata={"chunk_id": "second", "rrf_score": 0.03}),
    ]

    results = rerank_chunks("query", candidates, top_k=1)

    assert len(results) == 1
    assert results[0].metadata["chunk_id"] == "second"
    assert results[0].metadata["rrf_score"] == 0.03
    assert results[0].metadata["cohere_score"] == 0.98
    mock_client.rerank.assert_called_once_with(
        model="rerank-v3.5",
        query="query",
        documents=["first candidate", "second candidate"],
        top_n=1,
    )


@patch("app.services.retrieval.rerank_chunks")
@patch("app.services.retrieval.reciprocal_rank_fusion")
@patch("app.services.retrieval.retrieve_chunks")
def test_retrieve_reranked_chunks_expands_candidates_then_applies_rrf(
    mock_retrieve_chunks,
    mock_rrf,
    mock_rerank,
):
    retrieved = {"vector": [], "keyword": []}
    fused = [Document(page_content="fused")]
    reranked = [Document(page_content="reranked")]
    mock_retrieve_chunks.return_value = retrieved
    mock_rrf.return_value = fused
    mock_rerank.return_value = reranked

    result = retrieve_reranked_chunks("query", session_id=42, top_k=4)

    assert result == reranked
    mock_retrieve_chunks.assert_called_once_with("query", 42, top_k=12)
    mock_rrf.assert_called_once_with(retrieved, top_k=12, rrf_k=60)
    mock_rerank.assert_called_once_with("query", fused, top_k=4)
