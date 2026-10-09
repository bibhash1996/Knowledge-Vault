from functools import lru_cache
from typing import Any

from elasticsearch import Elasticsearch
from langchain_core.documents import Document

from app.core.config import get_settings


@lru_cache(maxsize=1)
def get_elasticsearch_client() -> Elasticsearch:
    """Return the shared Elasticsearch client used for BM25 indexing and search."""
    settings = get_settings()
    if not settings.elasticsearch_url:
        raise RuntimeError("ELASTICSEARCH_URL must be configured to use Elasticsearch")

    client_options: dict[str, Any] = {}
    if settings.elasticsearch_api_key:
        client_options["api_key"] = settings.elasticsearch_api_key

    return Elasticsearch(settings.elasticsearch_url, **client_options)


def ensure_chunks_index() -> None:
    """Create the chunk index and its BM25-searchable fields when absent."""
    settings = get_settings()
    client = get_elasticsearch_client()
    if client.indices.exists(index=settings.elasticsearch_index_name):
        return

    client.indices.create(
        index=settings.elasticsearch_index_name,
        mappings={
            "properties": {
                "chunk_id": {"type": "keyword"},
                "session_id": {"type": "integer"},
                "knowledge_id": {"type": "integer"},
                "file_url": {"type": "keyword"},
                "content": {"type": "text"},
                "original_content": {"type": "text"},
            }
        },
    )


def search_chunks(
    query: str,
    session_id: int,
    top_k: int = 10,
    knowledge_id: int | None = None,
) -> list[Document]:
    """Run session-scoped BM25 search and return LangChain documents."""
    if not query.strip():
        return []
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    settings = get_settings()
    client = get_elasticsearch_client()
    if not client.indices.exists(index=settings.elasticsearch_index_name):
        return []

    filters: list[dict[str, Any]] = [{"term": {"session_id": int(session_id)}}]
    if knowledge_id is not None:
        filters.append({"term": {"knowledge_id": int(knowledge_id)}})

    response = client.search(
        index=settings.elasticsearch_index_name,
        size=top_k,
        query={
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["content^2", "original_content"],
                        }
                    }
                ],
                "filter": filters,
            }
        },
    )

    documents = []
    for hit in response["hits"]["hits"]:
        source = hit["_source"]
        documents.append(
            Document(
                page_content=source.get("content", ""),
                metadata={
                    "chunk_id": source.get("chunk_id", hit["_id"]),
                    "session_id": source["session_id"],
                    "knowledge_id": source.get("knowledge_id"),
                    "file_url": source.get("file_url", ""),
                    "original_content": source.get("original_content", ""),
                    "bm25_score": hit["_score"],
                },
            )
        )
    return documents
