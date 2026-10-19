from langchain_core.documents import Document
import sys
from pathlib import Path
from typing import Any
import cohere
import json
import logging
import threading
import time
from sqlalchemy import select
from cohere.errors import TooManyRequestsError

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from app.db.elasticsearch_store import search_chunks
from app.db.vector_store import get_vector_store
from app.db.session import SessionLocal
from app.core.config import get_settings
from app.models.original_content import OriginalContent

import dotenv
dotenv.load_dotenv() 

logger = logging.getLogger(__name__)
_cohere_rate_limit_lock = threading.Lock()
_next_cohere_request_at = 0.0


def retrieve_chunks(
    query: str,
    session_id: int,
    top_k: int = 10,
) -> dict[str, list[Document]]:
    """Retrieve session-scoped chunks from vector and BM25 search."""
    if not query.strip():
        return {"vector": [], "keyword": []}
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    print(f"Retrieving chunks for query: {query}, session_id: {session_id}, top_k: {top_k}")
    vector_store = get_vector_store()
    vector_chunks = vector_store.similarity_search(
        query,
        k=top_k,
        filter={"session_id": int(session_id)},
    )
    print(f"Retrieved {len(vector_chunks)} vector chunks for query: {query}")
    keyword_chunks = search_chunks(
        query,
        session_id=int(session_id),
        top_k=top_k,
    )
    print(f"Retrieved {len(vector_chunks)} vector chunks and {len(keyword_chunks)} keyword chunks for query: {query}")

    return {
        "vector": vector_chunks,
        "keyword": keyword_chunks,
    }


def reciprocal_rank_fusion(
    retrieved_chunks: dict[str, list[Document]],
    top_k: int = 10,
    rrf_k: int = 60,
) -> list[Document]:
    """Fuse ranked retrieval lists with RRF, deduplicating shared chunk IDs."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if rrf_k < 0:
        raise ValueError("rrf_k cannot be negative")

    fused: dict[str, dict[str, Any]] = {}
    for source, chunks in retrieved_chunks.items():
        for rank, chunk in enumerate(chunks, start=1):
            metadata = dict(chunk.metadata or {})
            chunk_id = metadata.get("chunk_id")
            key = str(chunk_id) if chunk_id is not None else f"{source}:{rank}"

            entry = fused.setdefault(
                key,
                {
                    "document": chunk,
                    "score": 0.0,
                    "sources": [],
                },
            )
            entry["score"] += 1.0 / (rrf_k + rank)
            if source not in entry["sources"]:
                entry["sources"].append(source)

    ranked_entries = sorted(
        fused.values(),
        key=lambda entry: entry["score"],
        reverse=True,
    )[:top_k]

    fused_documents = []
    for entry in ranked_entries:
        document = entry["document"]
        metadata = dict(document.metadata or {})
        metadata["rrf_score"] = entry["score"]
        metadata["retrieval_sources"] = entry["sources"]
        fused_documents.append(
            Document(page_content=document.page_content, metadata=metadata)
        )
    return fused_documents


def rerank_chunks(
    query: str,
    chunks: list[Document],
    top_k: int = 10,
) -> list[Document]:
    """Rerank RRF candidate chunks with Cohere and return the strongest matches."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not query.strip() or not chunks:
        return []

    settings = get_settings()
    if not settings.cohere_api_key:
        raise RuntimeError("COHERE_API_KEY must be configured to rerank chunks")
    if settings.cohere_rerank_min_interval_seconds < 0:
        raise ValueError("COHERE_RERANK_MIN_INTERVAL_SECONDS cannot be negative")

    client = cohere.ClientV2(api_key=settings.cohere_api_key)
    request = {
        "model": settings.cohere_rerank_model,
        "query": query,
        "documents": [chunk.page_content for chunk in chunks],
        "top_n": min(top_k, len(chunks)),
    }
    _wait_for_cohere_rate_limit(settings.cohere_rerank_min_interval_seconds)
    try:
        response = client.rerank(**request)
    except TooManyRequestsError as exc:
        retry_after = _cohere_retry_after_seconds(exc)
        logger.warning(
            "Cohere rate limit reached; retrying rerank after %.1f seconds",
            retry_after,
        )
        time.sleep(retry_after)
        _wait_for_cohere_rate_limit(settings.cohere_rerank_min_interval_seconds)
        response = client.rerank(**request)

    reranked_chunks = []
    for result in response.results[:top_k]:
        chunk = chunks[result.index]
        metadata = dict(chunk.metadata or {})
        metadata["cohere_score"] = result.relevance_score
        reranked_chunks.append(
            Document(page_content=chunk.page_content, metadata=metadata)
        )
    return reranked_chunks


def _wait_for_cohere_rate_limit(min_interval_seconds: float) -> None:
    global _next_cohere_request_at

    with _cohere_rate_limit_lock:
        delay = max(0.0, _next_cohere_request_at - time.monotonic())
        if delay:
            time.sleep(delay)
        _next_cohere_request_at = time.monotonic() + min_interval_seconds


def _cohere_retry_after_seconds(error: TooManyRequestsError) -> float:
    headers = error.headers or {}
    retry_after = next(
        (
            value
            for key, value in headers.items()
            if key.lower() == "retry-after"
        ),
        None,
    )
    if retry_after is not None:
        try:
            return max(0.0, float(retry_after))
        except (TypeError, ValueError):
            logger.warning("Ignoring invalid Cohere Retry-After value: %r", retry_after)
    return 60.0


def retrieve_reranked_chunks(
    query: str,
    session_id: int,
    top_k: int = 10,
    candidate_k: int | None = None,
    rrf_k: int = 60,
) -> list[Document]:
    """Retrieve an expanded candidate set, fuse it with RRF, then rerank it."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if candidate_k is None:
        candidate_k = top_k * 3
    if candidate_k < top_k:
        raise ValueError("candidate_k must be greater than or equal to top_k")
    if not query.strip():
        return []

    retrieved = retrieve_chunks(query, session_id, top_k=candidate_k)
    fused_chunks = reciprocal_rank_fusion(
        retrieved,
        top_k=candidate_k,
        rrf_k=rrf_k,
    )
    reranked_chunks = rerank_chunks(query, fused_chunks, top_k=top_k)
    return populate_content_references(reranked_chunks, session_id=session_id)


def populate_content_references(
    chunks: list[Document],
    session_id: int,
) -> list[Document]:
    """Add stored table/image values to each chunk's original-content references."""
    parsed_content: list[dict[str, Any] | None] = []
    referenced_ids: set[int] = set()

    for chunk in chunks:
        metadata = dict(chunk.metadata or {})
        original_content = metadata.get("original_content")
        if not original_content:
            parsed_content.append(None)
            continue

        content = (
            json.loads(original_content)
            if isinstance(original_content, str)
            else dict(original_content)
        )
        references = content.get("content_references", [])
        if not isinstance(references, list):
            raise ValueError("original_content.content_references must be a list")
        for reference in references:
            if not isinstance(reference, dict) or "id" not in reference:
                raise ValueError("Each content reference must be an object with an id")
            referenced_ids.add(int(reference["id"]))
        parsed_content.append(content)

    if not referenced_ids:
        return chunks

    with SessionLocal() as db:
        rows = db.scalars(
            select(OriginalContent).where(
                OriginalContent.id.in_(referenced_ids),
                OriginalContent.session_id == int(session_id),
            )
        ).all()

    rows_by_id = {row.id: row for row in rows}
    hydrated_chunks = []
    for chunk, content in zip(chunks, parsed_content):
        if content is None:
            hydrated_chunks.append(chunk)
            continue

        hydrated_references = []
        for reference in content.get("content_references", []):
            row_id = int(reference["id"])
            row = rows_by_id.get(row_id)
            if row is None:
                raise ValueError(
                    f"Original content row {row_id} was not found for "
                    f"session_id={session_id}"
                )
            hydrated_references.append({**reference, "value": row.value})

        content["content_references"] = hydrated_references
        metadata = dict(chunk.metadata or {})
        metadata["original_content"] = json.dumps(content)
        hydrated_chunks.append(
            Document(page_content=chunk.page_content, metadata=metadata)
        )
    return hydrated_chunks



if __name__ == "__main__":
    # Example usage
    query = "What problems does Docker solve in the software development and deployment lifecycle?"
    session_id = 1
    top_k = 5

    reranked_results = retrieve_reranked_chunks(query, 1,5,10)
    print("\nCohere Reranked Results:")
    print(f"Reranked Results: {len(reranked_results)} chunks found.")
    print(reranked_results[0])