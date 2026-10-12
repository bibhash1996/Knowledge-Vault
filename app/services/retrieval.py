from langchain_core.documents import Document
import sys
from pathlib import Path
from typing import Any
import cohere

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from app.db.elasticsearch_store import search_chunks
from app.db.vector_store import get_vector_store
from app.core.config import get_settings

import dotenv
dotenv.load_dotenv() 

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

    vector_store = get_vector_store()
    vector_chunks = vector_store.similarity_search(
        query,
        k=top_k,
        filter={"session_id": int(session_id)},
    )
    keyword_chunks = search_chunks(
        query,
        session_id=int(session_id),
        top_k=top_k,
    )

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

    client = cohere.ClientV2(api_key=settings.cohere_api_key)
    response = client.rerank(
        model=settings.cohere_rerank_model,
        query=query,
        documents=[chunk.page_content for chunk in chunks],
        top_n=min(top_k, len(chunks)),
    )

    reranked_chunks = []
    for result in response.results[:top_k]:
        chunk = chunks[result.index]
        metadata = dict(chunk.metadata or {})
        metadata["cohere_score"] = result.relevance_score
        reranked_chunks.append(
            Document(page_content=chunk.page_content, metadata=metadata)
        )
    return reranked_chunks


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
    return rerank_chunks(query, fused_chunks, top_k=top_k)



if __name__ == "__main__":
    # Example usage
    query = "what are the key recommendations or guidelines provided in the document?"
    session_id = 1
    top_k = 10

    candidate_k = top_k * 3
    results = retrieve_chunks(query, session_id, candidate_k)
    print("Vector Search Results:")
    # for doc in results["vector"]:
    #     print(doc)
    print(f"Vector Search Results: {len(results['vector'])} chunks found.")

    print("\nKeyword Search Results:")
    # for doc in results["keyword"]:
    #     print(doc)
    print(f"Keyword Search Results: {len(results['keyword'])} chunks found.")
    rrf_results = reciprocal_rank_fusion(
        results,
        top_k=candidate_k,
        rrf_k=60,
    )
    print("\nReciprocal Rank Fusion Results:")
    print(f"RRF Results: {len(rrf_results)} chunks found.")
    reranked_results = rerank_chunks(query, rrf_results, top_k=top_k)
    print("\nCohere Reranked Results:")
    print(f"Reranked Results: {len(reranked_results)} chunks found.")
    print(reranked_results[0])