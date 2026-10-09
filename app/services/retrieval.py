from langchain_core.documents import Document
import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from app.db.elasticsearch_store import search_chunks
from app.db.vector_store import get_vector_store

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



if __name__ == "__main__":
    # Example usage
    query = "what are the key recommendations or guidelines provided in the document?"
    session_id = 1
    top_k = 10

    results = retrieve_chunks(query, session_id, top_k)
    print("Vector Search Results:")
    # for doc in results["vector"]:
    #     print(doc)
    print(f"Vector Search Results: {len(results['vector'])} chunks found.")

    print("\nKeyword Search Results:")
    # for doc in results["keyword"]:
    #     print(doc)
    print(f"Keyword Search Results: {len(results['keyword'])} chunks found.")