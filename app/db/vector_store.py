from functools import lru_cache

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from app.core.config import get_settings


@lru_cache(maxsize=1)
def get_vector_store() -> PineconeVectorStore:
    """Return the shared LangChain vector store backed by Pinecone."""
    settings = get_settings()
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY must be configured to use the vector store")
    if not settings.pinecone_index_name:
        raise RuntimeError("PINECONE_INDEX_NAME must be configured to use the vector store")

    pinecone = Pinecone(api_key=settings.pinecone_api_key)
    index = pinecone.Index(settings.pinecone_index_name)
    embeddings = OpenAIEmbeddings(model=settings.openai_embedding_model)

    return PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=settings.pinecone_namespace or None,
    )