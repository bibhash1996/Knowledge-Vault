from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "FastAPI Backend"
    environment: str = "development"
    debug: bool = False

    database_url: str

    allowed_origins: str = ""

    # Pinecone vector store settings
    pinecone_api_key: str = ""
    pinecone_index_name: str = ""
    pinecone_namespace: str = ""

    # Elasticsearch BM25 settings
    elasticsearch_url: str = ""
    elasticsearch_api_key: str = ""
    elasticsearch_index_name: str = "knowledge-chunks"

    # OpenAI API settings
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"

    # Kafka settings
    kafka_bootstrap_servers: str = ""
    kafka_producer_topic: str = ""
    kafka_consumer_topic: str = ""
    kafka_consumer_group: str = "app-consumer"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def cors_origins(self) -> list[str]:
        if not self.allowed_origins:
            return []

        return [
            origin.strip()
            for origin in self.allowed_origins.split(",")
            if origin.strip()
        ]


@lru_cache
def get_settings() -> Settings:
    return Settings()