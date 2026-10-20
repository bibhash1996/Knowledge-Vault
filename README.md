# AdvancedDocRAG

AdvancedDocRAG is a session-scoped document ingestion and retrieval service. It accepts PDF URLs, extracts and chunks document content, stores searchable chunks in Pinecone and Elasticsearch, and provides Python services for dense, keyword, hybrid, and reranked retrieval.

> Retrieval is currently exposed as Python service functions; there is no HTTP search endpoint yet.

## Contents

- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Configuration](#configuration)
- [Run locally](#run-locally)
- [Database migrations](#database-migrations)
- [API](#api)
- [Ingestion flow](#ingestion-flow)
- [Retrieval](#retrieval)
- [Retrieval evaluation](#retrieval-evaluation)
- [Docker](#docker)
- [Tests](#tests)

## Architecture

```text
Client
  ├── POST /api/v1/sessions/ ───────────────> FastAPI ──> PostgreSQL
  └── POST /api/v1/import/knowledge ────────> FastAPI ──> PostgreSQL
                                                     └──> Kafka: doc_import
                                                            │
                                                            v
                                                     Kafka consumer
                                                            │
                                      PDF URL ──> partition and chunk
                                                            │
                                          ┌─────────────────┴──────────────────┐
                                          v                                    v
                               PostgreSQL original_content           Pinecone vectors
                                                                            +
                                                                Elasticsearch BM25

Retrieval service:
  Pinecone dense search + Elasticsearch BM25
                └──> Reciprocal Rank Fusion (RRF)
                         └──> Cohere reranker
                                  └──> hydrate table/image references from PostgreSQL
```

## Project structure

| Path | Purpose |
| --- | --- |
| `app/main.py` | Creates the FastAPI app, configures CORS, and starts/stops Kafka services. |
| `app/api/v1/` | Health, session, and knowledge-import endpoints. |
| `app/core/config.py` | Environment-backed application settings. |
| `app/db/session.py` | SQLAlchemy engine, session factory, and declarative base. |
| `app/db/vector_store.py` | Pinecone vector-store client. |
| `app/db/elasticsearch_store.py` | Elasticsearch client, index mapping, and BM25 search. |
| `app/models/` | SQLAlchemy `Session`, `Knowledge`, and `OriginalContent` models. |
| `app/repositories/` | Database persistence helpers. |
| `app/kafka/` | Kafka producer, consumer, manager, and document-import handler. |
| `app/services/partition.py` | PDF partitioning, chunking, summaries, large-content persistence, and indexing. |
| `app/services/retrieval.py` | Dense search, BM25 search, RRF, Cohere reranking, and content hydration. |
| `app/services/retrieval_metrics.py` | Recall, precision, NDCG, MRR, and macro aggregation. |
| `alembic/versions/` | Database schema migrations. |
| `data/retrieval_eval_questions.jsonl` | Retrieval evaluation questions and answer/evidence guidance. |
| `scripts/evaluate_retrieval.py` | Retrieval candidate export and metric evaluation. |
| `tests/` | Focused tests for ingestion, indexing, retrieval, and metric behavior. |

## Requirements

- Python 3.11 (the Docker image uses Python 3.11).
- PostgreSQL for session, knowledge, and extracted table/image content.
- Pinecone for dense-vector storage.
- Elasticsearch for BM25 keyword search.
- OpenAI API access for embeddings and document summaries.
- Cohere API access for the optional reranking stage.
- Kafka for asynchronous URL ingestion. Kafka is optional for starting the API, but the import endpoint publishes its work through Kafka.
- PDF system packages are needed for local partitioning. The supplied Dockerfile installs `libmagic1`, `poppler-utils`, and `tesseract-ocr`.

Install Python dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configuration

The application loads settings from environment variables and `.env` in the working directory. Copy the following template to `.env` and replace each value with credentials and service addresses from your own environment. Do not commit `.env`.

```dotenv
# Required: PostgreSQL connection URL
DATABASE_URL="postgresql+psycopg2://<user>:<password>@<host>:5432/advanced_doc_rag"

# Optional API behavior
APP_NAME=AdvancedDocRAG
ENVIRONMENT=development
DEBUG=false
ALLOWED_ORIGINS=http://localhost:3000

# Pinecone dense vector search
PINECONE_API_KEY=
PINECONE_INDEX_NAME=
PINECONE_NAMESPACE=

# Elasticsearch keyword/BM25 search
ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_API_KEY=
ELASTICSEARCH_INDEX_NAME=knowledge-chunks

# OpenAI embeddings and summaries
OPENAI_API_KEY=
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Cohere reranking
COHERE_API_KEY=
COHERE_RERANK_MODEL=rerank-v3.5
COHERE_RERANK_MIN_INTERVAL_SECONDS=6.2

# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_PRODUCER_TOPIC=file-events
KAFKA_CONSUMER_TOPIC=file-events
KAFKA_CONSUMER_GROUP=app-consumer
```

`DATABASE_URL` is required to construct the application settings. Pinecone, Elasticsearch, OpenAI, Cohere, and Kafka settings are required only when the corresponding operations are used. Cohere request pacing defaults to 6.2 seconds between requests in a process to accommodate trial-key limits; a 429 response is retried once, honoring `Retry-After` when present.

## Run locally

Apply migrations, then start the API:

```bash
alembic upgrade head
uvicorn app.main:app --reload
```

Alternatively:

```bash
python -m uvicorn app.main:app --reload
```

The API listens on `http://127.0.0.1:8000` by default. OpenAPI docs are available at `/docs` and `/redoc` outside the production environment. The health endpoint is `GET /api/v1/health`.

## Database migrations

The SQL schema is managed by Alembic:

```bash
alembic upgrade head       # apply migrations
alembic current            # show current database revision
alembic history            # show migration history
alembic downgrade -1       # downgrade one revision
```

Current schema:

- `sessions`: unique email and generated session ID.
- `knowledge`: imported URL and processing status, associated with a session.
- `original_content`: extracted table/image values and their `entity_type`, associated with both a session and knowledge record.

## API

### Health

```http
GET /api/v1/health
```

Returns `{"status": "ok"}`.

### Create or retrieve a session

```http
POST /api/v1/sessions/
Content-Type: application/json

{"email": "user@example.com"}
```

Returns the session ID and email. An existing session with the same email is returned instead of creating another.

### Import a PDF URL

```http
POST /api/v1/import/knowledge
Content-Type: application/json
x-session-id: 1

{"url": "https://example.com/document.pdf"}
```

The `x-session-id` header must refer to an existing session. A knowledge record is created in `pending` state and a `doc_import` message is published to the configured producer topic (default `file-events`). The response includes the new `knowledge_id`, `session_id`, and whether publishing succeeded. The Kafka consumer updates the knowledge status while processing (`processing`, then `completed` or `failed`).

## Ingestion flow

1. The Kafka handler receives the PDF URL, `session_id`, and `knowledge_id`.
2. `partition_pdf_file` downloads remote PDFs to a temporary file and runs Unstructured PDF partitioning with table structure and image payload extraction enabled.
3. `create_chunks_by_title` makes title-aware chunks (up to 3,000 characters, with a target new-chunk threshold of 2,400 characters).
4. `summarise_chunks` creates enhanced descriptions for chunks containing tables or images.
5. Table HTML and image base64 payloads are stored in PostgreSQL `original_content`; chunk metadata carries compact references to those rows.
6. Each chunk receives a shared `chunk_id` and is indexed into Pinecone and Elasticsearch. Session and knowledge IDs scope the stored records.

PDF URLs must be reachable by the service. The configured OpenAI, Pinecone, Elasticsearch, PostgreSQL, and Kafka services must be available for their respective pipeline stages.

## Retrieval

Retrieval is currently a Python service, not a FastAPI route:

```python
from app.services.retrieval import (
    retrieve_chunks,
    reciprocal_rank_fusion,
    rerank_chunks,
    retrieve_reranked_chunks,
)

query = "How are Docker images distributed?"

# Separate dense and BM25 lists, filtered to a session
ranked_by_store = retrieve_chunks(
    query=query,
    session_id=1,
    top_k=30,
)

# Fuse result lists with Reciprocal Rank Fusion
fused = reciprocal_rank_fusion(ranked_by_store, top_k=30, rrf_k=60)

# Rerank fused candidates with Cohere
reranked = rerank_chunks(query, fused, top_k=10)

# Or run the complete pipeline; candidate_k defaults to 3 * top_k.
# Table/image references are hydrated from PostgreSQL in final results.
final_chunks = retrieve_reranked_chunks(
    query=query,
    session_id=1,
    top_k=10,
    candidate_k=30,
)
```

`retrieve_chunks` returns `{"vector": [...], "keyword": [...]}`. `retrieve_reranked_chunks` returns LangChain `Document` objects. Each result’s metadata includes the stable `chunk_id`; `original_content` is JSON containing the raw text and hydrated `content_references`. RRF metadata and Cohere relevance scores are included when those stages run.

The vector and keyword indexes must contain the same chunk IDs for hybrid deduplication and evaluation. Elasticsearch applies an exact `session_id` filter; Pinecone search also uses a session metadata filter.

## Retrieval evaluation

The evaluation question set currently contains 39 queries across semantic, keyword, table/figure, specific-fact, unanswerable, and advanced categories. It includes expected evidence guidance, not gold chunk labels. `relevant_chunk_grades` is initially `null`; fill it with verified chunk IDs and graded relevance before interpreting retrieval metrics:

```json
{
  "query_id": "S1",
  "category": "semantic",
  "query": "What problems does Docker solve in the software development and deployment lifecycle?",
  "expected_evidence": "Portability, packaging, and consistency across environments.",
  "relevant_chunk_grades": {
    "actual-chunk-id": 3,
    "another-relevant-chunk-id": 1
  }
}
```

Use relevance grades `0` (not relevant), `1` (related), `2` (partially answers), and `3` (directly answers). Only positive grades count as relevant. Leave labels `null` while a query is unjudged. An empty object `{}` means that the query was judged to have no relevant chunks; such queries are excluded from Recall/Precision/NDCG/MRR averages because those metrics require at least one relevant item. Assess abstention separately for unanswerable questions.

Run evaluation for a session whose documents have been indexed:

```bash
python scripts/evaluate_retrieval.py --session-id 1
```

Options:

```text
--dataset PATH          JSONL question dataset
--session-id INTEGER    Required session ID used to filter retrieval
--top-k INTEGER         Final Cohere rerank result count (default 10)
--candidate-k INTEGER   Candidates per dense/keyword retrieval (default 3 * top-k)
--rrf-k INTEGER         RRF rank constant (default 60)
--candidates-out PATH   Candidate ranking JSONL output path
```

The runner compares vector, keyword/BM25, RRF, and Cohere-reranked results. It exports ranked IDs, text, and metadata to `data/retrieval_eval_candidates.jsonl` for human relevance labeling, then reports per-query, per-category, and overall macro-average metrics for labeled queries.

Implemented metrics include Recall@1/3/5/10, Precision@1/3/5/10, NDCG@1/3/5/10 using graded relevance, and MRR@10. Unlabeled queries and queries with no positively graded chunk are reported as excluded. The runner currently invokes Cohere once per query; with the default trial-key pacing, evaluating all 39 queries takes several minutes.

## Docker

The Docker image installs system packages used by PDF partitioning. See [README.docker.md](README.docker.md) for build and run notes.

## Tests

Focused tests live in `tests/`. If pytest is installed in the active environment:

```bash
python -m pytest -q
```
