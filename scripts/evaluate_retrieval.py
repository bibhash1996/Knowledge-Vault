import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.retrieval import reciprocal_rank_fusion, rerank_chunks, retrieve_chunks
from app.services.retrieval_metrics import (
    aggregate_query_metrics,
    calculate_query_metrics,
)


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as dataset_file:
        records = [json.loads(line) for line in dataset_file if line.strip()]

    seen_ids: set[str] = set()
    for record in records:
        query_id = record.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            raise ValueError("Every evaluation record must have a query_id")
        if query_id in seen_ids:
            raise ValueError(f"Duplicate query_id in dataset: {query_id}")
        seen_ids.add(query_id)
        if not isinstance(record.get("query"), str) or not record["query"].strip():
            raise ValueError(f"Query {query_id} has no query text")
        grades = record.get("relevant_chunk_grades")
        if grades is not None and (
            not isinstance(grades, dict)
            or any(not isinstance(grade, int) for grade in grades.values())
        ):
            raise ValueError(
                f"Query {query_id}: relevant_chunk_grades must be null "
                "or a chunk_id-to-integer-grade object"
            )
    return records


def chunk_id(chunk: Any) -> str | None:
    value = (chunk.metadata or {}).get("chunk_id")
    return str(value) if value is not None else None


def serialize_ranking(chunks: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "chunk_id": chunk_id(chunk),
            "content": chunk.page_content,
            "metadata": dict(chunk.metadata or {}),
        }
        for rank, chunk in enumerate(chunks, start=1)
    ]


def evaluate(
    dataset: list[dict[str, Any]],
    session_id: int,
    top_k: int,
    candidate_k: int,
    rrf_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metric_cutoffs = (1, 3, 5, 10)
    system_query_metrics: dict[str, list[dict[str, float | None]]] = {
        "vector": [],
        "keyword": [],
        "rrf": [],
        "cohere_reranked": [],
    }
    category_query_metrics: dict[
        str, dict[str, list[dict[str, float | None]]]
    ] = {}
    candidate_records = []
    excluded_unlabeled = []
    excluded_no_relevant = []
    per_query: dict[str, Any] = {}

    total_queries = len(dataset)
    for query_number, record in enumerate(dataset, start=1):
        query_id = record["query_id"]
        query = record["query"]
        print(
            f"[{query_number}/{total_queries}] {query_id}: retrieving "
            f"top {candidate_k} candidates per search",
            flush=True,
        )
        results = retrieve_chunks(query, session_id, top_k=candidate_k)
        print(
            f"[{query_id}] Retrieved vector={len(results['vector'])}, "
            f"keyword={len(results['keyword'])}",
            flush=True,
        )
        fused = reciprocal_rank_fusion(
            results,
            top_k=candidate_k,
            rrf_k=rrf_k,
        )
        print(f"[{query_id}] RRF fused {len(fused)} candidates", flush=True)
        print(f"[{query_id}] Reranking top {top_k} with Cohere...", flush=True)
        reranked = rerank_chunks(query, fused, top_k=top_k)
        print(f"[{query_id}] Cohere returned {len(reranked)} chunks", flush=True)
        rankings = {
            "vector": results["vector"],
            "keyword": results["keyword"],
            "rrf": fused,
            "cohere_reranked": reranked,
        }
        candidate_records.append(
            {
                "query_id": query_id,
                "query": query,
                "rankings": {
                    name: serialize_ranking(chunks)
                    for name, chunks in rankings.items()
                },
            }
        )

        grades = record.get("relevant_chunk_grades")
        if grades is None:
            excluded_unlabeled.append(query_id)
            print(f"[{query_id}] Metrics skipped: relevance labels are null", flush=True)
            continue
        if not any(grade > 0 for grade in grades.values()):
            excluded_no_relevant.append(query_id)
            print(
                f"[{query_id}] Metrics skipped: no positively relevant chunks labeled",
                flush=True,
            )
            continue

        per_query[query_id] = {}
        category = record.get("category", "uncategorized")
        category_metrics = category_query_metrics.setdefault(
            category,
            {system: [] for system in system_query_metrics},
        )
        for system, chunks in rankings.items():
            metrics = calculate_query_metrics(
                [
                    value
                    for value in (
                        chunk_id(doc)
                        for doc in chunks
                    )
                    if value is not None
                ],
                grades,
                cutoffs=metric_cutoffs,
            )
            system_query_metrics[system].append(metrics)
            category_metrics[system].append(metrics)
            per_query[query_id][system] = metrics

        print(f"[{query_id}] Metrics calculated for all retrieval systems", flush=True)

    summary = {
        "session_id": session_id,
        "top_k": top_k,
        "candidate_k": candidate_k,
        "rrf_k": rrf_k,
        "judged_query_count": len(per_query),
        "excluded_unlabeled_query_ids": excluded_unlabeled,
        "excluded_queries_without_relevant_chunks": excluded_no_relevant,
        "metrics_by_system": {
            system: aggregate_query_metrics(metrics)
            for system, metrics in system_query_metrics.items()
        },
        "metrics_by_category": {
            category: {
                system: aggregate_query_metrics(metrics)
                for system, metrics in system_metrics.items()
            }
            for category, system_metrics in category_query_metrics.items()
        },
        "per_query": per_query,
    }
    return summary, candidate_records


def main() -> None:
    print("Evaluating retrieval systems...")
    parser = argparse.ArgumentParser(
        description="Evaluate vector, BM25, RRF, and Cohere retrieval rankings."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data/retrieval_eval_questions.jsonl",
    )
    parser.add_argument("--session-id", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--candidates-out",
        type=Path,
        default=PROJECT_ROOT / "data/retrieval_eval_candidates.jsonl",
        help="Write ranked chunk IDs and text here for relevance labeling.",
    )
    args = parser.parse_args()
    print(f"Loading dataset from {args.dataset}...")
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    candidate_k = args.candidate_k or args.top_k * 3
    if candidate_k < args.top_k:
        parser.error("--candidate-k must be at least --top-k")

    dataset = load_dataset(args.dataset)
    print(f"Loaded {len(dataset)} evaluation records from {args.dataset}")
    summary, candidate_records = evaluate(
        dataset,
        session_id=args.session_id,
        top_k=args.top_k,
        candidate_k=candidate_k,
        rrf_k=args.rrf_k,
    )
    args.candidates_out.parent.mkdir(parents=True, exist_ok=True)
    with args.candidates_out.open("w", encoding="utf-8") as candidates_file:
        for record in candidate_records:
            candidates_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nCandidate rankings written to {args.candidates_out}")


if __name__ == "__main__":
    main()
