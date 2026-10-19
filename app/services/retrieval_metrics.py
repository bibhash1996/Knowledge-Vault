from collections.abc import Sequence
import math
from typing import Any


def calculate_query_metrics(
    ranked_chunk_ids: Sequence[str],
    relevant_chunk_grades: dict[str, int],
    cutoffs: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float | None]:
    """Calculate standard ranked-retrieval metrics for one judged query."""
    if any(cutoff < 1 for cutoff in cutoffs):
        raise ValueError("Metric cutoffs must be at least 1")
    if any(grade < 0 for grade in relevant_chunk_grades.values()):
        raise ValueError("Relevance grades cannot be negative")

    unique_ranked_ids = list(dict.fromkeys(str(chunk_id) for chunk_id in ranked_chunk_ids))
    relevant = {
        str(chunk_id): grade
        for chunk_id, grade in relevant_chunk_grades.items()
        if grade > 0
    }
    total_relevant = len(relevant)
    metrics: dict[str, float | None] = {}

    for cutoff in cutoffs:
        top_ids = unique_ranked_ids[:cutoff]
        relevant_retrieved = sum(chunk_id in relevant for chunk_id in top_ids)
        metrics[f"recall@{cutoff}"] = (
            relevant_retrieved / total_relevant if total_relevant else None
        )
        metrics[f"precision@{cutoff}"] = (
            relevant_retrieved / cutoff if total_relevant else None
        )

        dcg = sum(
            (2**relevant[chunk_id] - 1) / _discount(rank)
            for rank, chunk_id in enumerate(top_ids, start=1)
            if chunk_id in relevant
        )
        ideal_grades = sorted(relevant.values(), reverse=True)[:cutoff]
        ideal_dcg = sum(
            (2**grade - 1) / _discount(rank)
            for rank, grade in enumerate(ideal_grades, start=1)
        )
        metrics[f"ndcg@{cutoff}"] = dcg / ideal_dcg if ideal_dcg else None

    reciprocal_rank_cutoff = max(cutoffs, default=10)
    first_relevant_rank = next(
        (
            rank
            for rank, chunk_id in enumerate(
                unique_ranked_ids[:reciprocal_rank_cutoff],
                start=1,
            )
            if chunk_id in relevant
        ),
        None,
    )
    metrics[f"mrr@{reciprocal_rank_cutoff}"] = (
        1 / first_relevant_rank if first_relevant_rank is not None else 0.0
    ) if relevant else None
    return metrics


def aggregate_query_metrics(
    query_metrics: Sequence[dict[str, float | None]],
) -> dict[str, Any]:
    """Macro-average each metric across queries where it is defined."""
    metric_names = {
        metric_name
        for metrics in query_metrics
        for metric_name in metrics
    }
    aggregates: dict[str, Any] = {"query_count": len(query_metrics)}
    for metric_name in sorted(metric_names):
        values = [
            metrics[metric_name]
            for metrics in query_metrics
            if metrics.get(metric_name) is not None
        ]
        aggregates[metric_name] = sum(values) / len(values) if values else None
        aggregates[f"{metric_name}_query_count"] = len(values)
    return aggregates


def _discount(rank: int) -> float:
    return math.log2(rank + 1)
