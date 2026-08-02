"""Compare FAISS and pgvector HNSW quality and latency."""

import argparse
import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter, perf_counter_ns

import numpy as np

from semanticcart.latency import summarize_latencies
from semanticcart.pgvector_index import (
    PgvectorItemIndex,
    PgvectorSearchConfig,
)
from semanticcart.serving import ServingBundle


DEFAULT_SERVING_ROOT = Path(
    "data/artifacts/video_games_5core/serving"
)
DEFAULT_OUTPUT = Path(
    "results/video_games_5core_vector_retrieval.json"
)


def parse_args() -> argparse.Namespace:
    """Parse reproducible retrieval-benchmark settings."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare FAISS and pgvector HNSW retrieval."
        )
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv(
            "SEMANTICCART_DATABASE_URL"
        ),
    )
    parser.add_argument(
        "--serving-root",
        type=Path,
        default=DEFAULT_SERVING_ROOT,
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--self-queries",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--ef-search",
        type=int,
        nargs="+",
        default=[128, 512, 1000],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Reject invalid benchmark arguments."""
    if not args.database_url:
        raise ValueError(
            "Set SEMANTICCART_DATABASE_URL or "
            "pass --database-url."
        )
    if args.queries <= 0:
        raise ValueError(
            "--queries must be positive."
        )
    if args.warmup < 0:
        raise ValueError(
            "--warmup cannot be negative."
        )
    if args.self_queries <= 0:
        raise ValueError(
            "--self-queries must be positive."
        )
    if args.k <= 0:
        raise ValueError(
            "--k must be positive."
        )
    if (
        not args.ef_search
        or any(value <= 0 for value in args.ef_search)
    ):
        raise ValueError(
            "--ef-search values must be positive."
        )


def sample_positions(
    population: int,
    count: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """Sample deterministic positions from one population."""
    if population <= 0:
        raise ValueError(
            "Population must be positive."
        )

    return generator.choice(
        population,
        size=count,
        replace=count > population,
    )


def benchmark_search(
    index,
    warmup_queries: np.ndarray,
    measured_queries: np.ndarray,
    k: int,
) -> tuple[np.ndarray, dict]:
    """Measure sequential warm search latency and return rankings."""
    for query in warmup_queries:
        index.search(query, k=k)

    latencies_ms = []
    rankings = []
    wall_started = perf_counter()

    for query in measured_queries:
        started = perf_counter_ns()
        _, indices = index.search(
            query,
            k=k,
        )
        elapsed_ms = (
            perf_counter_ns() - started
        ) / 1_000_000

        latencies_ms.append(elapsed_ms)
        rankings.append(indices[0].copy())

    wall_seconds = perf_counter() - wall_started
    metrics = summarize_latencies(
        latencies_ms,
        wall_seconds,
    )

    return (
        np.vstack(rankings),
        metrics.to_dict(),
    )


def compare_rankings(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float]:
    """Measure candidate agreement with FAISS Top-K rankings."""
    if reference.shape != candidate.shape:
        raise ValueError(
            "Ranking matrices must have equal shapes."
        )

    overlaps = []

    for reference_row, candidate_row in zip(
        reference,
        candidate,
        strict=True,
    ):
        overlaps.append(
            len(
                set(reference_row)
                & set(candidate_row)
            )
            / reference.shape[1]
        )

    return {
        "mean_top_k_overlap_with_faiss": float(
            np.mean(overlaps)
        ),
        "top_1_agreement_with_faiss": float(
            np.mean(
                reference[:, 0]
                == candidate[:, 0]
            )
        ),
        "exact_top_k_set_agreement_with_faiss": float(
            np.mean(
                np.asarray(overlaps) == 1.0
            )
        ),
    }


def self_recall_at_one(
    index,
    item_vectors: np.ndarray,
    expected_positions: np.ndarray,
) -> float:
    """Measure whether an indexed item retrieves itself first."""
    _, retrieved = index.search(
        item_vectors,
        k=1,
    )

    return float(
        np.mean(
            retrieved[:, 0]
            == expected_positions
        )
    )


def print_result(result: dict) -> None:
    """Print one concise benchmark row."""
    latency = result["latency"]

    print(
        f"{result['engine']:18} "
        f"ef={str(result['ef_search']):>4} "
        f"overlap="
        f"{result['quality']['mean_top_k_overlap_with_faiss']:.4f} "
        f"top1="
        f"{result['quality']['top_1_agreement_with_faiss']:.4f} "
        f"self@1="
        f"{result['quality']['self_recall_at_1']:.4f} "
        f"p50={latency['p50_ms']:.3f} ms "
        f"p95={latency['p95_ms']:.3f} ms "
        f"qps={latency['requests_per_second']:.1f}"
    )


def main() -> None:
    """Run the frozen profile-query comparison and save JSON."""
    args = parse_args()
    validate_args(args)

    print("Loading the immutable serving bundle...")
    bundle = ServingBundle.load(
        args.serving_root,
        verify_checksums=True,
    )

    generator = np.random.default_rng(
        args.seed
    )
    profile_positions = sample_positions(
        len(bundle.profiles.user_profiles),
        args.queries + args.warmup,
        generator,
    )
    self_query_count = min(
        args.self_queries,
        len(bundle.item_index.item_vectors),
    )
    self_positions = np.linspace(
        0,
        len(bundle.item_index.item_vectors) - 1,
        num=self_query_count,
        dtype=np.int64,
    )
    warmup_queries = bundle.profiles.user_profiles[
        profile_positions[: args.warmup]
    ]
    measured_queries = bundle.profiles.user_profiles[
        profile_positions[args.warmup :]
    ]
    self_vectors = bundle.item_index.item_vectors[
        self_positions
    ]

    print(
        f"Benchmarking {args.queries:,} profile queries "
        f"after {args.warmup:,} warmups..."
    )

    faiss_rankings, faiss_latency = benchmark_search(
        bundle.item_index,
        warmup_queries,
        measured_queries,
        args.k,
    )
    faiss_self_recall = self_recall_at_one(
        bundle.item_index,
        self_vectors,
        self_positions,
    )

    results = [
        {
            "engine": "FAISS HNSW",
            "ef_search": (
                bundle.item_index.config.ef_search
            ),
            "quality": {
                "mean_top_k_overlap_with_faiss": 1.0,
                "top_1_agreement_with_faiss": 1.0,
                "exact_top_k_set_agreement_with_faiss": 1.0,
                "self_recall_at_1": faiss_self_recall,
            },
            "latency": faiss_latency,
        }
    ]
    print_result(results[0])

    for ef_search in args.ef_search:
        print(
            "Benchmarking pgvector with "
            f"ef_search={ef_search}..."
        )

        config = PgvectorSearchConfig(
            ef_search=ef_search,
        )

        with PgvectorItemIndex(
            database_url=args.database_url,
            model_version=bundle.version,
            item_ids=bundle.item_index.item_ids,
            dimensions=(
                bundle.item_index.item_vectors.shape[1]
            ),
            config=config,
        ) as pgvector_index:
            rankings, latency = benchmark_search(
                pgvector_index,
                warmup_queries,
                measured_queries,
                args.k,
            )
            self_recall = self_recall_at_one(
                pgvector_index,
                self_vectors,
                self_positions,
            )

        quality = compare_rankings(
            faiss_rankings,
            rankings,
        )
        quality["self_recall_at_1"] = (
            self_recall
        )

        result = {
            "engine": "pgvector HNSW",
            "ef_search": ef_search,
            "quality": quality,
            "latency": latency,
        }
        results.append(result)
        print_result(result)

    payload = {
        "dataset": bundle.manifest["dataset"],
        "model_version": bundle.version,
        "created_at_utc": datetime.now(
            UTC
        ).isoformat(),
        "platform": platform.platform(),
        "benchmark": {
            "query_type": (
                "frozen returning-user dense profiles"
            ),
            "queries": args.queries,
            "warmup_queries": args.warmup,
            "self_queries": args.self_queries,
            "k": args.k,
            "seed": args.seed,
            "concurrency": 1,
            "latency_scope": (
                "warm sequential retrieval call"
            ),
        },
        "results": results,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()