"""Measure warm end-to-end SemanticCart API recommendation latency."""

import argparse
import json
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial
from importlib.metadata import version
from pathlib import Path
from urllib.parse import quote

import httpx2
import numpy as np
import pandas as pd

from semanticcart.latency import (
    summarize_latencies,
)


DEFAULT_INTERACTIONS = Path(
    "data/processed/amazon_video_games_5core/"
    "test.parquet"
)
DEFAULT_OUTPUT = Path(
    "results/video_games_5core_api_latency.json"
)


def parse_args() -> argparse.Namespace:
    """Parse benchmark configuration."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark returning-user recommendation latency "
            "against a running SemanticCart API."
        )
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--interactions",
        type=Path,
        default=DEFAULT_INTERACTIONS,
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 8],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:
    """Reject invalid benchmark settings."""
    if args.requests <= 0:
        raise ValueError(
            "--requests must be positive."
        )
    if args.warmup < 0:
        raise ValueError(
            "--warmup cannot be negative."
        )
    if args.k <= 0:
        raise ValueError("--k must be positive.")
    if args.timeout <= 0:
        raise ValueError(
            "--timeout must be positive."
        )
    if (
        not args.concurrency
        or any(value <= 0 for value in args.concurrency)
    ):
        raise ValueError(
            "--concurrency values must be positive."
        )


def sample_users(
    path: Path,
    sample_size: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Sample deterministic returning users from test interactions."""
    if not path.exists():
        raise FileNotFoundError(
            f"Interaction file not found: {path}"
        )

    interactions = pd.read_parquet(
        path,
        columns=["user_id"],
    )
    users = np.sort(
        interactions["user_id"]
        .astype(str)
        .unique()
    )

    if len(users) == 0:
        raise ValueError(
            "Interaction file contains no users."
        )

    generator = np.random.default_rng(seed)
    sampled = generator.choice(
        users,
        size=sample_size,
        replace=sample_size > len(users),
    )

    return sampled, len(users)


def timed_recommendation(
    client: httpx2.Client,
    user_id: str,
    k: int,
) -> float:
    """Issue one request and return its latency in milliseconds."""
    path = (
        "/recommendations/"
        f"{quote(str(user_id), safe='')}"
    )

    started = time.perf_counter_ns()
    response = client.get(
        path,
        params={"k": k},
    )
    elapsed_ms = (
        time.perf_counter_ns() - started
    ) / 1_000_000

    response.raise_for_status()
    body = response.json()

    if body.get("count") != k:
        raise RuntimeError(
            f"Expected {k} recommendations for "
            f"{user_id}; received {body.get('count')}."
        )

    recommendations = body.get(
        "recommendations"
    )

    if (
        not isinstance(recommendations, list)
        or len(recommendations) != k
    ):
        raise RuntimeError(
            "Recommendation response has an invalid schema."
        )

    return elapsed_ms


def run_scenario(
    base_url: str,
    users: np.ndarray,
    warmup_users: np.ndarray,
    k: int,
    concurrency: int,
    timeout: float,
) -> dict:
    """Measure one fixed-concurrency workload."""
    limits = httpx2.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
    )

    with httpx2.Client(
        base_url=base_url,
        timeout=timeout,
        limits=limits,
        headers={
            "User-Agent": (
                "semanticcart-latency-benchmark/1.0"
            )
        },
    ) as client:
        request = partial(
            timed_recommendation,
            client,
            k=k,
        )

        if concurrency == 1:
            for user_id in warmup_users:
                request(user_id)

            wall_started = time.perf_counter()
            latencies = [
                request(user_id)
                for user_id in users
            ]
            wall_seconds = (
                time.perf_counter()
                - wall_started
            )
        else:
            with ThreadPoolExecutor(
                max_workers=concurrency
            ) as executor:
                list(
                    executor.map(
                        request,
                        warmup_users,
                    )
                )

                wall_started = time.perf_counter()
                latencies = list(
                    executor.map(
                        request,
                        users,
                    )
                )
                wall_seconds = (
                    time.perf_counter()
                    - wall_started
                )

    metrics = summarize_latencies(
        latencies,
        wall_seconds,
    ).to_dict()

    return {
        "concurrency": concurrency,
        "workload": (
            "returning_user_recent_profile"
        ),
        "k": k,
        **metrics,
    }


def main() -> None:
    """Run warm sequential and concurrent API benchmarks."""
    args = parse_args()
    validate_args(args)

    total_sample_size = max(
        args.requests,
        args.warmup,
    )
    sampled_users, available_users = (
        sample_users(
            args.interactions,
            total_sample_size,
            args.seed,
        )
    )
    measured_users = sampled_users[
        : args.requests
    ]
    warmup_users = sampled_users[
        : args.warmup
    ]
    base_url = args.base_url.rstrip("/")

    with httpx2.Client(
        base_url=base_url,
        timeout=args.timeout,
    ) as client:
        health_response = client.get(
            "/health"
        )
        health_response.raise_for_status()
        health = health_response.json()

        model_response = client.get(
            "/model-info"
        )
        model_response.raise_for_status()
        model_info = model_response.json()

    scenarios = []

    for concurrency in dict.fromkeys(
        args.concurrency
    ):
        print(
            f"Benchmarking concurrency={concurrency} "
            f"with {args.requests:,} requests..."
        )
        scenario = run_scenario(
            base_url=base_url,
            users=measured_users,
            warmup_users=warmup_users,
            k=args.k,
            concurrency=concurrency,
            timeout=args.timeout,
        )
        scenarios.append(scenario)

        print(
            f"p50={scenario['p50_ms']:.3f} ms "
            f"p95={scenario['p95_ms']:.3f} ms "
            f"p99={scenario['p99_ms']:.3f} ms "
            f"throughput="
            f"{scenario['requests_per_second']:.1f} req/s"
        )

    report = {
        "benchmark": (
            "semanticcart_http_recommendation_latency"
        ),
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "base_url": base_url,
        "transport": "HTTP/1.1 loopback",
        "server_process": (
            "one warm Uvicorn worker"
        ),
        "startup_and_model_load_excluded": True,
        "connection_reuse": True,
        "request_count_per_scenario": (
            args.requests
        ),
        "warmup_requests_per_scenario": (
            args.warmup
        ),
        "random_seed": args.seed,
        "interaction_source": str(
            args.interactions
        ),
        "available_test_users": available_users,
        "model_version": health[
            "model_version"
        ],
        "dataset": model_info["dataset"],
        "client_environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "httpx2": version("httpx2"),
        },
        "scenarios": scenarios,
        "notes": [
            (
                "Measurements include HTTP serialization, "
                "routing, inference, reranking, and response "
                "deserialization."
            ),
            (
                "Measurements use local loopback and do not "
                "include internet or cross-host network latency."
            ),
            (
                "Returning users use persisted recent profiles; "
                "no live session events are submitted."
            ),
        ],
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()