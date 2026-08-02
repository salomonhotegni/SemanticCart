"""Build portfolio-facing vector retrieval reports."""

import json
import math
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results")
INPUT_PATH = (
    RESULTS_DIR
    / "video_games_5core_vector_retrieval.json"
)
CSV_PATH = (
    RESULTS_DIR
    / "video_games_5core_vector_retrieval.csv"
)
MARKDOWN_PATH = (
    RESULTS_DIR
    / "video_games_5core_vector_retrieval.md"
)

QUALITY_METRICS = {
    "mean_top_k_overlap_with_faiss",
    "top_1_agreement_with_faiss",
    "exact_top_k_set_agreement_with_faiss",
    "self_recall_at_1",
}

LATENCY_METRICS = {
    "requests",
    "requests_per_second",
    "mean_ms",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "maximum_ms",
}


def load_result() -> dict:
    """Load the required retrieval benchmark."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing retrieval result: {INPUT_PATH}"
        )

    with INPUT_PATH.open(
        encoding="utf-8"
    ) as source:
        return json.load(source)


def validate_result(result: dict) -> None:
    """Validate benchmark methodology and measurements."""
    benchmark = result.get("benchmark")

    if not isinstance(benchmark, dict):
        raise ValueError(
            "Benchmark metadata is missing."
        )
    if benchmark.get("k") != 10:
        raise ValueError(
            "Retrieval benchmark must use K=10."
        )
    if benchmark.get("concurrency") != 1:
        raise ValueError(
            "Retrieval comparison must use concurrency 1."
        )
    if benchmark.get("queries", 0) <= 0:
        raise ValueError(
            "Retrieval query count must be positive."
        )
    if benchmark.get("warmup_queries", -1) < 0:
        raise ValueError(
            "Warmup count cannot be negative."
        )
    if benchmark.get("self_queries", 0) <= 0:
        raise ValueError(
            "Self-query count must be positive."
        )

    rows = result.get("results")

    if not isinstance(rows, list) or not rows:
        raise ValueError(
            "Retrieval result rows are missing."
        )

    engine_settings = {
        (
            row.get("engine"),
            row.get("ef_search"),
        )
        for row in rows
    }
    required_settings = {
        ("FAISS HNSW", 128),
        ("pgvector HNSW", 128),
        ("pgvector HNSW", 512),
        ("pgvector HNSW", 1000),
    }

    if not required_settings.issubset(
        engine_settings
    ):
        raise ValueError(
            "Required engine settings are missing."
        )

    for row in rows:
        quality = row.get("quality")
        latency = row.get("latency")

        if not isinstance(quality, dict):
            raise ValueError(
                "Quality metrics are missing."
            )
        if not isinstance(latency, dict):
            raise ValueError(
                "Latency metrics are missing."
            )

        missing_quality = (
            QUALITY_METRICS - set(quality)
        )
        missing_latency = (
            LATENCY_METRICS - set(latency)
        )

        if missing_quality:
            raise ValueError(
                "Missing quality metrics: "
                f"{sorted(missing_quality)}"
            )
        if missing_latency:
            raise ValueError(
                "Missing latency metrics: "
                f"{sorted(missing_latency)}"
            )

        for metric in QUALITY_METRICS:
            value = quality[metric]

            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(
                    f"Invalid quality metric: {metric}"
                )

        for metric in LATENCY_METRICS:
            value = latency[metric]

            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    f"Invalid latency metric: {metric}"
                )


def build_table(result: dict) -> pd.DataFrame:
    """Create one row per engine and search setting."""
    rows = []

    for record in result["results"]:
        quality = record["quality"]
        latency = record["latency"]

        rows.append(
            {
                "engine": record["engine"],
                "ef_search": record["ef_search"],
                "top_10_overlap_with_faiss": quality[
                    "mean_top_k_overlap_with_faiss"
                ],
                "top_1_agreement_with_faiss": quality[
                    "top_1_agreement_with_faiss"
                ],
                "exact_top_10_set_agreement": quality[
                    "exact_top_k_set_agreement_with_faiss"
                ],
                "systematic_self_recall_at_1": quality[
                    "self_recall_at_1"
                ],
                "p50_ms": latency["p50_ms"],
                "p95_ms": latency["p95_ms"],
                "p99_ms": latency["p99_ms"],
                "requests_per_second": latency[
                    "requests_per_second"
                ],
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    """Validate and generate CSV and Markdown reports."""
    result = load_result()
    validate_result(result)
    table = build_table(result)

    table.to_csv(
        CSV_PATH,
        index=False,
    )

    faiss = table.loc[
        table["engine"] == "FAISS HNSW"
    ].iloc[0]
    selected = table.loc[
        (table["engine"] == "pgvector HNSW")
        & (table["ef_search"] == 128)
    ].iloc[0]
    broad = table.loc[
        (table["engine"] == "pgvector HNSW")
        & (table["ef_search"] == 1000)
    ].iloc[0]

    p50_multiplier = (
        selected["p50_ms"]
        / faiss["p50_ms"]
    )
    p95_multiplier = (
        selected["p95_ms"]
        / faiss["p95_ms"]
    )
    broad_multiplier = (
        broad["p50_ms"]
        / selected["p50_ms"]
    )

    lines = [
        "# SemanticCart Vector Retrieval",
        "",
        (
            f"Model version: "
            f"`{result['model_version']}`  "
        ),
        f"Dataset: `{result['dataset']}`  ",
        (
            "Workload: frozen returning-user dense "
            "profiles  "
        ),
        (
            "Protocol: 500 measured queries, "
            "25 warmups, concurrency 1  "
        ),
        "",
        "## Results",
        "",
        (
            "| Engine | ef_search | Top-10 overlap "
            "with FAISS | Top-1 agreement | "
            "Self Recall@1 | p50 (ms) | p95 (ms) | "
            "Throughput (queries/s) |"
        ),
        (
            "|---|---:|---:|---:|---:|---:|---:|---:|"
        ),
    ]

    for row in table.to_dict(
        orient="records"
    ):
        lines.append(
            f"| {row['engine']} "
            f"| {row['ef_search']} "
            f"| {row['top_10_overlap_with_faiss']:.4f} "
            f"| {row['top_1_agreement_with_faiss']:.4f} "
            f"| {row['systematic_self_recall_at_1']:.4f} "
            f"| {row['p50_ms']:.3f} "
            f"| {row['p95_ms']:.3f} "
            f"| {row['requests_per_second']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                f"- pgvector at `ef_search=128` retained "
                f"{selected['top_10_overlap_with_faiss']:.2%} "
                "mean Top-10 overlap and full Top-1 "
                "agreement on user-profile queries."
            ),
            (
                f"- Its p50 and p95 were "
                f"{p50_multiplier:.1f}x and "
                f"{p95_multiplier:.1f}x FAISS latency, "
                "respectively."
            ),
            (
                "- Increasing pgvector to `ef_search=512` "
                "did not improve measured profile overlap "
                "or systematic self-retrieval."
            ),
            (
                f"- `ef_search=1000` improved systematic "
                f"self-retrieval to "
                f"{broad['systematic_self_recall_at_1']:.2%}, "
                f"but raised p50 by {broad_multiplier:.1f}x "
                "relative to pgvector at 128."
            ),
            (
                "- FAISS remains the deployed in-process "
                "retrieval engine; pgvector is a validated "
                "durable alternative for deployments that "
                "prioritize centralized vector storage."
            ),
            "",
            "## Methodology",
            "",
            (
                "- Latency excludes model loading, pool "
                "startup, and warmup queries."
            ),
            (
                "- Top-10 overlap uses FAISS HNSW as the "
                "reference, not an exact brute-force oracle."
            ),
            (
                "- Self Recall@1 uses 100 systematic "
                "catalogue positions, including both "
                "catalogue endpoints."
            ),
        ]
    )

    MARKDOWN_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(table.to_string(index=False))
    print()
    print(f"CSV:      {CSV_PATH}")
    print(f"Markdown: {MARKDOWN_PATH}")


if __name__ == "__main__":
    main()