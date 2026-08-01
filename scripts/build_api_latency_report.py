"""Build portfolio-facing API latency reports."""

import json
import math
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results")
INPUT_PATH = (
    RESULTS_DIR
    / "video_games_5core_api_latency.json"
)
CSV_PATH = (
    RESULTS_DIR
    / "video_games_5core_api_latency.csv"
)
MARKDOWN_PATH = (
    RESULTS_DIR
    / "video_games_5core_api_latency.md"
)

REQUIRED_METRICS = {
    "requests",
    "wall_seconds",
    "requests_per_second",
    "mean_ms",
    "minimum_ms",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "maximum_ms",
}


def load_report() -> dict:
    """Load the required latency benchmark result."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing latency result: {INPUT_PATH}"
        )

    with INPUT_PATH.open(
        encoding="utf-8"
    ) as source:
        return json.load(source)


def validate_report(
    report: dict,
) -> None:
    """Validate benchmark methodology and measurements."""
    if report.get("benchmark") != (
        "semanticcart_http_recommendation_latency"
    ):
        raise ValueError(
            "Unexpected latency benchmark type."
        )
    if report.get(
        "startup_and_model_load_excluded"
    ) is not True:
        raise ValueError(
            "Benchmark must describe warm serving latency."
        )
    if report.get("connection_reuse") is not True:
        raise ValueError(
            "Benchmark must reuse HTTP connections."
        )

    scenarios = report.get("scenarios")

    if not isinstance(scenarios, list):
        raise ValueError(
            "Latency scenarios are missing."
        )

    concurrency_values = {
        scenario.get("concurrency")
        for scenario in scenarios
    }

    if not {1, 8}.issubset(
        concurrency_values
    ):
        raise ValueError(
            "Concurrency 1 and 8 results are required."
        )

    for scenario in scenarios:
        missing = (
            REQUIRED_METRICS - set(scenario)
        )

        if missing:
            raise ValueError(
                f"Missing latency metrics: "
                f"{sorted(missing)}"
            )
        if scenario.get("k") != 10:
            raise ValueError(
                "Latency benchmark must use K=10."
            )
        if scenario.get("workload") != (
            "returning_user_recent_profile"
        ):
            raise ValueError(
                "Unexpected latency workload."
            )

        for metric in REQUIRED_METRICS:
            value = scenario[metric]

            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    f"Invalid latency metric: {metric}"
                )


def scenario_table(
    report: dict,
) -> pd.DataFrame:
    """Build one row per concurrency scenario."""
    rows = []

    for scenario in sorted(
        report["scenarios"],
        key=lambda value: value["concurrency"],
    ):
        rows.append(
            {
                "concurrency": scenario[
                    "concurrency"
                ],
                "requests": scenario["requests"],
                "k": scenario["k"],
                "p50_ms": scenario["p50_ms"],
                "p95_ms": scenario["p95_ms"],
                "p99_ms": scenario["p99_ms"],
                "mean_ms": scenario["mean_ms"],
                "maximum_ms": scenario[
                    "maximum_ms"
                ],
                "requests_per_second": scenario[
                    "requests_per_second"
                ],
                "wall_seconds": scenario[
                    "wall_seconds"
                ],
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    """Generate CSV and Markdown latency reports."""
    report = load_report()
    validate_report(report)
    table = scenario_table(report)

    table.to_csv(
        CSV_PATH,
        index=False,
    )

    sequential = table.loc[
        table["concurrency"] == 1
    ].iloc[0]
    concurrent = table.loc[
        table["concurrency"] == 8
    ].iloc[0]

    p95_multiplier = (
        concurrent["p95_ms"]
        / sequential["p95_ms"]
    )
    throughput_ratio = (
        concurrent["requests_per_second"]
        / sequential["requests_per_second"]
    )

    lines = [
        "# SemanticCart Online API Latency",
        "",
        (
            f"Model version: `{report['model_version']}`  "
        ),
        f"Dataset: `{report['dataset']}`  ",
        (
            f"Workload: returning-user hybrid Top-10 "
            f"recommendations  "
        ),
        (
            f"Transport: {report['transport']}  "
        ),
        (
            f"Server: {report['server_process']}  "
        ),
        "",
        "## Results",
        "",
        (
            "| Concurrency | Requests | p50 (ms) | "
            "p95 (ms) | p99 (ms) | Throughput (req/s) |"
        ),
        "|---:|---:|---:|---:|---:|---:|",
    ]

    for row in table.to_dict(
        orient="records"
    ):
        lines.append(
            f"| {row['concurrency']} "
            f"| {row['requests']} "
            f"| {row['p50_ms']:.3f} "
            f"| {row['p95_ms']:.3f} "
            f"| {row['p99_ms']:.3f} "
            f"| {row['requests_per_second']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                f"- Warm low-load latency is "
                f"{sequential['p50_ms']:.3f} ms p50 and "
                f"{sequential['p95_ms']:.3f} ms p95."
            ),
            (
                f"- At concurrency 8, p95 latency is "
                f"{concurrent['p95_ms']:.3f} ms, or "
                f"{p95_multiplier:.1f}x the low-load p95."
            ),
            (
                f"- Concurrency-8 throughput is "
                f"{throughput_ratio:.2f}x low-load throughput, "
                "showing saturation in the single CPU-bound worker."
            ),
            (
                "- Measurements include HTTP handling, JSON "
                "serialization, ALS retrieval, semantic scoring, "
                "MMR reranking, and response deserialization."
            ),
            (
                "- Model startup and checksum verification are excluded."
            ),
            (
                "- Loopback measurements exclude cross-host network "
                "latency and should not be presented as internet latency."
            ),
            "",
        ]
    )

    MARKDOWN_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(table.to_string(index=False))
    print()
    print(f"CSV:      {CSV_PATH}")
    print(f"Markdown: {MARKDOWN_PATH}")


if __name__ == "__main__":
    main()