"""Build the held-out test comparison and generalization report."""

import json
from math import isclose
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results")
CSV_PATH = RESULTS_DIR / "video_games_5core_final_test.csv"
MARKDOWN_PATH = RESULTS_DIR / "video_games_5core_final_test.md"
K = 10

TEST_SPECS = [
    (
        "Collaborative ALS",
        RESULTS_DIR / "video_games_5core_als_test.json",
        "64 latent factors",
    ),
    (
        "OpenAI content",
        RESULTS_DIR / "video_games_5core_openai_semantic_test.json",
        "512d embeddings with FAISS HNSW",
    ),
    (
        "Hybrid",
        RESULTS_DIR / "video_games_5core_hybrid_test.json",
        "ALS Top-10 reranked; semantic weight 0.6",
    ),
]

VALIDATION_PATHS = {
    "Collaborative ALS": RESULTS_DIR / "video_games_5core_als.json",
    "OpenAI content": (
        RESULTS_DIR / "video_games_5core_openai_semantic.json"
    ),
    "Hybrid": RESULTS_DIR / "video_games_5core_hybrid.json",
}


def load_result(path: Path) -> dict:
    """Load one tracked JSON result."""
    if not path.exists():
        raise FileNotFoundError(f"Missing result: {path}")

    with path.open(encoding="utf-8") as source:
        return json.load(source)


def relative_change(new_value: float, baseline: float) -> float:
    """Return percentage change relative to a baseline."""
    return 100.0 * (new_value / baseline - 1.0)


def timing_fields(
    model: str,
    result: dict,
) -> tuple[str, float, float]:
    """Return explicitly scoped bulk-stage timing."""
    if model == "Hybrid":
        seconds = result["ranking_seconds"]
        users = result["recommendation_rows"] / result["k"]
        return "reranking only", seconds, users / seconds

    return (
        "candidate generation",
        result["recommendation_seconds"],
        result["recommendation_users_per_second"],
    )


def validate_test_protocol(model: str, result: dict) -> None:
    """Reject results that do not follow the frozen test protocol."""
    if result.get("evaluation_split") != "test":
        raise ValueError(f"{model} is not a test result.")

    if result.get("test_used_for_tuning") is not False:
        raise ValueError(f"{model} does not declare frozen tuning.")

    if result.get("k") != K:
        raise ValueError(f"{model} does not use K={K}.")

    if result.get("fit_splits") != ["train", "validation"]:
        raise ValueError(
            f"{model} was not fitted through validation."
        )


def main() -> None:
    """Generate final CSV and portfolio-facing Markdown reports."""
    rows = []
    test_results = {}
    validation_results = {}

    for model, path, notes in TEST_SPECS:
        result = load_result(path)
        validate_test_protocol(model, result)
        test_results[model] = result

        timing_scope, seconds, throughput = timing_fields(
            model,
            result,
        )

        rows.append(
            {
                "model": model,
                "recall_at_10": result["recall_at_k"],
                "ndcg_at_10": result["ndcg_at_k"],
                "mrr_at_10": result["mrr_at_k"],
                "catalog_coverage": result[
                    "catalog_coverage"
                ],
                "timing_scope": timing_scope,
                "stage_seconds": seconds,
                "stage_users_per_second": throughput,
                "notes": notes,
                "result_path": path.as_posix(),
            }
        )

        validation_results[model] = load_result(
            VALIDATION_PATHS[model]
        )

    als = test_results["Collaborative ALS"]
    semantic = test_results["OpenAI content"]
    hybrid = test_results["Hybrid"]

    if not isclose(
        hybrid["recall_at_k"],
        als["recall_at_k"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Hybrid does not preserve ALS Recall@10.")

    if not isclose(
        hybrid["catalog_coverage"],
        als["catalog_coverage"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Hybrid does not preserve ALS coverage.")

    table = pd.DataFrame(rows)
    table.to_csv(CSV_PATH, index=False)

    lines = [
        "# Video Games 5-Core Final Test Results",
        "",
        (
            "Protocol: models were fitted on train plus validation "
            "(719,824 interactions and 25,600 products), then evaluated "
            "on 94,762 untouched chronological test events with K=10."
        ),
        (
            "All configurations were fixed using validation results before "
            "test evaluation. No parameter was selected using test metrics."
        ),
        (
            "See the complete "
            "[validation ablation](video_games_5core_ablation.md) for "
            "popularity, TF-IDF, OpenAI, ALS, and hybrid comparisons."
        ),
        "",
        "## Held-Out Ranking Quality",
        "",
        (
            "| Model | Recall@10 | NDCG@10 | MRR@10 | "
            "Coverage | Notes |"
        ),
        "|---|---:|---:|---:|---:|---|",
    ]

    for row in rows:
        lines.append(
            f"| {row['model']} "
            f"| {row['recall_at_10']:.6f} "
            f"| {row['ndcg_at_10']:.6f} "
            f"| {row['mrr_at_10']:.6f} "
            f"| {row['catalog_coverage']:.6f} "
            f"| {row['notes']} |"
        )

    lines.extend(
        [
            "",
            "## Validation-to-Test Generalization",
            "",
            (
                "| Model | Validation Recall@10 | Test Recall@10 | "
                "Validation NDCG@10 | Test NDCG@10 | NDCG change |"
            ),
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    for model, _, _ in TEST_SPECS:
        validation = validation_results[model]
        test = test_results[model]
        ndcg_change = relative_change(
            test["ndcg_at_k"],
            validation["ndcg_at_k"],
        )

        lines.append(
            f"| {model} "
            f"| {validation['recall_at_k']:.6f} "
            f"| {test['recall_at_k']:.6f} "
            f"| {validation['ndcg_at_k']:.6f} "
            f"| {test['ndcg_at_k']:.6f} "
            f"| {ndcg_change:+.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Bulk Performance Context",
            "",
            "| Model | Measured stage | Seconds | Users/second |",
            "|---|---|---:|---:|",
        ]
    )

    for row in rows:
        lines.append(
            f"| {row['model']} "
            f"| {row['timing_scope']} "
            f"| {row['stage_seconds']:.2f} "
            f"| {row['stage_users_per_second']:,.0f} |"
        )

    ndcg_gain = relative_change(
        hybrid["ndcg_at_k"],
        als["ndcg_at_k"],
    )
    mrr_gain = relative_change(
        hybrid["mrr_at_k"],
        als["mrr_at_k"],
    )
    coverage_ratio = (
        semantic["catalog_coverage"]
        / als["catalog_coverage"]
    )

    lines.extend(
        [
            "",
            (
                "Hybrid timing measures reranking after candidates already "
                "exist, not end-to-end serving latency."
            ),
            "",
            "## Findings",
            "",
            (
                f"- The fixed hybrid improves test NDCG@10 by "
                f"{ndcg_gain:.3f}% and MRR@10 by {mrr_gain:.3f}% "
                "over ALS."
            ),
            (
                "- Conservative reranking preserves ALS Recall@10 and "
                "catalogue coverage exactly."
            ),
            (
                f"- OpenAI semantic retrieval covers {coverage_ratio:.1f}x "
                "as much of the fit catalogue as ALS."
            ),
            (
                "- All three models score lower on the later test horizon, "
                "showing why chronological holdout evaluation matters."
            ),
            (
                "- Online p50 and p95 latency remain unreported until the "
                "serving API is benchmarked."
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