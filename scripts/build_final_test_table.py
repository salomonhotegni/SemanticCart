"""Build the held-out test comparison and generalization report."""

import json
from dataclasses import dataclass
from math import isclose
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results")
CSV_PATH = RESULTS_DIR / "video_games_5core_final_test.csv"
MARKDOWN_PATH = RESULTS_DIR / "video_games_5core_final_test.md"
K = 10


@dataclass(frozen=True)
class ModelSpec:
    """Describe one model result and its metric schema."""

    name: str
    test_path: Path
    validation_path: Path
    notes: str
    timing_kind: str
    test_metrics_key: str | None = None
    validation_metrics_key: str | None = None


DIVERSITY_TEST_PATH = (
    RESULTS_DIR / "video_games_5core_diversity_test.json"
)
DIVERSITY_TUNING_PATH = (
    RESULTS_DIR / "video_games_5core_diversity_tuning.json"
)

MODEL_SPECS = [
    ModelSpec(
        name="Collaborative ALS",
        test_path=RESULTS_DIR / "video_games_5core_als_test.json",
        validation_path=RESULTS_DIR / "video_games_5core_als.json",
        notes="64 latent factors",
        timing_kind="candidate_generation",
    ),
    ModelSpec(
        name="OpenAI content",
        test_path=(
            RESULTS_DIR
            / "video_games_5core_openai_semantic_test.json"
        ),
        validation_path=(
            RESULTS_DIR
            / "video_games_5core_openai_semantic.json"
        ),
        notes="512d embeddings with FAISS HNSW",
        timing_kind="candidate_generation",
    ),
    ModelSpec(
        name="Long-term hybrid",
        test_path=RESULTS_DIR / "video_games_5core_hybrid_test.json",
        validation_path=RESULTS_DIR / "video_games_5core_hybrid.json",
        notes="ALS Top-10 reranked by long-term semantics; weight 0.6",
        timing_kind="legacy_reranking",
    ),
    ModelSpec(
        name="Returning-user hybrid",
        test_path=(
            RESULTS_DIR
            / "video_games_5core_returning_user_test.json"
        ),
        validation_path=(
            RESULTS_DIR
            / "video_games_5core_returning_user_tuning.json"
        ),
        notes="ALS Top-10 reranked by one-item session intent; weight 0.5",
        timing_kind="legacy_reranking",
        validation_metrics_key="best_validation_result",
    ),
    ModelSpec(
        name="Top-25 returning-user",
        test_path=DIVERSITY_TEST_PATH,
        validation_path=DIVERSITY_TUNING_PATH,
        notes="Top-10 selected from 25 ALS candidates; session weight 0.5",
        timing_kind="deep_relevance",
        test_metrics_key="baseline_metrics",
        validation_metrics_key="full_validation_baseline",
    ),
    ModelSpec(
        name="Diversity-aware reranker",
        test_path=DIVERSITY_TEST_PATH,
        validation_path=DIVERSITY_TUNING_PATH,
        notes="MMR with semantic, category, and price redundancy",
        timing_kind="diversity_reranking",
        test_metrics_key="selected_metrics",
        validation_metrics_key="full_validation_selected",
    ),
]


def load_result(path: Path) -> dict:
    """Load one required tracked JSON result."""
    if not path.exists():
        raise FileNotFoundError(f"Missing result: {path}")

    with path.open(encoding="utf-8") as source:
        return json.load(source)


def metric_block(
    result: dict,
    key: str | None,
) -> dict:
    """Extract flat or nested ranking metrics."""
    return result if key is None else result[key]


def relative_change(
    new_value: float,
    baseline: float,
) -> float:
    """Return percentage change relative to a baseline."""
    return 100.0 * (new_value / baseline - 1.0)


def timing_fields(
    spec: ModelSpec,
    result: dict,
) -> tuple[str, float, float]:
    """Return explicitly scoped offline bulk-stage timing."""
    if spec.timing_kind == "candidate_generation":
        return (
            "candidate generation",
            result["recommendation_seconds"],
            result["recommendation_users_per_second"],
        )

    if spec.timing_kind == "legacy_reranking":
        seconds = result["ranking_seconds"]
        users = result.get("candidate_users")

        if users is None:
            users = (
                result["recommendation_rows"]
                / result["k"]
            )

        return "Top-10 reranking only", seconds, users / seconds

    users = result["candidate_users"]

    if spec.timing_kind == "deep_relevance":
        seconds = result["relevance_ranking_seconds"]
        return (
            "Top-25 relevance reranking only",
            seconds,
            users / seconds,
        )

    if spec.timing_kind == "diversity_reranking":
        seconds = result["diversity_reranking_seconds"]
        return (
            "MMR Top-10 selection only",
            seconds,
            users / seconds,
        )

    raise ValueError(
        f"Unknown timing kind: {spec.timing_kind}"
    )


def validate_test_protocol(
    spec: ModelSpec,
    result: dict,
) -> None:
    """Reject results that do not follow the frozen test protocol."""
    if result.get("evaluation_split") != "test":
        raise ValueError(
            f"{spec.name} is not a test result."
        )
    if result.get("test_used_for_tuning") is not False:
        raise ValueError(
            f"{spec.name} does not declare frozen tuning."
        )
    if result.get("k") != K:
        raise ValueError(
            f"{spec.name} does not use K={K}."
        )
    if result.get("fit_splits") != ["train", "validation"]:
        raise ValueError(
            f"{spec.name} was not fitted through validation."
        )

    if spec.name == "Returning-user hybrid":
        if result.get("session_weight") != 0.5:
            raise ValueError(
                "Returning-user weight does not match validation."
            )
        if (
            result.get(
                "test_ground_truth_loaded_after_ranking"
            )
            is not True
        ):
            raise ValueError(
                "Returning-user labels were not loaded after ranking."
            )

    if spec.test_path == DIVERSITY_TEST_PATH:
        if result.get("candidate_k") != 25:
            raise ValueError(
                "Diversity result does not use Top-25 candidates."
            )
        if result.get("selected_session_weight") != 0.5:
            raise ValueError(
                "Deep session weight does not match validation."
            )
        if (
            result.get(
                "test_ground_truth_loaded_after_ranking"
            )
            is not True
        ):
            raise ValueError(
                "Diversity labels were not loaded after ranking."
            )


def validate_frozen_diversity(
    test_result: dict,
    tuning_result: dict,
) -> None:
    """Confirm test diversity settings equal validation selections."""
    if tuning_result.get("selection_split") != "validation":
        raise ValueError(
            "Diversity tuning did not use validation."
        )
    if tuning_result.get("test_used_for_tuning") is not False:
        raise ValueError(
            "Diversity tuning does not exclude test."
        )
    if (
        test_result["selected_session_weight"]
        != tuning_result["selected_session_weight"]
    ):
        raise ValueError(
            "Diversity session weight changed on test."
        )
    if (
        test_result["selected_diversity_config"]
        != tuning_result["selected_diversity_config"]
    ):
        raise ValueError(
            "Diversity configuration changed on test."
        )


def main() -> None:
    """Generate final CSV and portfolio-facing Markdown reports."""
    rows = []
    test_metrics = {}
    validation_metrics = {}
    raw_test_results = {}

    for spec in MODEL_SPECS:
        test_result = load_result(spec.test_path)
        validation_result = load_result(
            spec.validation_path
        )

        validate_test_protocol(spec, test_result)

        if spec.test_path == DIVERSITY_TEST_PATH:
            validate_frozen_diversity(
                test_result,
                validation_result,
            )

        test = metric_block(
            test_result,
            spec.test_metrics_key,
        )
        validation = metric_block(
            validation_result,
            spec.validation_metrics_key,
        )

        test_metrics[spec.name] = test
        validation_metrics[spec.name] = validation
        raw_test_results[spec.name] = test_result

        timing_scope, seconds, throughput = timing_fields(
            spec,
            test_result,
        )

        rows.append(
            {
                "model": spec.name,
                "recall_at_10": test["recall_at_k"],
                "ndcg_at_10": test["ndcg_at_k"],
                "mrr_at_10": test["mrr_at_k"],
                "catalog_coverage": test[
                    "catalog_coverage"
                ],
                "timing_scope": timing_scope,
                "stage_seconds": seconds,
                "stage_users_per_second": throughput,
                "notes": spec.notes,
                "result_path": spec.test_path.as_posix(),
            }
        )

    als = test_metrics["Collaborative ALS"]
    semantic = test_metrics["OpenAI content"]
    long_term = test_metrics["Long-term hybrid"]
    returning = test_metrics["Returning-user hybrid"]
    deep_relevance = test_metrics["Top-25 returning-user"]
    diversified = test_metrics["Diversity-aware reranker"]

    for name in (
        "Long-term hybrid",
        "Returning-user hybrid",
    ):
        reranker = test_metrics[name]

        if not isclose(
            reranker["recall_at_k"],
            als["recall_at_k"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{name} does not preserve ALS Recall@10."
            )
        if not isclose(
            reranker["catalog_coverage"],
            als["catalog_coverage"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{name} does not preserve ALS coverage."
            )

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
            "[validation ablation](video_games_5core_ablation.md) and "
            "[diversity report](video_games_5core_diversity.md)."
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

    for spec in MODEL_SPECS:
        validation = validation_metrics[spec.name]
        test = test_metrics[spec.name]
        ndcg_change = relative_change(
            test["ndcg_at_k"],
            validation["ndcg_at_k"],
        )

        lines.append(
            f"| {spec.name} "
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

    final_recall_gain = relative_change(
        diversified["recall_at_k"],
        als["recall_at_k"],
    )
    final_ndcg_gain = relative_change(
        diversified["ndcg_at_k"],
        als["ndcg_at_k"],
    )
    final_mrr_gain = relative_change(
        diversified["mrr_at_k"],
        als["mrr_at_k"],
    )
    final_coverage_gain = relative_change(
        diversified["catalog_coverage"],
        als["catalog_coverage"],
    )
    depth_ndcg_gain = relative_change(
        deep_relevance["ndcg_at_k"],
        returning["ndcg_at_k"],
    )
    diversity_ndcg_gain = relative_change(
        diversified["ndcg_at_k"],
        deep_relevance["ndcg_at_k"],
    )
    semantic_coverage_ratio = (
        semantic["catalog_coverage"]
        / als["catalog_coverage"]
    )

    lines.extend(
        [
            "",
            (
                "Timing rows measure different offline bulk stages and are "
                "not end-to-end serving latency."
            ),
            "",
            "## Findings",
            "",
            (
                f"- The final diversity-aware model improves Recall@10 by "
                f"{final_recall_gain:.3f}%, NDCG@10 by "
                f"{final_ndcg_gain:.3f}%, and MRR@10 by "
                f"{final_mrr_gain:.3f}% over ALS."
            ),
            (
                f"- Final catalogue coverage is "
                f"{final_coverage_gain:.3f}% higher than ALS coverage."
            ),
            (
                f"- Expanding the returning-user candidate pool from 10 to "
                f"25 improves NDCG@10 by {depth_ndcg_gain:.3f}%."
            ),
            (
                f"- Frozen MMR adds another "
                f"{diversity_ndcg_gain:.3f}% NDCG@10 while improving "
                "semantic and category diversity."
            ),
            (
                f"- OpenAI semantic retrieval covers "
                f"{semantic_coverage_ratio:.1f}x as much of the fit "
                "catalogue as ALS."
            ),
            (
                "- All six models score lower on the later test horizon, "
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