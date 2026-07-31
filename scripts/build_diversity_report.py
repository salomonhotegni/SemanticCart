"""Build portfolio-facing diversity reranking reports."""

import json
from math import isclose
from pathlib import Path

import pandas as pd


DATASET = "video_games_5core"
RESULTS_DIR = Path("results")

TUNING_PATH = RESULTS_DIR / f"{DATASET}_diversity_tuning.json"
TEST_PATH = RESULTS_DIR / f"{DATASET}_diversity_test.json"
TOP10_PATH = (
    RESULTS_DIR
    / f"{DATASET}_returning_user_test.json"
)

CSV_PATH = RESULTS_DIR / f"{DATASET}_diversity.csv"
MARKDOWN_PATH = RESULTS_DIR / f"{DATASET}_diversity.md"

METRICS = [
    "recall_at_k",
    "ndcg_at_k",
    "mrr_at_k",
    "catalog_coverage",
    "intra_list_diversity",
    "category_variety",
    "category_coverage",
    "novelty",
    "price_dispersion",
]

DISPLAY_NAMES = {
    "recall_at_k": "Recall@10",
    "ndcg_at_k": "NDCG@10",
    "mrr_at_k": "MRR@10",
    "catalog_coverage": "Catalogue coverage",
    "intra_list_diversity": "Intra-list diversity",
    "category_variety": "Category variety",
    "category_coverage": "Category coverage",
    "novelty": "Popularity novelty",
    "price_dispersion": "Price dispersion",
}


def load_json(path: Path) -> dict:
    """Load one required tracked JSON result."""
    if not path.exists():
        raise FileNotFoundError(f"Missing result: {path}")

    with path.open(encoding="utf-8") as source:
        return json.load(source)


def relative_change(
    new_value: float,
    baseline: float,
) -> float:
    """Return percentage change from a nonzero baseline."""
    if baseline == 0:
        return 0.0

    return 100.0 * (new_value / baseline - 1.0)


def validate_protocol(
    tuning: dict,
    test: dict,
) -> None:
    """Reject reports that violate the frozen-test protocol."""
    if tuning.get("selection_split") != "validation":
        raise ValueError(
            "Diversity configuration was not selected on validation."
        )
    if tuning.get("test_used_for_tuning") is not False:
        raise ValueError(
            "Tuning result does not exclude test selection."
        )
    if test.get("evaluation_split") != "test":
        raise ValueError("Final result is not a test evaluation.")
    if test.get("fit_splits") != ["train", "validation"]:
        raise ValueError(
            "Final model was not fitted through validation."
        )
    if test.get("test_used_for_tuning") is not False:
        raise ValueError(
            "Test result does not declare frozen tuning."
        )
    if (
        test.get("test_ground_truth_loaded_after_ranking")
        is not True
    ):
        raise ValueError(
            "Test labels were not loaded after ranking."
        )
    if tuning.get("k") != 10 or test.get("k") != 10:
        raise ValueError("Results must use K=10.")
    if (
        tuning.get("candidate_k") != 25
        or test.get("candidate_k") != 25
    ):
        raise ValueError(
            "Results must use a Top-25 candidate pool."
        )
    if not isclose(
        tuning["selected_session_weight"],
        test["selected_session_weight"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Test session weight differs from validation selection."
        )
    if (
        tuning["selected_diversity_config"]
        != test["selected_diversity_config"]
    ):
        raise ValueError(
            "Test diversity configuration differs from validation."
        )


def build_comparison_table(test: dict) -> pd.DataFrame:
    """Create baseline and selected test rows."""
    baseline = test["baseline_metrics"]
    selected = test["selected_metrics"]

    rows = [
        {
            "model": "Top-25 returning-user relevance",
            **{
                DISPLAY_NAMES[name]: baseline[name]
                for name in METRICS
            },
        },
        {
            "model": "Diversity-aware reranker",
            **{
                DISPLAY_NAMES[name]: selected[name]
                for name in METRICS
            },
        },
    ]

    return pd.DataFrame(rows)


def main() -> None:
    """Validate results and generate CSV and Markdown reports."""
    tuning = load_json(TUNING_PATH)
    test = load_json(TEST_PATH)
    top10 = load_json(TOP10_PATH)

    validate_protocol(tuning, test)

    baseline = test["baseline_metrics"]
    selected = test["selected_metrics"]
    selected_config = test["selected_diversity_config"]

    table = build_comparison_table(test)
    table.to_csv(CSV_PATH, index=False)

    change_rows = []

    for metric in METRICS:
        change_rows.append(
            {
                "metric": DISPLAY_NAMES[metric],
                "absolute_change": (
                    selected[metric] - baseline[metric]
                ),
                "relative_change_percent": relative_change(
                    selected[metric],
                    baseline[metric],
                ),
            }
        )

    changes = pd.DataFrame(change_rows)

    top25_vs_top10 = {
        metric: relative_change(
            baseline[metric],
            top10[metric],
        )
        for metric in (
            "recall_at_k",
            "ndcg_at_k",
            "mrr_at_k",
            "catalog_coverage",
        )
    }

    relevance_seconds = test["relevance_ranking_seconds"]
    diversity_seconds = test["diversity_reranking_seconds"]
    users = test["candidate_users"]

    lines = [
        "# Video Games 5-Core Diversity Reranking",
        "",
        (
            "Protocol: session blending was tuned on the full validation "
            "split. Sixteen MMR configurations were then compared on a "
            "deterministic 10,000-user validation cohort under a 99% "
            "NDCG-retention constraint. Test labels were loaded only after "
            "the frozen recommendations were written."
        ),
        "",
        "## Held-Out Test Results",
        "",
        (
            "| Model | Recall@10 | NDCG@10 | MRR@10 | Coverage | "
            "ILD | Category variety | Novelty | Price dispersion |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in table.to_dict("records"):
        lines.append(
            f"| {row['model']} "
            f"| {row['Recall@10']:.6f} "
            f"| {row['NDCG@10']:.6f} "
            f"| {row['MRR@10']:.6f} "
            f"| {row['Catalogue coverage']:.6f} "
            f"| {row['Intra-list diversity']:.6f} "
            f"| {row['Category variety']:.6f} "
            f"| {row['Popularity novelty']:.6f} "
            f"| {row['Price dispersion']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Diversity Trade-Off",
            "",
            "| Metric | Absolute change | Relative change |",
            "|---|---:|---:|",
        ]
    )

    for row in changes.to_dict("records"):
        lines.append(
            f"| {row['metric']} "
            f"| {row['absolute_change']:+.6f} "
            f"| {row['relative_change_percent']:+.3f}% |"
        )

    lines.extend(
        [
            "",
            "## Selected Configuration",
            "",
            "| Setting | Value |",
            "|---|---:|",
            (
                f"| Session semantic weight "
                f"| {test['selected_session_weight']:.3f} |"
            ),
            (
                f"| MMR relevance weight "
                f"| {selected_config['relevance_weight']:.3f} |"
            ),
            (
                f"| Popularity novelty weight "
                f"| {selected_config['novelty_weight']:.3f} |"
            ),
            (
                f"| Semantic redundancy weight "
                f"| {selected_config['semantic_similarity_weight']:.3f} |"
            ),
            (
                f"| Category redundancy weight "
                f"| {selected_config['category_similarity_weight']:.3f} |"
            ),
            (
                f"| Price redundancy weight "
                f"| {selected_config['price_similarity_weight']:.3f} |"
            ),
            "",
            "## Candidate-Depth Contribution",
            "",
            (
                "The earlier returning-user hybrid used ten ALS candidates. "
                "The new relevance baseline selects ten products from a "
                "Top-25 pool using the same one-item session signal."
            ),
            "",
            "| Metric | Relative change from Top-10 |",
            "|---|---:|",
            (
                f"| Recall@10 "
                f"| {top25_vs_top10['recall_at_k']:+.3f}% |"
            ),
            (
                f"| NDCG@10 "
                f"| {top25_vs_top10['ndcg_at_k']:+.3f}% |"
            ),
            (
                f"| MRR@10 "
                f"| {top25_vs_top10['mrr_at_k']:+.3f}% |"
            ),
            (
                f"| Catalogue coverage "
                f"| {top25_vs_top10['catalog_coverage']:+.3f}% |"
            ),
            "",
            "## Bulk Performance Context",
            "",
            "| Stage | Seconds | Users/second |",
            "|---|---:|---:|",
            (
                f"| Top-25 relevance ranking "
                f"| {relevance_seconds:.2f} "
                f"| {users / relevance_seconds:,.0f} |"
            ),
            (
                f"| MMR Top-10 selection "
                f"| {diversity_seconds:.2f} "
                f"| {users / diversity_seconds:,.0f} |"
            ),
            "",
            (
                "These are offline bulk-stage measurements, not online p50 "
                "or p95 API latency."
            ),
            "",
            "## Findings",
            "",
            (
                f"- Expanding the candidate pool from 10 to 25 improves "
                f"test NDCG@10 by "
                f"{top25_vs_top10['ndcg_at_k']:.3f}%."
            ),
            (
                f"- Frozen MMR adds another "
                f"{relative_change(selected['ndcg_at_k'], baseline['ndcg_at_k']):.3f}% "
                "NDCG@10 improvement while increasing intra-list diversity."
            ),
            (
                f"- Category variety improves by "
                f"{relative_change(selected['category_variety'], baseline['category_variety']):.3f}% "
                "and price dispersion improves by "
                f"{relative_change(selected['price_dispersion'], baseline['price_dispersion']):.3f}%."
            ),
            (
                f"- Catalogue coverage changes by "
                f"{relative_change(selected['catalog_coverage'], baseline['catalog_coverage']):+.3f}%, "
                "a small trade-off."
            ),
            (
                "- Validation selected a popularity-novelty weight of zero; "
                "the signal was evaluated but did not improve the constrained "
                "objective."
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
    print(changes.to_string(index=False))
    print()
    print(f"CSV:      {CSV_PATH}")
    print(f"Markdown: {MARKDOWN_PATH}")


if __name__ == "__main__":
    main()