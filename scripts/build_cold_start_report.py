"""Build portfolio-facing cold-start result tables."""

import json
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results")
SOURCE_PATH = RESULTS_DIR / "video_games_5core_cold_start.json"
CSV_PATH = RESULTS_DIR / "video_games_5core_cold_start.csv"
MARKDOWN_PATH = RESULTS_DIR / "video_games_5core_cold_start.md"
SESSION_LENGTHS = (1, 3, 5)


def load_results() -> dict:
    """Load and validate the frozen cold-start evaluation."""
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Missing cold-start result: {SOURCE_PATH}"
        )

    with SOURCE_PATH.open(encoding="utf-8") as source:
        result = json.load(source)

    if result.get("evaluation_split") != "test":
        raise ValueError("Cold-start result is not a test result.")

    if result.get("test_used_for_tuning") is not False:
        raise ValueError("Test data was not declared frozen.")

    if result.get("candidate_catalog_scope") != "full_catalogue":
        raise ValueError("Cold products were not retrieval candidates.")

    if result.get("faiss_build_threads") != 1:
        raise ValueError(
            "Cold-start index was not built deterministically."
        )

    return result


def main() -> None:
    """Generate CSV and Markdown cold-start reports."""
    result = load_results()
    cold = result["cold_item"]
    natural = result["simulated_new_user_sessions"]
    controlled = result[
        "controlled_common_session_cohort"
    ]

    rows = [
        {
            "cohort": "real_cold_items",
            "scope": "real test cohort",
            "history_items": None,
            "users": cold["cohort_users"],
            "events": cold["cohort_events"],
            "recall_at_10": cold["recall_at_k"],
            "ndcg_at_10": cold["ndcg_at_k"],
            "mrr_at_10": cold["mrr_at_k"],
            "catalog_coverage": cold["catalog_coverage"],
            "recall_ci_95_low": (
                cold["recall_at_k_wilson_95_low"]
            ),
            "recall_ci_95_high": (
                cold["recall_at_k_wilson_95_high"]
            ),
        }
    ]

    for session_length in SESSION_LENGTHS:
        session = natural[str(session_length)]

        rows.append(
            {
                "cohort": "simulated_new_user",
                "scope": "natural eligible cohort",
                "history_items": session_length,
                "users": session["eligible_users"],
                "events": session["evaluation_events"],
                "recall_at_10": session["recall_at_k"],
                "ndcg_at_10": session["ndcg_at_k"],
                "mrr_at_10": session["mrr_at_k"],
                "catalog_coverage": session[
                    "catalog_coverage"
                ],
                "recall_ci_95_low": None,
                "recall_ci_95_high": None,
            }
        )

    for session_length in SESSION_LENGTHS:
        session = controlled["results"][str(session_length)]

        rows.append(
            {
                "cohort": "simulated_new_user",
                "scope": "controlled common cohort",
                "history_items": session_length,
                "users": session["eligible_users"],
                "events": session["evaluation_events"],
                "recall_at_10": session["recall_at_k"],
                "ndcg_at_10": session["ndcg_at_k"],
                "mrr_at_10": session["mrr_at_k"],
                "catalog_coverage": session[
                    "catalog_coverage"
                ],
                "recall_ci_95_low": None,
                "recall_ci_95_high": None,
            }
        )

    table = pd.DataFrame(rows)
    table["history_items"] = table[
        "history_items"
    ].astype("Int64")
    table.to_csv(CSV_PATH, index=False)

    lines = [
        "# Video Games 5-Core Cold-Start Evaluation",
        "",
        (
            "Protocol: embeddings cover the complete 25,612-product "
            "catalogue, while behavioral fitting uses only train plus "
            "validation interactions. Test data was not used for tuning."
        ),
        (
            "See the [final warm-user results]"
            "(video_games_5core_final_test.md) for ALS, semantic, and "
            "hybrid comparisons."
        ),
        "",
        "## Real New-Item Cohort",
        "",
        (
            "These products have metadata and embeddings but no interactions "
            "in the final fitting data."
        ),
        "",
        "| Events | Users | Products | Hits@10 | Recall@10 | 95% CI | "
        "NDCG@10 | MRR@10 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {cold['cohort_events']:,} "
            f"| {cold['cohort_users']:,} "
            f"| {cold['cohort_items']:,} "
            f"| {cold['hit_count_at_k']:,} "
            f"| {cold['recall_at_k']:.6f} "
            f"| [{cold['recall_at_k_wilson_95_low']:.6f}, "
            f"{cold['recall_at_k_wilson_95_high']:.6f}] "
            f"| {cold['ndcg_at_k']:.6f} "
            f"| {cold['mrr_at_k']:.6f} |"
        ),
        "",
        (
            f"The semantic model surfaced "
            f"{cold['catalogue_only_items_recommended']} of "
            f"{result['catalogue_only_items']} new products and produced "
            f"{cold['hit_count_at_k']} correct Top-10 matches."
        ),
        (
            "ALS cannot represent these products because they have no "
            "fitted interaction factors."
        ),
        "",
        "## Simulated New Users",
        "",
        (
            "Each profile uses only the user's most recent N unique viewed "
            "products. User identity and older behavior are not model "
            "features."
        ),
        "",
        "| History items | Eligible users | Recall@10 | NDCG@10 | "
        "MRR@10 | Coverage | Users/second |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for session_length in SESSION_LENGTHS:
        session = natural[str(session_length)]

        lines.append(
            f"| {session_length} "
            f"| {session['eligible_users']:,} "
            f"| {session['recall_at_k']:.6f} "
            f"| {session['ndcg_at_k']:.6f} "
            f"| {session['mrr_at_k']:.6f} "
            f"| {session['catalog_coverage']:.6f} "
            f"| {session['recommendation_users_per_second']:,.0f} |"
        )

    lines.extend(
        [
            "",
            "## Controlled Session-Length Comparison",
            "",
            (
                f"All session lengths below use the same "
                f"{controlled['eligible_users']:,}-user cohort."
            ),
            "",
            "| History items | Recall@10 | NDCG@10 | MRR@10 | Coverage |",
            "|---:|---:|---:|---:|---:|",
        ]
    )

    for session_length in SESSION_LENGTHS:
        session = controlled["results"][str(session_length)]

        lines.append(
            f"| {session_length} "
            f"| {session['recall_at_k']:.6f} "
            f"| {session['ndcg_at_k']:.6f} "
            f"| {session['mrr_at_k']:.6f} "
            f"| {session['catalog_coverage']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation and Limitations",
            "",
            (
                "- One-item session intent performs best under the current "
                "recency-weighted averaging strategy."
            ),
            (
                "- This is a dataset-specific observation, not evidence that "
                "shorter histories are universally better."
            ),
            (
                "- The real cold-item cohort contains only 69 events, so its "
                "wide confidence interval must accompany the headline recall."
            ),
            (
                "- Simulated new users are established users whose histories "
                "were deliberately truncated; they are not organically new "
                "production users."
            ),
            (
                "- Two complete runs produced identical recommendation "
                "artifact hashes using single-threaded HNSW construction."
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