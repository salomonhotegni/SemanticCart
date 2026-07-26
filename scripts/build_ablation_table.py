"""Build reproducible model-quality and performance comparison tables."""

import json
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results")
CSV_PATH = RESULTS_DIR / "video_games_5core_ablation.csv"
MARKDOWN_PATH = RESULTS_DIR / "video_games_5core_ablation.md"

MODEL_SPECS = [
    (
        "Popularity",
        RESULTS_DIR / "video_games_5core_popularity.json",
        "Global Top-K",
    ),
    (
        "TF-IDF content",
        RESULTS_DIR / "video_games_5core_tfidf.json",
        "50,000-feature sparse retrieval",
    ),
    (
        "OpenAI content",
        RESULTS_DIR / "video_games_5core_openai_semantic.json",
        "512d embeddings with FAISS HNSW",
    ),
    (
        "Collaborative ALS",
        RESULTS_DIR / "video_games_5core_als.json",
        "64 latent factors",
    ),
    (
        "Hybrid",
        RESULTS_DIR / "video_games_5core_hybrid.json",
        "ALS Top-10 reranked; semantic weight 0.6",
    ),
]


def load_result(path: Path) -> dict:
    """Load one baseline result artifact."""
    if not path.exists():
        raise FileNotFoundError(f"Missing result: {path}")

    with path.open(encoding="utf-8") as source:
        return json.load(source)


def timing_fields(
    name: str,
    result: dict,
) -> tuple[str, float | None, float | None]:
    """Return explicitly scoped bulk timing measurements."""
    if name == "Hybrid":
        seconds = result["ranking_seconds"]
        users = result["recommendation_rows"] / result["k"]
        return "reranking only", seconds, users / seconds

    seconds = result.get("recommendation_seconds")
    throughput = result.get(
        "recommendation_users_per_second"
    )

    if seconds is None:
        return "not recorded", None, None

    return "candidate generation", seconds, throughput


def relative_gain(new_value: float, baseline: float) -> float:
    """Calculate percentage improvement over a baseline."""
    return 100.0 * (new_value / baseline - 1.0)


def main() -> None:
    """Build CSV and Markdown ablation artifacts from tracked JSON."""
    rows = []

    for name, path, notes in MODEL_SPECS:
        result = load_result(path)
        timing_scope, stage_seconds, stage_throughput = (
            timing_fields(name, result)
        )

        rows.append(
            {
                "model": name,
                "recall_at_10": result["recall_at_k"],
                "ndcg_at_10": result["ndcg_at_k"],
                "mrr_at_10": result["mrr_at_k"],
                "catalog_coverage": result[
                    "catalog_coverage"
                ],
                "timing_scope": timing_scope,
                "stage_seconds": stage_seconds,
                "stage_users_per_second": stage_throughput,
                "notes": notes,
                "result_path": path.as_posix(),
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(CSV_PATH, index=False)

    indexed = table.set_index("model")
    tfidf = indexed.loc["TF-IDF content"]
    openai = indexed.loc["OpenAI content"]
    als = indexed.loc["Collaborative ALS"]
    hybrid = indexed.loc["Hybrid"]

    lines = [
        "# Video Games 5-Core Validation Ablation",
        "",
        (
            "Protocol: leave-last-two chronological split, one validation "
            "target per user, 94,762 users, 25,527 training products, K=10."
        ),
        (
            "Hybrid alpha was selected on validation; the test split remains "
            "reserved for final evaluation."
        ),
        "",
        "## Ranking Quality",
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
            "## Bulk Performance Context",
            "",
            (
                "| Model | Measured stage | Seconds | "
                "Users/second |"
            ),
            "|---|---|---:|---:|",
        ]
    )

    for row in rows:
        seconds = (
            "-"
            if row["stage_seconds"] is None
            else f"{row['stage_seconds']:.2f}"
        )
        throughput = (
            "-"
            if row["stage_users_per_second"] is None
            else f"{row['stage_users_per_second']:,.0f}"
        )

        lines.append(
            f"| {row['model']} "
            f"| {row['timing_scope']} "
            f"| {seconds} "
            f"| {throughput} |"
        )

    lines.extend(
        [
            "",
            (
                "Hybrid timing measures reranking after candidates already "
                "exist. It is not end-to-end serving latency."
            ),
            "",
            "## Findings",
            "",
            (
                f"- OpenAI embeddings improve NDCG@10 by "
                f"{relative_gain(openai['ndcg_at_10'], tfidf['ndcg_at_10']):.1f}% "
                "over TF-IDF."
            ),
            (
                f"- Conservative hybrid reranking improves NDCG@10 by "
                f"{relative_gain(hybrid['ndcg_at_10'], als['ndcg_at_10']):.1f}% "
                "over ALS while preserving Recall@10 and coverage."
            ),
            (
                "- OpenAI content reaches broad catalogue coverage, while ALS "
                "remains the strongest source for warm-user recall."
            ),
            (
                "- p50 and p95 online latency are intentionally omitted until "
                "the serving API is benchmarked."
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