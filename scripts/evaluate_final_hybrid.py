"""Evaluate the validation-selected conservative hybrid on test."""

import json
from dataclasses import asdict
from math import isclose
from pathlib import Path
from time import perf_counter

import pandas as pd

from semanticcart.evaluation import evaluate_ranking
from semanticcart.hybrid import (
    HybridConfig,
    rerank_collaborative_candidates,
)


DATASET = "video_games_5core"
DATA_DIR = Path("data/processed/amazon_video_games_5core")
ARTIFACT_ROOT = Path("data/artifacts") / DATASET / "final"

ALS_CANDIDATES_PATH = (
    ARTIFACT_ROOT / "als" / "test_recommendations.parquet"
)
SEMANTIC_CANDIDATES_PATH = (
    ARTIFACT_ROOT
    / "openai_semantic"
    / "test_recommendations.parquet"
)
HYBRID_ARTIFACT_DIR = ARTIFACT_ROOT / "hybrid"

ALS_RESULTS_PATH = (
    Path("results") / f"{DATASET}_als_test.json"
)
SEMANTIC_RESULTS_PATH = (
    Path("results") / f"{DATASET}_openai_semantic_test.json"
)
RESULTS_PATH = (
    Path("results") / f"{DATASET}_hybrid_test.json"
)

K = 10
SEMANTIC_WEIGHT = 0.6


def load_candidates() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the two frozen final candidate sets."""
    collaborative = pd.read_parquet(ALS_CANDIDATES_PATH)
    semantic = pd.read_parquet(SEMANTIC_CANDIDATES_PATH)

    for frame in (collaborative, semantic):
        frame["user_id"] = frame["user_id"].astype(str)
        frame["item_id"] = frame["item_id"].astype(str)

    if collaborative["rank"].max() != K:
        raise ValueError(
            f"Expected ALS candidate depth {K}."
        )

    if semantic["rank"].max() != K:
        raise ValueError(
            f"Expected semantic candidate depth {K}."
        )

    return collaborative, semantic


def verify_candidate_set(
    collaborative: pd.DataFrame,
    hybrid: pd.DataFrame,
) -> None:
    """Confirm conservative reranking preserves every ALS candidate."""
    expected = (
        collaborative[["user_id", "item_id"]]
        .sort_values(["user_id", "item_id"])
        .reset_index(drop=True)
    )
    actual = (
        hybrid[["user_id", "item_id"]]
        .sort_values(["user_id", "item_id"])
        .reset_index(drop=True)
    )

    if not expected.equals(actual):
        raise ValueError(
            "Conservative reranking changed the ALS candidate set."
        )


def main() -> None:
    """Rerank at fixed alpha and evaluate once on test."""
    collaborative, semantic = load_candidates()

    print("Strategy:        conservative")
    print(f"Semantic weight: {SEMANTIC_WEIGHT:.1f}")
    print(f"ALS candidates:  {len(collaborative):,}")
    print(f"Semantic candidates: {len(semantic):,}")
    print("No test parameter tuning will be performed.")

    ranking_start = perf_counter()
    recommendations = rerank_collaborative_candidates(
        collaborative,
        semantic,
        HybridConfig(
            semantic_weight=SEMANTIC_WEIGHT,
            k=K,
        ),
    )
    ranking_seconds = perf_counter() - ranking_start

    verify_candidate_set(collaborative, recommendations)

    overlap = collaborative[
        ["user_id", "item_id"]
    ].merge(
        semantic[["user_id", "item_id"]],
        on=["user_id", "item_id"],
        how="inner",
        validate="one_to_one",
    )

    # Test ground truth is loaded only after ranking is complete.
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    validation = pd.read_parquet(
        DATA_DIR / "validation.parquet"
    )
    test = pd.read_parquet(DATA_DIR / "test.parquet")

    for frame in (train, validation, test):
        frame["user_id"] = frame["user_id"].astype(str)
        frame["item_id"] = frame["item_id"].astype(str)

    fit_events = pd.concat(
        [train, validation],
        ignore_index=True,
    )
    fit_users = pd.Index(fit_events["user_id"].unique())
    fit_items = pd.Index(fit_events["item_id"].unique())
    test_users = pd.Index(test["user_id"].unique())

    metrics = evaluate_ranking(
        recommendations=recommendations,
        ground_truth=test,
        catalog_size=len(fit_items),
        k=K,
    )

    with ALS_RESULTS_PATH.open(
        encoding="utf-8"
    ) as source:
        als_results = json.load(source)

    with SEMANTIC_RESULTS_PATH.open(
        encoding="utf-8"
    ) as source:
        semantic_results = json.load(source)

    if not isclose(
        metrics.recall_at_k,
        als_results["recall_at_k"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Hybrid Recall@10 differs from preserved ALS candidates."
        )

    if not isclose(
        metrics.catalog_coverage,
        als_results["catalog_coverage"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Hybrid coverage differs from preserved ALS candidates."
        )

    cold_user_rate = float(
        (~test_users.isin(fit_users)).mean()
    )
    unseen_item_event_rate = float(
        (~test["item_id"].isin(fit_items)).mean()
    )

    metric_names = (
        "recall_at_k",
        "ndcg_at_k",
        "mrr_at_k",
        "catalog_coverage",
    )
    source_metrics = {
        "als": {
            name: als_results[name]
            for name in metric_names
        },
        "openai_semantic": {
            name: semantic_results[name]
            for name in metric_names
        },
    }
    delta_vs_als = {
        name: getattr(metrics, name) - als_results[name]
        for name in metric_names
    }

    results = {
        "dataset": DATASET,
        "model": "normalized_als_openai_hybrid",
        "strategy": "conservative",
        "evaluation_split": "test",
        "fit_splits": ["train", "validation"],
        "selection_split": "validation",
        "selection_metric": "ndcg_at_k",
        "semantic_weight": SEMANTIC_WEIGHT,
        "test_used_for_tuning": False,
        "k": K,
        "candidate_depth_per_model": K,
        "fit_interactions": len(fit_events),
        "test_interactions": len(test),
        "fit_users": len(fit_users),
        "fit_items": len(fit_items),
        "evaluated_users": len(test_users),
        "als_candidate_rows": len(collaborative),
        "semantic_candidate_rows": len(semantic),
        "candidate_overlap_rows": len(overlap),
        "users_with_candidate_overlap": (
            overlap["user_id"].nunique()
        ),
        "recommendation_rows": len(recommendations),
        "cold_user_rate": cold_user_rate,
        "unseen_item_event_rate": unseen_item_event_rate,
        "ranking_seconds": ranking_seconds,
        "source_metrics": source_metrics,
        "delta_vs_als": delta_vs_als,
        **asdict(metrics),
    }

    HYBRID_ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    recommendations.to_parquet(
        HYBRID_ARTIFACT_DIR / "test_recommendations.parquet",
        index=False,
    )

    with RESULTS_PATH.open("w", encoding="utf-8") as output:
        json.dump(results, output, indent=2)

    print()
    print(f"Recall@{K}:     {metrics.recall_at_k:.6f}")
    print(f"NDCG@{K}:       {metrics.ndcg_at_k:.6f}")
    print(f"MRR@{K}:        {metrics.mrr_at_k:.6f}")
    print(f"Coverage:      {metrics.catalog_coverage:.6f}")
    print(
        f"NDCG delta:    {delta_vs_als['ndcg_at_k']:+.6f}"
    )
    print(
        f"MRR delta:     {delta_vs_als['mrr_at_k']:+.6f}"
    )
    print(f"Overlap rows:  {len(overlap):,}")
    print(f"Ranking time:  {ranking_seconds:.2f} seconds")
    print(f"Results:       {RESULTS_PATH}")


if __name__ == "__main__":
    main()