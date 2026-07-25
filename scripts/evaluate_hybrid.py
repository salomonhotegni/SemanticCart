"""Tune normalized ALS and OpenAI semantic score blending."""

import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from math import isclose

import pandas as pd

from semanticcart.evaluation import evaluate_ranking
from semanticcart.hybrid import (
    HybridConfig,
    rank_hybrid_candidates,
)


DATASET = "video_games_5core"
DATA_DIR = Path("data/processed/amazon_video_games_5core")
ARTIFACT_ROOT = Path("data/artifacts") / DATASET

ALS_CANDIDATES_PATH = (
    ARTIFACT_ROOT
    / "als"
    / "validation_recommendations.parquet"
)
SEMANTIC_CANDIDATES_PATH = (
    ARTIFACT_ROOT
    / "openai_semantic"
    / "validation_recommendations.parquet"
)
HYBRID_ARTIFACT_DIR = ARTIFACT_ROOT / "hybrid"

ALS_RESULTS_PATH = Path("results") / f"{DATASET}_als.json"
SEMANTIC_RESULTS_PATH = (
    Path("results")
    / f"{DATASET}_openai_semantic.json"
)
TUNING_RESULTS_PATH = (
    Path("results")
    / f"{DATASET}_hybrid_tuning.json"
)
BEST_RESULTS_PATH = (
    Path("results")
    / f"{DATASET}_hybrid.json"
)

K = 10
ALPHA_GRID = tuple(step / 10 for step in range(11))


def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    int,
    dict,
    dict,
]:
    """Load source candidates, held-out events, and source metrics."""
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    validation = pd.read_parquet(
        DATA_DIR / "validation.parquet"
    )
    collaborative = pd.read_parquet(
        ALS_CANDIDATES_PATH
    )
    semantic = pd.read_parquet(
        SEMANTIC_CANDIDATES_PATH
    )

    for frame in (
        train,
        validation,
        collaborative,
        semantic,
    ):
        frame["user_id"] = frame["user_id"].astype(str)
        frame["item_id"] = frame["item_id"].astype(str)

    catalog_size = train["item_id"].nunique()

    with ALS_RESULTS_PATH.open(encoding="utf-8") as source:
        als_results = json.load(source)

    with SEMANTIC_RESULTS_PATH.open(
        encoding="utf-8"
    ) as source:
        semantic_results = json.load(source)

    return (
        train,
        validation,
        collaborative,
        semantic,
        catalog_size,
        als_results,
        semantic_results,
    )


def evaluate_alpha(
    collaborative: pd.DataFrame,
    semantic: pd.DataFrame,
    validation: pd.DataFrame,
    catalog_size: int,
    alpha: float,
) -> tuple[pd.DataFrame, dict]:
    """Rank and evaluate one semantic blending weight."""
    start = perf_counter()

    recommendations = rank_hybrid_candidates(
        collaborative,
        semantic,
        HybridConfig(
            semantic_weight=alpha,
            k=K,
        ),
    )

    ranking_seconds = perf_counter() - start

    metrics = evaluate_ranking(
        recommendations=recommendations,
        ground_truth=validation,
        catalog_size=catalog_size,
        k=K,
    )

    record = {
        "semantic_weight": alpha,
        "ranking_seconds": ranking_seconds,
        "recommendation_rows": len(recommendations),
        **asdict(metrics),
    }

    print(
        f"alpha={alpha:.1f} "
        f"Recall@{K}={metrics.recall_at_k:.6f} "
        f"NDCG@{K}={metrics.ndcg_at_k:.6f} "
        f"MRR@{K}={metrics.mrr_at_k:.6f} "
        f"Coverage={metrics.catalog_coverage:.6f} "
        f"time={ranking_seconds:.2f}s"
    )

    return recommendations, record


def verify_endpoint(
    record: dict,
    source_results: dict,
    source_name: str,
) -> None:
    """Confirm an endpoint reproduces its source baseline metrics."""
    metric_names = [
        "recall_at_k",
        "ndcg_at_k",
        "mrr_at_k",
        "catalog_coverage",
    ]

    for metric_name in metric_names:
        if not isclose(
            record[metric_name],
            source_results[metric_name],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{source_name} endpoint mismatch for "
                f"{metric_name}: {record[metric_name]} versus "
                f"{source_results[metric_name]}"
            )


def main() -> None:
    """Sweep alpha, select validation NDCG, and save hybrid results."""
    (
        train,
        validation,
        collaborative,
        semantic,
        catalog_size,
        als_results,
        semantic_results,
    ) = load_inputs()

    if collaborative["rank"].max() != K:
        raise ValueError(
            f"Expected ALS candidate depth {K}."
        )
    if semantic["rank"].max() != K:
        raise ValueError(
            f"Expected semantic candidate depth {K}."
        )

    print(f"ALS candidates:      {len(collaborative):,}")
    print(f"Semantic candidates: {len(semantic):,}")
    print(f"Alpha values:        {len(ALPHA_GRID)}")
    print()

    tuning_records = []

    for alpha in ALPHA_GRID:
        recommendations, record = evaluate_alpha(
            collaborative=collaborative,
            semantic=semantic,
            validation=validation,
            catalog_size=catalog_size,
            alpha=alpha,
        )

        tuning_records.append(record)
        del recommendations

    als_endpoint = next(
        record
        for record in tuning_records
        if record["semantic_weight"] == 0.0
    )
    semantic_endpoint = next(
        record
        for record in tuning_records
        if record["semantic_weight"] == 1.0
    )

    verify_endpoint(
        als_endpoint,
        als_results,
        "ALS",
    )
    verify_endpoint(
        semantic_endpoint,
        semantic_results,
        "OpenAI semantic",
    )

    best_record = max(
        tuning_records,
        key=lambda record: (
            record["ndcg_at_k"],
            record["recall_at_k"],
            record["mrr_at_k"],
            record["catalog_coverage"],
        ),
    )
    best_alpha = best_record["semantic_weight"]

    print()
    print(f"Selected alpha: {best_alpha:.1f}")
    print("Regenerating best recommendations...")

    best_recommendations, final_record = evaluate_alpha(
        collaborative=collaborative,
        semantic=semantic,
        validation=validation,
        catalog_size=catalog_size,
        alpha=best_alpha,
    )

    train_users = pd.Index(train["user_id"].unique())
    validation_users = pd.Index(
        validation["user_id"].unique()
    )
    train_items = pd.Index(train["item_id"].unique())

    cold_user_rate = float(
        (~validation_users.isin(train_users)).mean()
    )
    unseen_item_event_rate = float(
        (~validation["item_id"].isin(train_items)).mean()
    )

    source_metric_names = [
        "recall_at_k",
        "ndcg_at_k",
        "mrr_at_k",
        "catalog_coverage",
    ]

    source_metrics = {
        "als": {
            name: als_results[name]
            for name in source_metric_names
        },
        "openai_semantic": {
            name: semantic_results[name]
            for name in source_metric_names
        },
    }

    tuning_results = {
        "dataset": DATASET,
        "model": "normalized_als_openai_hybrid",
        "candidate_depth_per_model": K,
        "selection_split": "validation",
        "selection_metric": "ndcg_at_k",
        "alpha_grid": list(ALPHA_GRID),
        "source_metrics": source_metrics,
        "tuning_seconds": sum(
            record["ranking_seconds"]
            for record in tuning_records
        ),
        "trials": tuning_records,
    }

    best_results = {
        "dataset": DATASET,
        "model": "normalized_als_openai_hybrid",
        "k": K,
        "semantic_weight": best_alpha,
        "candidate_depth_per_model": K,
        "selection_split": "validation",
        "selection_metric": "ndcg_at_k",
        "train_interactions": len(train),
        "validation_interactions": len(validation),
        "train_users": train["user_id"].nunique(),
        "train_items": catalog_size,
        "als_candidate_rows": len(collaborative),
        "semantic_candidate_rows": len(semantic),
        "cold_user_rate": cold_user_rate,
        "unseen_item_event_rate": unseen_item_event_rate,
        "source_metrics": source_metrics,
        **final_record,
    }

    HYBRID_ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    BEST_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_recommendations.to_parquet(
        HYBRID_ARTIFACT_DIR
        / "validation_recommendations.parquet",
        index=False,
    )

    with TUNING_RESULTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(tuning_results, output, indent=2)

    with BEST_RESULTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(best_results, output, indent=2)

    print()
    print(f"Best alpha:    {best_alpha:.1f}")
    print(
        f"Recall@{K}:     "
        f"{final_record['recall_at_k']:.6f}"
    )
    print(
        f"NDCG@{K}:       "
        f"{final_record['ndcg_at_k']:.6f}"
    )
    print(
        f"MRR@{K}:        "
        f"{final_record['mrr_at_k']:.6f}"
    )
    print(
        f"Coverage:      "
        f"{final_record['catalog_coverage']:.6f}"
    )
    print(f"Tuning:        {TUNING_RESULTS_PATH}")
    print(f"Best result:   {BEST_RESULTS_PATH}")


if __name__ == "__main__":
    main()