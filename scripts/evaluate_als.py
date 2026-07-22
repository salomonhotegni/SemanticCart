import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import pandas as pd

from semanticcart.collaborative import ALSConfig, ALSRecommender
from semanticcart.evaluation import evaluate_ranking


DATASET = "video_games_5core"
DATA_DIR = Path("data/processed/amazon_video_games_5core")
ARTIFACT_DIR = Path("data/artifacts") / DATASET / "als"
RESULTS_PATH = Path("results") / f"{DATASET}_als.json"
K = 10


def main() -> None:
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    validation = pd.read_parquet(
        DATA_DIR / "validation.parquet"
    )

    config = ALSConfig(
        factors=64,
        regularization=0.05,
        alpha=20.0,
        iterations=20,
        random_state=42,
        batch_size=1024,
    )

    training_start = perf_counter()
    model = ALSRecommender.fit(train, config=config)
    training_seconds = perf_counter() - training_start

    train_users = pd.Index(train["user_id"].unique())
    validation_users = pd.Index(
        validation["user_id"].unique()
    )
    train_items = pd.Index(train["item_id"].unique())

    warm_validation_users = validation_users[
        validation_users.isin(train_users)
    ]

    recommendation_start = perf_counter()
    recommendations = model.recommend_for_users(
        user_ids=warm_validation_users,
        k=K,
    )
    recommendation_seconds = (
        perf_counter() - recommendation_start
    )

    metrics = evaluate_ranking(
        recommendations=recommendations,
        ground_truth=validation,
        catalog_size=len(train_items),
        k=K,
    )

    cold_user_rate = float(
        (~validation_users.isin(train_users)).mean()
    )
    unseen_item_event_rate = float(
        (~validation["item_id"].isin(train_items)).mean()
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(ARTIFACT_DIR)

    recommendations.to_parquet(
        ARTIFACT_DIR / "validation_recommendations.parquet",
        index=False,
    )

    users_per_second = (
        len(warm_validation_users) / recommendation_seconds
        if recommendation_seconds > 0
        else 0.0
    )

    results = {
        "dataset": DATASET,
        "model": "implicit_als",
        "k": K,
        "config": asdict(config),
        "train_interactions": len(train),
        "validation_interactions": len(validation),
        "train_users": train["user_id"].nunique(),
        "train_items": train["item_id"].nunique(),
        "cold_user_rate": cold_user_rate,
        "unseen_item_event_rate": unseen_item_event_rate,
        "training_seconds": training_seconds,
        "recommendation_seconds": recommendation_seconds,
        "recommendation_users_per_second": users_per_second,
        **asdict(metrics),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS_PATH.open("w", encoding="utf-8") as output:
        json.dump(results, output, indent=2)

    print(f"Dataset:       {DATASET}")
    print(f"Model:         implicit ALS")
    print(f"Recall@{K}:     {metrics.recall_at_k:.6f}")
    print(f"NDCG@{K}:       {metrics.ndcg_at_k:.6f}")
    print(f"MRR@{K}:        {metrics.mrr_at_k:.6f}")
    print(f"Coverage:      {metrics.catalog_coverage:.6f}")
    print(f"Cold users:    {cold_user_rate:.2%}")
    print(f"Unseen items:  {unseen_item_event_rate:.2%}")
    print(f"Training time: {training_seconds:.2f} seconds")
    print(
        f"Recommendation:{recommendation_seconds:.2f} seconds "
        f"({users_per_second:,.0f} users/second)"
    )
    print(f"Results:       {RESULTS_PATH}")


if __name__ == "__main__":
    main()