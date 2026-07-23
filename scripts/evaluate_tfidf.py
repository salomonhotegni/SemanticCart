"""Train and evaluate the TF-IDF semantic recommender."""

import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import pandas as pd

from semanticcart.evaluation import evaluate_ranking
from semanticcart.tfidf import TfidfConfig, TfidfRecommender


DATASET = "video_games_5core"
DATA_DIR = Path("data/processed/amazon_video_games_5core")
ARTIFACT_DIR = Path("data/artifacts") / DATASET / "tfidf"
RESULTS_PATH = Path("results") / f"{DATASET}_tfidf.json"

K = 10
PROGRESS_CHUNK_SIZE = 5_000


def main() -> None:
    """Fit TF-IDF on training products and evaluate validation users."""
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    validation = pd.read_parquet(
        DATA_DIR / "validation.parquet"
    )
    catalog = pd.read_parquet(DATA_DIR / "catalog.parquet")

    train["user_id"] = train["user_id"].astype(str)
    train["item_id"] = train["item_id"].astype(str)
    validation["user_id"] = validation["user_id"].astype(str)
    validation["item_id"] = validation["item_id"].astype(str)
    catalog["item_id"] = catalog["item_id"].astype(str)

    train_item_ids = pd.Index(
        train["item_id"].unique(),
        name="item_id",
    )

    training_catalog = catalog.loc[
        catalog["item_id"].isin(train_item_ids)
    ].copy()

    catalog_item_ids = pd.Index(training_catalog["item_id"])
    missing_train_items = train_item_ids.difference(
        catalog_item_ids
    )

    if len(missing_train_items):
        raise ValueError(
            f"Missing metadata for {len(missing_train_items)} "
            "training products."
        )

    config = TfidfConfig(
        max_features=50_000,
        min_df=2,
        recency_decay=0.85,
        batch_size=256,
    )

    print(
        f"Training TF-IDF on "
        f"{len(training_catalog):,} products..."
    )

    training_start = perf_counter()
    model = TfidfRecommender.fit(
        catalog=training_catalog,
        interactions=train,
        config=config,
    )
    training_seconds = perf_counter() - training_start

    train_users = pd.Index(train["user_id"].unique())
    validation_users = pd.Index(
        validation["user_id"].unique()
    )

    warm_validation_users = validation_users[
        validation_users.isin(train_users)
    ].tolist()

    print(
        f"Generating recommendations for "
        f"{len(warm_validation_users):,} users..."
    )

    recommendation_start = perf_counter()
    recommendation_frames = []

    for start in range(
        0,
        len(warm_validation_users),
        PROGRESS_CHUNK_SIZE,
    ):
        stop = min(
            start + PROGRESS_CHUNK_SIZE,
            len(warm_validation_users),
        )

        recommendation_frames.append(
            model.recommend_for_users(
                warm_validation_users[start:stop],
                k=K,
            )
        )

        print(
            f"Recommended: {stop:,}/"
            f"{len(warm_validation_users):,}"
        )

    recommendations = pd.concat(
        recommendation_frames,
        ignore_index=True,
    )

    recommendation_seconds = (
        perf_counter() - recommendation_start
    )

    metrics = evaluate_ranking(
        recommendations=recommendations,
        ground_truth=validation,
        catalog_size=len(train_item_ids),
        k=K,
    )

    cold_user_rate = float(
        (~validation_users.isin(train_users)).mean()
    )

    unseen_item_event_rate = float(
        (~validation["item_id"].isin(train_item_ids)).mean()
    )

    users_per_second = (
        len(warm_validation_users) / recommendation_seconds
        if recommendation_seconds > 0
        else 0.0
    )

    model.save(ARTIFACT_DIR)

    recommendations.to_parquet(
        ARTIFACT_DIR / "validation_recommendations.parquet",
        index=False,
    )

    results = {
        "dataset": DATASET,
        "model": "tfidf_content",
        "k": K,
        "config": asdict(config),
        "train_interactions": len(train),
        "validation_interactions": len(validation),
        "train_users": train["user_id"].nunique(),
        "train_items": len(train_item_ids),
        "vocabulary_size": len(model.vectorizer.vocabulary_),
        "item_vector_nonzeros": int(model.item_vectors.nnz),
        "user_profile_nonzeros": int(model.user_profiles.nnz),
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
    print(f"Model:         TF-IDF content")
    print(f"Vocabulary:    {len(model.vectorizer.vocabulary_):,}")
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