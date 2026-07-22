"""
Evaluation script for the popularity-based recommendation model.
This script evaluates the performance of the popularity-based recommendation model
on a given dataset and computes various ranking metrics such as Recall@K, NDCG@ K, MRR@K, and catalog coverage.
"""

from pathlib import Path

import pandas as pd

from semanticcart.evaluation import evaluate_ranking
from semanticcart.popularity import PopularityModel


DATA_DIR = Path("data/processed/amazon_all_beauty")
MODEL_PATH = Path("data/artifacts/popularity_ranking.parquet")
K = 10


def main() -> None:
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    validation = pd.read_parquet(DATA_DIR / "validation.parquet")

    model = PopularityModel.fit(train)
    recommendations = model.recommend_for_users(
        user_ids=validation["user_id"].unique(),
        seen_interactions=train,
        k=K,
    )

    metrics = evaluate_ranking(
        recommendations=recommendations,
        ground_truth=validation,
        catalog_size=train["item_id"].nunique(),
        k=K,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.ranking.to_parquet(MODEL_PATH, index=False)

    train_users = pd.Index(train["user_id"].unique())
    validation_users = pd.Index(validation["user_id"].unique())
    train_items = pd.Index(train["item_id"].unique())

    cold_user_rate = (~validation_users.isin(train_users)).mean()
    cold_item_event_rate = (
        ~validation["item_id"].isin(train_items)
    ).mean()

    print(f"Recall@{K}:  {metrics.recall_at_k:.6f}")
    print(f"NDCG@{K}:    {metrics.ndcg_at_k:.6f}")
    print(f"MRR@{K}:     {metrics.mrr_at_k:.6f}")
    print(f"Coverage:    {metrics.catalog_coverage:.6f}")
    print(f"Cold validation users: {cold_user_rate:.2%}")
    print(f"Events with unseen items: {cold_item_event_rate:.2%}")


if __name__ == "__main__":
    main()