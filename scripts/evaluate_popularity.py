import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from semanticcart.evaluation import evaluate_ranking
from semanticcart.popularity import PopularityModel


DATASET_DIRS = {
    "all_beauty_0core": Path(
        "data/processed/amazon_all_beauty"
    ),
    "video_games_5core": Path(
        "data/processed/amazon_video_games_5core"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the popularity recommender."
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_DIRS,
        required=True,
    )
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = DATASET_DIRS[args.dataset]

    train = pd.read_parquet(data_dir / "train.parquet")
    validation = pd.read_parquet(
        data_dir / "validation.parquet"
    )

    model = PopularityModel.fit(train)
    recommendations = model.recommend_for_users(
        user_ids=validation["user_id"].unique(),
        seen_interactions=train,
        k=args.k,
    )

    metrics = evaluate_ranking(
        recommendations=recommendations,
        ground_truth=validation,
        catalog_size=train["item_id"].nunique(),
        k=args.k,
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

    model_path = (
        Path("data/artifacts")
        / args.dataset
        / "popularity_ranking.parquet"
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.ranking.to_parquet(model_path, index=False)

    results = {
        "dataset": args.dataset,
        "model": "popularity",
        "k": args.k,
        "train_interactions": len(train),
        "validation_interactions": len(validation),
        "train_users": train["user_id"].nunique(),
        "train_items": train["item_id"].nunique(),
        "cold_user_rate": cold_user_rate,
        "unseen_item_event_rate": unseen_item_event_rate,
        **asdict(metrics),
    }

    results_path = (
        Path("results")
        / f"{args.dataset}_popularity.json"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("w", encoding="utf-8") as output:
        json.dump(results, output, indent=2)

    print(f"Dataset:     {args.dataset}")
    print(f"Recall@{args.k}:   {metrics.recall_at_k:.6f}")
    print(f"NDCG@{args.k}:     {metrics.ndcg_at_k:.6f}")
    print(f"MRR@{args.k}:      {metrics.mrr_at_k:.6f}")
    print(f"Coverage:    {metrics.catalog_coverage:.6f}")
    print(f"Cold users:  {cold_user_rate:.2%}")
    print(f"Unseen items:{unseen_item_event_rate:.2%}")
    print(f"Results:     {results_path}")


if __name__ == "__main__":
    main()