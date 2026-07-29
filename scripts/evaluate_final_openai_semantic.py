"""Refit the dense semantic model and evaluate it on the test split."""

import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import pandas as pd

from semanticcart.cached_embeddings import (
    load_cached_catalog_embeddings,
)
from semanticcart.dense_index import DenseItemIndex, HnswConfig
from semanticcart.dense_profiles import (
    DenseProfileConfig,
    DenseUserProfiles,
)
from semanticcart.dense_semantic import (
    DenseSemanticConfig,
    DenseSemanticRecommender,
)
from semanticcart.embedding_cache import EmbeddingConfig
from semanticcart.evaluation import evaluate_ranking


DATASET = "video_games_5core"
DATA_DIR = Path("data/processed/amazon_video_games_5core")
CACHE_PATH = (
    Path("data/artifacts")
    / DATASET
    / "openai_embeddings"
    / "embedding_cache.parquet"
)
ARTIFACT_DIR = (
    Path("data/artifacts")
    / DATASET
    / "final"
    / "openai_semantic"
)
RESULTS_PATH = (
    Path("results")
    / f"{DATASET}_openai_semantic_test.json"
)

K = 10
PROGRESS_CHUNK_SIZE = 5_000

EMBEDDING_CONFIG = EmbeddingConfig(
    model="text-embedding-3-small",
    dimensions=512,
)
INDEX_CONFIG = HnswConfig(
    connections=32,
    ef_construction=200,
    ef_search=128,
)
PROFILE_CONFIG = DenseProfileConfig(
    recency_decay=0.85,
)
RETRIEVAL_CONFIG = DenseSemanticConfig(
    batch_size=512,
    candidate_multiplier=10,
)


def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load final-fit interactions, test events, and fit catalogue."""
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    validation = pd.read_parquet(
        DATA_DIR / "validation.parquet"
    )
    test = pd.read_parquet(DATA_DIR / "test.parquet")
    catalog = pd.read_parquet(DATA_DIR / "catalog.parquet")

    for frame in (train, validation, test):
        frame["user_id"] = frame["user_id"].astype(str)
        frame["item_id"] = frame["item_id"].astype(str)

    catalog["item_id"] = catalog["item_id"].astype(str)

    fit_events = pd.concat(
        [train, validation],
        ignore_index=True,
    )
    fit_item_ids = pd.Index(
        fit_events["item_id"].unique(),
        name="item_id",
    )

    fit_catalog = catalog.loc[
        catalog["item_id"].isin(fit_item_ids)
    ].copy()

    missing_items = fit_item_ids.difference(
        pd.Index(fit_catalog["item_id"])
    )

    if len(missing_items):
        raise ValueError(
            f"Missing metadata for {len(missing_items):,} "
            "final-fit products."
        )

    if len(fit_catalog) != len(fit_item_ids):
        raise ValueError(
            "Final-fit catalogue contains duplicate product IDs."
        )

    return train, validation, fit_events, test, fit_catalog


def generate_recommendations(
    model: DenseSemanticRecommender,
    fit_events: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], float]:
    """Generate test recommendations in bounded user batches."""
    fit_users = pd.Index(fit_events["user_id"].unique())
    test_users = pd.Index(test["user_id"].unique())

    warm_users = test_users[
        test_users.isin(fit_users)
    ].tolist()

    print(
        f"Generating recommendations for "
        f"{len(warm_users):,} test users..."
    )

    start_time = perf_counter()
    frames = []

    for start in range(
        0,
        len(warm_users),
        PROGRESS_CHUNK_SIZE,
    ):
        stop = min(
            start + PROGRESS_CHUNK_SIZE,
            len(warm_users),
        )

        frames.append(
            model.recommend_for_users(
                warm_users[start:stop],
                k=K,
            )
        )

        print(f"Recommended: {stop:,}/{len(warm_users):,}")

    if frames:
        recommendations = pd.concat(
            frames,
            ignore_index=True,
        )
    else:
        recommendations = pd.DataFrame(
            columns=model.RESULT_COLUMNS
        )

    elapsed_seconds = perf_counter() - start_time
    return recommendations, warm_users, elapsed_seconds


def main() -> None:
    """Build final semantic artifacts and evaluate untouched test events."""
    train, validation, fit_events, test, fit_catalog = (
        load_inputs()
    )

    print(
        f"Loading cached embeddings for "
        f"{len(fit_catalog):,} products..."
    )

    start_time = perf_counter()
    embedded_catalog = load_cached_catalog_embeddings(
        fit_catalog,
        CACHE_PATH,
        id_column="item_id",
        config=EMBEDDING_CONFIG,
    )
    embedding_load_seconds = perf_counter() - start_time

    print("Building final FAISS HNSW item index...")

    start_time = perf_counter()
    item_index = DenseItemIndex.from_catalog(
        embedded_catalog,
        config=INDEX_CONFIG,
    )
    index_build_seconds = perf_counter() - start_time

    print(
        f"Building final dense profiles for "
        f"{fit_events['user_id'].nunique():,} users..."
    )

    start_time = perf_counter()
    profiles = DenseUserProfiles.build(
        item_index,
        fit_events,
        config=PROFILE_CONFIG,
    )
    profile_build_seconds = perf_counter() - start_time

    model = DenseSemanticRecommender(
        profiles,
        config=RETRIEVAL_CONFIG,
    )

    recommendations, warm_users, recommendation_seconds = (
        generate_recommendations(
            model,
            fit_events,
            test,
        )
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

    cold_user_rate = float(
        (~test_users.isin(fit_users)).mean()
    )
    unseen_item_event_rate = float(
        (~test["item_id"].isin(fit_items)).mean()
    )
    users_per_second = (
        len(warm_users) / recommendation_seconds
        if recommendation_seconds > 0
        else 0.0
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    model.item_index.save(ARTIFACT_DIR)
    model.profiles.save(ARTIFACT_DIR)

    with (ARTIFACT_DIR / "retrieval_config.json").open(
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(
            asdict(RETRIEVAL_CONFIG),
            output,
            indent=2,
        )

    recommendations.to_parquet(
        ARTIFACT_DIR / "test_recommendations.parquet",
        index=False,
    )

    results = {
        "dataset": DATASET,
        "model": "openai_hnsw_content",
        "evaluation_split": "test",
        "fit_splits": ["train", "validation"],
        "hyperparameter_selection_split": "validation",
        "test_used_for_tuning": False,
        "k": K,
        "embedding_config": asdict(EMBEDDING_CONFIG),
        "index_config": asdict(INDEX_CONFIG),
        "profile_config": asdict(PROFILE_CONFIG),
        "retrieval_config": asdict(RETRIEVAL_CONFIG),
        "train_interactions": len(train),
        "validation_interactions": len(validation),
        "fit_interactions": len(fit_events),
        "test_interactions": len(test),
        "fit_users": fit_events["user_id"].nunique(),
        "fit_items": len(fit_items),
        "unique_embedding_texts": (
            embedded_catalog["content_hash"].nunique()
        ),
        "evaluated_users": len(test_users),
        "warm_evaluated_users": len(warm_users),
        "recommendation_rows": len(recommendations),
        "cold_user_rate": cold_user_rate,
        "unseen_item_event_rate": unseen_item_event_rate,
        "embedding_load_seconds": embedding_load_seconds,
        "index_build_seconds": index_build_seconds,
        "profile_build_seconds": profile_build_seconds,
        "recommendation_seconds": recommendation_seconds,
        "recommendation_users_per_second": users_per_second,
        "embedding_cache_bytes": CACHE_PATH.stat().st_size,
        "faiss_index_bytes": (
            ARTIFACT_DIR / "item_index.faiss"
        ).stat().st_size,
        **asdict(metrics),
    }

    with RESULTS_PATH.open("w", encoding="utf-8") as output:
        json.dump(results, output, indent=2)

    print(f"Dataset:       {DATASET}")
    print("Model:         final OpenAI semantic + FAISS HNSW")
    print("Fit through:   validation")
    print("Evaluation:    test")
    print(f"Recall@{K}:     {metrics.recall_at_k:.6f}")
    print(f"NDCG@{K}:       {metrics.ndcg_at_k:.6f}")
    print(f"MRR@{K}:        {metrics.mrr_at_k:.6f}")
    print(f"Coverage:      {metrics.catalog_coverage:.6f}")
    print(f"Cold users:    {cold_user_rate:.2%}")
    print(f"Unseen items:  {unseen_item_event_rate:.2%}")
    print(
        f"Embedding load:{embedding_load_seconds:.2f} seconds"
    )
    print(f"Index build:   {index_build_seconds:.2f} seconds")
    print(f"Profile build: {profile_build_seconds:.2f} seconds")
    print(
        f"Recommendation:{recommendation_seconds:.2f} seconds "
        f"({users_per_second:,.0f} users/second)"
    )
    print(f"Results:       {RESULTS_PATH}")


if __name__ == "__main__":
    main()