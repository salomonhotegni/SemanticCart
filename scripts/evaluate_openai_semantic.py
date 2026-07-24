"""Build and evaluate the OpenAI dense semantic recommender."""

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
    / "openai_semantic"
)
RESULTS_PATH = (
    Path("results")
    / f"{DATASET}_openai_semantic.json"
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


def load_training_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load validation data and restrict candidates to training products."""
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    validation = pd.read_parquet(
        DATA_DIR / "validation.parquet"
    )
    catalog = pd.read_parquet(DATA_DIR / "catalog.parquet")

    for frame in (train, validation):
        frame["user_id"] = frame["user_id"].astype(str)
        frame["item_id"] = frame["item_id"].astype(str)

    catalog["item_id"] = catalog["item_id"].astype(str)

    train_item_ids = pd.Index(
        train["item_id"].unique(),
        name="item_id",
    )

    training_catalog = catalog.loc[
        catalog["item_id"].isin(train_item_ids)
    ].copy()

    available_item_ids = pd.Index(
        training_catalog["item_id"]
    )
    missing_items = train_item_ids.difference(
        available_item_ids
    )

    if len(missing_items):
        raise ValueError(
            f"Missing metadata for {len(missing_items):,} "
            "training products."
        )

    if len(training_catalog) != len(train_item_ids):
        raise ValueError(
            "Training catalogue contains duplicate product IDs."
        )

    return train, validation, training_catalog


def build_model(
    train: pd.DataFrame,
    training_catalog: pd.DataFrame,
) -> tuple[
    DenseSemanticRecommender,
    pd.DataFrame,
    dict[str, float],
]:
    """Load cached vectors and build the index and user profiles."""
    print(
        f"Loading cached embeddings for "
        f"{len(training_catalog):,} products..."
    )

    embedding_start = perf_counter()
    embedded_catalog = load_cached_catalog_embeddings(
        training_catalog,
        CACHE_PATH,
        id_column="item_id",
        config=EMBEDDING_CONFIG,
    )
    embedding_seconds = perf_counter() - embedding_start

    print("Building FAISS HNSW item index...")

    index_start = perf_counter()
    item_index = DenseItemIndex.from_catalog(
        embedded_catalog,
        config=INDEX_CONFIG,
    )
    index_seconds = perf_counter() - index_start

    print(
        f"Building dense profiles for "
        f"{train['user_id'].nunique():,} users..."
    )

    profile_start = perf_counter()
    profiles = DenseUserProfiles.build(
        item_index,
        train,
        config=PROFILE_CONFIG,
    )
    profile_seconds = perf_counter() - profile_start

    model = DenseSemanticRecommender(
        profiles,
        config=RETRIEVAL_CONFIG,
    )

    timings = {
        "embedding_load_seconds": embedding_seconds,
        "index_build_seconds": index_seconds,
        "profile_build_seconds": profile_seconds,
    }

    return model, embedded_catalog, timings


def generate_validation_recommendations(
    model: DenseSemanticRecommender,
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], float]:
    """Generate Top-K recommendations for established validation users.

    Args:
        model: Fitted dense semantic recommender.
        train: Training interactions used to identify established users.
        validation: Held-out events defining evaluation users.

    Returns:
        Recommendations, evaluated user IDs, and elapsed retrieval seconds.
    """
    train_users = pd.Index(
        train["user_id"].unique()
    )
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

    if recommendation_frames:
        recommendations = pd.concat(
            recommendation_frames,
            ignore_index=True,
        )
    else:
        recommendations = pd.DataFrame(
            columns=model.RESULT_COLUMNS
        )

    recommendation_seconds = (
        perf_counter() - recommendation_start
    )

    users_per_second = (
        len(warm_validation_users) / recommendation_seconds
        if recommendation_seconds > 0
        else 0.0
    )

    print(
        f"Retrieval throughput: "
        f"{users_per_second:,.0f} users/second"
    )

    return (
        recommendations,
        warm_validation_users,
        recommendation_seconds,
    )
    

def evaluate_and_save(
    model: DenseSemanticRecommender,
    embedded_catalog: pd.DataFrame,
    timings: dict[str, float],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    recommendations: pd.DataFrame,
    warm_users: list[str],
    recommendation_seconds: float,
) -> dict:
    """Evaluate recommendations and persist model and result artifacts."""
    train_item_ids = pd.Index(train["item_id"].unique())
    train_users = pd.Index(train["user_id"].unique())
    validation_users = pd.Index(
        validation["user_id"].unique()
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
        len(warm_users) / recommendation_seconds
        if recommendation_seconds > 0
        else 0.0
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

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
        ARTIFACT_DIR / "validation_recommendations.parquet",
        index=False,
    )

    model_artifact_names = [
        "item_index.faiss",
        "item_vectors.npy",
        "items.parquet",
        "index_config.json",
        "user_profiles.npy",
        "user_items.npz",
        "users.parquet",
        "profile_config.json",
        "retrieval_config.json",
    ]

    model_artifact_bytes = sum(
        (ARTIFACT_DIR / filename).stat().st_size
        for filename in model_artifact_names
    )

    results = {
        "dataset": DATASET,
        "model": "openai_hnsw_content",
        "retrieval": "faiss_hnsw",
        "k": K,
        "embedding_config": {
            "model": EMBEDDING_CONFIG.model,
            "dimensions": EMBEDDING_CONFIG.dimensions,
        },
        "index_config": asdict(INDEX_CONFIG),
        "profile_config": asdict(PROFILE_CONFIG),
        "retrieval_config": asdict(RETRIEVAL_CONFIG),
        "train_interactions": len(train),
        "validation_interactions": len(validation),
        "train_users": train["user_id"].nunique(),
        "train_items": len(train_item_ids),
        "unique_embedding_texts": (
            embedded_catalog["content_hash"].nunique()
        ),
        "evaluated_users": len(warm_users),
        "recommendation_rows": len(recommendations),
        "cold_user_rate": cold_user_rate,
        "unseen_item_event_rate": unseen_item_event_rate,
        **timings,
        "recommendation_seconds": recommendation_seconds,
        "recommendation_users_per_second": users_per_second,
        "embedding_cache_bytes": CACHE_PATH.stat().st_size,
        "faiss_index_bytes": (
            ARTIFACT_DIR / "item_index.faiss"
        ).stat().st_size,
        "model_artifact_bytes": model_artifact_bytes,
        **asdict(metrics),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS_PATH.open("w", encoding="utf-8") as output:
        json.dump(results, output, indent=2)

    print(f"Dataset:       {DATASET}")
    print("Model:         OpenAI semantic + FAISS HNSW")
    print(f"Recall@{K}:     {metrics.recall_at_k:.6f}")
    print(f"NDCG@{K}:       {metrics.ndcg_at_k:.6f}")
    print(f"MRR@{K}:        {metrics.mrr_at_k:.6f}")
    print(f"Coverage:      {metrics.catalog_coverage:.6f}")
    print(f"Cold users:    {cold_user_rate:.2%}")
    print(f"Unseen items:  {unseen_item_event_rate:.2%}")
    print(
        f"Embedding load:{timings['embedding_load_seconds']:.2f} "
        "seconds"
    )
    print(
        f"Index build:   {timings['index_build_seconds']:.2f} "
        "seconds"
    )
    print(
        f"Profile build: {timings['profile_build_seconds']:.2f} "
        "seconds"
    )
    print(
        f"Recommendation:{recommendation_seconds:.2f} seconds "
        f"({users_per_second:,.0f} users/second)"
    )
    print(
        f"FAISS index:   "
        f"{results['faiss_index_bytes'] / 2**20:.2f} MiB"
    )
    print(
        f"Model artifacts:"
        f"{model_artifact_bytes / 2**20:.2f} MiB"
    )
    print(f"Results:       {RESULTS_PATH}")

    return results


def main() -> None:
    """Build, evaluate, persist, and report the semantic recommender."""
    train, validation, training_catalog = (
        load_training_inputs()
    )

    model, embedded_catalog, timings = build_model(
        train,
        training_catalog,
    )

    recommendations, warm_users, recommendation_seconds = (
        generate_validation_recommendations(
            model,
            train,
            validation,
        )
    )

    evaluate_and_save(
        model=model,
        embedded_catalog=embedded_catalog,
        timings=timings,
        train=train,
        validation=validation,
        recommendations=recommendations,
        warm_users=warm_users,
        recommendation_seconds=recommendation_seconds,
    )


if __name__ == "__main__":
    main()