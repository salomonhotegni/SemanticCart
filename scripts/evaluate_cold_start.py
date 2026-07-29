"""Evaluate real cold items and simulated new-user sessions."""

import json
from dataclasses import asdict
from math import sqrt
from pathlib import Path
from time import perf_counter

import faiss
import pandas as pd

from semanticcart.cached_embeddings import (
    load_cached_catalog_embeddings,
)
from semanticcart.cold_start import (
    select_cold_item_events,
    select_recent_session_interactions,
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
    / "cold_start"
)
RESULTS_PATH = (
    Path("results")
    / f"{DATASET}_cold_start.json"
)

K = 10
SESSION_LENGTHS = (1, 3, 5)
PROGRESS_CHUNK_SIZE = 5_000
FAISS_BUILD_THREADS = 1

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


def wilson_interval(
    successes: int,
    trials: int,
    z_score: float = 1.96,
) -> tuple[float, float]:
    """Calculate a binomial Wilson confidence interval."""
    if trials <= 0:
        return 0.0, 0.0

    proportion = successes / trials
    denominator = 1.0 + z_score**2 / trials
    centre = (
        proportion + z_score**2 / (2.0 * trials)
    ) / denominator
    margin = (
        z_score
        * sqrt(
            (
                proportion * (1.0 - proportion)
                + z_score**2 / (4.0 * trials)
            )
            / trials
        )
        / denominator
    )

    return max(0.0, centre - margin), min(
        1.0,
        centre + margin,
    )


def generate_recommendations(
    model: DenseSemanticRecommender,
    user_ids: list[str],
    label: str,
) -> tuple[pd.DataFrame, float]:
    """Generate Top-K recommendations in bounded user batches."""
    print(
        f"Generating {label} recommendations for "
        f"{len(user_ids):,} users..."
    )

    start_time = perf_counter()
    frames = []

    for start in range(
        0,
        len(user_ids),
        PROGRESS_CHUNK_SIZE,
    ):
        stop = min(
            start + PROGRESS_CHUNK_SIZE,
            len(user_ids),
        )

        frames.append(
            model.recommend_for_users(
                user_ids[start:stop],
                k=K,
            )
        )

        print(f"Recommended: {stop:,}/{len(user_ids):,}")

    if frames:
        recommendations = pd.concat(
            frames,
            ignore_index=True,
        )
    else:
        recommendations = pd.DataFrame(
            columns=model.RESULT_COLUMNS
        )

    return recommendations, perf_counter() - start_time


def main() -> None:
    """Build the full-catalogue index and run cold-start evaluations."""
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

    if catalog["item_id"].duplicated().any():
        raise ValueError(
            "Full catalogue contains duplicate product IDs."
        )

    fit_events = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    fit_item_ids = set(fit_events["item_id"])
    full_item_ids = set(catalog["item_id"])
    catalogue_only_ids = full_item_ids - fit_item_ids

    print(
        f"Loading full-catalogue embeddings for "
        f"{len(catalog):,} products..."
    )

    start_time = perf_counter()
    embedded_catalog = load_cached_catalog_embeddings(
        catalog,
        CACHE_PATH,
        id_column="item_id",
        config=EMBEDDING_CONFIG,
    )
    embedding_load_seconds = perf_counter() - start_time

    print("Building deterministic full-catalogue HNSW index...")

    faiss_search_threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(FAISS_BUILD_THREADS)

    start_time = perf_counter()

    try:
        item_index = DenseItemIndex.from_catalog(
            embedded_catalog,
            config=INDEX_CONFIG,
        )
    finally:
        index_build_seconds = perf_counter() - start_time
        faiss.omp_set_num_threads(faiss_search_threads)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    item_index.save(ARTIFACT_DIR)

    unique_embedding_texts = embedded_catalog[
        "content_hash"
    ].nunique()
    del embedded_catalog

    # Real new-item evaluation.
    cold_events = select_cold_item_events(
        fit_events,
        test,
    )

    print(
        f"Building long-term profiles for "
        f"{fit_events['user_id'].nunique():,} users..."
    )

    start_time = perf_counter()
    long_term_profiles = DenseUserProfiles.build(
        item_index,
        fit_events,
        config=PROFILE_CONFIG,
    )
    long_term_profile_seconds = perf_counter() - start_time

    long_term_model = DenseSemanticRecommender(
        long_term_profiles,
        config=RETRIEVAL_CONFIG,
    )

    cold_users = cold_events[
        "user_id"
    ].drop_duplicates().tolist()

    cold_recommendations, cold_recommendation_seconds = (
        generate_recommendations(
            long_term_model,
            cold_users,
            label="cold-item",
        )
    )

    cold_metrics = evaluate_ranking(
        recommendations=cold_recommendations,
        ground_truth=cold_events,
        catalog_size=len(catalog),
        k=K,
    )

    cold_hits = cold_recommendations[
        ["user_id", "item_id"]
    ].merge(
        cold_events[["user_id", "item_id"]],
        on=["user_id", "item_id"],
        how="inner",
        validate="one_to_one",
    )

    confidence_low, confidence_high = wilson_interval(
        successes=len(cold_hits),
        trials=len(cold_events),
    )

    cold_product_recommendations = cold_recommendations.loc[
        cold_recommendations["item_id"].isin(
            catalogue_only_ids
        )
    ]

    cold_recommendations.to_parquet(
        ARTIFACT_DIR / "cold_item_recommendations.parquet",
        index=False,
    )

    cold_users_per_second = (
        len(cold_users) / cold_recommendation_seconds
        if cold_recommendation_seconds > 0
        else 0.0
    )

    cold_item_results = {
        "cohort_events": len(cold_events),
        "cohort_users": cold_events["user_id"].nunique(),
        "cohort_items": cold_events["item_id"].nunique(),
        "hit_count_at_k": len(cold_hits),
        "recall_at_k_wilson_95_low": confidence_low,
        "recall_at_k_wilson_95_high": confidence_high,
        "catalogue_only_recommendation_rows": len(
            cold_product_recommendations
        ),
        "catalogue_only_items_recommended": (
            cold_product_recommendations[
                "item_id"
            ].nunique()
        ),
        "catalogue_only_item_exposure": (
            cold_product_recommendations[
                "item_id"
            ].nunique()
            / len(catalogue_only_ids)
        ),
        "als_representable_cold_items": 0,
        "profile_build_seconds": long_term_profile_seconds,
        "recommendation_seconds": (
            cold_recommendation_seconds
        ),
        "recommendation_users_per_second": (
            cold_users_per_second
        ),
        **asdict(cold_metrics),
    }

    del long_term_model
    del long_term_profiles
    del cold_recommendations

    # Simulated anonymous new-user evaluation.
    session_results = {}

    for session_length in SESSION_LENGTHS:
        print()
        print(
            f"Building {session_length}-item session cohort..."
        )

        sessions = select_recent_session_interactions(
            fit_events,
            session_length=session_length,
        )
        session_users = sessions[
            "user_id"
        ].drop_duplicates().tolist()

        session_ground_truth = test.loc[
            test["user_id"].isin(session_users)
        ].copy()

        start_time = perf_counter()
        session_profiles = DenseUserProfiles.build(
            item_index,
            sessions,
            config=PROFILE_CONFIG,
        )
        profile_seconds = perf_counter() - start_time

        session_model = DenseSemanticRecommender(
            session_profiles,
            config=RETRIEVAL_CONFIG,
        )

        recommendations, recommendation_seconds = (
            generate_recommendations(
                session_model,
                session_users,
                label=f"{session_length}-item session",
            )
        )

        metrics = evaluate_ranking(
            recommendations=recommendations,
            ground_truth=session_ground_truth,
            catalog_size=len(catalog),
            k=K,
        )

        recommendations.to_parquet(
            ARTIFACT_DIR
            / f"session_{session_length}_recommendations.parquet",
            index=False,
        )

        users_per_second = (
            len(session_users) / recommendation_seconds
            if recommendation_seconds > 0
            else 0.0
        )

        session_results[str(session_length)] = {
            "session_length": session_length,
            "eligible_users": len(session_users),
            "session_interactions": len(sessions),
            "evaluation_events": len(
                session_ground_truth
            ),
            "profile_build_seconds": profile_seconds,
            "recommendation_seconds": (
                recommendation_seconds
            ),
            "recommendation_users_per_second": (
                users_per_second
            ),
            **asdict(metrics),
        }

        print(
            f"Session {session_length}: "
            f"Recall@{K}={metrics.recall_at_k:.6f} "
            f"NDCG@{K}={metrics.ndcg_at_k:.6f} "
            f"Coverage={metrics.catalog_coverage:.6f}"
        )

        del session_model
        del session_profiles
        del recommendations
        del sessions

    # Compare every session length on the same five-item-eligible cohort.
    common_sessions = select_recent_session_interactions(
        fit_events,
        session_length=max(SESSION_LENGTHS),
    )
    common_user_ids = set(common_sessions["user_id"])
    common_ground_truth = test.loc[
        test["user_id"].isin(common_user_ids)
    ].copy()

    common_session_results = {}

    for session_length in SESSION_LENGTHS:
        recommendations = pd.read_parquet(
            ARTIFACT_DIR
            / f"session_{session_length}_recommendations.parquet"
        )
        recommendations["user_id"] = recommendations[
            "user_id"
        ].astype(str)

        common_recommendations = recommendations.loc[
            recommendations["user_id"].isin(common_user_ids)
        ]

        metrics = evaluate_ranking(
            recommendations=common_recommendations,
            ground_truth=common_ground_truth,
            catalog_size=len(catalog),
            k=K,
        )

        common_session_results[str(session_length)] = {
            "session_length": session_length,
            "eligible_users": len(common_user_ids),
            "evaluation_events": len(common_ground_truth),
            **asdict(metrics),
        }

        del recommendations
        del common_recommendations

    results = {
        "dataset": DATASET,
        "model": "openai_hnsw_cold_start",
        "evaluation_split": "test",
        "fit_splits": ["train", "validation"],
        "test_used_for_tuning": False,
        "candidate_catalog_scope": "full_catalogue",
        "k": K,
        "embedding_config": asdict(EMBEDDING_CONFIG),
        "index_config": asdict(INDEX_CONFIG),
        "profile_config": asdict(PROFILE_CONFIG),
        "retrieval_config": asdict(RETRIEVAL_CONFIG),
        "fit_interactions": len(fit_events),
        "fit_users": fit_events["user_id"].nunique(),
        "fit_items": len(fit_item_ids),
        "catalogue_items": len(catalog),
        "catalogue_only_items": len(catalogue_only_ids),
        "unique_embedding_texts": unique_embedding_texts,
        "embedding_load_seconds": embedding_load_seconds,
        "index_build_seconds": index_build_seconds,
        "cold_item": cold_item_results,
        "simulated_new_user_sessions": session_results,
        "controlled_common_session_cohort": {
            "required_history_length": max(SESSION_LENGTHS),
            "eligible_users": len(common_user_ids),
            "results": common_session_results,
        },
        "faiss_build_threads": FAISS_BUILD_THREADS,
        "faiss_search_threads": faiss_search_threads,
    }

    with RESULTS_PATH.open("w", encoding="utf-8") as output:
        json.dump(results, output, indent=2)

    print()
    print("Cold-start evaluation complete")
    print(
        f"Cold-item Recall@{K}: "
        f"{cold_metrics.recall_at_k:.6f} "
        f"({len(cold_hits)}/{len(cold_events)} hits)"
    )
    print(
        "Cold-item Recall 95% Wilson interval: "
        f"[{confidence_low:.6f}, {confidence_high:.6f}]"
    )
    print()
    print(
        f"Controlled session cohort: "
        f"{len(common_user_ids):,} users"
    )

    for session_length in SESSION_LENGTHS:
        controlled = common_session_results[
            str(session_length)
        ]

        print(
            f"Session {session_length}: "
            f"Recall@{K}={controlled['recall_at_k']:.6f} "
            f"NDCG@{K}={controlled['ndcg_at_k']:.6f} "
            f"Coverage={controlled['catalog_coverage']:.6f}"
        )
    print(f"Results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()