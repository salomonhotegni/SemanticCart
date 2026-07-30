"""Prepare recent-session semantic scores for ALS candidates."""

import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import faiss
import pandas as pd

from semanticcart.cached_embeddings import (
    load_cached_catalog_embeddings,
)
from semanticcart.cold_start import (
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


DATASET = "video_games_5core"
DATA_DIR = Path("data/processed/amazon_video_games_5core")
CACHE_PATH = (
    Path("data/artifacts")
    / DATASET
    / "openai_embeddings"
    / "embedding_cache.parquet"
)
ARTIFACT_ROOT = Path("data/artifacts") / DATASET
OUTPUT_DIR = ARTIFACT_ROOT / "returning_user"

ALS_CANDIDATE_PATHS = {
    "validation": (
        ARTIFACT_ROOT
        / "als"
        / "validation_recommendations.parquet"
    ),
    "test": (
        ARTIFACT_ROOT
        / "final"
        / "als"
        / "test_recommendations.parquet"
    ),
}
OUTPUT_PATHS = {
    split: OUTPUT_DIR / f"{split}_session_scores.parquet"
    for split in ALS_CANDIDATE_PATHS
}
MANIFEST_PATH = OUTPUT_DIR / "candidate_manifest.json"

SESSION_LENGTH = 1
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
SCORING_CONFIG = DenseSemanticConfig(
    batch_size=512,
    candidate_multiplier=10,
)


def select_fit_catalog(
    catalog: pd.DataFrame,
    fit_events: pd.DataFrame,
) -> pd.DataFrame:
    """Restrict catalogue candidates to products observed during fitting."""
    fit_item_ids = pd.Index(
        fit_events["item_id"].unique()
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
            "fit products."
        )

    if len(fit_catalog) != len(fit_item_ids):
        raise ValueError(
            "Fit catalogue contains duplicate product IDs."
        )

    return fit_catalog


def verify_candidate_pairs(
    collaborative: pd.DataFrame,
    semantic: pd.DataFrame,
) -> None:
    """Confirm semantic scoring preserves every ALS candidate pair."""
    expected = (
        collaborative[["user_id", "item_id"]]
        .sort_values(["user_id", "item_id"])
        .reset_index(drop=True)
    )
    actual = (
        semantic[["user_id", "item_id"]]
        .sort_values(["user_id", "item_id"])
        .reset_index(drop=True)
    )

    if not expected.equals(actual):
        raise ValueError(
            "Session scoring changed the ALS candidate set."
        )


def prepare_scope(
    split: str,
    fit_events: pd.DataFrame,
    catalog: pd.DataFrame,
) -> dict:
    """Build one-item session profiles and score one ALS candidate set."""
    print()
    print(f"Preparing {split} recent-session scores...")

    collaborative = pd.read_parquet(
        ALS_CANDIDATE_PATHS[split]
    )
    collaborative["user_id"] = collaborative[
        "user_id"
    ].astype(str)
    collaborative["item_id"] = collaborative[
        "item_id"
    ].astype(str)

    fit_catalog = select_fit_catalog(
        catalog,
        fit_events,
    )

    start_time = perf_counter()
    embedded_catalog = load_cached_catalog_embeddings(
        fit_catalog,
        CACHE_PATH,
        id_column="item_id",
        config=EMBEDDING_CONFIG,
    )
    embedding_load_seconds = perf_counter() - start_time

    search_threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(FAISS_BUILD_THREADS)

    start_time = perf_counter()

    try:
        item_index = DenseItemIndex.from_catalog(
            embedded_catalog,
            config=INDEX_CONFIG,
        )
    finally:
        index_build_seconds = perf_counter() - start_time
        faiss.omp_set_num_threads(search_threads)

    sessions = select_recent_session_interactions(
        fit_events,
        session_length=SESSION_LENGTH,
    )

    start_time = perf_counter()
    profiles = DenseUserProfiles.build(
        item_index,
        sessions,
        config=PROFILE_CONFIG,
    )
    profile_build_seconds = perf_counter() - start_time

    model = DenseSemanticRecommender(
        profiles,
        config=SCORING_CONFIG,
    )

    print(
        f"Scoring {len(collaborative):,} ALS candidates "
        f"for {collaborative['user_id'].nunique():,} users..."
    )

    start_time = perf_counter()
    semantic = model.score_candidates(
        collaborative[["user_id", "item_id"]]
    )
    scoring_seconds = perf_counter() - start_time

    verify_candidate_pairs(collaborative, semantic)

    semantic.to_parquet(
        OUTPUT_PATHS[split],
        index=False,
    )

    candidates_per_second = (
        len(semantic) / scoring_seconds
        if scoring_seconds > 0
        else 0.0
    )

    record = {
        "split": split,
        "fit_interactions": len(fit_events),
        "fit_users": fit_events["user_id"].nunique(),
        "fit_items": len(fit_catalog),
        "session_length": SESSION_LENGTH,
        "session_interactions": len(sessions),
        "candidate_users": collaborative[
            "user_id"
        ].nunique(),
        "candidate_rows": len(collaborative),
        "semantic_score_rows": len(semantic),
        "embedding_load_seconds": embedding_load_seconds,
        "index_build_seconds": index_build_seconds,
        "profile_build_seconds": profile_build_seconds,
        "scoring_seconds": scoring_seconds,
        "scoring_candidates_per_second": (
            candidates_per_second
        ),
        "faiss_build_threads": FAISS_BUILD_THREADS,
        "faiss_search_threads": search_threads,
        "output_path": OUTPUT_PATHS[split].as_posix(),
    }

    print(f"Fit products:     {len(fit_catalog):,}")
    print(f"Session users:    {sessions['user_id'].nunique():,}")
    print(f"Scored rows:      {len(semantic):,}")
    print(f"Scoring time:     {scoring_seconds:.2f} seconds")
    print(
        f"Scoring throughput:{candidates_per_second:,.0f} "
        "candidates/second"
    )
    print(f"Output:           {OUTPUT_PATHS[split]}")

    return record


def main() -> None:
    """Prepare validation and test features without loading test labels."""
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    validation = pd.read_parquet(
        DATA_DIR / "validation.parquet"
    )
    catalog = pd.read_parquet(DATA_DIR / "catalog.parquet")

    for frame in (train, validation):
        frame["user_id"] = frame["user_id"].astype(str)
        frame["item_id"] = frame["item_id"].astype(str)

    catalog["item_id"] = catalog["item_id"].astype(str)

    if catalog["item_id"].duplicated().any():
        raise ValueError(
            "Catalogue contains duplicate product IDs."
        )

    final_fit = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = {
        "validation": prepare_scope(
            split="validation",
            fit_events=train,
            catalog=catalog,
        ),
        "test": prepare_scope(
            split="test",
            fit_events=final_fit,
            catalog=catalog,
        ),
    }

    manifest = {
        "dataset": DATASET,
        "feature": "one_item_recent_session_similarity",
        "session_length": SESSION_LENGTH,
        "test_ground_truth_loaded": False,
        "embedding_config": asdict(EMBEDDING_CONFIG),
        "index_config": asdict(INDEX_CONFIG),
        "profile_config": asdict(PROFILE_CONFIG),
        "scoring_config": asdict(SCORING_CONFIG),
        "records": records,
    }

    with MANIFEST_PATH.open("w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)

    print()
    print("Returning-user candidate preparation complete.")
    print("Test ground truth loaded: False")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()