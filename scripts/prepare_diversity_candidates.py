"""Prepare deeper ALS candidates with recent-session semantic scores."""

import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from threadpoolctl import threadpool_limits

import faiss
import pandas as pd

from semanticcart.cached_embeddings import (
    load_cached_catalog_embeddings,
)
from semanticcart.cold_start import (
    select_recent_session_interactions,
)
from semanticcart.collaborative import ALSRecommender
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
ARTIFACT_ROOT = Path("data/artifacts") / DATASET
OUTPUT_DIR = ARTIFACT_ROOT / "diversity"
CACHE_PATH = (
    ARTIFACT_ROOT
    / "openai_embeddings"
    / "embedding_cache.parquet"
)
MANIFEST_PATH = OUTPUT_DIR / "candidate_manifest.json"

MODEL_DIRS = {
    "validation": ARTIFACT_ROOT / "als",
    "test": ARTIFACT_ROOT / "final" / "als",
}
OUTPUT_PATHS = {
    split: OUTPUT_DIR / f"{split}_candidates.parquet"
    for split in MODEL_DIRS
}

CANDIDATE_K = 25
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


def select_model_catalog(
    catalog: pd.DataFrame,
    model: ALSRecommender,
) -> pd.DataFrame:
    """Select exactly the products represented by one ALS model."""
    model_items = pd.Index(model.item_ids)
    selected = catalog.loc[
        catalog["item_id"].isin(model_items)
    ].copy()

    missing_items = model_items.difference(
        pd.Index(selected["item_id"])
    )

    if len(missing_items):
        raise ValueError(
            f"Missing metadata for {len(missing_items):,} "
            "ALS products."
        )
    if len(selected) != len(model_items):
        raise ValueError(
            "Model catalogue contains duplicate product IDs."
        )

    return selected


def verify_candidate_pool(
    collaborative: pd.DataFrame,
    semantic: pd.DataFrame,
    expected_users: int,
) -> None:
    """Verify candidate depth, uniqueness, and semantic score coverage."""
    if collaborative.duplicated(
        ["user_id", "item_id"]
    ).any():
        raise ValueError(
            "ALS candidate pairs must be unique."
        )

    depths = collaborative.groupby("user_id").size()

    if (
        len(depths) != expected_users
        or not depths.eq(CANDIDATE_K).all()
    ):
        raise ValueError(
            f"Every user must have exactly "
            f"{CANDIDATE_K} candidates."
        )

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
            "Session scoring changed the candidate pairs."
        )


def prepare_scope(
    split: str,
    fit_events: pd.DataFrame,
    catalog: pd.DataFrame,
) -> dict:
    """Generate and score one validation or test candidate pool."""
    print()
    print(f"Preparing {split} Top-{CANDIDATE_K} candidates...")

    model_load_start = perf_counter()

    with threadpool_limits(
        limits=1,
        user_api="blas",
    ):
        model = ALSRecommender.load(
            MODEL_DIRS[split]
        )

    model_load_seconds = perf_counter() - model_load_start

    candidate_start = perf_counter()

    with threadpool_limits(
        limits=1,
        user_api="blas",
    ):
        collaborative = model.recommend_for_users(
            model.user_ids,
            k=CANDIDATE_K,
        )

    candidate_seconds = perf_counter() - candidate_start

    collaborative["user_id"] = collaborative[
        "user_id"
    ].astype(str)
    collaborative["item_id"] = collaborative[
        "item_id"
    ].astype(str)

    model_catalog = select_model_catalog(
        catalog,
        model,
    )

    embedding_start = perf_counter()
    embedded_catalog = load_cached_catalog_embeddings(
        model_catalog,
        CACHE_PATH,
        id_column="item_id",
        config=EMBEDDING_CONFIG,
    )
    embedding_seconds = perf_counter() - embedding_start

    original_threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(FAISS_BUILD_THREADS)
    index_start = perf_counter()

    try:
        item_index = DenseItemIndex.from_catalog(
            embedded_catalog,
            config=INDEX_CONFIG,
        )
    finally:
        index_seconds = perf_counter() - index_start
        faiss.omp_set_num_threads(original_threads)

    sessions = select_recent_session_interactions(
        fit_events,
        session_length=SESSION_LENGTH,
    )

    profile_start = perf_counter()
    profiles = DenseUserProfiles.build(
        item_index,
        sessions,
        config=PROFILE_CONFIG,
    )
    profile_seconds = perf_counter() - profile_start

    if set(model.user_ids) != set(profiles.user_ids):
        raise ValueError(
            "Session profiles do not cover every ALS user."
        )

    scorer = DenseSemanticRecommender(
        profiles,
        config=SCORING_CONFIG,
    )

    print(
        f"Scoring {len(collaborative):,} candidates "
        f"for {len(model.user_ids):,} users..."
    )

    scoring_start = perf_counter()
    semantic = scorer.score_candidates(
        collaborative[["user_id", "item_id"]]
    )
    scoring_seconds = perf_counter() - scoring_start

    verify_candidate_pool(
        collaborative,
        semantic,
        expected_users=len(model.user_ids),
    )

    combined = collaborative.merge(
        semantic[
            ["user_id", "item_id", "semantic_score"]
        ],
        on=["user_id", "item_id"],
        how="left",
        validate="one_to_one",
    )

    if combined["semantic_score"].isna().any():
        raise ValueError(
            "Some candidates are missing session scores."
        )

    combined.to_parquet(
        OUTPUT_PATHS[split],
        index=False,
    )

    candidate_users_per_second = (
        len(model.user_ids) / candidate_seconds
        if candidate_seconds > 0
        else 0.0
    )
    scoring_candidates_per_second = (
        len(combined) / scoring_seconds
        if scoring_seconds > 0
        else 0.0
    )

    record = {
        "split": split,
        "model_directory": MODEL_DIRS[split].as_posix(),
        "fit_interactions": len(fit_events),
        "fit_users": fit_events["user_id"].nunique(),
        "fit_items": len(model.item_ids),
        "candidate_k": CANDIDATE_K,
        "candidate_users": len(model.user_ids),
        "candidate_rows": len(combined),
        "session_length": SESSION_LENGTH,
        "session_interactions": len(sessions),
        "model_load_seconds": model_load_seconds,
        "candidate_generation_seconds": candidate_seconds,
        "candidate_users_per_second": (
            candidate_users_per_second
        ),
        "embedding_load_seconds": embedding_seconds,
        "index_build_seconds": index_seconds,
        "profile_build_seconds": profile_seconds,
        "semantic_scoring_seconds": scoring_seconds,
        "semantic_candidates_per_second": (
            scoring_candidates_per_second
        ),
        "output_path": OUTPUT_PATHS[split].as_posix(),
    }

    print(f"Model products:    {len(model.item_ids):,}")
    print(f"Candidate users:   {len(model.user_ids):,}")
    print(f"Candidate rows:    {len(combined):,}")
    print(
        f"ALS generation:    {candidate_seconds:.2f} seconds "
        f"({candidate_users_per_second:,.0f} users/second)"
    )
    print(
        f"Session scoring:   {scoring_seconds:.2f} seconds "
        f"({scoring_candidates_per_second:,.0f} "
        "candidates/second)"
    )
    print(f"Output:            {OUTPUT_PATHS[split]}")

    return record


def main() -> None:
    """Prepare deeper pools without loading test ground-truth items."""
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    validation = pd.read_parquet(
        DATA_DIR / "validation.parquet"
    )
    catalog = pd.read_parquet(
        DATA_DIR / "catalog.parquet"
    )

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
        "artifact": "diversity_candidate_pool",
        "candidate_k": CANDIDATE_K,
        "session_length": SESSION_LENGTH,
        "test_ground_truth_loaded": False,
        "embedding_config": asdict(EMBEDDING_CONFIG),
        "index_config": asdict(INDEX_CONFIG),
        "profile_config": asdict(PROFILE_CONFIG),
        "scoring_config": asdict(SCORING_CONFIG),
        "records": records,
    }

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(manifest, output, indent=2)

    print()
    print("Diversity candidate preparation complete.")
    print("Test ground truth loaded: False")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()