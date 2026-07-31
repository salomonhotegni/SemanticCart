"""Build a versioned, load-validated SemanticCart serving bundle."""

import hashlib
import json
import platform
import shutil
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import faiss
import pandas as pd
from threadpoolctl import threadpool_limits

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
from semanticcart.embedding_cache import EmbeddingConfig


DATASET = "video_games_5core"
DATA_DIR = Path("data/processed/amazon_video_games_5core")
ARTIFACT_ROOT = Path("data/artifacts") / DATASET
SOURCE_ALS_DIR = ARTIFACT_ROOT / "final" / "als"
CACHE_PATH = (
    ARTIFACT_ROOT
    / "openai_embeddings"
    / "embedding_cache.parquet"
)
SERVING_ROOT = ARTIFACT_ROOT / "serving"
CURRENT_PATH = SERVING_ROOT / "CURRENT"

RESULT_PATH = (
    Path("results")
    / f"{DATASET}_diversity_test.json"
)

ALS_FILES = (
    "als_model.npz",
    "user_items.npz",
    "users.parquet",
    "items.parquet",
    "config.json",
)

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
SESSION_LENGTH = 1
FAISS_BUILD_THREADS = 1


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()

    with path.open("rb") as source:
        for block in iter(
            lambda: source.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def required_source_hashes() -> dict[str, str]:
    """Hash every source artifact that defines the serving version."""
    sources = {
        f"als/{name}": SOURCE_ALS_DIR / name
        for name in ALS_FILES
    }
    sources.update(
        {
            "catalog": DATA_DIR / "catalog.parquet",
            "train": DATA_DIR / "train.parquet",
            "validation": DATA_DIR / "validation.parquet",
            "embedding_cache": CACHE_PATH,
            "frozen_result": RESULT_PATH,
        }
    )

    missing = [
        path
        for path in sources.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            f"Missing serving sources: {missing}"
        )

    return {
        name: sha256_file(path)
        for name, path in sources.items()
    }


def build_model_version(
    source_hashes: dict[str, str],
    ranking_config: dict,
) -> str:
    """Derive a deterministic version from sources and configuration."""
    payload = {
        "dataset": DATASET,
        "source_hashes": source_hashes,
        "embedding_config": asdict(EMBEDDING_CONFIG),
        "index_config": asdict(INDEX_CONFIG),
        "profile_config": asdict(PROFILE_CONFIG),
        "session_length": SESSION_LENGTH,
        "ranking_config": ranking_config,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    digest = hashlib.sha256(encoded).hexdigest()[:16]
    return f"semanticcart-{digest}"


def copy_als_artifacts(destination: Path) -> None:
    """Copy the complete fitted ALS model into the bundle."""
    destination.mkdir(parents=True, exist_ok=True)

    for name in ALS_FILES:
        shutil.copy2(
            SOURCE_ALS_DIR / name,
            destination / name,
        )


def artifact_inventory(directory: Path) -> dict[str, dict]:
    """Record bundle file sizes and SHA-256 checksums."""
    inventory = {}

    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "manifest.json":
            continue

        relative_path = path.relative_to(
            directory
        ).as_posix()

        inventory[relative_path] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    return inventory


def main() -> None:
    """Build all immutable artifacts and publish one current version."""
    frozen_result = json.loads(
        RESULT_PATH.read_text(encoding="utf-8")
    )

    ranking_config = {
        "model": frozen_result["model"],
        "k": frozen_result["k"],
        "candidate_k": frozen_result["candidate_k"],
        "session_length": SESSION_LENGTH,
        "session_weight": (
            frozen_result["selected_session_weight"]
        ),
        "diversity": (
            frozen_result["selected_diversity_config"]
        ),
    }

    source_hashes = required_source_hashes()
    model_version = build_model_version(
        source_hashes,
        ranking_config,
    )

    SERVING_ROOT.mkdir(parents=True, exist_ok=True)
    bundle_directory = SERVING_ROOT / model_version

    if bundle_directory.exists():
        raise FileExistsError(
            f"Serving bundle already exists: {bundle_directory}"
        )

    print(f"Model version: {model_version}")
    print("Loading final train-plus-validation data...")

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

    fit_events = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    with threadpool_limits(
        limits=1,
        user_api="blas",
    ):
        als = ALSRecommender.load(SOURCE_ALS_DIR)

    if set(als.item_ids) - set(catalog["item_id"]):
        raise ValueError(
            "ALS contains products missing from the catalogue."
        )

    print(
        f"Loading embeddings for "
        f"{len(catalog):,} catalogue products..."
    )

    embedding_start = perf_counter()
    embedded_catalog = load_cached_catalog_embeddings(
        catalog,
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

    if set(als.user_ids) != set(profiles.user_ids):
        raise ValueError(
            "ALS users and dense session-profile users differ."
        )

    popularity = fit_events.groupby(
        "item_id"
    ).size()

    serving_catalog = catalog[
        [
            "item_id",
            "title",
            "main_category",
            "categories",
            "store",
            "price",
            "image_url",
        ]
    ].copy()
    serving_catalog["popularity"] = (
        serving_catalog["item_id"]
        .map(popularity)
        .fillna(0)
        .astype("int64")
    )

    with tempfile.TemporaryDirectory(
        dir=SERVING_ROOT,
        prefix=".build-",
    ) as temporary_directory:
        staging = Path(temporary_directory)

        copy_als_artifacts(staging / "als")
        item_index.save(staging / "dense_index")
        profiles.save(staging / "recent_profiles")
        serving_catalog.to_parquet(
            staging / "catalog.parquet",
            index=False,
        )

        with (
            staging / "ranking_config.json"
        ).open("w", encoding="utf-8") as output:
            json.dump(
                ranking_config,
                output,
                indent=2,
            )

        print("Validating staged serving artifacts...")

        with threadpool_limits(
            limits=1,
            user_api="blas",
        ):
            loaded_als = ALSRecommender.load(
                staging / "als"
            )

        loaded_index = DenseItemIndex.load(
            staging / "dense_index"
        )
        loaded_profiles = DenseUserProfiles.load(
            loaded_index,
            staging / "recent_profiles",
        )

        if len(loaded_als.item_ids) != 25_600:
            raise ValueError(
                "Unexpected serving ALS item count."
            )
        if len(loaded_index.item_ids) != 25_612:
            raise ValueError(
                "Unexpected semantic catalogue item count."
            )
        if len(loaded_profiles.user_ids) != 94_762:
            raise ValueError(
                "Unexpected serving profile user count."
            )

        inventory = artifact_inventory(staging)

        manifest = {
            "dataset": DATASET,
            "model_version": model_version,
            "created_at_utc": datetime.now(
                UTC
            ).isoformat(),
            "python_version": platform.python_version(),
            "faiss_version": faiss.__version__,
            "fit_splits": ["train", "validation"],
            "fit_interactions": len(fit_events),
            "users": len(loaded_als.user_ids),
            "als_items": len(loaded_als.item_ids),
            "semantic_items": len(loaded_index.item_ids),
            "catalog_only_items": (
                len(loaded_index.item_ids)
                - len(loaded_als.item_ids)
            ),
            "embedding_config": asdict(
                EMBEDDING_CONFIG
            ),
            "index_config": asdict(INDEX_CONFIG),
            "profile_config": asdict(PROFILE_CONFIG),
            "ranking_config": ranking_config,
            "build_seconds": {
                "embedding_load": embedding_seconds,
                "index_build": index_seconds,
                "profile_build": profile_seconds,
            },
            "source_hashes": source_hashes,
            "artifacts": inventory,
        }

        with (
            staging / "manifest.json"
        ).open("w", encoding="utf-8") as output:
            json.dump(manifest, output, indent=2)

        staging.rename(bundle_directory)

    CURRENT_PATH.write_text(
        f"{model_version}\n",
        encoding="utf-8",
    )

    total_bytes = sum(
        record["bytes"]
        for record in inventory.values()
    )

    print()
    print("Serving bundle complete")
    print(f"Version:          {model_version}")
    print(f"Users:            {len(als.user_ids):,}")
    print(f"ALS products:     {len(als.item_ids):,}")
    print(f"Semantic products:{len(item_index.item_ids):,}")
    print(
        f"Catalogue-only:   "
        f"{len(item_index.item_ids) - len(als.item_ids):,}"
    )
    print(
        f"Bundle size:      "
        f"{total_bytes / (1024 ** 2):.2f} MiB"
    )
    print(f"Directory:        {bundle_directory}")
    print(f"Current pointer:  {CURRENT_PATH}")


if __name__ == "__main__":
    main()