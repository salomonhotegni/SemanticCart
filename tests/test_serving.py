import json
from pathlib import Path

import pandas as pd
import pytest

from semanticcart.collaborative import (
    ALSConfig,
    ALSRecommender,
)
from semanticcart.dense_index import (
    DenseItemIndex,
    HnswConfig,
)
from semanticcart.dense_profiles import (
    DenseProfileConfig,
    DenseUserProfiles,
)
from semanticcart.serving import (
    ServingBundle,
    sha256_file,
)


@pytest.fixture
def serving_root(tmp_path: Path) -> Path:
    """Build a complete tiny serving bundle."""
    interactions = pd.DataFrame(
        {
            "user_id": [
                "u1",
                "u1",
                "u2",
                "u2",
                "u3",
                "u3",
                "u4",
                "u4",
            ],
            "item_id": [
                "a",
                "b",
                "b",
                "c",
                "c",
                "d",
                "d",
                "a",
            ],
            "timestamp": [
                1,
                2,
                1,
                2,
                1,
                2,
                1,
                2,
            ],
        }
    )

    als = ALSRecommender.fit(
        interactions,
        ALSConfig(
            factors=4,
            regularization=0.1,
            alpha=10.0,
            iterations=3,
            random_state=42,
            batch_size=2,
        ),
    )

    embedded_catalog = pd.DataFrame(
        {
            "item_id": ["a", "b", "c", "d", "e"],
            "embedding": [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
                [0.7, 0.7],
            ],
        }
    )
    item_index = DenseItemIndex.from_catalog(
        embedded_catalog,
        HnswConfig(
            connections=8,
            ef_construction=40,
            ef_search=32,
        ),
    )
    profiles = DenseUserProfiles.build(
        item_index,
        interactions,
        DenseProfileConfig(recency_decay=0.85),
    )

    root = tmp_path / "serving"
    version = "semanticcart-test"
    bundle = root / version
    bundle.mkdir(parents=True)

    als.save(bundle / "als")
    item_index.save(bundle / "dense_index")
    profiles.save(bundle / "recent_profiles")

    catalog = pd.DataFrame(
        {
            "item_id": ["a", "b", "c", "d", "e"],
            "title": ["A", "B", "C", "D", "E"],
            "main_category": ["Games"] * 5,
            "categories": [
                "Action",
                "Action",
                "Puzzle",
                "Puzzle",
                "Strategy",
            ],
            "store": ["Store"] * 5,
            "price": [10.0, 20.0, 30.0, 40.0, None],
            "image_url": [""] * 5,
            "popularity": [2, 2, 2, 2, 0],
        }
    )
    catalog.to_parquet(
        bundle / "catalog.parquet",
        index=False,
    )

    ranking_config = {
        "model": "test_hybrid",
        "k": 2,
        "candidate_k": 3,
        "session_length": 1,
        "session_weight": 0.5,
        "diversity": {
            "k": 2,
            "relevance_weight": 0.85,
            "novelty_weight": 0.0,
            "semantic_similarity_weight": 0.7,
            "category_similarity_weight": 0.2,
            "price_similarity_weight": 0.1,
        },
    }

    with (
        bundle / "ranking_config.json"
    ).open("w", encoding="utf-8") as output:
        json.dump(ranking_config, output, indent=2)

    inventory = {}

    for path in sorted(bundle.rglob("*")):
        if not path.is_file():
            continue

        relative = path.relative_to(bundle).as_posix()
        inventory[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    manifest = {
        "dataset": "synthetic",
        "model_version": version,
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        "fit_splits": ["train", "validation"],
        "users": 4,
        "als_items": 4,
        "semantic_items": 5,
        "catalog_only_items": 1,
        "embedding_config": {
            "model": "test-embedding",
            "dimensions": 2,
        },
        "ranking_config": ranking_config,
        "artifacts": inventory,
    }

    with (
        bundle / "manifest.json"
    ).open("w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)

    root.mkdir(parents=True, exist_ok=True)
    (root / "CURRENT").write_text(
        f"{version}\n",
        encoding="utf-8",
    )

    return root


def test_loads_current_bundle(
    serving_root: Path,
) -> None:
    bundle = ServingBundle.load(serving_root)

    assert bundle.version == "semanticcart-test"
    assert len(bundle.als.user_ids) == 4
    assert len(bundle.als.item_ids) == 4
    assert len(bundle.item_index.item_ids) == 5
    assert len(bundle.catalog) == 5
    assert bundle.catalog["item_id"].tolist() == (
        bundle.item_index.item_ids.tolist()
    )
    assert len(bundle.item_features) == 5

    info = bundle.model_info()

    assert info["dataset"] == "synthetic"
    assert info["catalog_only_items"] == 1
    assert info["ranking_config"]["candidate_k"] == 3


def test_loads_explicit_version(
    serving_root: Path,
) -> None:
    bundle = ServingBundle.load(
        serving_root,
        version="semanticcart-test",
        verify_checksums=False,
    )

    assert bundle.version == "semanticcart-test"


def test_rejects_corrupted_artifact(
    serving_root: Path,
) -> None:
    catalog_path = (
        serving_root
        / "semanticcart-test"
        / "catalog.parquet"
    )

    with catalog_path.open("ab") as output:
        output.write(b"corruption")

    with pytest.raises(
        ValueError,
        match="Artifact size mismatch",
    ):
        ServingBundle.load(serving_root)


def test_rejects_ranking_configuration_drift(
    serving_root: Path,
) -> None:
    ranking_path = (
        serving_root
        / "semanticcart-test"
        / "ranking_config.json"
    )
    ranking = json.loads(
        ranking_path.read_text(encoding="utf-8")
    )
    ranking["candidate_k"] = 4
    ranking_path.write_text(
        json.dumps(ranking, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="does not match manifest",
    ):
        ServingBundle.load(serving_root)


def test_rejects_manifest_path_traversal(
    serving_root: Path,
) -> None:
    manifest_path = (
        serving_root
        / "semanticcart-test"
        / "manifest.json"
    )
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    manifest["artifacts"]["../outside"] = {
        "bytes": 0,
        "sha256": "",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="escapes bundle",
    ):
        ServingBundle.load(serving_root)


@pytest.mark.parametrize(
    "version",
    ["", ".", "..", "../escape", "nested/version"],
)
def test_rejects_invalid_version_pointer(
    serving_root: Path,
    version: str,
) -> None:
    (serving_root / "CURRENT").write_text(
        version,
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="one directory name",
    ):
        ServingBundle.load(serving_root)