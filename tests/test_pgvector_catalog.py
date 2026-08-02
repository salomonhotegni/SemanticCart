from types import SimpleNamespace

import numpy as np
import pytest

from semanticcart.pgvector_catalog import (
    _validate_bundle,
    sync_serving_embeddings,
)


def make_bundle(
    *,
    version: str = "semanticcart-test",
    item_ids: list[str] | None = None,
    vectors: np.ndarray | None = None,
    dimensions: int = 512,
    semantic_items: int | None = None,
):
    """Build a minimal serving-bundle substitute for validation tests."""
    if item_ids is None:
        item_ids = ["a", "b", "c"]

    if vectors is None:
        vectors = np.zeros(
            (len(item_ids), 512),
            dtype=np.float32,
        )

        for position in range(len(item_ids)):
            vectors[position, position] = 1.0

    if semantic_items is None:
        semantic_items = len(item_ids)

    return SimpleNamespace(
        version=version,
        manifest={
            "embedding_config": {
                "model": "text-embedding-3-small",
                "dimensions": dimensions,
            },
            "semantic_items": semantic_items,
        },
        item_index=SimpleNamespace(
            item_ids=np.asarray(item_ids),
            item_vectors=vectors,
        ),
    )


def test_validates_normalized_serving_snapshot() -> None:
    bundle = make_bundle()

    (
        model_version,
        embedding_model,
        item_ids,
        vectors,
    ) = _validate_bundle(bundle)

    assert model_version == "semanticcart-test"
    assert embedding_model == "text-embedding-3-small"
    assert item_ids.tolist() == ["a", "b", "c"]
    assert vectors.shape == (3, 512)
    assert vectors.dtype == np.float32
    assert vectors.flags.c_contiguous


def test_rejects_empty_model_version() -> None:
    with pytest.raises(
        ValueError,
        match="Model version",
    ):
        _validate_bundle(
            make_bundle(version=" ")
        )


def test_rejects_missing_embedding_configuration() -> None:
    bundle = make_bundle()
    bundle.manifest["embedding_config"] = None

    with pytest.raises(
        ValueError,
        match="configuration",
    ):
        _validate_bundle(bundle)


def test_rejects_wrong_embedding_dimensions() -> None:
    with pytest.raises(
        ValueError,
        match="512 dimensions",
    ):
        _validate_bundle(
            make_bundle(dimensions=256)
        )


def test_rejects_duplicate_item_ids() -> None:
    with pytest.raises(
        ValueError,
        match="unique",
    ):
        _validate_bundle(
            make_bundle(
                item_ids=["a", "a", "b"],
            )
        )


def test_rejects_nonfinite_vectors() -> None:
    vectors = make_bundle().item_index.item_vectors.copy()
    vectors[0, 0] = np.nan

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        _validate_bundle(
            make_bundle(vectors=vectors)
        )


def test_rejects_zero_vectors() -> None:
    vectors = make_bundle().item_index.item_vectors.copy()
    vectors[0] = 0.0

    with pytest.raises(
        ValueError,
        match="zero vectors",
    ):
        _validate_bundle(
            make_bundle(vectors=vectors)
        )


def test_rejects_unnormalized_vectors() -> None:
    vectors = make_bundle().item_index.item_vectors.copy()
    vectors[0, 0] = 2.0

    with pytest.raises(
        ValueError,
        match="normalized",
    ):
        _validate_bundle(
            make_bundle(vectors=vectors)
        )


def test_rejects_manifest_item_count_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="semantic-item count",
    ):
        _validate_bundle(
            make_bundle(semantic_items=99)
        )


def test_rejects_empty_database_url_before_connecting() -> None:
    with pytest.raises(
        ValueError,
        match="database_url",
    ):
        sync_serving_embeddings(
            "",
            make_bundle(),
        )


def test_rejects_nonboolean_index_option() -> None:
    with pytest.raises(
        ValueError,
        match="rebuild_index",
    ):
        sync_serving_embeddings(
            "postgresql://unused",
            make_bundle(),
            rebuild_index=1,
        )