import numpy as np
import pandas as pd
import pytest

from semanticcart.cached_embeddings import (
    load_cached_catalog_embeddings,
)
from semanticcart.embedding_cache import (
    EmbeddingConfig,
    content_hash,
)


CONFIG = EmbeddingConfig(
    model="text-embedding-3-small",
    dimensions=2,
)


def cache_row(
    text,
    embedding,
    model=CONFIG.model,
    dimensions=CONFIG.dimensions,
):
    """Build one versioned cache record."""
    return {
        "content_hash": content_hash(text),
        "model": model,
        "dimensions": dimensions,
        "embedding": embedding,
    }


def write_cache(tmp_path, rows):
    """Write cache rows to a temporary Parquet file."""
    path = tmp_path / "embedding_cache.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_loads_compatible_embeddings_in_catalog_order(tmp_path):
    catalog = pd.DataFrame(
        {
            "item_id": ["p2", "p1"],
            "catalog_text": ["second product", "first product"],
        }
    )

    cache_path = write_cache(
        tmp_path,
        [
            cache_row("first product", [1.0, 0.0]),
            cache_row("second product", [0.0, 1.0]),
            cache_row(
                "first product",
                [0.5, 0.5],
                model="different-model",
            ),
        ],
    )

    embedded = load_cached_catalog_embeddings(
        catalog,
        cache_path,
        config=CONFIG,
    )

    assert embedded["item_id"].tolist() == ["p2", "p1"]
    assert embedded["content_hash"].tolist() == [
        content_hash("second product"),
        content_hash("first product"),
    ]

    vectors = np.vstack(embedded["embedding"])

    assert vectors.dtype == np.float32
    assert np.allclose(vectors, [[0.0, 1.0], [1.0, 0.0]])


def test_keeps_latest_duplicate_cache_entry(tmp_path):
    catalog = pd.DataFrame(
        {
            "item_id": ["p1"],
            "catalog_text": ["first product"],
        }
    )

    cache_path = write_cache(
        tmp_path,
        [
            cache_row("first product", [1.0, 0.0]),
            cache_row("first product", [0.5, 0.5]),
        ],
    )

    embedded = load_cached_catalog_embeddings(
        catalog,
        cache_path,
        config=CONFIG,
    )

    assert np.allclose(
        embedded.iloc[0]["embedding"],
        [0.5, 0.5],
    )


def test_rejects_incomplete_cache_coverage(tmp_path):
    catalog = pd.DataFrame(
        {
            "item_id": ["p1", "p2"],
            "catalog_text": ["first product", "second product"],
        }
    )

    cache_path = write_cache(
        tmp_path,
        [cache_row("first product", [1.0, 0.0])],
    )

    with pytest.raises(
        ValueError,
        match="Missing compatible cached embeddings for 1 products",
    ):
        load_cached_catalog_embeddings(
            catalog,
            cache_path,
            config=CONFIG,
        )


@pytest.mark.parametrize(
    ("embedding", "message"),
    [
        ([1.0], "shape"),
        ([np.nan, 1.0], "non-finite"),
        ([0.0, 0.0], "zero vectors"),
    ],
)
def test_rejects_invalid_cached_vectors(
    tmp_path,
    embedding,
    message,
):
    catalog = pd.DataFrame(
        {
            "item_id": ["p1"],
            "catalog_text": ["first product"],
        }
    )

    cache_path = write_cache(
        tmp_path,
        [cache_row("first product", embedding)],
    )

    with pytest.raises(ValueError, match=message):
        load_cached_catalog_embeddings(
            catalog,
            cache_path,
            config=CONFIG,
        )


def test_rejects_invalid_cache_schema(tmp_path):
    catalog = pd.DataFrame(
        {
            "item_id": ["p1"],
            "catalog_text": ["first product"],
        }
    )

    cache_path = tmp_path / "embedding_cache.parquet"
    pd.DataFrame(
        {
            "content_hash": [content_hash("first product")],
            "model": [CONFIG.model],
            "dimensions": [CONFIG.dimensions],
        }
    ).to_parquet(cache_path, index=False)

    with pytest.raises(ValueError, match="Missing cache columns"):
        load_cached_catalog_embeddings(
            catalog,
            cache_path,
            config=CONFIG,
        )


def test_rejects_duplicate_product_ids(tmp_path):
    catalog = pd.DataFrame(
        {
            "item_id": ["p1", "p1"],
            "catalog_text": ["first product", "first product"],
        }
    )

    cache_path = write_cache(
        tmp_path,
        [cache_row("first product", [1.0, 0.0])],
    )

    with pytest.raises(ValueError, match="must be unique"):
        load_cached_catalog_embeddings(
            catalog,
            cache_path,
            config=CONFIG,
        )