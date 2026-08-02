import os
from pathlib import Path

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from semanticcart.serving import ServingBundle

import numpy as np
import pytest

from semanticcart.pgvector_index import (
    PgvectorItemIndex,
    PgvectorSearchConfig,
)


DATABASE_URL = "postgresql://unused"
LIVE_DATABASE_URL = os.getenv(
    "SEMANTICCART_TEST_DATABASE_URL"
)
SERVING_ROOT = Path(
    "data/artifacts/video_games_5core/serving"
)
LIVE_TEST_AVAILABLE = (
    LIVE_DATABASE_URL is not None
    and (SERVING_ROOT / "CURRENT").exists()
)


def make_index(
    *,
    database_url: str = DATABASE_URL,
    model_version: str = "semanticcart-test",
    item_ids: list[str] | None = None,
    dimensions: int = 512,
    config: PgvectorSearchConfig | None = None,
) -> PgvectorItemIndex:
    """Construct an unopened adapter for validation tests."""
    return PgvectorItemIndex(
        database_url=database_url,
        model_version=model_version,
        item_ids=np.asarray(
            (
                item_ids
                if item_ids is not None
                else ["a", "b", "c"]
            )
        ),
        dimensions=dimensions,
        config=config,
    )


def test_default_search_configuration() -> None:
    assert PgvectorSearchConfig() == (
        PgvectorSearchConfig(
            ef_search=128,
            min_pool_size=1,
            max_pool_size=8,
            timeout_seconds=10.0,
        )
    )


def test_constructs_item_position_mapping() -> None:
    index = make_index()

    assert index.item_to_index == {
        "a": 0,
        "b": 1,
        "c": 2,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("database_url", " "),
        ("model_version", " "),
    ],
)
def test_rejects_empty_connection_metadata(
    field: str,
    value: str,
) -> None:
    arguments = {field: value}

    with pytest.raises(ValueError):
        make_index(**arguments)


@pytest.mark.parametrize(
    "dimensions",
    [0, -1, True, 512.0],
)
def test_rejects_invalid_dimensions(
    dimensions,
) -> None:
    with pytest.raises(
        ValueError,
        match="dimensions",
    ):
        make_index(dimensions=dimensions)


@pytest.mark.parametrize(
    "item_ids",
    [
        [],
        ["a", " ", "c"],
        ["a", "a", "c"],
    ],
)
def test_rejects_invalid_item_ids(
    item_ids: list[str],
) -> None:
    with pytest.raises(ValueError):
        make_index(item_ids=item_ids)


@pytest.mark.parametrize(
    "config",
    [
        PgvectorSearchConfig(ef_search=0),
        PgvectorSearchConfig(min_pool_size=0),
        PgvectorSearchConfig(
            min_pool_size=2,
            max_pool_size=1,
        ),
        PgvectorSearchConfig(timeout_seconds=0),
    ],
)
def test_rejects_invalid_configuration(
    config: PgvectorSearchConfig,
) -> None:
    with pytest.raises(ValueError):
        make_index(config=config)


def test_normalizes_single_query_vector() -> None:
    index = make_index(dimensions=3)

    query = index._normalize_queries(
        np.asarray([3.0, 4.0, 0.0])
    )

    assert query.shape == (1, 3)
    assert query.dtype == np.float32
    assert np.allclose(
        np.linalg.norm(query, axis=1),
        1.0,
    )


def test_normalizes_query_matrix() -> None:
    index = make_index(dimensions=3)

    queries = index._normalize_queries(
        np.asarray(
            [
                [3.0, 4.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        )
    )

    assert queries.shape == (2, 3)
    assert np.allclose(
        np.linalg.norm(queries, axis=1),
        1.0,
    )


def test_rejects_wrong_query_dimensions() -> None:
    index = make_index(dimensions=3)

    with pytest.raises(
        ValueError,
        match="dimensions",
    ):
        index._normalize_queries(
            np.ones(4)
        )


def test_rejects_nonfinite_query() -> None:
    index = make_index(dimensions=3)

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        index._normalize_queries(
            np.asarray([1.0, np.nan, 0.0])
        )


def test_rejects_zero_query() -> None:
    index = make_index(dimensions=3)

    with pytest.raises(
        ValueError,
        match="zero",
    ):
        index._normalize_queries(
            np.zeros(3)
        )


def test_rejects_search_before_open() -> None:
    index = make_index()

    with pytest.raises(
        RuntimeError,
        match="not open",
    ):
        index.search(
            np.ones(512),
            k=10,
        )


def test_close_before_open_is_idempotent() -> None:
    index = make_index()

    index.close()
    index.close()


@pytest.mark.skipif(
    not LIVE_TEST_AVAILABLE,
    reason=(
        "Loaded serving artifacts and PostgreSQL "
        "test URL are required."
    ),
)
def test_live_pgvector_search_matches_faiss_and_uses_hnsw() -> None:
    """Verify live retrieval quality, mapping, and query planning."""
    assert LIVE_DATABASE_URL is not None

    bundle = ServingBundle.load(
        SERVING_ROOT
    )
    query = bundle.profiles.user_profiles[0]

    _, faiss_indices = bundle.item_index.search(
        query,
        k=10,
    )

    with PgvectorItemIndex(
        database_url=LIVE_DATABASE_URL,
        model_version=bundle.version,
        item_ids=bundle.item_index.item_ids,
        config=PgvectorSearchConfig(
            ef_search=128
        ),
    ) as index:
        scores, pgvector_indices = index.search(
            query,
            k=10,
        )

    assert scores.shape == (1, 10)
    assert pgvector_indices.shape == (1, 10)
    assert np.isfinite(scores).all()
    assert np.all(
        np.diff(scores[0]) <= 1e-6
    )
    assert len(
        np.unique(pgvector_indices[0])
    ) == 10

    overlap = len(
        set(faiss_indices[0])
        & set(pgvector_indices[0])
    )

    assert overlap >= 9

    with psycopg.connect(
        LIVE_DATABASE_URL
    ) as connection:
        register_vector(connection)

        plan_rows = connection.execute(
            """
            EXPLAIN (FORMAT TEXT)
            SELECT item_id
            FROM catalog_embeddings
            WHERE model_version = %s
            ORDER BY embedding <=> %s
            LIMIT 10
            """,
            (
                bundle.version,
                Vector(query),
            ),
        ).fetchall()

    plan = "\n".join(
        str(row[0])
        for row in plan_rows
    )

    assert (
        "catalog_embeddings_embedding_hnsw_idx"
        in plan
    )