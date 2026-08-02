"""Synchronize immutable serving embeddings with PostgreSQL."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

if TYPE_CHECKING:
    from semanticcart.serving import ServingBundle


EXPECTED_DIMENSIONS = 512
NORMALIZATION_TOLERANCE = 1e-4
SYNC_LOCK_NAME = "semanticcart.catalog_embeddings.sync"


@dataclass(frozen=True)
class PgvectorSyncResult:
    """Describe one successfully published embedding snapshot."""

    model_version: str
    embedding_model: str
    rows: int
    dimensions: int
    minimum_norm: float
    maximum_norm: float
    replaced_rows: int
    index_rebuilt: bool


def _validate_bundle(
    bundle: "ServingBundle",
) -> tuple[str, str, np.ndarray, np.ndarray]:
    """Validate and extract one serving embedding snapshot."""
    model_version = str(bundle.version).strip()

    if not model_version:
        raise ValueError("Model version cannot be empty.")

    embedding_config = bundle.manifest.get(
        "embedding_config"
    )

    if not isinstance(embedding_config, dict):
        raise ValueError(
            "Serving embedding configuration is missing."
        )

    embedding_model = str(
        embedding_config.get("model", "")
    ).strip()
    dimensions = embedding_config.get("dimensions")

    if not embedding_model:
        raise ValueError("Embedding model cannot be empty.")
    if (
        isinstance(dimensions, bool)
        or not isinstance(dimensions, int)
        or dimensions != EXPECTED_DIMENSIONS
    ):
        raise ValueError(
            "Serving embeddings must have 512 dimensions."
        )

    item_ids = np.asarray(
        bundle.item_index.item_ids,
        dtype=str,
    )
    vectors = np.asarray(
        bundle.item_index.item_vectors,
        dtype=np.float32,
    )

    if item_ids.ndim != 1 or len(item_ids) == 0:
        raise ValueError(
            "Serving item IDs must be a nonempty vector."
        )
    if np.any(np.char.strip(item_ids) == ""):
        raise ValueError(
            "Serving item IDs cannot be empty."
        )
    if len(np.unique(item_ids)) != len(item_ids):
        raise ValueError(
            "Serving item IDs must be unique."
        )
    if vectors.ndim != 2:
        raise ValueError(
            "Serving embeddings must be a matrix."
        )
    if vectors.shape != (
        len(item_ids),
        EXPECTED_DIMENSIONS,
    ):
        raise ValueError(
            "Serving item IDs and embeddings do not match."
        )
    if not np.isfinite(vectors).all():
        raise ValueError(
            "Serving embeddings must contain finite values."
        )

    norms = np.linalg.norm(vectors, axis=1)

    if np.any(norms == 0):
        raise ValueError(
            "Serving embeddings cannot contain zero vectors."
        )
    if not np.allclose(
        norms,
        1.0,
        atol=NORMALIZATION_TOLERANCE,
    ):
        raise ValueError(
            "Serving embeddings must be normalized."
        )

    expected_items = bundle.manifest.get(
        "semantic_items"
    )

    if expected_items != len(item_ids):
        raise ValueError(
            "Manifest semantic-item count does not match."
        )

    return (
        model_version,
        embedding_model,
        item_ids,
        np.ascontiguousarray(vectors),
    )


def _require_schema(
    connection: psycopg.Connection,
) -> None:
    """Verify that the vector table and HNSW index exist."""
    row = connection.execute(
        """
        SELECT
            to_regclass(
                'public.catalog_embeddings'
            ),
            to_regclass(
                'public.catalog_embeddings_embedding_hnsw_idx'
            )
        """
    ).fetchone()

    if row is None or row[0] is None:
        raise RuntimeError(
            "catalog_embeddings is missing; "
            "run migration 002."
        )
    if row[1] is None:
        raise RuntimeError(
            "The catalogue HNSW index is missing; "
            "run migration 002."
        )


def _copy_to_stage(
    connection: psycopg.Connection,
    model_version: str,
    embedding_model: str,
    item_ids: np.ndarray,
    vectors: np.ndarray,
) -> None:
    """Binary-copy one validated snapshot into a temporary table."""
    connection.execute(
        """
        CREATE TEMP TABLE
            semanticcart_embedding_stage
        (
            LIKE public.catalog_embeddings
            INCLUDING DEFAULTS
            INCLUDING CONSTRAINTS
        )
        ON COMMIT DROP
        """
    )

    copy_sql = """
        COPY semanticcart_embedding_stage (
            model_version,
            item_id,
            embedding_model,
            embedding_dimensions,
            embedding
        )
        FROM STDIN
        WITH (FORMAT BINARY)
    """

    with connection.cursor().copy(copy_sql) as copy:
        copy.set_types(
            [
                "text",
                "text",
                "text",
                "int2",
                "vector",
            ]
        )

        for item_id, vector in zip(
            item_ids,
            vectors,
            strict=True,
        ):
            copy.write_row(
                (
                    model_version,
                    str(item_id),
                    embedding_model,
                    EXPECTED_DIMENSIONS,
                    Vector(vector),
                )
            )


def _validate_stage(
    connection: psycopg.Connection,
    model_version: str,
    embedding_model: str,
    expected_rows: int,
) -> tuple[float, float]:
    """Validate the staged snapshot using database calculations."""
    row = connection.execute(
        """
        SELECT
            COUNT(*)::BIGINT,
            COUNT(DISTINCT item_id)::BIGINT,
            COUNT(*) FILTER (
                WHERE model_version <> %s
                   OR embedding_model <> %s
            )::BIGINT,
            MIN(embedding_dimensions)::INTEGER,
            MAX(embedding_dimensions)::INTEGER,
            MIN(vector_dims(embedding))::INTEGER,
            MAX(vector_dims(embedding))::INTEGER,
            MIN(vector_norm(embedding))::DOUBLE PRECISION,
            MAX(vector_norm(embedding))::DOUBLE PRECISION
        FROM semanticcart_embedding_stage
        """,
        (
            model_version,
            embedding_model,
        ),
    ).fetchone()

    if row is None:
        raise RuntimeError(
            "Staged embedding validation returned no result."
        )

    (
        rows,
        unique_items,
        metadata_mismatches,
        minimum_declared_dimensions,
        maximum_declared_dimensions,
        minimum_vector_dimensions,
        maximum_vector_dimensions,
        minimum_norm,
        maximum_norm,
    ) = row

    if rows != expected_rows:
        raise ValueError(
            "Staged embedding row count does not match."
        )
    if unique_items != expected_rows:
        raise ValueError(
            "Staged embedding item IDs are not unique."
        )
    if metadata_mismatches != 0:
        raise ValueError(
            "Staged embedding metadata does not match."
        )
    if {
        minimum_declared_dimensions,
        maximum_declared_dimensions,
        minimum_vector_dimensions,
        maximum_vector_dimensions,
    } != {EXPECTED_DIMENSIONS}:
        raise ValueError(
            "Staged embedding dimensions do not match."
        )
    if (
        minimum_norm is None
        or maximum_norm is None
        or abs(float(minimum_norm) - 1.0)
        > NORMALIZATION_TOLERANCE
        or abs(float(maximum_norm) - 1.0)
        > NORMALIZATION_TOLERANCE
    ):
        raise ValueError(
            "Staged embeddings are not normalized."
        )

    return float(minimum_norm), float(maximum_norm)


def sync_serving_embeddings(
    database_url: str,
    bundle: "ServingBundle",
    rebuild_index: bool = True,
) -> PgvectorSyncResult:
    """Publish one validated serving snapshot transactionally.

    The current database snapshot is replaced only after binary COPY and
    database-side validation succeed. Any exception rolls back the complete
    operation and preserves the previously published vectors.
    """
    database_url = str(database_url).strip()

    if not database_url:
        raise ValueError(
            "database_url cannot be empty."
        )
    if not isinstance(rebuild_index, bool):
        raise ValueError(
            "rebuild_index must be boolean."
        )

    (
        model_version,
        embedding_model,
        item_ids,
        vectors,
    ) = _validate_bundle(bundle)

    with psycopg.connect(database_url) as connection:
        register_vector(connection)
        _require_schema(connection)

        connection.execute(
            """
            SELECT pg_advisory_xact_lock(
                hashtext(%s)
            )
            """,
            (SYNC_LOCK_NAME,),
        )

        previous_row = connection.execute(
            """
            SELECT COUNT(*)::BIGINT
            FROM catalog_embeddings
            """
        ).fetchone()
        replaced_rows = int(previous_row[0])

        _copy_to_stage(
            connection=connection,
            model_version=model_version,
            embedding_model=embedding_model,
            item_ids=item_ids,
            vectors=vectors,
        )
        minimum_norm, maximum_norm = _validate_stage(
            connection=connection,
            model_version=model_version,
            embedding_model=embedding_model,
            expected_rows=len(item_ids),
        )

        connection.execute(
            """
            DELETE FROM catalog_embeddings
            """
        )
        connection.execute(
            """
            INSERT INTO catalog_embeddings (
                model_version,
                item_id,
                embedding_model,
                embedding_dimensions,
                embedding,
                loaded_at
            )
            SELECT
                model_version,
                item_id,
                embedding_model,
                embedding_dimensions,
                embedding,
                loaded_at
            FROM semanticcart_embedding_stage
            """
        )

        if rebuild_index:
            connection.execute(
                """
                REINDEX INDEX
                    catalog_embeddings_embedding_hnsw_idx
                """
            )

        connection.execute(
            """
            ANALYZE catalog_embeddings
            """
        )

        published_row = connection.execute(
            """
            SELECT COUNT(*)::BIGINT
            FROM catalog_embeddings
            WHERE model_version = %s
            """,
            (model_version,),
        ).fetchone()

        if (
            published_row is None
            or int(published_row[0]) != len(item_ids)
        ):
            raise RuntimeError(
                "Published embedding count does not match."
            )

    return PgvectorSyncResult(
        model_version=model_version,
        embedding_model=embedding_model,
        rows=len(item_ids),
        dimensions=EXPECTED_DIMENSIONS,
        minimum_norm=minimum_norm,
        maximum_norm=maximum_norm,
        replaced_rows=replaced_rows,
        index_rebuilt=rebuild_index,
    )