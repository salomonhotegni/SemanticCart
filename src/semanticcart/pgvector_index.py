"""Retrieve semantic products through PostgreSQL HNSW."""

from dataclasses import dataclass
from threading import RLock

import numpy as np
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool


@dataclass(frozen=True)
class PgvectorSearchConfig:
    """Configure pgvector retrieval and connection pooling."""

    ef_search: int = 128
    min_pool_size: int = 1
    max_pool_size: int = 8
    timeout_seconds: float = 10.0


class PgvectorItemIndex:
    """Expose PostgreSQL HNSW through the dense-index search contract."""

    def __init__(
        self,
        database_url: str,
        model_version: str,
        item_ids: np.ndarray,
        dimensions: int = 512,
        config: PgvectorSearchConfig | None = None,
    ) -> None:
        config = config or PgvectorSearchConfig()
        database_url = str(database_url).strip()
        model_version = str(model_version).strip()

        if not database_url:
            raise ValueError(
                "database_url cannot be empty."
            )
        if not model_version:
            raise ValueError(
                "model_version cannot be empty."
            )
        if (
            isinstance(dimensions, bool)
            or not isinstance(dimensions, int)
            or dimensions <= 0
        ):
            raise ValueError(
                "dimensions must be positive."
            )

        for name, value in (
            ("ef_search", config.ef_search),
            ("min_pool_size", config.min_pool_size),
            ("max_pool_size", config.max_pool_size),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be positive."
                )

        if (
            config.max_pool_size
            < config.min_pool_size
        ):
            raise ValueError(
                "max_pool_size must be at least "
                "min_pool_size."
            )
        if (
            isinstance(config.timeout_seconds, bool)
            or config.timeout_seconds <= 0
        ):
            raise ValueError(
                "timeout_seconds must be positive."
            )

        item_ids = np.asarray(
            item_ids,
            dtype=str,
        )

        if item_ids.ndim != 1 or len(item_ids) == 0:
            raise ValueError(
                "item_ids must be a nonempty vector."
            )
        if np.any(np.char.strip(item_ids) == ""):
            raise ValueError(
                "item_ids cannot contain empty values."
            )
        if len(np.unique(item_ids)) != len(item_ids):
            raise ValueError(
                "item_ids must be unique."
            )

        self.model_version = model_version
        self.item_ids = item_ids
        self.dimensions = dimensions
        self.config = config
        self.item_to_index = {
            item_id: position
            for position, item_id in enumerate(
                self.item_ids
            )
        }

        self._pool = ConnectionPool(
            conninfo=database_url,
            min_size=config.min_pool_size,
            max_size=config.max_pool_size,
            timeout=float(config.timeout_seconds),
            configure=register_vector,
            open=False,
            name="semanticcart-pgvector",
        )
        self._is_open = False
        self._state_lock = RLock()

    def open(self) -> None:
        """Open the pool and verify the published vector snapshot."""
        with self._state_lock:
            if self._is_open:
                return

            self._pool.open(
                wait=True,
                timeout=self.config.timeout_seconds,
            )

            try:
                with self._pool.connection() as connection:
                    summary = connection.execute(
                        """
                        SELECT
                            COUNT(*)::BIGINT,
                            COUNT(
                                DISTINCT item_id
                            )::BIGINT,
                            MIN(
                                vector_dims(embedding)
                            )::INTEGER,
                            MAX(
                                vector_dims(embedding)
                            )::INTEGER
                        FROM catalog_embeddings
                        WHERE model_version = %s
                        """,
                        (self.model_version,),
                    ).fetchone()

                    if summary is None:
                        raise RuntimeError(
                            "Vector snapshot verification "
                            "returned no result."
                        )

                    (
                        row_count,
                        unique_items,
                        minimum_dimensions,
                        maximum_dimensions,
                    ) = summary

                    if (
                        row_count != len(self.item_ids)
                        or unique_items
                        != len(self.item_ids)
                    ):
                        raise RuntimeError(
                            "PostgreSQL vector item count "
                            "does not match the serving bundle."
                        )
                    if {
                        minimum_dimensions,
                        maximum_dimensions,
                    } != {self.dimensions}:
                        raise RuntimeError(
                            "PostgreSQL vector dimensions "
                            "do not match the serving bundle."
                        )

                    rows = connection.execute(
                        """
                        SELECT item_id
                        FROM catalog_embeddings
                        WHERE model_version = %s
                        ORDER BY item_id
                        """,
                        (self.model_version,),
                    ).fetchall()

                    database_items = np.asarray(
                        [
                            str(row[0])
                            for row in rows
                        ],
                        dtype=str,
                    )
                    expected_items = np.sort(
                        self.item_ids
                    )

                    if not np.array_equal(
                        database_items,
                        expected_items,
                    ):
                        raise RuntimeError(
                            "PostgreSQL vector item IDs "
                            "do not match the serving bundle."
                        )
            except Exception:
                self._pool.close()
                raise

            self._is_open = True

    def close(self) -> None:
        """Close every pooled PostgreSQL connection."""
        with self._state_lock:
            if not self._is_open:
                return

            self._pool.close()
            self._is_open = False

    def _require_open(self) -> None:
        """Reject retrieval outside the adapter lifecycle."""
        if not self._is_open:
            raise RuntimeError(
                "Pgvector item index is not open."
            )

    def _normalize_queries(
        self,
        query_vectors: np.ndarray,
    ) -> np.ndarray:
        """Validate and normalize query vectors for cosine search."""
        queries = np.asarray(
            query_vectors,
            dtype=np.float32,
        )

        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        if queries.ndim != 2:
            raise ValueError(
                "Query vectors must be a matrix."
            )
        if queries.shape[1] != self.dimensions:
            raise ValueError(
                "Query and product dimensions do not match."
            )
        if not np.isfinite(queries).all():
            raise ValueError(
                "Query vectors must contain finite values."
            )

        norms = np.linalg.norm(
            queries,
            axis=1,
            keepdims=True,
        )

        if np.any(norms == 0):
            raise ValueError(
                "Query vectors cannot be zero vectors."
            )

        return np.ascontiguousarray(
            queries / norms,
            dtype=np.float32,
        )

    def search(
        self,
        query_vectors: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return cosine scores and serving-bundle item positions."""
        self._require_open()

        if (
            isinstance(k, bool)
            or not isinstance(k, int)
            or k <= 0
        ):
            raise ValueError(
                "k must be greater than zero."
            )

        queries = self._normalize_queries(
            query_vectors
        )
        result_count = min(
            k,
            len(self.item_ids),
        )
        ef_search = max(
            self.config.ef_search,
            result_count,
        )

        scores = np.empty(
            (len(queries), result_count),
            dtype=np.float32,
        )
        indices = np.empty(
            (len(queries), result_count),
            dtype=np.int64,
        )

        search_sql = """
            SELECT
                item_id,
                1.0 - (embedding <=> %s)
                    AS cosine_similarity
            FROM catalog_embeddings
            WHERE model_version = %s
            ORDER BY embedding <=> %s
            LIMIT %s
        """

        with self._pool.connection() as connection:
            connection.execute(
                """
                SELECT set_config(
                    'hnsw.ef_search',
                    %s,
                    true
                )
                """,
                (str(ef_search),),
            )
            connection.execute(
                """
                SELECT set_config(
                    'hnsw.iterative_scan',
                    'strict_order',
                    true
                )
                """
            )

            for query_position, query in enumerate(
                queries
            ):
                vector = Vector(query)
                rows = connection.execute(
                    search_sql,
                    (
                        vector,
                        self.model_version,
                        vector,
                        result_count,
                    ),
                ).fetchall()

                if len(rows) != result_count:
                    raise RuntimeError(
                        "PostgreSQL returned an incomplete "
                        "nearest-neighbour result."
                    )

                for rank, row in enumerate(rows):
                    item_id = str(row[0])
                    score = float(row[1])

                    if item_id not in self.item_to_index:
                        raise RuntimeError(
                            "PostgreSQL returned an unknown item."
                        )
                    if not np.isfinite(score):
                        raise RuntimeError(
                            "PostgreSQL returned a nonfinite score."
                        )

                    indices[
                        query_position,
                        rank,
                    ] = self.item_to_index[item_id]
                    scores[
                        query_position,
                        rank,
                    ] = score

        return scores, indices

    def __enter__(self) -> "PgvectorItemIndex":
        """Open the context-managed index."""
        self.open()
        return self

    def __exit__(
        self,
        exception_type,
        exception,
        traceback,
    ) -> None:
        """Close the context-managed index."""
        self.close()