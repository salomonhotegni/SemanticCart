"""Persist online interaction events in PostgreSQL."""

from datetime import datetime
from threading import RLock

from psycopg_pool import ConnectionPool

from semanticcart.events import (
    EventStore,
    EventType,
    StoredInteractionEvent,
    create_interaction_event,
)


RECENT_ITEMS_SQL = """
WITH latest_per_item AS (
    SELECT DISTINCT ON (item_id)
        item_id,
        occurred_at,
        ingested_at,
        event_id
    FROM interaction_events
    WHERE user_id = %s
    ORDER BY
        item_id,
        occurred_at DESC,
        ingested_at DESC,
        event_id DESC
)
SELECT item_id
FROM latest_per_item
ORDER BY
    occurred_at DESC,
    ingested_at DESC,
    event_id DESC
LIMIT %s
"""


class PostgresEventStore:
    """Store interaction histories through a Psycopg connection pool."""
    backend = "postgresql"
    
    def __init__(
        self,
        database_url: str,
        min_pool_size: int = 1,
        max_pool_size: int = 8,
        timeout_seconds: float = 10.0,
    ) -> None:
        database_url = str(database_url).strip()

        if not database_url:
            raise ValueError(
                "database_url cannot be empty."
            )
        if (
            isinstance(min_pool_size, bool)
            or not isinstance(min_pool_size, int)
            or min_pool_size <= 0
        ):
            raise ValueError(
                "min_pool_size must be positive."
            )
        if (
            isinstance(max_pool_size, bool)
            or not isinstance(max_pool_size, int)
            or max_pool_size < min_pool_size
        ):
            raise ValueError(
                "max_pool_size must be at least min_pool_size."
            )
        if (
            isinstance(timeout_seconds, bool)
            or timeout_seconds <= 0
        ):
            raise ValueError(
                "timeout_seconds must be positive."
            )

        self.timeout_seconds = float(
            timeout_seconds
        )
        self._pool = ConnectionPool(
            conninfo=database_url,
            min_size=min_pool_size,
            max_size=max_pool_size,
            timeout=self.timeout_seconds,
            open=False,
            name="semanticcart-events",
        )
        self._is_open = False
        self._state_lock = RLock()

    def open(self) -> None:
        """Open the pool and verify the migrated event table."""
        with self._state_lock:
            if self._is_open:
                return

            self._pool.open(
                wait=True,
                timeout=self.timeout_seconds,
            )

            try:
                with self._pool.connection() as connection:
                    row = connection.execute(
                        """
                        SELECT to_regclass(
                            'public.interaction_events'
                        )
                        """
                    ).fetchone()

                    if row is None or row[0] is None:
                        raise RuntimeError(
                            "interaction_events is missing; "
                            "run the PostgreSQL migration."
                        )
            except Exception:
                self._pool.close()
                raise

            self._is_open = True

    def close(self) -> None:
        """Close all pooled PostgreSQL connections."""
        with self._state_lock:
            if not self._is_open:
                return

            self._pool.close()
            self._is_open = False

    def _require_open(self) -> None:
        """Reject operations before startup or after shutdown."""
        if not self._is_open:
            raise RuntimeError(
                "PostgreSQL event store is not open."
            )

    def append(
        self,
        user_id: str,
        item_id: str,
        event_type: EventType,
        occurred_at: datetime | None = None,
    ) -> StoredInteractionEvent:
        """Validate and persist one interaction transactionally."""
        self._require_open()

        event = create_interaction_event(
            user_id=user_id,
            item_id=item_id,
            event_type=event_type,
            occurred_at=occurred_at,
        )

        with self._pool.connection() as connection:
            connection.execute(
                """
                INSERT INTO interaction_events (
                    event_id,
                    user_id,
                    item_id,
                    event_type,
                    occurred_at
                )
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    event.event_id,
                    event.user_id,
                    event.item_id,
                    event.event_type,
                    event.occurred_at,
                ),
            )

        return event

    def recent_item_ids(
        self,
        user_id: str,
        max_items: int,
    ) -> list[str]:
        """Load latest unique products in chronological order."""
        self._require_open()

        user_id = str(user_id).strip()

        if not user_id:
            raise ValueError(
                "user_id cannot be empty."
            )
        if (
            isinstance(max_items, bool)
            or not isinstance(max_items, int)
            or max_items <= 0
        ):
            raise ValueError(
                "max_items must be positive."
            )

        with self._pool.connection() as connection:
            rows = connection.execute(
                RECENT_ITEMS_SQL,
                (
                    user_id,
                    max_items,
                ),
            ).fetchall()

        item_ids = [
            str(row[0])
            for row in rows
        ]
        item_ids.reverse()
        return item_ids

    def event_count(
        self,
        user_id: str,
    ) -> int:
        """Count persisted events for one user."""
        self._require_open()

        with self._pool.connection() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*)
                FROM interaction_events
                WHERE user_id = %s
                """,
                (str(user_id).strip(),),
            ).fetchone()

        return int(row[0])

    def __enter__(self) -> "PostgresEventStore":
        """Open the store for context-managed scripts."""
        self.open()
        return self

    def __exit__(
        self,
        exception_type,
        exception,
        traceback,
    ) -> None:
        """Close the context-managed store."""
        self.close()