import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import psycopg
import pytest

from semanticcart.events import EventStore
from semanticcart.postgres_events import (
    PostgresEventStore,
)


DATABASE_URL = os.getenv(
    "SEMANTICCART_TEST_DATABASE_URL"
)

pytestmark = pytest.mark.skipif(
    DATABASE_URL is None,
    reason=(
        "SEMANTICCART_TEST_DATABASE_URL "
        "is not configured."
    ),
)


@pytest.fixture
def postgres_store():
    """Open an isolated PostgreSQL history and remove it afterward."""
    assert DATABASE_URL is not None

    user_id = (
        f"postgres-test-{uuid4().hex}"
    )
    store = PostgresEventStore(
        DATABASE_URL,
        min_pool_size=1,
        max_pool_size=4,
    )
    store.open()

    try:
        yield store, user_id
    finally:
        with psycopg.connect(
            DATABASE_URL
        ) as connection:
            connection.execute(
                """
                DELETE FROM interaction_events
                WHERE user_id = %s
                """,
                (user_id,),
            )

        store.close()


def test_postgres_store_satisfies_contract(
    postgres_store,
) -> None:
    store, _ = postgres_store

    assert isinstance(store, EventStore)


def test_persists_normalized_event(
    postgres_store,
) -> None:
    store, user_id = postgres_store
    timestamp = datetime(
        2026,
        8,
        2,
        12,
        0,
        tzinfo=timezone(
            timedelta(hours=2)
        ),
    )

    event = store.append(
        user_id=f"  {user_id}  ",
        item_id="  product-a  ",
        event_type="view",
        occurred_at=timestamp,
    )

    assert event.event_id
    assert event.user_id == user_id
    assert event.item_id == "product-a"
    assert event.occurred_at == datetime(
        2026,
        8,
        2,
        10,
        0,
        tzinfo=timezone.utc,
    )
    assert store.event_count(user_id) == 1


def test_returns_latest_unique_items(
    postgres_store,
) -> None:
    store, user_id = postgres_store
    start = datetime(
        2026,
        8,
        2,
        tzinfo=timezone.utc,
    )

    for offset, item_id in enumerate(
        ["a", "b", "a", "c"]
    ):
        store.append(
            user_id=user_id,
            item_id=item_id,
            event_type="view",
            occurred_at=(
                start
                + timedelta(minutes=offset)
            ),
        )

    assert store.recent_item_ids(
        user_id,
        max_items=3,
    ) == ["b", "a", "c"]

    assert store.recent_item_ids(
        user_id,
        max_items=2,
    ) == ["a", "c"]


def test_events_survive_store_restart(
    postgres_store,
) -> None:
    store, user_id = postgres_store
    assert DATABASE_URL is not None

    store.append(
        user_id,
        "durable-product",
        "purchase",
    )
    store.close()

    replacement = PostgresEventStore(
        DATABASE_URL,
        min_pool_size=1,
        max_pool_size=2,
    )

    try:
        replacement.open()

        assert replacement.recent_item_ids(
            user_id,
            max_items=5,
        ) == ["durable-product"]
    finally:
        replacement.close()


def test_supports_concurrent_pooled_writes(
    postgres_store,
) -> None:
    store, user_id = postgres_store

    def append_event(index: int) -> None:
        store.append(
            user_id=user_id,
            item_id=f"product-{index}",
            event_type="view",
        )

    with ThreadPoolExecutor(
        max_workers=4
    ) as executor:
        list(
            executor.map(
                append_event,
                range(20),
            )
        )

    assert store.event_count(user_id) == 20


def test_rejects_operations_before_open() -> None:
    assert DATABASE_URL is not None

    store = PostgresEventStore(
        DATABASE_URL,
        min_pool_size=1,
        max_pool_size=1,
    )

    with pytest.raises(
        RuntimeError,
        match="not open",
    ):
        store.recent_item_ids(
            "user-1",
            max_items=5,
        )

    store.close()


@pytest.mark.parametrize(
    "arguments",
    [
        {"database_url": ""},
        {
            "database_url": "postgresql://unused",
            "min_pool_size": 0,
        },
        {
            "database_url": "postgresql://unused",
            "min_pool_size": 2,
            "max_pool_size": 1,
        },
        {
            "database_url": "postgresql://unused",
            "timeout_seconds": 0,
        },
    ],
)
def test_rejects_invalid_configuration(
    arguments: dict,
) -> None:
    with pytest.raises(ValueError):
        PostgresEventStore(**arguments)