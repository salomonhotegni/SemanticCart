from datetime import datetime, timedelta, timezone

import pytest

from semanticcart.events import (
    InMemoryEventStore,
)


def test_appends_normalized_event() -> None:
    store = InMemoryEventStore()
    timestamp = datetime(
        2026,
        8,
        1,
        12,
        30,
        tzinfo=timezone(
            timedelta(hours=2)
        ),
    )

    event = store.append(
        user_id="  user-1  ",
        item_id="  product-a  ",
        event_type="view",
        occurred_at=timestamp,
    )

    assert event.event_id
    assert event.user_id == "user-1"
    assert event.item_id == "product-a"
    assert event.event_type == "view"
    assert event.occurred_at == datetime(
        2026,
        8,
        1,
        10,
        30,
        tzinfo=timezone.utc,
    )
    assert store.event_count("user-1") == 1


def test_treats_naive_timestamp_as_utc() -> None:
    store = InMemoryEventStore()

    event = store.append(
        "user-1",
        "product-a",
        "cart",
        datetime(2026, 8, 1, 10, 30),
    )

    assert event.occurred_at.tzinfo == timezone.utc
    assert event.occurred_at.hour == 10


def test_generates_current_utc_timestamp() -> None:
    store = InMemoryEventStore()
    before = datetime.now(timezone.utc)

    event = store.append(
        "user-1",
        "product-a",
        "purchase",
    )

    after = datetime.now(timezone.utc)

    assert before <= event.occurred_at <= after


def test_returns_latest_unique_items_in_order() -> None:
    store = InMemoryEventStore()

    for item_id in ["a", "b", "a", "c"]:
        store.append(
            "user-1",
            item_id,
            "view",
        )

    assert store.recent_item_ids(
        "user-1",
        max_items=3,
    ) == ["b", "a", "c"]

    assert store.recent_item_ids(
        "user-1",
        max_items=2,
    ) == ["a", "c"]


def test_bounds_each_user_history() -> None:
    store = InMemoryEventStore(
        max_events_per_user=2,
    )

    for item_id in ["a", "b", "c"]:
        store.append(
            "user-1",
            item_id,
            "view",
        )

    store.append(
        "user-2",
        "d",
        "view",
    )

    assert store.event_count("user-1") == 2
    assert store.event_count("user-2") == 1
    assert store.recent_item_ids(
        "user-1",
        max_items=5,
    ) == ["b", "c"]


def test_unknown_user_has_empty_history() -> None:
    store = InMemoryEventStore()

    assert store.recent_item_ids(
        "unknown",
        max_items=5,
    ) == []


@pytest.mark.parametrize(
    "max_events",
    [0, -1, True, 1.5],
)
def test_rejects_invalid_history_capacity(
    max_events: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="max_events_per_user",
    ):
        InMemoryEventStore(
            max_events_per_user=max_events,
        )


@pytest.mark.parametrize(
    ("user_id", "item_id"),
    [
        ("", "product-a"),
        ("user-1", ""),
        ("   ", "product-a"),
        ("user-1", "   "),
    ],
)
def test_rejects_empty_identifiers(
    user_id: str,
    item_id: str,
) -> None:
    store = InMemoryEventStore()

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        store.append(
            user_id,
            item_id,
            "view",
        )


def test_rejects_unsupported_event_type() -> None:
    store = InMemoryEventStore()

    with pytest.raises(
        ValueError,
        match="Unsupported event type",
    ):
        store.append(
            "user-1",
            "product-a",
            "wishlist",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "max_items",
    [0, -1, True, 1.5],
)
def test_rejects_invalid_recent_item_depth(
    max_items: object,
) -> None:
    store = InMemoryEventStore()

    with pytest.raises(
        ValueError,
        match="max_items",
    ):
        store.recent_item_ids(
            "user-1",
            max_items=max_items,
        )