"""Store recent interaction events for online session personalization."""

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Literal
from uuid import uuid4


EventType = Literal["view", "cart", "purchase"]
VALID_EVENT_TYPES = frozenset(
    {"view", "cart", "purchase"}
)


@dataclass(frozen=True)
class StoredInteractionEvent:
    """Represent one normalized interaction accepted by the service."""

    event_id: str
    user_id: str
    item_id: str
    event_type: EventType
    occurred_at: datetime


class InMemoryEventStore:
    """Maintain bounded recent histories for local API serving.

    This implementation is process-local and intended for development and
    testing. A PostgreSQL implementation can later provide the same append
    and recent-item operations without changing recommendation logic.
    """

    def __init__(
        self,
        max_events_per_user: int = 100,
    ) -> None:
        if (
            isinstance(max_events_per_user, bool)
            or not isinstance(max_events_per_user, int)
            or max_events_per_user <= 0
        ):
            raise ValueError(
                "max_events_per_user must be positive."
            )

        self.max_events_per_user = (
            max_events_per_user
        )
        self._events: dict[
            str,
            list[StoredInteractionEvent],
        ] = {}
        self._lock = RLock()

    def append(
        self,
        user_id: str,
        item_id: str,
        event_type: EventType,
        occurred_at: datetime | None = None,
    ) -> StoredInteractionEvent:
        """Validate and append one interaction event."""
        user_id = str(user_id).strip()
        item_id = str(item_id).strip()

        if not user_id:
            raise ValueError("user_id cannot be empty.")
        if not item_id:
            raise ValueError("item_id cannot be empty.")
        if event_type not in VALID_EVENT_TYPES:
            raise ValueError(
                f"Unsupported event type: {event_type}"
            )

        timestamp = (
            occurred_at
            if occurred_at is not None
            else datetime.now(timezone.utc)
        )

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(
                tzinfo=timezone.utc
            )
        else:
            timestamp = timestamp.astimezone(
                timezone.utc
            )

        event = StoredInteractionEvent(
            event_id=uuid4().hex,
            user_id=user_id,
            item_id=item_id,
            event_type=event_type,
            occurred_at=timestamp,
        )

        with self._lock:
            history = self._events.setdefault(
                user_id,
                [],
            )
            history.append(event)

            excess = (
                len(history)
                - self.max_events_per_user
            )

            if excess > 0:
                del history[:excess]

        return event

    def recent_item_ids(
        self,
        user_id: str,
        max_items: int,
    ) -> list[str]:
        """Return latest unique products in chronological order."""
        if (
            isinstance(max_items, bool)
            or not isinstance(max_items, int)
            or max_items <= 0
        ):
            raise ValueError(
                "max_items must be positive."
            )

        user_id = str(user_id).strip()

        if not user_id:
            raise ValueError("user_id cannot be empty.")

        with self._lock:
            history = list(
                self._events.get(user_id, [])
            )

        seen = set()
        recent = []

        for event in reversed(history):
            if event.item_id in seen:
                continue

            seen.add(event.item_id)
            recent.append(event.item_id)

            if len(recent) == max_items:
                break

        recent.reverse()
        return recent

    def event_count(
        self,
        user_id: str,
    ) -> int:
        """Return the number of retained events for one user."""
        with self._lock:
            return len(
                self._events.get(
                    str(user_id).strip(),
                    [],
                )
            )