import pandas as pd
import pytest

from semanticcart.cold_start import (
    select_cold_item_events,
    select_recent_session_interactions,
)


def test_recent_sessions_select_latest_unique_items() -> None:
    events = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u1", "u2"],
            "item_id": ["a", "b", "a", "c", "x"],
            "timestamp": [1, 2, 3, 4, 1],
        }
    )

    sessions = select_recent_session_interactions(
        events,
        session_length=2,
    )

    assert sessions["user_id"].unique().tolist() == ["u1"]
    assert sessions["item_id"].tolist() == ["a", "c"]
    assert sessions["timestamp"].tolist() == [3, 4]


def test_recent_sessions_return_oldest_to_newest() -> None:
    events = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "item_id": ["newest", "oldest", "middle"],
            "timestamp": [30, 10, 20],
        }
    )

    sessions = select_recent_session_interactions(events, 3)

    assert sessions["item_id"].tolist() == [
        "oldest",
        "middle",
        "newest",
    ]


@pytest.mark.parametrize(
    "session_length",
    [0, -1, 1.5, True],
)
def test_recent_sessions_reject_invalid_length(
    session_length: object,
) -> None:
    events = pd.DataFrame(
        {
            "user_id": ["u1"],
            "item_id": ["a"],
            "timestamp": [1],
        }
    )

    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        select_recent_session_interactions(
            events,
            session_length,  # type: ignore[arg-type]
        )


def test_recent_sessions_reject_invalid_timestamp() -> None:
    events = pd.DataFrame(
        {
            "user_id": ["u1"],
            "item_id": ["a"],
            "timestamp": ["invalid"],
        }
    )

    with pytest.raises(ValueError, match="timestamps"):
        select_recent_session_interactions(events, 1)


def test_recent_sessions_require_columns() -> None:
    with pytest.raises(ValueError, match="timestamp"):
        select_recent_session_interactions(
            pd.DataFrame(
                {
                    "user_id": ["u1"],
                    "item_id": ["a"],
                }
            ),
            1,
        )


def test_cold_item_events_select_unseen_items() -> None:
    fit = pd.DataFrame({"item_id": [1, 2]})
    evaluation = pd.DataFrame(
        {
            "user_id": [10, 20, 30],
            "item_id": [2, 3, 4],
            "timestamp": [100, 200, 300],
        }
    )

    cold = select_cold_item_events(fit, evaluation)

    assert cold["user_id"].tolist() == ["20", "30"]
    assert cold["item_id"].tolist() == ["3", "4"]
    assert cold["timestamp"].tolist() == [200, 300]


def test_cold_item_events_require_identifiers() -> None:
    with pytest.raises(ValueError, match="user_id"):
        select_cold_item_events(
            pd.DataFrame({"item_id": ["a"]}),
            pd.DataFrame({"item_id": ["b"]}),
        )