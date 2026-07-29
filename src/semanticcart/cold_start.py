"""Construct reproducible cold-start evaluation cohorts."""

from numbers import Integral

import pandas as pd


SESSION_COLUMNS = ["user_id", "item_id", "timestamp"]


def select_recent_session_interactions(
    interactions: pd.DataFrame,
    session_length: int,
) -> pd.DataFrame:
    """Select each eligible user's N most recent unique products.

    Users with fewer than ``session_length`` unique products are excluded.
    Returned interactions are ordered oldest-to-newest within each session.

    Args:
        interactions: Chronological events with user, item, and timestamp.
        session_length: Exact number of recent unique products per user.

    Returns:
        Session interactions for users with sufficient history.

    Raises:
        ValueError: If configuration, columns, identifiers, or timestamps
            are invalid.
    """
    if (
        isinstance(session_length, bool)
        or not isinstance(session_length, Integral)
        or session_length <= 0
    ):
        raise ValueError(
            "session_length must be a positive integer."
        )

    missing = set(SESSION_COLUMNS) - set(interactions.columns)

    if missing:
        raise ValueError(
            f"Missing interaction columns: {sorted(missing)}"
        )

    events = interactions[SESSION_COLUMNS].copy()

    if events[["user_id", "item_id"]].isna().any().any():
        raise ValueError("User and item identifiers cannot be null.")

    events["user_id"] = events["user_id"].astype(str)
    events["item_id"] = events["item_id"].astype(str)
    events["timestamp"] = pd.to_numeric(
        events["timestamp"],
        errors="coerce",
    )

    if events["timestamp"].isna().any():
        raise ValueError("Interaction timestamps must be numeric.")

    events = (
        events.sort_values(
            ["user_id", "timestamp", "item_id"],
            ascending=[True, False, True],
        )
        .drop_duplicates(
            ["user_id", "item_id"],
            keep="first",
        )
    )

    unique_counts = events.groupby("user_id").size()
    eligible_users = unique_counts.loc[
        unique_counts >= session_length
    ].index

    sessions = (
        events.loc[events["user_id"].isin(eligible_users)]
        .groupby("user_id", sort=False)
        .head(session_length)
        .sort_values(
            ["user_id", "timestamp", "item_id"],
            ascending=[True, True, True],
        )
        .reset_index(drop=True)
    )

    return sessions


def select_cold_item_events(
    fit_interactions: pd.DataFrame,
    evaluation_events: pd.DataFrame,
) -> pd.DataFrame:
    """Return evaluation events for items absent during model fitting.

    Args:
        fit_interactions: Events used to fit the recommendation system.
        evaluation_events: Held-out events containing user and item IDs.

    Returns:
        Held-out events whose items never occur in fit interactions.

    Raises:
        ValueError: If required columns or identifiers are invalid.
    """
    if "item_id" not in fit_interactions.columns:
        raise ValueError(
            "fit_interactions must contain item_id."
        )

    required_evaluation = {"user_id", "item_id"}
    missing = required_evaluation - set(
        evaluation_events.columns
    )

    if missing:
        raise ValueError(
            f"Missing evaluation columns: {sorted(missing)}"
        )

    if fit_interactions["item_id"].isna().any():
        raise ValueError("Fit item identifiers cannot be null.")

    if (
        evaluation_events[["user_id", "item_id"]]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            "Evaluation user and item identifiers cannot be null."
        )

    fit_item_ids = set(
        fit_interactions["item_id"].astype(str)
    )

    cold_events = evaluation_events.copy()
    cold_events["user_id"] = cold_events["user_id"].astype(str)
    cold_events["item_id"] = cold_events["item_id"].astype(str)

    return cold_events.loc[
        ~cold_events["item_id"].isin(fit_item_ids)
    ].reset_index(drop=True)