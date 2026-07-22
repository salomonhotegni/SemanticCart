"""
This module provides functions to load and process 
user-item interaction data for recommendation systems.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "user_id",
    "parent_asin",
    "rating",
    "timestamp",
]

TRAIN_END_MS = 1628643414042
VALIDATION_END_MS = 1658002729837


@dataclass(frozen=True)
class InteractionSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def load_interactions(path: str | Path) -> pd.DataFrame:
    interactions = pd.read_csv(
        path,
        compression="gzip",
        usecols=REQUIRED_COLUMNS,
        dtype={
            "user_id": "string",
            "parent_asin": "string",
        },
    )

    interactions["user_id"] = interactions["user_id"].str.strip()
    interactions["parent_asin"] = interactions["parent_asin"].str.strip()
    interactions["rating"] = pd.to_numeric(
        interactions["rating"], errors="coerce"
    )
    interactions["timestamp"] = pd.to_numeric(
        interactions["timestamp"], errors="coerce"
    )

    interactions = interactions.dropna(subset=REQUIRED_COLUMNS)
    interactions = interactions.loc[
        (interactions["user_id"] != "")
        & (interactions["parent_asin"] != "")
    ]

    interactions["timestamp"] = interactions["timestamp"].astype("int64")
    interactions = interactions.rename(
        columns={"parent_asin": "item_id"}
    )

    interactions["event_time"] = pd.to_datetime(
        interactions["timestamp"],
        unit="ms",
        utc=True,
    )

    interactions = interactions.drop_duplicates(
        ["user_id", "item_id", "timestamp"],
        keep="last",
    )

    return interactions.sort_values(
        ["timestamp", "user_id", "item_id"]
    ).reset_index(drop=True)


def chronological_split(
    interactions: pd.DataFrame,
    train_end_ms: int = TRAIN_END_MS,
    validation_end_ms: int = VALIDATION_END_MS,
) -> InteractionSplits:
    if train_end_ms >= validation_end_ms:
        raise ValueError("Training cutoff must precede validation cutoff.")

    timestamps = interactions["timestamp"]

    train = interactions.loc[timestamps < train_end_ms].copy()
    validation = interactions.loc[
        (timestamps >= train_end_ms)
        & (timestamps < validation_end_ms)
    ].copy()
    test = interactions.loc[timestamps >= validation_end_ms].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError("Chronological split produced an empty partition.")

    return InteractionSplits(train, validation, test)

def leave_last_two_split(
    interactions: pd.DataFrame,
) -> InteractionSplits:
    required = {"user_id", "item_id", "timestamp"}
    missing = required - set(interactions.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    ordered = interactions.sort_values(
        ["user_id", "timestamp", "item_id"]
    ).copy()

    ordered["_position"] = ordered.groupby("user_id").cumcount()
    ordered["_user_count"] = ordered.groupby("user_id")[
        "item_id"
    ].transform("size")

    if (ordered["_user_count"] < 3).any():
        raise ValueError(
            "Every user must have at least three interactions."
        )

    ordered["_position_from_end"] = (
        ordered["_user_count"] - ordered["_position"] - 1
    )

    train = ordered.loc[
        ordered["_position_from_end"] >= 2
    ].copy()
    validation = ordered.loc[
        ordered["_position_from_end"] == 1
    ].copy()
    test = ordered.loc[
        ordered["_position_from_end"] == 0
    ].copy()

    helper_columns = [
        "_position",
        "_user_count",
        "_position_from_end",
    ]

    train = train.drop(columns=helper_columns).reset_index(drop=True)
    validation = validation.drop(
        columns=helper_columns
    ).reset_index(drop=True)
    test = test.drop(columns=helper_columns).reset_index(drop=True)

    return InteractionSplits(train, validation, test)