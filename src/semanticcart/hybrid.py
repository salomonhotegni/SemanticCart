"""Blend collaborative and semantic recommendation candidates."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HybridConfig:
    """Configure normalized hybrid ranking.

    Attributes:
        semantic_weight: Semantic contribution between zero and one.
        k: Maximum recommendations returned per user.
    """

    semantic_weight: float = 0.5
    k: int = 10


RESULT_COLUMNS = [
    "user_id",
    "item_id",
    "rank",
    "hybrid_score",
    "collaborative_score",
    "semantic_score",
    "collaborative_normalized",
    "semantic_normalized",
]


def _prepare_candidates(
    candidates: pd.DataFrame,
    score_column: str,
    prefix: str,
) -> pd.DataFrame:
    """Validate and min-max normalize one model's scores per user."""
    required = {"user_id", "item_id", "rank", score_column}
    missing = required - set(candidates.columns)

    if missing:
        raise ValueError(
            f"Missing {prefix} columns: {sorted(missing)}"
        )

    prepared = candidates[
        ["user_id", "item_id", "rank", score_column]
    ].copy()

    prepared["user_id"] = prepared["user_id"].astype(str)
    prepared["item_id"] = prepared["item_id"].astype(str)
    prepared["rank"] = pd.to_numeric(
        prepared["rank"],
        errors="coerce",
    )
    prepared[score_column] = pd.to_numeric(
        prepared[score_column],
        errors="coerce",
    )

    if prepared[["rank", score_column]].isna().any().any():
        raise ValueError(
            f"{prefix} ranks and scores must be numeric."
        )
    if (prepared["rank"] <= 0).any():
        raise ValueError(
            f"{prefix} ranks must be positive."
        )
    if not np.isfinite(
        prepared[score_column].to_numpy()
    ).all():
        raise ValueError(
            f"{prefix} scores must be finite."
        )

    prepared = (
        prepared.sort_values(
            ["user_id", "rank", score_column, "item_id"],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(
            ["user_id", "item_id"],
            keep="first",
        )
    )

    grouped_scores = prepared.groupby("user_id")[
        score_column
    ]
    score_minimum = grouped_scores.transform("min")
    score_maximum = grouped_scores.transform("max")
    score_range = score_maximum - score_minimum

    normalized = (
        (prepared[score_column] - score_minimum)
        / score_range.where(score_range > 0)
    ).fillna(1.0)

    return prepared.rename(
        columns={
            "rank": f"{prefix}_rank",
            score_column: f"{prefix}_score",
        }
    ).assign(
        **{f"{prefix}_normalized": normalized}
    )


def rank_hybrid_candidates(
    collaborative: pd.DataFrame,
    semantic: pd.DataFrame,
    config: HybridConfig | None = None,
) -> pd.DataFrame:
    """Blend the union of ALS and semantic candidates.

    Scores are normalized independently for each user and source model before
    applying the configured weighted sum. Missing source scores contribute
    zero. Source rank provides deterministic tie-breaking.

    Args:
        collaborative: ALS candidates with collaborative_score.
        semantic: Dense candidates with semantic_score.
        config: Semantic blend weight and output size.

    Returns:
        Ranked hybrid recommendations and both source components.

    Raises:
        ValueError: If configuration or candidate data is invalid.
    """
    config = config or HybridConfig()

    if not 0 <= config.semantic_weight <= 1:
        raise ValueError(
            "semantic_weight must be between zero and one."
        )
    if config.k <= 0:
        raise ValueError("k must be greater than zero.")

    collaborative_prepared = _prepare_candidates(
        collaborative,
        score_column="collaborative_score",
        prefix="collaborative",
    )
    semantic_prepared = _prepare_candidates(
        semantic,
        score_column="semantic_score",
        prefix="semantic",
    )

    candidates = collaborative_prepared.merge(
        semantic_prepared,
        on=["user_id", "item_id"],
        how="outer",
        validate="one_to_one",
    )

    if candidates.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    candidates["collaborative_normalized"] = candidates[
        "collaborative_normalized"
    ].fillna(0.0)
    candidates["semantic_normalized"] = candidates[
        "semantic_normalized"
    ].fillna(0.0)

    candidates["hybrid_score"] = (
        (1.0 - config.semantic_weight)
        * candidates["collaborative_normalized"]
        + config.semantic_weight
        * candidates["semantic_normalized"]
    )

    candidates["_collaborative_order"] = candidates[
        "collaborative_rank"
    ].fillna(np.inf)
    candidates["_semantic_order"] = candidates[
        "semantic_rank"
    ].fillna(np.inf)

    if config.semantic_weight >= 0.5:
        source_order = [
            "_semantic_order",
            "_collaborative_order",
        ]
    else:
        source_order = [
            "_collaborative_order",
            "_semantic_order",
        ]

    candidates = candidates.sort_values(
        ["user_id", "hybrid_score", *source_order, "item_id"],
        ascending=[True, False, True, True, True],
    )

    candidates["rank"] = (
        candidates.groupby("user_id").cumcount() + 1
    )
    candidates = candidates.loc[
        candidates["rank"] <= config.k
    ]

    return candidates[RESULT_COLUMNS].reset_index(drop=True)