"""Evaluate diversity, novelty, and price spread in recommendation lists."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DiversityMetrics:
    """Aggregate beyond-accuracy metrics for Top-K recommendations.

    Attributes:
        intra_list_diversity: Mean pairwise cosine distance within each list.
        category_variety: Mean fraction of distinct categories per list.
        category_coverage: Fraction of catalogue categories surfaced.
        novelty: Mean self-information of recommended-item popularity.
        price_dispersion: Mean pairwise log-price distance within each list.
    """

    intra_list_diversity: float
    category_variety: float
    category_coverage: float
    novelty: float
    price_dispersion: float


def _mean_pairwise(values: np.ndarray) -> float:
    """Return the mean upper-triangle value of a square matrix."""
    if len(values) < 2:
        return 0.0

    upper = np.triu_indices(len(values), k=1)
    return float(values[upper].mean())


def _validate_features(item_features: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize catalogue features used by the metrics."""
    required = {
        "item_id",
        "embedding",
        "categories",
        "price",
        "popularity",
    }
    missing = required - set(item_features.columns)

    if missing:
        raise ValueError(
            f"Missing item feature columns: {sorted(missing)}"
        )

    features = item_features[list(required)].copy()
    features["item_id"] = features["item_id"].astype(str)

    if features.empty:
        raise ValueError("Item features cannot be empty.")
    if features["item_id"].duplicated().any():
        raise ValueError("Item feature IDs must be unique.")

    vectors = [
        np.asarray(vector, dtype=np.float32).reshape(-1)
        for vector in features["embedding"]
    ]
    dimensions = {len(vector) for vector in vectors}

    if len(dimensions) != 1:
        raise ValueError("Embeddings must share one dimension.")

    matrix = np.vstack(vectors)

    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings must be finite.")

    norms = np.linalg.norm(matrix, axis=1)

    if np.any(norms == 0):
        raise ValueError("Embeddings cannot contain zero vectors.")

    features["embedding"] = list(
        matrix / norms[:, np.newaxis]
    )

    features["categories"] = (
        features["categories"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    original_prices = features["price"]
    features["price"] = pd.to_numeric(
        original_prices,
        errors="coerce",
    )

    invalid_prices = (
        original_prices.notna()
        & features["price"].isna()
    )
    if invalid_prices.any() or (
        features["price"].dropna() < 0
    ).any():
        raise ValueError(
            "Prices must be non-negative numeric values or missing."
        )

    features["popularity"] = pd.to_numeric(
        features["popularity"],
        errors="coerce",
    )

    popularity = features["popularity"].to_numpy()

    if (
        np.isnan(popularity).any()
        or not np.isfinite(popularity).all()
        or np.any(popularity < 0)
    ):
        raise ValueError(
            "Popularity values must be finite and non-negative."
        )

    return features


def evaluate_diversity(
    recommendations: pd.DataFrame,
    item_features: pd.DataFrame,
    k: int = 10,
) -> DiversityMetrics:
    """Evaluate beyond-accuracy properties of recommendation lists.

    Popularity novelty uses additive-one smoothing. Price dispersion is the
    mean absolute difference between log-transformed prices, which prevents
    a few expensive products from dominating the metric.

    Args:
        recommendations: Rows with user_id, item_id, and one-based rank.
        item_features: Catalogue features containing normalized metadata,
            popularity counts, and dense product embeddings.
        k: Largest recommendation rank included in evaluation.

    Returns:
        User-averaged diversity metrics and global category coverage.

    Raises:
        ValueError: If inputs, ranks, metadata, or embeddings are invalid.
    """
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    required = {"user_id", "item_id", "rank"}
    missing = required - set(recommendations.columns)

    if missing:
        raise ValueError(
            f"Missing recommendation columns: {sorted(missing)}"
        )

    features = _validate_features(item_features)

    recs = recommendations[
        ["user_id", "item_id", "rank"]
    ].copy()
    recs["user_id"] = recs["user_id"].astype(str)
    recs["item_id"] = recs["item_id"].astype(str)
    recs["rank"] = pd.to_numeric(
        recs["rank"],
        errors="coerce",
    )

    ranks = recs["rank"].to_numpy()

    if (
        np.isnan(ranks).any()
        or not np.isfinite(ranks).all()
        or np.any(ranks <= 0)
    ):
        raise ValueError("Ranks must be finite and positive.")

    recs = (
        recs.loc[recs["rank"] <= k]
        .sort_values(["user_id", "rank", "item_id"])
        .drop_duplicates(["user_id", "item_id"], keep="first")
    )

    if recs.duplicated(["user_id", "rank"]).any():
        raise ValueError(
            "Each user can have only one item at each rank."
        )

    if recs.empty:
        return DiversityMetrics(0.0, 0.0, 0.0, 0.0, 0.0)

    unknown_items = pd.Index(
        recs["item_id"].unique()
    ).difference(pd.Index(features["item_id"]))

    if len(unknown_items):
        raise ValueError(
            f"Missing features for items: {unknown_items[:5].tolist()}"
        )

    total_popularity = features["popularity"].sum()
    novelty_denominator = total_popularity + len(features)

    features["novelty"] = -np.log2(
        (features["popularity"] + 1.0)
        / novelty_denominator
    )

    merged = recs.merge(
        features,
        on="item_id",
        how="left",
        validate="many_to_one",
    )

    semantic_values = []
    category_values = []
    price_values = []

    for _, group in merged.groupby("user_id", sort=False):
        vectors = np.vstack(group["embedding"])
        similarity = np.clip(vectors @ vectors.T, -1.0, 1.0)
        semantic_values.append(
            _mean_pairwise(1.0 - similarity)
        )

        categories = group.loc[
            group["categories"].ne(""),
            "categories",
        ]
        category_values.append(
            categories.nunique() / len(group)
        )

        prices = group["price"].dropna().to_numpy()

        if len(prices) < 2:
            price_values.append(0.0)
        else:
            log_prices = np.log1p(prices)
            differences = np.abs(
                log_prices[:, None] - log_prices[None, :]
            )
            price_values.append(
                _mean_pairwise(differences)
            )

    catalogue_categories = set(
        features.loc[
            features["categories"].ne(""),
            "categories",
        ]
    )
    recommended_categories = set(
        merged.loc[
            merged["categories"].ne(""),
            "categories",
        ]
    )
    category_coverage = (
        len(recommended_categories) / len(catalogue_categories)
        if catalogue_categories
        else 0.0
    )

    return DiversityMetrics(
        intra_list_diversity=float(np.mean(semantic_values)),
        category_variety=float(np.mean(category_values)),
        category_coverage=float(category_coverage),
        novelty=float(merged["novelty"].mean()),
        price_dispersion=float(np.mean(price_values)),
    )