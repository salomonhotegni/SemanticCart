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


@dataclass(frozen=True)
class DiversityRerankConfig:
    """Configure relevance, novelty, and redundancy-aware reranking.

    Attributes:
        k: Maximum recommendations selected per user.
        relevance_weight: MMR tradeoff between utility and redundancy.
        novelty_weight: Popularity novelty contribution to item utility.
        semantic_similarity_weight: Semantic redundancy contribution.
        category_similarity_weight: Category redundancy contribution.
        price_similarity_weight: Price redundancy contribution.
    """

    k: int = 10
    relevance_weight: float = 0.90
    novelty_weight: float = 0.05
    semantic_similarity_weight: float = 0.70
    category_similarity_weight: float = 0.20
    price_similarity_weight: float = 0.10


DIVERSITY_RESULT_COLUMNS = [
    "user_id",
    "item_id",
    "rank",
    "diversity_score",
    "relevance_score",
    "relevance_normalized",
    "novelty_normalized",
    "redundancy_penalty",
    "source_rank",
]


def _validate_rerank_config(
    config: DiversityRerankConfig,
) -> None:
    """Reject invalid reranking weights and output sizes."""
    if config.k <= 0:
        raise ValueError("k must be greater than zero.")

    bounded_weights = {
        "relevance_weight": config.relevance_weight,
        "novelty_weight": config.novelty_weight,
    }

    for name, value in bounded_weights.items():
        if not 0 <= value <= 1:
            raise ValueError(
                f"{name} must be between zero and one."
            )

    similarity_weights = np.asarray(
        [
            config.semantic_similarity_weight,
            config.category_similarity_weight,
            config.price_similarity_weight,
        ],
        dtype=np.float64,
    )

    if (
        not np.isfinite(similarity_weights).all()
        or np.any(similarity_weights < 0)
    ):
        raise ValueError(
            "Similarity weights must be finite and non-negative."
        )

    if not np.isclose(similarity_weights.sum(), 1.0):
        raise ValueError(
            "Similarity weights must sum to one."
        )


def _prepare_rerank_candidates(
    candidates: pd.DataFrame,
    item_features: pd.DataFrame,
    score_column: str,
) -> pd.DataFrame:
    """Validate candidates and attach normalized reranking features."""
    required = {
        "user_id",
        "item_id",
        "rank",
        score_column,
    }
    missing = required - set(candidates.columns)

    if missing:
        raise ValueError(
            f"Missing candidate columns: {sorted(missing)}"
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

    numeric_values = prepared[
        ["rank", score_column]
    ].to_numpy()

    if (
        np.isnan(numeric_values).any()
        or not np.isfinite(numeric_values).all()
    ):
        raise ValueError(
            "Candidate ranks and relevance scores must be finite."
        )
    if (prepared["rank"] <= 0).any():
        raise ValueError(
            "Candidate ranks must be positive."
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

    if prepared.duplicated(["user_id", "rank"]).any():
        raise ValueError(
            "Each user can have only one candidate at each rank."
        )

    features = _validate_features(item_features)

    unknown_items = pd.Index(
        prepared["item_id"].unique()
    ).difference(pd.Index(features["item_id"]))

    if len(unknown_items):
        raise ValueError(
            f"Missing features for items: {unknown_items[:5].tolist()}"
        )

    maximum_popularity = features["popularity"].max()

    if maximum_popularity > 0:
        features["novelty_normalized"] = (
            1.0
            - np.log1p(features["popularity"])
            / np.log1p(maximum_popularity)
        )
    else:
        features["novelty_normalized"] = 1.0

    merged = prepared.merge(
        features,
        on="item_id",
        how="left",
        validate="many_to_one",
    )

    grouped_scores = merged.groupby("user_id")[
        score_column
    ]
    score_minimum = grouped_scores.transform("min")
    score_maximum = grouped_scores.transform("max")
    score_range = score_maximum - score_minimum

    merged["relevance_normalized"] = (
        (merged[score_column] - score_minimum)
        / score_range.where(score_range > 0)
    ).fillna(1.0)

    return merged.rename(
        columns={score_column: "relevance_score"}
    )


def _pairwise_redundancy(
    group: pd.DataFrame,
    config: DiversityRerankConfig,
) -> np.ndarray:
    """Build pairwise redundancy from available product signals."""
    vectors = np.vstack(group["embedding"])
    cosine_similarity = np.clip(
        vectors @ vectors.T,
        -1.0,
        1.0,
    )
    semantic_similarity = (
        cosine_similarity + 1.0
    ) / 2.0

    weighted_similarity = (
        config.semantic_similarity_weight
        * semantic_similarity
    )
    available_weight = np.full(
        semantic_similarity.shape,
        config.semantic_similarity_weight,
        dtype=np.float32,
    )

    categories = group["categories"].to_numpy()
    category_available = (
        categories[:, None].astype(bool)
        & categories[None, :].astype(bool)
    )
    category_similarity = (
        categories[:, None] == categories[None, :]
    )

    weighted_similarity += (
        config.category_similarity_weight
        * category_similarity
        * category_available
    )
    available_weight += (
        config.category_similarity_weight
        * category_available
    )

    prices = group["price"].to_numpy(dtype=np.float64)
    price_available = (
        np.isfinite(prices[:, None])
        & np.isfinite(prices[None, :])
    )
    log_prices = np.log1p(prices)
    price_similarity = np.exp(
        -np.abs(
            log_prices[:, None] - log_prices[None, :]
        )
    )
    price_similarity = np.where(
        price_available,
        price_similarity,
        0.0,
    )

    weighted_similarity += (
        config.price_similarity_weight
        * price_similarity
    )
    available_weight += (
        config.price_similarity_weight
        * price_available
    )

    redundancy = np.divide(
        weighted_similarity,
        available_weight,
        out=np.zeros_like(
            weighted_similarity,
            dtype=np.float32,
        ),
        where=available_weight > 0,
    )

    np.fill_diagonal(redundancy, 1.0)
    return redundancy


def _rerank_one_user(
    group: pd.DataFrame,
    config: DiversityRerankConfig,
) -> list[dict]:
    """Greedily select one user's recommendations with MMR."""
    group = group.reset_index(drop=True)
    redundancy = _pairwise_redundancy(
        group,
        config,
    )

    relevance = group[
        "relevance_normalized"
    ].to_numpy(dtype=np.float64)
    novelty = group[
        "novelty_normalized"
    ].to_numpy(dtype=np.float64)

    utility = (
        (1.0 - config.novelty_weight) * relevance
        + config.novelty_weight * novelty
    )

    source_ranks = group["rank"].to_numpy()
    item_ids = group["item_id"].to_numpy()
    remaining = np.ones(len(group), dtype=bool)
    maximum_redundancy = np.zeros(
        len(group),
        dtype=np.float64,
    )

    records = []
    output_size = min(config.k, len(group))

    for output_rank in range(1, output_size + 1):
        diversity_scores = (
            config.relevance_weight * utility
            - (1.0 - config.relevance_weight)
            * maximum_redundancy
        )

        available = np.flatnonzero(remaining)
        order = np.lexsort(
            (
                item_ids[available],
                source_ranks[available],
                -diversity_scores[available],
            )
        )
        selected = available[order[0]]

        records.append(
            {
                "user_id": group.at[selected, "user_id"],
                "item_id": item_ids[selected],
                "rank": output_rank,
                "diversity_score": diversity_scores[selected],
                "relevance_score": group.at[
                    selected,
                    "relevance_score",
                ],
                "relevance_normalized": relevance[selected],
                "novelty_normalized": novelty[selected],
                "redundancy_penalty": (
                    maximum_redundancy[selected]
                ),
                "source_rank": source_ranks[selected],
            }
        )

        remaining[selected] = False
        maximum_redundancy = np.maximum(
            maximum_redundancy,
            redundancy[:, selected],
        )

    return records


def rerank_diverse_candidates(
    candidates: pd.DataFrame,
    item_features: pd.DataFrame,
    config: DiversityRerankConfig | None = None,
    score_column: str = "hybrid_score",
) -> pd.DataFrame:
    """Select relevance-aware, novel, and non-redundant products.

    The function applies maximal marginal relevance independently to each
    user. Relevance and popularity novelty determine item utility, while
    semantic, category, and price similarity determine redundancy.

    Args:
        candidates: Deeper ranked candidate pool with a relevance score.
        item_features: Catalogue embeddings, metadata, and popularity counts.
        config: Optional reranking weights and final output size.
        score_column: Candidate column containing ranking relevance.

    Returns:
        Top-K diversified recommendations with auditable score components.

    Raises:
        ValueError: If configuration, candidates, or features are invalid.
    """
    config = config or DiversityRerankConfig()
    _validate_rerank_config(config)

    prepared = _prepare_rerank_candidates(
        candidates,
        item_features,
        score_column,
    )

    if prepared.empty:
        return pd.DataFrame(
            columns=DIVERSITY_RESULT_COLUMNS
        )

    records = []

    for _, group in prepared.groupby(
        "user_id",
        sort=False,
    ):
        records.extend(
            _rerank_one_user(group, config)
        )

    return pd.DataFrame.from_records(
        records,
        columns=DIVERSITY_RESULT_COLUMNS,
    )