import numpy as np
import pandas as pd
import pytest
from dataclasses import replace

from semanticcart.diversity import (
    DIVERSITY_RESULT_COLUMNS,
    DiversityMetrics,
    DiversityRerankConfig,
    evaluate_diversity,
    rerank_diverse_candidates,
)


@pytest.fixture
def item_features() -> pd.DataFrame:
    """Build a small catalogue with controlled feature relationships."""
    return pd.DataFrame(
        {
            "item_id": ["a", "b", "c", "d"],
            "embedding": [
                np.array([1.0, 0.0]),
                np.array([1.0, 0.0]),
                np.array([0.0, 1.0]),
                np.array([-1.0, 0.0]),
            ],
            "categories": [
                "Action",
                "Action",
                "Puzzle",
                "Strategy",
            ],
            "price": [10.0, 20.0, 40.0, 80.0],
            "popularity": [90, 9, 0, 0],
        }
    )


@pytest.fixture
def recommendations() -> pd.DataFrame:
    """Build two recommendation lists with known diversity values."""
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2"],
            "item_id": ["a", "b", "d", "a", "c"],
            "rank": [1, 2, 3, 1, 2],
        }
    )


def test_computes_expected_metrics(
    recommendations: pd.DataFrame,
    item_features: pd.DataFrame,
) -> None:
    metrics = evaluate_diversity(
        recommendations,
        item_features,
        k=2,
    )

    expected_novelty = np.mean(
        [
            -np.log2(91 / 103),
            -np.log2(10 / 103),
            -np.log2(91 / 103),
            -np.log2(1 / 103),
        ]
    )
    expected_price_dispersion = np.mean(
        [
            abs(np.log1p(10) - np.log1p(20)),
            abs(np.log1p(10) - np.log1p(40)),
        ]
    )

    assert metrics.intra_list_diversity == pytest.approx(0.5)
    assert metrics.category_variety == pytest.approx(0.75)
    assert metrics.category_coverage == pytest.approx(2 / 3)
    assert metrics.novelty == pytest.approx(expected_novelty)
    assert metrics.price_dispersion == pytest.approx(
        expected_price_dispersion
    )


def test_excludes_items_below_k(
    recommendations: pd.DataFrame,
    item_features: pd.DataFrame,
) -> None:
    metrics = evaluate_diversity(
        recommendations,
        item_features,
        k=2,
    )

    assert metrics.category_coverage == pytest.approx(2 / 3)


def test_empty_recommendations_return_zero_metrics(
    item_features: pd.DataFrame,
) -> None:
    recommendations = pd.DataFrame(
        columns=["user_id", "item_id", "rank"]
    )

    metrics = evaluate_diversity(
        recommendations,
        item_features,
    )

    assert metrics == DiversityMetrics(
        intra_list_diversity=0.0,
        category_variety=0.0,
        category_coverage=0.0,
        novelty=0.0,
        price_dispersion=0.0,
    )


def test_rejects_missing_item_features(
    item_features: pd.DataFrame,
) -> None:
    recommendations = pd.DataFrame(
        {
            "user_id": ["u1"],
            "item_id": ["missing"],
            "rank": [1],
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing features for items",
    ):
        evaluate_diversity(
            recommendations,
            item_features,
        )


def test_rejects_duplicate_ranks(
    item_features: pd.DataFrame,
) -> None:
    recommendations = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "item_id": ["a", "b"],
            "rank": [1, 1],
        }
    )

    with pytest.raises(
        ValueError,
        match="only one item at each rank",
    ):
        evaluate_diversity(
            recommendations,
            item_features,
        )


def test_rejects_duplicate_feature_ids(
    recommendations: pd.DataFrame,
    item_features: pd.DataFrame,
) -> None:
    duplicated = pd.concat(
        [item_features, item_features.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="Item feature IDs must be unique",
    ):
        evaluate_diversity(
            recommendations,
            duplicated,
        )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        (
            "embedding",
            np.array([0.0, 0.0]),
            "zero vectors",
        ),
        (
            "price",
            -1.0,
            "Prices must be non-negative",
        ),
        (
            "popularity",
            -1.0,
            "Popularity values must be finite",
        ),
    ],
)
def test_rejects_invalid_item_features(
    recommendations: pd.DataFrame,
    item_features: pd.DataFrame,
    column: str,
    value: object,
    message: str,
) -> None:
    invalid = item_features.copy()
    invalid.at[0, column] = value

    with pytest.raises(ValueError, match=message):
        evaluate_diversity(
            recommendations,
            invalid,
        )


@pytest.mark.parametrize("rank", [0, -1, np.nan])
def test_rejects_invalid_ranks(
    item_features: pd.DataFrame,
    rank: float,
) -> None:
    recommendations = pd.DataFrame(
        {
            "user_id": ["u1"],
            "item_id": ["a"],
            "rank": [rank],
        }
    )

    with pytest.raises(
        ValueError,
        match="Ranks must be finite and positive",
    ):
        evaluate_diversity(
            recommendations,
            item_features,
        )


@pytest.fixture
def rerank_candidates() -> pd.DataFrame:
    """Build candidates with decreasing relevance."""
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "item_id": ["a", "b", "c"],
            "rank": [1, 2, 3],
            "score": [1.0, 0.9, 0.8],
        }
    )


def test_relevance_only_preserves_source_order(
    rerank_candidates: pd.DataFrame,
    item_features: pd.DataFrame,
) -> None:
    result = rerank_diverse_candidates(
        rerank_candidates,
        item_features,
        config=DiversityRerankConfig(
            k=3,
            relevance_weight=1.0,
            novelty_weight=0.0,
        ),
        score_column="score",
    )

    assert result["item_id"].tolist() == ["a", "b", "c"]
    assert result["rank"].tolist() == [1, 2, 3]
    assert result["source_rank"].tolist() == [1, 2, 3]


def test_mmr_replaces_redundant_candidate(
    rerank_candidates: pd.DataFrame,
    item_features: pd.DataFrame,
) -> None:
    result = rerank_diverse_candidates(
        rerank_candidates,
        item_features,
        config=DiversityRerankConfig(
            k=2,
            relevance_weight=0.5,
            novelty_weight=0.0,
        ),
        score_column="score",
    )

    assert result["item_id"].tolist() == ["a", "c"]
    assert result["redundancy_penalty"].iloc[0] == 0.0
    assert result["redundancy_penalty"].iloc[1] > 0.0


def test_novelty_can_promote_unpopular_item(
    item_features: pd.DataFrame,
) -> None:
    candidates = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "item_id": ["a", "c"],
            "rank": [1, 2],
            "score": [1.0, 1.0],
        }
    )

    result = rerank_diverse_candidates(
        candidates,
        item_features,
        config=DiversityRerankConfig(
            k=2,
            relevance_weight=1.0,
            novelty_weight=1.0,
        ),
        score_column="score",
    )

    assert result["item_id"].tolist() == ["c", "a"]
    assert (
        result["novelty_normalized"].iloc[0]
        > result["novelty_normalized"].iloc[1]
    )


def test_category_redundancy_changes_selection(
    rerank_candidates: pd.DataFrame,
    item_features: pd.DataFrame,
) -> None:
    result = rerank_diverse_candidates(
        rerank_candidates,
        item_features,
        config=DiversityRerankConfig(
            k=2,
            relevance_weight=0.4,
            novelty_weight=0.0,
            semantic_similarity_weight=0.0,
            category_similarity_weight=1.0,
            price_similarity_weight=0.0,
        ),
        score_column="score",
    )

    assert result["item_id"].tolist() == ["a", "c"]


def test_price_redundancy_changes_selection(
    rerank_candidates: pd.DataFrame,
    item_features: pd.DataFrame,
) -> None:
    result = rerank_diverse_candidates(
        rerank_candidates,
        item_features,
        config=DiversityRerankConfig(
            k=2,
            relevance_weight=0.3,
            novelty_weight=0.0,
            semantic_similarity_weight=0.0,
            category_similarity_weight=0.0,
            price_similarity_weight=1.0,
        ),
        score_column="score",
    )

    assert result["item_id"].tolist() == ["a", "c"]


def test_missing_optional_metadata_is_neutral(
    rerank_candidates: pd.DataFrame,
    item_features: pd.DataFrame,
) -> None:
    features = item_features.copy()
    features.loc[
        features["item_id"] == "b",
        ["categories", "price"],
    ] = ["", np.nan]

    result = rerank_diverse_candidates(
        rerank_candidates,
        features,
        config=DiversityRerankConfig(k=2),
        score_column="score",
    )

    assert len(result) == 2
    assert result["item_id"].is_unique


def test_empty_candidates_return_stable_schema(
    item_features: pd.DataFrame,
) -> None:
    candidates = pd.DataFrame(
        columns=["user_id", "item_id", "rank", "score"]
    )

    result = rerank_diverse_candidates(
        candidates,
        item_features,
        score_column="score",
    )

    assert result.empty
    assert result.columns.tolist() == DIVERSITY_RESULT_COLUMNS


@pytest.mark.parametrize(
    "changes",
    [
        {"k": 0},
        {"relevance_weight": -0.1},
        {"novelty_weight": 1.1},
        {"semantic_similarity_weight": -0.1},
        {
            "semantic_similarity_weight": 0.5,
            "category_similarity_weight": 0.2,
            "price_similarity_weight": 0.1,
        },
    ],
)
def test_rejects_invalid_rerank_configuration(
    rerank_candidates: pd.DataFrame,
    item_features: pd.DataFrame,
    changes: dict,
) -> None:
    config = replace(
        DiversityRerankConfig(),
        **changes,
    )

    with pytest.raises(ValueError):
        rerank_diverse_candidates(
            rerank_candidates,
            item_features,
            config=config,
            score_column="score",
        )


def test_rejects_unknown_candidate_items(
    rerank_candidates: pd.DataFrame,
    item_features: pd.DataFrame,
) -> None:
    candidates = rerank_candidates.copy()
    candidates.loc[0, "item_id"] = "missing"

    with pytest.raises(
        ValueError,
        match="Missing features for items",
    ):
        rerank_diverse_candidates(
            candidates,
            item_features,
            score_column="score",
        )


def test_rejects_duplicate_candidate_ranks(
    item_features: pd.DataFrame,
) -> None:
    candidates = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "item_id": ["a", "b"],
            "rank": [1, 1],
            "score": [1.0, 0.9],
        }
    )

    with pytest.raises(
        ValueError,
        match="only one candidate at each rank",
    ):
        rerank_diverse_candidates(
            candidates,
            item_features,
            score_column="score",
        )