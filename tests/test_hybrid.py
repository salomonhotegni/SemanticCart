import numpy as np
import pandas as pd
import pytest

from semanticcart.hybrid import (
    HybridConfig,
    RESULT_COLUMNS,
    rank_hybrid_candidates,
)


@pytest.fixture
def candidate_lists():
    collaborative = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "item_id": ["a", "b", "c"],
            "rank": [1, 2, 3],
            "collaborative_score": [100.0, 60.0, 0.0],
        }
    )

    semantic = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "item_id": ["c", "b", "d"],
            "rank": [1, 2, 3],
            "semantic_score": [0.9, 0.8, 0.7],
        }
    )

    return collaborative, semantic


@pytest.mark.parametrize(
    ("semantic_weight", "expected_items"),
    [
        (0.0, ["a", "b", "c"]),
        (1.0, ["c", "b", "d"]),
    ],
)
def test_endpoint_weights_reproduce_source_rankings(
    candidate_lists,
    semantic_weight,
    expected_items,
):
    collaborative, semantic = candidate_lists

    recommendations = rank_hybrid_candidates(
        collaborative,
        semantic,
        HybridConfig(
            semantic_weight=semantic_weight,
            k=3,
        ),
    )

    assert recommendations["item_id"].tolist() == expected_items
    assert recommendations["rank"].tolist() == [1, 2, 3]


def test_agreement_between_models_improves_rank(candidate_lists):
    collaborative, semantic = candidate_lists

    recommendations = rank_hybrid_candidates(
        collaborative,
        semantic,
        HybridConfig(semantic_weight=0.5, k=3),
    )

    assert recommendations.iloc[0]["item_id"] == "b"
    assert recommendations.iloc[0]["hybrid_score"] == pytest.approx(
        0.55
    )


def test_normalization_removes_raw_score_scale(candidate_lists):
    collaborative, semantic = candidate_lists

    baseline = rank_hybrid_candidates(
        collaborative,
        semantic,
        HybridConfig(semantic_weight=0.4, k=3),
    )

    scaled = collaborative.copy()
    scaled["collaborative_score"] = (
        scaled["collaborative_score"] * 1_000 + 7
    )

    scaled_result = rank_hybrid_candidates(
        scaled,
        semantic,
        HybridConfig(semantic_weight=0.4, k=3),
    )

    assert scaled_result["item_id"].tolist() == (
        baseline["item_id"].tolist()
    )
    assert scaled_result["hybrid_score"].tolist() == pytest.approx(
        baseline["hybrid_score"].tolist()
    )


def test_duplicate_candidates_keep_best_rank(candidate_lists):
    collaborative, semantic = candidate_lists

    duplicate = pd.DataFrame(
        {
            "user_id": ["u1"],
            "item_id": ["b"],
            "rank": [99],
            "collaborative_score": [999.0],
        }
    )

    duplicated = pd.concat(
        [collaborative, duplicate],
        ignore_index=True,
    )

    recommendations = rank_hybrid_candidates(
        duplicated,
        semantic,
        HybridConfig(semantic_weight=0.0, k=3),
    )

    assert recommendations["item_id"].tolist() == ["a", "b", "c"]
    assert recommendations["item_id"].is_unique


def test_union_preserves_source_components(candidate_lists):
    collaborative, semantic = candidate_lists

    recommendations = rank_hybrid_candidates(
        collaborative,
        semantic,
        HybridConfig(semantic_weight=0.5, k=4),
    ).set_index("item_id")

    assert np.isnan(recommendations.loc["a", "semantic_score"])
    assert np.isnan(
        recommendations.loc["d", "collaborative_score"]
    )
    assert recommendations.loc[
        "a", "semantic_normalized"
    ] == 0.0
    assert recommendations.loc[
        "d", "collaborative_normalized"
    ] == 0.0


def test_empty_candidates_return_expected_schema(candidate_lists):
    collaborative, semantic = candidate_lists

    recommendations = rank_hybrid_candidates(
        collaborative.iloc[0:0],
        semantic.iloc[0:0],
    )

    assert recommendations.empty
    assert recommendations.columns.tolist() == RESULT_COLUMNS


@pytest.mark.parametrize(
    "config",
    [
        HybridConfig(semantic_weight=-0.1),
        HybridConfig(semantic_weight=1.1),
        HybridConfig(k=0),
    ],
)
def test_rejects_invalid_configuration(
    candidate_lists,
    config,
):
    collaborative, semantic = candidate_lists

    with pytest.raises(ValueError):
        rank_hybrid_candidates(
            collaborative,
            semantic,
            config,
        )


def test_rejects_non_finite_scores(candidate_lists):
    collaborative, semantic = candidate_lists
    collaborative = collaborative.copy()
    collaborative.loc[0, "collaborative_score"] = np.inf

    with pytest.raises(ValueError, match="must be finite"):
        rank_hybrid_candidates(
            collaborative,
            semantic,
        )