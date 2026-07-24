import pandas as pd
import pytest

from semanticcart.dense_index import DenseItemIndex, HnswConfig
from semanticcart.dense_profiles import (
    DenseProfileConfig,
    DenseUserProfiles,
)
from semanticcart.dense_semantic import (
    DenseSemanticConfig,
    DenseSemanticRecommender,
)


@pytest.fixture(scope="module")
def semantic_model():
    """Build a small warm-user semantic recommender."""
    catalog = pd.DataFrame(
        {
            "item_id": ["a", "b", "c", "d", "e"],
            "embedding": [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.95, 0.05],
                [0.05, 0.95],
                [-1.0, 0.0],
            ],
        }
    )

    item_index = DenseItemIndex.from_catalog(
        catalog,
        HnswConfig(
            connections=8,
            ef_construction=40,
            ef_search=32,
        ),
    )

    interactions = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "item_id": ["b", "a", "a", "b"],
            "timestamp": [1, 2, 1, 2],
        }
    )

    profiles = DenseUserProfiles.build(
        item_index,
        interactions,
        DenseProfileConfig(recency_decay=0.1),
    )

    model = DenseSemanticRecommender(
        profiles,
        DenseSemanticConfig(
            batch_size=1,
            candidate_multiplier=2,
        ),
    )

    return model, interactions


def test_returns_ranked_unseen_products(semantic_model):
    model, interactions = semantic_model

    recommendations = model.recommend_for_users(
        ["u1", "u2"],
        k=2,
    )

    assert len(recommendations) == 4

    for user_id, user_recommendations in recommendations.groupby(
        "user_id",
        sort=False,
    ):
        seen_items = set(
            interactions.loc[
                interactions["user_id"] == user_id,
                "item_id",
            ]
        )

        assert user_recommendations["rank"].tolist() == [1, 2]
        assert set(
            user_recommendations["item_id"]
        ).isdisjoint(seen_items)
        assert user_recommendations[
            "semantic_score"
        ].is_monotonic_decreasing


def test_recent_item_drives_first_recommendation(semantic_model):
    model, _ = semantic_model

    recommendations = model.recommend_for_users(
        ["u1", "u2"],
        k=1,
    )

    first_items = recommendations.set_index("user_id")[
        "item_id"
    ].to_dict()

    assert first_items == {
        "u1": "c",
        "u2": "d",
    }


def test_duplicate_requested_users_are_evaluated_once(
    semantic_model,
):
    model, _ = semantic_model

    recommendations = model.recommend_for_users(
        ["u1", "u1"],
        k=2,
    )

    assert recommendations["user_id"].unique().tolist() == ["u1"]
    assert len(recommendations) == 2


def test_returns_available_unseen_products_when_k_is_large(
    semantic_model,
):
    model, _ = semantic_model

    recommendations = model.recommend_for_users(
        ["u1"],
        k=10,
    )

    assert len(recommendations) == 3
    assert recommendations["rank"].tolist() == [1, 2, 3]


def test_empty_user_request_returns_empty_frame(semantic_model):
    model, _ = semantic_model

    recommendations = model.recommend_for_users([], k=2)

    assert recommendations.empty
    assert recommendations.columns.tolist() == model.RESULT_COLUMNS


def test_rejects_unknown_users(semantic_model):
    model, _ = semantic_model

    with pytest.raises(ValueError, match="Unknown users"):
        model.recommend_for_users(["new-user"], k=2)


def test_rejects_invalid_retrieval_configuration(semantic_model):
    model, _ = semantic_model

    with pytest.raises(ValueError, match="batch_size"):
        DenseSemanticRecommender(
            model.profiles,
            DenseSemanticConfig(batch_size=0),
        )

    with pytest.raises(ValueError, match="greater than zero"):
        model.recommend_for_users(["u1"], k=0)
        

@pytest.mark.parametrize(
    ("history", "expected_first_item"),
    [
        (["b", "a"], "c"),
        (["a", "b"], "d"),
    ],
)
def test_session_recency_changes_first_recommendation(
    semantic_model,
    history,
    expected_first_item,
):
    model, _ = semantic_model

    recommendations = model.recommend_from_history(
        history,
        k=2,
        recency_decay=0.1,
    )

    assert recommendations.iloc[0]["item_id"] == expected_first_item


def test_session_filters_viewed_products(semantic_model):
    model, _ = semantic_model
    history = ["b", "a"]

    recommendations = model.recommend_from_history(
        history,
        k=2,
        recency_decay=0.1,
    )

    assert recommendations["rank"].tolist() == [1, 2]
    assert set(recommendations["item_id"]).isdisjoint(history)
    assert recommendations[
        "semantic_score"
    ].is_monotonic_decreasing


def test_repeated_session_items_keep_latest_position(
    semantic_model,
):
    model, _ = semantic_model

    baseline = model.recommend_from_history(
        ["b", "a"],
        k=2,
        recency_decay=0.1,
    )
    repeated = model.recommend_from_history(
        ["a", "b", "a"],
        k=2,
        recency_decay=0.1,
    )

    assert repeated["item_id"].tolist() == baseline["item_id"].tolist()
    assert repeated["semantic_score"].tolist() == pytest.approx(
        baseline["semantic_score"].tolist()
    )


def test_session_returns_only_available_unseen_products(
    semantic_model,
):
    model, _ = semantic_model

    recommendations = model.recommend_from_history(
        ["a", "b", "c", "d"],
        k=10,
    )

    assert recommendations["item_id"].tolist() == ["e"]
    assert recommendations["rank"].tolist() == [1]


def test_session_returns_empty_frame_when_every_item_was_viewed(
    semantic_model,
):
    model, _ = semantic_model

    recommendations = model.recommend_from_history(
        ["a", "b", "c", "d", "e"],
        k=2,
    )

    assert recommendations.empty
    assert recommendations.columns.tolist() == (
        model.SESSION_RESULT_COLUMNS
    )


def test_session_rejects_invalid_inputs(semantic_model):
    model, _ = semantic_model

    with pytest.raises(ValueError, match="At least one"):
        model.recommend_from_history([], k=2)

    with pytest.raises(ValueError, match="Unknown products"):
        model.recommend_from_history(["unknown"], k=2)

    with pytest.raises(ValueError, match="between zero and one"):
        model.recommend_from_history(
            ["a"],
            k=2,
            recency_decay=0,
        )