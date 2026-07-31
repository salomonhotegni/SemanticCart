import pandas as pd
import pytest
import numpy as np

from semanticcart.collaborative import ALSConfig, ALSRecommender


@pytest.fixture(scope="module")
def trained_model():
    interactions = pd.DataFrame(
        {
            "user_id": [
                "u1", "u1", "u1",
                "u2", "u2",
                "u3", "u3",
                "u4", "u4",
                "u5", "u5",
            ],
            "item_id": [
                "a", "a", "b",
                "b", "c",
                "c", "d",
                "d", "e",
                "e", "a",
            ],
        }
    )

    config = ALSConfig(
        factors=4,
        regularization=0.1,
        alpha=10.0,
        iterations=3,
        random_state=42,
        batch_size=2,
    )

    model = ALSRecommender.fit(
        interactions,
        config=config,
    )

    return model, interactions


def test_training_builds_binary_sparse_matrix(trained_model):
    model, _ = trained_model

    assert model.user_items.shape == (5, 5)
    assert set(model.user_items.data) == {1.0}


def test_recommendations_exclude_seen_items(trained_model):
    model, interactions = trained_model

    recommendations = model.recommend_for_users(
        ["u1", "u2"],
        k=2,
    )

    assert recommendations.groupby("user_id").size().to_dict() == {
        "u1": 2,
        "u2": 2,
    }

    for user_id, user_recommendations in recommendations.groupby(
        "user_id"
    ):
        seen_items = set(
            interactions.loc[
                interactions["user_id"] == user_id,
                "item_id",
            ]
        )

        recommended_items = set(user_recommendations["item_id"])

        assert recommended_items.isdisjoint(seen_items)
        assert user_recommendations["rank"].tolist() == [1, 2]


def test_unknown_user_is_rejected(trained_model):
    model, _ = trained_model

    with pytest.raises(ValueError, match="Unknown users"):
        model.recommend_for_users(["new-user"], k=2)


def test_model_artifacts_are_saved(trained_model, tmp_path):
    model, _ = trained_model

    model.save(tmp_path)

    assert (tmp_path / "als_model.npz").exists()
    assert (tmp_path / "users.parquet").exists()
    assert (tmp_path / "items.parquet").exists()
    assert (tmp_path / "user_items.npz").exists()
    assert (tmp_path / "config.json").exists()


def test_saved_model_round_trip_preserves_recommendations(
    trained_model,
    tmp_path,
):
    model, _ = trained_model
    model.save(tmp_path)

    loaded = ALSRecommender.load(tmp_path)

    assert loaded.config == model.config
    assert np.array_equal(
        loaded.user_ids,
        model.user_ids,
    )
    assert np.array_equal(
        loaded.item_ids,
        model.item_ids,
    )
    assert (
        loaded.user_items != model.user_items
    ).nnz == 0

    expected = model.recommend_for_users(
        ["u1", "u2"],
        k=2,
    )
    actual = loaded.recommend_for_users(
        ["u1", "u2"],
        k=2,
    )

    assert actual[
        ["user_id", "item_id", "rank"]
    ].equals(
        expected[["user_id", "item_id", "rank"]]
    )
    np.testing.assert_allclose(
        actual["collaborative_score"],
        expected["collaborative_score"],
    )


def test_load_rejects_missing_artifacts(tmp_path):
    with pytest.raises(
        FileNotFoundError,
        match="Missing ALS artifacts",
    ):
        ALSRecommender.load(tmp_path)


def test_load_rejects_noncontiguous_mapping(
    trained_model,
    tmp_path,
):
    model, _ = trained_model
    model.save(tmp_path)

    users_path = tmp_path / "users.parquet"
    users = pd.read_parquet(users_path)
    users.loc[0, "user_index"] = len(users) + 1
    users.to_parquet(users_path, index=False)

    with pytest.raises(
        ValueError,
        match="Mapping indexes must be contiguous",
    ):
        ALSRecommender.load(tmp_path)