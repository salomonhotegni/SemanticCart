import pandas as pd
import pytest

from semanticcart.tfidf import (
    TfidfConfig,
    TfidfRecommender,
)


@pytest.fixture(scope="module")
def trained_tfidf_model():
    catalog = pd.DataFrame(
        {
            "item_id": ["a", "b", "c", "d", "e"],
            "catalog_text": [
                "space galaxy starship adventure",
                "racing car speed competition",
                "space planet alien exploration",
                "football stadium sports tournament",
                "racing vehicle track championship",
            ],
        }
    )

    interactions = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "item_id": ["b", "a", "c", "d"],
            "timestamp": [1, 2, 1, 2],
        }
    )

    config = TfidfConfig(
        max_features=100,
        min_df=1,
        recency_decay=0.1,
        batch_size=1,
    )

    model = TfidfRecommender.fit(
        catalog=catalog,
        interactions=interactions,
        config=config,
    )

    return model, interactions


def test_tfidf_builds_item_and_user_vectors(
    trained_tfidf_model,
):
    model, _ = trained_tfidf_model

    assert model.item_vectors.shape[0] == 5
    assert model.user_profiles.shape[0] == 2
    assert model.item_vectors.shape[1] > 0


def test_tfidf_returns_ranked_unseen_items(
    trained_tfidf_model,
):
    model, interactions = trained_tfidf_model

    recommendations = model.recommend_for_users(
        ["u1"],
        k=2,
    )

    seen_items = set(
        interactions.loc[
            interactions["user_id"] == "u1",
            "item_id",
        ]
    )

    assert recommendations["rank"].tolist() == [1, 2]
    assert recommendations["item_id"].is_unique
    assert set(recommendations["item_id"]).isdisjoint(
        seen_items
    )
    assert recommendations["semantic_score"].is_monotonic_decreasing


def test_recent_space_product_drives_first_result(
    trained_tfidf_model,
):
    model, _ = trained_tfidf_model

    recommendations = model.recommend_for_users(
        ["u1"],
        k=2,
    )

    assert recommendations.iloc[0]["item_id"] == "c"


def test_tfidf_rejects_unknown_user(
    trained_tfidf_model,
):
    model, _ = trained_tfidf_model

    with pytest.raises(ValueError, match="Unknown users"):
        model.recommend_for_users(["new-user"], k=2)


def test_tfidf_artifacts_are_saved(
    trained_tfidf_model,
    tmp_path,
):
    model, _ = trained_tfidf_model

    model.save(tmp_path)

    expected_files = {
        "tfidf_vectorizer.joblib",
        "item_vectors.npz",
        "user_profiles.npz",
        "user_items.npz",
        "users.parquet",
        "items.parquet",
    }

    assert expected_files == {
        path.name for path in tmp_path.iterdir()
    }