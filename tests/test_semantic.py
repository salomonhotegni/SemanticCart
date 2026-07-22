import numpy as np
import pandas as pd
import pytest

from semanticcart.semantic import recommend_semantic


@pytest.fixture
def synthetic_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_id": [
                "viewed_old",
                "viewed_new",
                "old_match",
                "recent_match",
                "balanced_match",
            ],
            "title": [
                "Old view",
                "New view",
                "Old preference match",
                "Recent preference match",
                "Balanced match",
            ],
            "embedding": [
                np.array([1.0, 0.0]),
                np.array([0.0, 1.0]),
                np.array([1.0, 0.0]),
                np.array([0.0, 1.0]),
                np.array([1.0, 1.0]),
            ],
        }
    )


def test_returns_ranked_unique_unseen_products(
    synthetic_catalog: pd.DataFrame,
) -> None:
    viewed = ["viewed_old", "viewed_new"]

    recommendations = recommend_semantic(
        synthetic_catalog,
        viewed_product_ids=viewed,
        k=3,
        recency_decay=0.1,
    )

    assert recommendations["rank"].tolist() == [1, 2, 3]
    assert recommendations["product_id"].is_unique
    assert not recommendations["product_id"].isin(viewed).any()
    assert recommendations["semantic_score"].is_monotonic_decreasing


def test_recent_view_has_more_influence(
    synthetic_catalog: pd.DataFrame,
) -> None:
    recommendations = recommend_semantic(
        synthetic_catalog,
        viewed_product_ids=["viewed_old", "viewed_new"],
        k=3,
        recency_decay=0.1,
    )

    assert recommendations.iloc[0]["product_id"] == "recent_match"


def test_unknown_product_raises_error(
    synthetic_catalog: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="Unknown product IDs"):
        recommend_semantic(
            synthetic_catalog,
            viewed_product_ids=["missing_product"],
        )