import numpy as np
import pandas as pd
import pytest

from semanticcart.dense_index import DenseItemIndex, HnswConfig
from semanticcart.dense_profiles import (
    DenseProfileConfig,
    DenseUserProfiles,
)


@pytest.fixture(scope="module")
def item_index():
    """Build a small dense product index for profile tests."""
    catalog = pd.DataFrame(
        {
            "item_id": ["a", "b", "c", "d"],
            "embedding": [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.9, 0.1],
                [0.1, 0.9],
            ],
        }
    )

    return DenseItemIndex.from_catalog(
        catalog,
        HnswConfig(
            connections=8,
            ef_construction=40,
            ef_search=32,
        ),
    )


@pytest.fixture(scope="module")
def user_profiles(item_index):
    """Build profiles containing duplicates and an unknown product."""
    interactions = pd.DataFrame(
        {
            "user_id": [
                "u2",
                "u1",
                "u1",
                "u1",
                "u1",
                "u2",
            ],
            "item_id": [
                "c",
                "b",
                "b",
                "a",
                "outside-index",
                "d",
            ],
            "timestamp": [1, 1, 2, 3, 4, 2],
        }
    )

    return DenseUserProfiles.build(
        item_index,
        interactions,
        DenseProfileConfig(recency_decay=0.1),
    )


def test_builds_sorted_normalized_profiles(user_profiles):
    assert user_profiles.user_ids.tolist() == ["u1", "u2"]
    assert user_profiles.user_profiles.shape == (2, 2)
    assert np.allclose(
        np.linalg.norm(user_profiles.user_profiles, axis=1),
        1.0,
    )


def test_keeps_one_event_per_user_item(user_profiles):
    assert user_profiles.user_items.nnz == 4

    u1_index = user_profiles.user_to_index["u1"]
    assert user_profiles.user_items[u1_index].nnz == 2


def test_most_recent_product_receives_largest_weight(user_profiles):
    u1_index = user_profiles.user_to_index["u1"]

    expected = np.array([1.0, 0.1], dtype=np.float32)
    expected /= np.linalg.norm(expected)

    assert np.allclose(
        user_profiles.user_profiles[u1_index],
        expected,
    )


def test_resolves_user_indices_in_requested_order(user_profiles):
    indices = user_profiles.indices_for(["u2", "u1"])

    assert user_profiles.user_ids[indices].tolist() == [
        "u2",
        "u1",
    ]


def test_rejects_unknown_users(user_profiles):
    with pytest.raises(ValueError, match="Unknown users"):
        user_profiles.indices_for(["new-user"])


def test_rejects_interactions_without_indexed_products(item_index):
    interactions = pd.DataFrame(
        {
            "user_id": ["u1"],
            "item_id": ["outside-index"],
            "timestamp": [1],
        }
    )

    with pytest.raises(ValueError, match="No interactions match"):
        DenseUserProfiles.build(item_index, interactions)


def test_rejects_zero_user_profiles():
    opposite_index = DenseItemIndex.from_catalog(
        pd.DataFrame(
            {
                "item_id": ["a", "b"],
                "embedding": [[1.0, 0.0], [-1.0, 0.0]],
            }
        )
    )

    interactions = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "item_id": ["a", "b"],
            "timestamp": [1, 2],
        }
    )

    with pytest.raises(ValueError, match="zero user profile"):
        DenseUserProfiles.build(
            opposite_index,
            interactions,
            DenseProfileConfig(recency_decay=1.0),
        )