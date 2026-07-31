import numpy as np
import pandas as pd
import pytest
import json

from scipy.sparse import load_npz
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
        
    
def test_dense_profile_artifacts_are_readable(
    user_profiles,
    tmp_path,
):
    user_profiles.save(tmp_path)

    assert {
        path.name for path in tmp_path.iterdir()
    } == {
        "user_profiles.npy",
        "user_items.npz",
        "users.parquet",
        "profile_config.json",
    }

    saved_profiles = np.load(
        tmp_path / "user_profiles.npy",
        allow_pickle=False,
    )
    saved_items = load_npz(
        tmp_path / "user_items.npz"
    )
    saved_users = pd.read_parquet(
        tmp_path / "users.parquet"
    )
    saved_config = json.loads(
        (tmp_path / "profile_config.json").read_text()
    )

    assert np.allclose(
        saved_profiles,
        user_profiles.user_profiles,
    )
    assert (
        saved_items != user_profiles.user_items
    ).nnz == 0
    assert saved_users["user_id"].tolist() == (
        user_profiles.user_ids.tolist()
    )
    assert saved_config["recency_decay"] == pytest.approx(0.1)

def test_saved_profiles_round_trip(
    item_index,
    user_profiles,
    tmp_path,
):
    user_profiles.save(tmp_path)

    loaded = DenseUserProfiles.load(
        item_index,
        tmp_path,
    )

    assert loaded.config == user_profiles.config
    assert np.array_equal(
        loaded.user_ids,
        user_profiles.user_ids,
    )
    assert np.allclose(
        loaded.user_profiles,
        user_profiles.user_profiles,
    )
    assert (
        loaded.user_items
        != user_profiles.user_items
    ).nnz == 0
    assert loaded.indices_for(
        ["u2", "u1"]
    ).tolist() == [1, 0]


def test_load_rejects_missing_profile_artifacts(
    item_index,
    tmp_path,
):
    with pytest.raises(
        FileNotFoundError,
        match="Missing profile artifacts",
    ):
        DenseUserProfiles.load(
            item_index,
            tmp_path,
        )


def test_load_rejects_mismatched_item_index(
    user_profiles,
    tmp_path,
):
    user_profiles.save(tmp_path)

    smaller_index = DenseItemIndex.from_catalog(
        pd.DataFrame(
            {
                "item_id": ["a", "b"],
                "embedding": [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
            }
        )
    )

    with pytest.raises(
        ValueError,
        match="Profile user-item matrix has shape",
    ):
        DenseUserProfiles.load(
            smaller_index,
            tmp_path,
        )