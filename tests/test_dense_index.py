import numpy as np
import pandas as pd
import pytest
import json

import faiss

from semanticcart.dense_index import DenseItemIndex, HnswConfig


@pytest.fixture(scope="module")
def dense_index():
    """Build a small deterministic semantic product index."""
    catalog = pd.DataFrame(
        {
            "item_id": ["c", "a", "b"],
            "embedding": [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.8, 0.2],
            ],
        }
    )

    return DenseItemIndex.from_catalog(
        catalog,
        config=HnswConfig(
            connections=8,
            ef_construction=40,
            ef_search=32,
        ),
    )


def test_builds_stable_normalized_item_index(dense_index):
    assert dense_index.item_ids.tolist() == ["a", "b", "c"]
    assert dense_index.index.ntotal == 3

    norms = np.linalg.norm(
        dense_index.item_vectors,
        axis=1,
    )
    assert np.allclose(norms, 1.0)


def test_search_ranks_products_by_cosine_similarity(dense_index):
    scores, indices = dense_index.search(
        np.array([1.0, 0.0]),
        k=2,
    )

    item_ids = dense_index.item_ids[indices[0]]

    assert item_ids.tolist() == ["a", "b"]
    assert scores[0, 0] == pytest.approx(1.0)
    assert scores[0, 0] >= scores[0, 1]


def test_search_supports_multiple_queries(dense_index):
    scores, indices = dense_index.search(
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        k=1,
    )

    assert scores.shape == (2, 1)
    assert dense_index.item_ids[indices[:, 0]].tolist() == [
        "a",
        "c",
    ]


@pytest.mark.parametrize(
    ("catalog", "message"),
    [
        (
            pd.DataFrame(
                {
                    "item_id": ["a", "a"],
                    "embedding": [[1.0, 0.0], [0.0, 1.0]],
                }
            ),
            "unique",
        ),
        (
            pd.DataFrame(
                {
                    "item_id": ["a"],
                    "embedding": [[0.0, 0.0]],
                }
            ),
            "zero vectors",
        ),
        (
            pd.DataFrame(
                {
                    "item_id": ["a", "b"],
                    "embedding": [[1.0, 0.0], [1.0]],
                }
            ),
            "share one dimension",
        ),
    ],
)
def test_rejects_invalid_catalog_embeddings(catalog, message):
    with pytest.raises(ValueError, match=message):
        DenseItemIndex.from_catalog(catalog)


def test_rejects_incompatible_query_dimensions(dense_index):
    with pytest.raises(ValueError, match="dimensions do not match"):
        dense_index.search(
            np.array([1.0, 0.0, 0.0]),
            k=1,
        )
        
        
def test_dense_index_artifacts_are_readable(
    dense_index,
    tmp_path,
):
    dense_index.save(tmp_path)

    assert {
        path.name for path in tmp_path.iterdir()
    } == {
        "item_index.faiss",
        "item_vectors.npy",
        "items.parquet",
        "index_config.json",
    }

    saved_index = faiss.read_index(
        str(tmp_path / "item_index.faiss")
    )
    saved_vectors = np.load(
        tmp_path / "item_vectors.npy",
        allow_pickle=False,
    )
    saved_items = pd.read_parquet(
        tmp_path / "items.parquet"
    )
    saved_config = json.loads(
        (tmp_path / "index_config.json").read_text()
    )

    assert saved_index.ntotal == len(dense_index.item_ids)
    assert np.allclose(
        saved_vectors,
        dense_index.item_vectors,
    )
    assert saved_items["item_id"].tolist() == (
        dense_index.item_ids.tolist()
    )
    assert saved_config["connections"] == 8