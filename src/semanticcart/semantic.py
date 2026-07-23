"""Rank products using recency-weighted semantic embedding similarity."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


def recommend_semantic(
    catalog: pd.DataFrame,
    viewed_product_ids: Sequence[str],
    k: int = 10,
    recency_decay: float = 0.85,
) -> pd.DataFrame:
    """Recommend unseen products from a sequence of viewed product embeddings.

    The last viewed product is treated as the most recent. Product and user
    vectors are L2-normalized, making the resulting score cosine similarity.

    Args:
        catalog: Products containing product_id, title, and embedding.
        viewed_product_ids: Ordered product history from oldest to newest.
        k: Maximum number of unique unseen products to return.
        recency_decay: Multiplicative weight applied per step into the past.

    Returns:
        Products ranked by descending semantic_score with one-based ranks.

    Raises:
        ValueError: If required columns are missing, inputs are invalid, or a
            viewed product is absent from the catalogue.
    """

    required = {"product_id", "title", "embedding"}
    missing_columns = required - set(catalog.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")
    if not viewed_product_ids:
        raise ValueError("At least one viewed product is required.")
    if k <= 0:
        raise ValueError("k must be greater than zero.")
    if not 0 < recency_decay <= 1:
        raise ValueError("recency_decay must be between 0 and 1.")

    embedding_lookup = catalog.set_index("product_id")["embedding"]

    unknown_ids = [
        product_id
        for product_id in viewed_product_ids
        if product_id not in embedding_lookup.index
    ]
    if unknown_ids:
        raise ValueError(f"Unknown product IDs: {unknown_ids}")

    viewed_matrix = np.vstack(
        [embedding_lookup.loc[product_id] for product_id in viewed_product_ids]
    ).astype(np.float32)
    viewed_matrix = normalize(viewed_matrix)

    ages = np.arange(len(viewed_product_ids) - 1, -1, -1)
    weights = recency_decay**ages

    user_vector = np.average(viewed_matrix, axis=0, weights=weights)
    user_vector = normalize(user_vector.reshape(1, -1))[0]

    candidates = catalog.loc[
        ~catalog["product_id"].isin(viewed_product_ids)
    ].copy()

    if candidates.empty:
        return pd.DataFrame(
            columns=["rank", "product_id", "title", "semantic_score"]
        )

    candidate_matrix = normalize(
        np.vstack(candidates["embedding"]).astype(np.float32)
    )
    scores = candidate_matrix @ user_vector

    top_indices = np.argsort(-scores, kind="stable")[: min(k, len(scores))]
    recommendations = candidates.iloc[top_indices][
        ["product_id", "title"]
    ].copy()

    recommendations["semantic_score"] = scores[top_indices]
    recommendations.insert(
        0, "rank", np.arange(1, len(recommendations) + 1)
    )

    return recommendations.reset_index(drop=True)
