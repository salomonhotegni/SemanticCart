"""Attach versioned cached embeddings to a product catalogue."""

from pathlib import Path

import numpy as np
import pandas as pd

from semanticcart.embedding_cache import (
    EmbeddingConfig,
    content_hash,
)


CACHE_COLUMNS = {
    "content_hash",
    "model",
    "dimensions",
    "embedding",
}


def load_cached_catalog_embeddings(
    catalog: pd.DataFrame,
    cache_path: str | Path,
    id_column: str = "item_id",
    config: EmbeddingConfig = EmbeddingConfig(),
) -> pd.DataFrame:
    """Load compatible cached vectors without making API requests.

    Args:
        catalog: Products containing an identifier and catalog_text.
        cache_path: Versioned Parquet embedding cache.
        id_column: Product identifier column used by the catalogue.
        config: Required embedding model and dimensionality.

    Returns:
        Catalogue rows with content_hash and float32 embedding columns.

    Raises:
        FileNotFoundError: If the cache does not exist.
        ValueError: If catalogue data, cache schema, coverage, or vectors are
            invalid.
    """
    required_catalog_columns = {id_column, "catalog_text"}
    missing_catalog_columns = (
        required_catalog_columns - set(catalog.columns)
    )

    if missing_catalog_columns:
        raise ValueError(
            "Missing catalogue columns: "
            f"{sorted(missing_catalog_columns)}"
        )

    work = catalog.copy()

    if work.empty:
        raise ValueError("Catalogue cannot be empty.")
    if work[id_column].isna().any():
        raise ValueError("Product IDs cannot be missing.")
    if work["catalog_text"].isna().any():
        raise ValueError("Catalogue text cannot be missing.")

    work[id_column] = work[id_column].astype(str)
    work["catalog_text"] = work["catalog_text"].astype(str)

    if work[id_column].duplicated().any():
        raise ValueError("Product IDs must be unique.")

    cache_path = Path(cache_path)

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Embedding cache not found: {cache_path}"
        )

    cache = pd.read_parquet(cache_path)
    missing_cache_columns = CACHE_COLUMNS - set(cache.columns)

    if missing_cache_columns:
        raise ValueError(
            f"Missing cache columns: {sorted(missing_cache_columns)}"
        )

    compatible_cache = (
        cache.loc[
            (cache["model"] == config.model)
            & (cache["dimensions"] == config.dimensions)
        ]
        .drop_duplicates("content_hash", keep="last")
    )

    work["content_hash"] = work["catalog_text"].map(
        content_hash
    )

    work = work.merge(
        compatible_cache[["content_hash", "embedding"]],
        on="content_hash",
        how="left",
        validate="many_to_one",
    )

    missing_embeddings = work["embedding"].isna()

    if missing_embeddings.any():
        missing_ids = work.loc[
            missing_embeddings,
            id_column,
        ].head(5).tolist()

        raise ValueError(
            "Missing compatible cached embeddings for "
            f"{int(missing_embeddings.sum()):,} products. "
            f"Examples: {missing_ids}"
        )

    try:
        vectors = np.vstack(
            work["embedding"].to_numpy()
        ).astype(np.float32)
    except ValueError as error:
        raise ValueError(
            "Cached embeddings do not share one dimension."
        ) from error

    expected_shape = (len(work), config.dimensions)

    if vectors.shape != expected_shape:
        raise ValueError(
            f"Embedding matrix has shape {vectors.shape}; "
            f"expected {expected_shape}."
        )
    if not np.isfinite(vectors).all():
        raise ValueError(
            "Cached embeddings contain non-finite values."
        )
    if np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError(
            "Cached embeddings contain zero vectors."
        )

    work["embedding"] = list(vectors)
    return work