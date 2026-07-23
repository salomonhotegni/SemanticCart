"""Generate and cache versioned OpenAI product embeddings."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configure OpenAI catalogue embedding requests.

    Attributes:
        model: Embedding model identifier.
        dimensions: Number of dimensions requested for each vector.
        batch_size: Maximum number of catalogue texts sent per API call.
    """

    model: str = "text-embedding-3-small"
    dimensions: int = 512
    batch_size: int = 64


def content_hash(text: str) -> str:
    """Return a stable SHA-256 identifier for normalized product text."""

    return sha256(text.encode("utf-8")).hexdigest()


def embed_catalog(
    catalog: pd.DataFrame,
    cache_path: str | Path,
    config: EmbeddingConfig = EmbeddingConfig(),
) -> pd.DataFrame:
    """Attach cached or newly generated embeddings to a product catalogue.

    Cache entries are keyed by content hash, model, and dimensions. An OpenAI
    client is created only when at least one unique text requires embedding.

    Args:
        catalog: Products containing product_id and catalog_text.
        cache_path: Parquet file used to persist versioned embeddings.
        config: Embedding model, dimensionality, and request batch size.

    Returns:
        The catalogue with an embedding column joined by product_id.

    Raises:
        OpenAIError: If new embeddings are required and API authentication or
            the embeddings request fails.
    """

    cache_path = Path(cache_path)
    work = catalog[["product_id", "catalog_text"]].copy()
    work["content_hash"] = work["catalog_text"].map(content_hash)

    cache_columns = [
        "content_hash", "model", "dimensions", "embedding"
    ]

    if cache_path.exists():
        cache = pd.read_parquet(cache_path)
    else:
        cache = pd.DataFrame(columns=cache_columns)

    current_cache = cache.loc[
        (cache["model"] == config.model)
        & (cache["dimensions"] == config.dimensions)
    ].drop_duplicates("content_hash", keep="last")

    work = work.merge(
        current_cache[["content_hash", "embedding"]],
        on="content_hash",
        how="left",
    )

    missing = (
        work.loc[work["embedding"].isna(), ["content_hash", "catalog_text"]]
        .drop_duplicates("content_hash")
        .to_dict("records")
    )

    generated = {}

    if missing:
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        client = OpenAI()

        for start in range(0, len(missing), config.batch_size):
            batch = missing[start : start + config.batch_size]
            response = client.embeddings.create(
                model=config.model,
                dimensions=config.dimensions,
                encoding_format="float",
                input=[row["catalog_text"] for row in batch],
            )

            response_items = sorted(response.data, key=lambda item: item.index)
            for row, item in zip(batch, response_items):
                generated[row["content_hash"]] = item.embedding

    for index in work.index[work["embedding"].isna()]:
        work.at[index, "embedding"] = generated[
            work.at[index, "content_hash"]
        ]

    new_cache = work[["content_hash", "embedding"]].copy()
    new_cache["model"] = config.model
    new_cache["dimensions"] = config.dimensions

    updated_cache = pd.concat(
        [cache, new_cache[cache_columns]], ignore_index=True
    ).drop_duplicates(
        ["content_hash", "model", "dimensions"], keep="last"
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    updated_cache.to_parquet(cache_path, index=False)

    return catalog.merge(
        work[["product_id", "embedding"]],
        on="product_id",
        how="left",
    )
