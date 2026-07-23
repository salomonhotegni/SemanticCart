"""Prepare OpenAI Batch API requests for training-catalogue embeddings."""

from pathlib import Path

import pandas as pd

from semanticcart.openai_batch import (
    BatchEmbeddingConfig,
    prepare_embedding_batches,
)


DATA_DIR = Path("data/processed/amazon_video_games_5core")
ARTIFACT_ROOT = Path(
    "data/artifacts/video_games_5core/openai_embeddings"
)
CACHE_PATH = ARTIFACT_ROOT / "embedding_cache.parquet"
BATCH_ROOT = ARTIFACT_ROOT / "batches"


def main() -> None:
    """Generate resumable JSONL chunks without calling the API."""
    train = pd.read_parquet(
        DATA_DIR / "train.parquet",
        columns=["item_id"],
    )
    catalog = pd.read_parquet(
        DATA_DIR / "catalog.parquet",
        columns=["item_id", "catalog_text"],
    )

    train_item_ids = pd.Index(
        train["item_id"].astype(str).unique()
    )

    training_catalog = catalog.loc[
        catalog["item_id"].astype(str).isin(train_item_ids)
    ].copy()

    missing_metadata = train_item_ids.difference(
        pd.Index(training_catalog["item_id"].astype(str))
    )

    if len(missing_metadata):
        raise ValueError(
            f"Missing metadata for {len(missing_metadata)} "
            "training products."
        )

    config = BatchEmbeddingConfig(
        model="text-embedding-3-small",
        dimensions=512,
        max_input_tokens=8_192,
        max_batch_requests=50_000,
        max_batch_tokens=2_500_000,
    )

    manifest_path, manifest = prepare_embedding_batches(
        catalog=training_catalog,
        cache_path=CACHE_PATH,
        output_root=BATCH_ROOT,
        config=config,
    )

    print(f"Workload ID:      {manifest['workload_id']}")
    print(f"Catalogue items:  {manifest['catalog_products']:,}")
    print(f"Unique texts:     {manifest['unique_texts']:,}")
    print(f"Cached texts:     {manifest['cached_texts']:,}")
    print(f"Pending requests: {manifest['pending_requests']:,}")
    print(f"Pending tokens:   {manifest['pending_tokens']:,}")
    print(f"Batch chunks:     {len(manifest['chunks']):,}")
    print()

    total_requests = 0
    total_tokens = 0
    total_bytes = 0

    for chunk in manifest["chunks"]:
        total_requests += chunk["request_count"]
        total_tokens += chunk["token_count"]
        total_bytes += chunk["file_size_bytes"]

        size_mb = chunk["file_size_bytes"] / (1024 * 1024)

        print(
            f"Chunk {chunk['chunk_number']:04d}: "
            f"requests={chunk['request_count']:,} "
            f"tokens={chunk['token_count']:,} "
            f"size={size_mb:.2f} MB "
            f"status={chunk['status']}"
        )

    if total_requests != manifest["pending_requests"]:
        raise ValueError("Chunk request totals do not match manifest.")

    if total_tokens != manifest["pending_tokens"]:
        raise ValueError("Chunk token totals do not match manifest.")

    print()
    print(f"Total JSONL size: {total_bytes / (1024 * 1024):.2f} MB")
    print(f"Manifest:         {manifest_path}")
    print("No OpenAI API request was made.")


if __name__ == "__main__":
    main()