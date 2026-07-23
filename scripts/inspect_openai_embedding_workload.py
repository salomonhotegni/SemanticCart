"""Inspect token volume and cost before embedding the product catalogue."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tiktoken

from semanticcart.embedding_cache import content_hash


MODEL = "text-embedding-3-small"
DIMENSIONS = 512
MAX_INPUT_TOKENS = 8_192

SYNC_PRICE_PER_MILLION = 0.02
BATCH_DISCOUNT = 0.50
PRICING_AS_OF = "2026-07-23"

DATA_DIR = Path("data/processed/amazon_video_games_5core")
CACHE_PATH = (
    Path("data/artifacts/video_games_5core/openai_embeddings")
    / "embedding_cache.parquet"
)
RESULTS_PATH = Path(
    "results/video_games_5core_openai_embedding_workload.json"
)


def percentile(values: pd.Series, percentage: int) -> int:
    """Return an integer percentile, or zero for an empty series."""
    if values.empty:
        return 0

    return int(np.percentile(values, percentage))


def main() -> None:
    """Measure catalogue tokens, cache reuse, and estimated API cost."""
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

    work = catalog.loc[
        catalog["item_id"].astype(str).isin(train_item_ids),
        ["item_id", "catalog_text"],
    ].copy()

    work["item_id"] = work["item_id"].astype(str)
    work["catalog_text"] = (
        work["catalog_text"].fillna("").astype(str)
    )
    work["content_hash"] = work["catalog_text"].map(
        content_hash
    )

    unique_texts = (
        work.drop_duplicates("content_hash")
        .reset_index(drop=True)
    )

    encoding = tiktoken.encoding_for_model(MODEL)
    unique_texts["token_count"] = unique_texts[
        "catalog_text"
    ].map(lambda text: len(encoding.encode(text)))

    cached_hashes: set[str] = set()

    if CACHE_PATH.exists():
        cache = pd.read_parquet(
            CACHE_PATH,
            columns=["content_hash", "model", "dimensions"],
        )

        cached_hashes = set(
            cache.loc[
                (cache["model"] == MODEL)
                & (cache["dimensions"] == DIMENSIONS),
                "content_hash",
            ]
        )

    missing = unique_texts.loc[
        ~unique_texts["content_hash"].isin(cached_hashes)
    ]

    invalid_inputs = unique_texts.loc[
        (unique_texts["token_count"] == 0)
        | (unique_texts["token_count"] > MAX_INPUT_TOKENS)
    ]

    total_tokens = int(unique_texts["token_count"].sum())
    missing_tokens = int(missing["token_count"].sum())

    full_sync_cost = (
        total_tokens / 1_000_000
    ) * SYNC_PRICE_PER_MILLION

    full_batch_cost = full_sync_cost * BATCH_DISCOUNT

    incremental_batch_cost = (
        missing_tokens / 1_000_000
    ) * SYNC_PRICE_PER_MILLION * BATCH_DISCOUNT

    report = {
        "model": MODEL,
        "dimensions": DIMENSIONS,
        "pricing_as_of": PRICING_AS_OF,
        "sync_price_per_million_tokens_usd": (
            SYNC_PRICE_PER_MILLION
        ),
        "batch_discount": BATCH_DISCOUNT,
        "training_products": len(work),
        "unique_content_hashes": len(unique_texts),
        "cached_content_hashes": len(cached_hashes),
        "missing_embedding_requests": len(missing),
        "total_tokens": total_tokens,
        "missing_tokens": missing_tokens,
        "tokens_p50": percentile(
            unique_texts["token_count"], 50
        ),
        "tokens_p95": percentile(
            unique_texts["token_count"], 95
        ),
        "tokens_max": int(
            unique_texts["token_count"].max()
        ),
        "invalid_inputs": len(invalid_inputs),
        "estimated_full_sync_cost_usd": full_sync_cost,
        "estimated_full_batch_cost_usd": full_batch_cost,
        "estimated_incremental_batch_cost_usd": (
            incremental_batch_cost
        ),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS_PATH.open("w", encoding="utf-8") as output:
        json.dump(report, output, indent=2)

    print(f"Training products:  {len(work):,}")
    print(f"Unique texts:       {len(unique_texts):,}")
    print(f"Cached texts:       {len(cached_hashes):,}")
    print(f"Missing requests:   {len(missing):,}")
    print(f"Total tokens:       {total_tokens:,}")
    print(f"Missing tokens:     {missing_tokens:,}")
    print(
        "Tokens per text:   "
        f"p50={report['tokens_p50']:,} "
        f"p95={report['tokens_p95']:,} "
        f"max={report['tokens_max']:,}"
    )
    print(f"Invalid inputs:     {len(invalid_inputs):,}")
    print(f"Full sync cost:     ${full_sync_cost:.4f}")
    print(f"Full batch cost:    ${full_batch_cost:.4f}")
    print(
        "Incremental batch: "
        f"${incremental_batch_cost:.4f}"
    )
    print(f"Results:            {RESULTS_PATH}")


if __name__ == "__main__":
    main()