"""Prepare deterministic, resumable OpenAI embedding Batch workloads."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import tiktoken

from semanticcart.embedding_cache import content_hash


@dataclass(frozen=True)
class BatchEmbeddingConfig:
    """Configure offline OpenAI embedding Batch requests.

    Attributes:
        model: OpenAI embedding model identifier.
        dimensions: Number of output vector dimensions.
        max_input_tokens: Maximum permitted tokens in one input.
        max_batch_requests: Maximum requests written to one batch file.
        max_batch_tokens: Local token ceiling used to support lower tiers.
    """

    model: str = "text-embedding-3-small"
    dimensions: int = 512
    max_input_tokens: int = 8_192
    max_batch_requests: int = 50_000
    max_batch_tokens: int = 2_500_000


def prepare_embedding_batches(
    catalog: pd.DataFrame,
    cache_path: str | Path,
    output_root: str | Path,
    config: BatchEmbeddingConfig | None = None,
) -> tuple[Path, dict]:
    """Write missing embedding requests to resumable JSONL chunks.

    Product texts are deduplicated by content hash. Cache entries are reused
    only when both model and dimensions match the current configuration. The
    deterministic workload ID prevents reruns from overwriting another job.

    Args:
        catalog: Products containing item_id and catalog_text.
        cache_path: Parquet cache containing versioned embeddings.
        output_root: Parent directory for generated workload directories.
        config: Optional model and chunking configuration.

    Returns:
        The manifest path and parsed manifest dictionary.

    Raises:
        ValueError: If required columns are missing, text is empty, an input
            exceeds the model limit, or a request exceeds the chunk limit.
    """
    config = config or BatchEmbeddingConfig()
    required = {"item_id", "catalog_text"}
    missing_columns = required - set(catalog.columns)

    if missing_columns:
        raise ValueError(
            f"Missing catalogue columns: {sorted(missing_columns)}"
        )

    work = catalog[["item_id", "catalog_text"]].copy()
    work["item_id"] = work["item_id"].astype(str)
    work["catalog_text"] = (
        work["catalog_text"].fillna("").astype(str)
    )
    work["content_hash"] = work["catalog_text"].map(content_hash)

    unique_texts = (
        work.drop_duplicates("content_hash")
        .sort_values("content_hash")
        .reset_index(drop=True)
    )

    encoding = tiktoken.encoding_for_model(config.model)
    unique_texts["token_count"] = unique_texts[
        "catalog_text"
    ].map(lambda text: len(encoding.encode(text)))

    invalid = unique_texts.loc[
        (unique_texts["token_count"] == 0)
        | (
            unique_texts["token_count"]
            > config.max_input_tokens
        )
    ]

    if not invalid.empty:
        raise ValueError(
            f"{len(invalid)} embedding inputs are empty or exceed "
            f"{config.max_input_tokens:,} tokens."
        )

    cache_path = Path(cache_path)
    cached_hashes: set[str] = set()

    if cache_path.exists():
        cache = pd.read_parquet(
            cache_path,
            columns=["content_hash", "model", "dimensions"],
        )

        cached_hashes = set(
            cache.loc[
                (cache["model"] == config.model)
                & (cache["dimensions"] == config.dimensions),
                "content_hash",
            ]
        )
    
    catalog_hashes = set(unique_texts["content_hash"])
    cached_hashes.intersection_update(catalog_hashes)

    pending = unique_texts.loc[
        ~unique_texts["content_hash"].isin(cached_hashes)
    ].reset_index(drop=True)

    fingerprint_parts = [
        config.model,
        str(config.dimensions),
        *pending["content_hash"].tolist(),
    ]
    workload_id = content_hash(
        "\n".join(fingerprint_parts)
    )[:16]

    run_directory = Path(output_root) / workload_id
    manifest_path = run_directory / "manifest.json"

    if manifest_path.exists():
        with manifest_path.open(
            "r",
            encoding="utf-8",
        ) as source:
            return manifest_path, json.load(source)

    run_directory.mkdir(parents=True, exist_ok=True)

    chunks: list[tuple[list, int]] = []
    current_rows: list = []
    current_tokens = 0

    for row in pending.itertuples(index=False):
        if row.token_count > config.max_batch_tokens:
            raise ValueError(
                f"Input {row.content_hash} exceeds the local "
                "batch-token ceiling."
            )

        exceeds_request_limit = (
            len(current_rows) >= config.max_batch_requests
        )
        exceeds_token_limit = (
            current_tokens + row.token_count
            > config.max_batch_tokens
        )

        if current_rows and (
            exceeds_request_limit or exceeds_token_limit
        ):
            chunks.append((current_rows, current_tokens))
            current_rows = []
            current_tokens = 0

        current_rows.append(row)
        current_tokens += row.token_count

    if current_rows:
        chunks.append((current_rows, current_tokens))

    chunk_metadata = []

    for chunk_number, (rows, token_count) in enumerate(
        chunks,
        start=1,
    ):
        filename = f"batch_{chunk_number:04d}.jsonl"
        chunk_path = run_directory / filename

        with chunk_path.open("w", encoding="utf-8") as output:
            for row in rows:
                request = {
                    "custom_id": row.content_hash,
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": {
                        "model": config.model,
                        "input": row.catalog_text,
                        "dimensions": config.dimensions,
                        "encoding_format": "float",
                    },
                }

                output.write(
                    json.dumps(
                        request,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

        chunk_metadata.append(
            {
                "chunk_number": chunk_number,
                "filename": filename,
                "request_count": len(rows),
                "token_count": token_count,
                "file_size_bytes": chunk_path.stat().st_size,
                "status": "prepared",
                "input_file_id": None,
                "batch_id": None,
                "output_file_id": None,
                "error_file_id": None,
            }
        )

    manifest = {
        "schema_version": 1,
        "workload_id": workload_id,
        "config": asdict(config),
        "catalog_products": len(work),
        "unique_texts": len(unique_texts),
        "cached_texts": len(cached_hashes),
        "pending_requests": len(pending),
        "pending_tokens": int(pending["token_count"].sum()),
        "chunks": chunk_metadata,
    }

    with manifest_path.open("w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)

    return manifest_path, manifest