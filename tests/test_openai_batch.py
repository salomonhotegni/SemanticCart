import json

import pandas as pd
import pytest

from semanticcart.embedding_cache import content_hash
from semanticcart.openai_batch import (
    BatchEmbeddingConfig,
    prepare_embedding_batches,
)


def read_jsonl(path):
    """Read a JSONL file into a list of dictionaries."""
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def test_preparation_deduplicates_and_partitions(tmp_path):
    catalog = pd.DataFrame(
        {
            "item_id": ["a", "b", "c", "d"],
            "catalog_text": [
                "alpha game",
                "alpha game",
                "beta game",
                "gamma game",
            ],
        }
    )

    config = BatchEmbeddingConfig(
        max_batch_requests=2,
        max_batch_tokens=1_000,
    )

    manifest_path, manifest = prepare_embedding_batches(
        catalog=catalog,
        cache_path=tmp_path / "missing_cache.parquet",
        output_root=tmp_path / "batches",
        config=config,
    )

    assert manifest["catalog_products"] == 4
    assert manifest["unique_texts"] == 3
    assert manifest["pending_requests"] == 3
    assert len(manifest["chunks"]) == 2

    assert [
        chunk["request_count"]
        for chunk in manifest["chunks"]
    ] == [2, 1]

    requests = []

    for chunk in manifest["chunks"]:
        chunk_path = manifest_path.parent / chunk["filename"]
        assert chunk_path.exists()
        assert chunk["file_size_bytes"] > 0
        requests.extend(read_jsonl(chunk_path))

    assert len(requests) == 3
    assert len({request["custom_id"] for request in requests}) == 3

    for request in requests:
        assert request["method"] == "POST"
        assert request["url"] == "/v1/embeddings"
        assert request["body"]["model"] == (
            "text-embedding-3-small"
        )
        assert request["body"]["dimensions"] == 512
        assert request["body"]["encoding_format"] == "float"


def test_preparation_excludes_cache_and_resumes(tmp_path):
    catalog = pd.DataFrame(
        {
            "item_id": ["a", "b"],
            "catalog_text": ["alpha game", "beta game"],
        }
    )

    cache_path = tmp_path / "embedding_cache.parquet"

    pd.DataFrame(
        {
            "content_hash": [
                content_hash("alpha game"),
                content_hash("unrelated cached text"),
            ],
            "model": [
                "text-embedding-3-small",
                "text-embedding-3-small",
            ],
            "dimensions": [512, 512],
        }
    ).to_parquet(cache_path, index=False)

    manifest_path, manifest = prepare_embedding_batches(
        catalog=catalog,
        cache_path=cache_path,
        output_root=tmp_path / "batches",
    )

    assert manifest["cached_texts"] == 1
    assert manifest["pending_requests"] == 1

    chunk = manifest["chunks"][0]
    requests = read_jsonl(
        manifest_path.parent / chunk["filename"]
    )

    assert requests[0]["custom_id"] == content_hash(
        "beta game"
    )

    manifest["chunks"][0]["status"] = "submitted"

    with manifest_path.open("w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)

    repeated_path, repeated_manifest = (
        prepare_embedding_batches(
            catalog=catalog,
            cache_path=cache_path,
            output_root=tmp_path / "batches",
        )
    )

    assert repeated_path == manifest_path
    assert repeated_manifest["chunks"][0]["status"] == (
        "submitted"
    )


def test_preparation_rejects_empty_text(tmp_path):
    catalog = pd.DataFrame(
        {
            "item_id": ["a"],
            "catalog_text": [""],
        }
    )

    with pytest.raises(
        ValueError,
        match="empty or exceed",
    ):
        prepare_embedding_batches(
            catalog=catalog,
            cache_path=tmp_path / "cache.parquet",
            output_root=tmp_path / "batches",
        )