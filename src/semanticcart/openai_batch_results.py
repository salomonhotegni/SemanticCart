"""Collect completed OpenAI Batch embeddings into the local cache."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from semanticcart.openai_batch_jobs import (
    refresh_batch_statuses,
    save_manifest,
)


def _download_file(client: Any, file_id: str, destination: Path) -> None:
    """Download an OpenAI file atomically and reuse it after interrupted runs."""
    if destination.exists():
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(f".{destination.name}.tmp")

    response = client.files.content(file_id)
    response.write_to_file(temporary_path)
    temporary_path.replace(destination)


def _load_expected_ids(input_path: Path) -> set[str]:
    """Read and validate custom IDs from an input Batch JSONL file."""
    expected_ids: set[str] = set()

    with input_path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue

            request = json.loads(line)
            custom_id = request.get("custom_id")

            if not custom_id:
                raise ValueError(
                    f"Missing custom_id in {input_path} at line {line_number}."
                )
            if custom_id in expected_ids:
                raise ValueError(f"Duplicate input custom_id: {custom_id}")

            expected_ids.add(custom_id)

    return expected_ids


def _parse_output(
    output_path: Path,
    expected_ids: set[str],
    dimensions: int,
) -> dict[str, list[float]]:
    """Validate successful Batch responses and return vectors by custom ID."""
    embeddings: dict[str, list[float]] = {}
    observed_ids: set[str] = set()

    with output_path.open(encoding="utf-8") as output_file:
        for line_number, line in enumerate(output_file, start=1):
            if not line.strip():
                continue

            result = json.loads(line)
            custom_id = result.get("custom_id")

            if custom_id not in expected_ids:
                raise ValueError(
                    f"Unexpected custom_id at output line {line_number}: {custom_id}"
                )
            if custom_id in observed_ids:
                raise ValueError(f"Duplicate output custom_id: {custom_id}")

            observed_ids.add(custom_id)
            response = result.get("response")

            if not response or response.get("status_code") != 200:
                continue

            data = response.get("body", {}).get("data", [])
            if len(data) != 1:
                raise ValueError(
                    f"Expected one embedding for custom_id {custom_id}."
                )

            vector = np.asarray(data[0].get("embedding"), dtype=np.float32)

            if vector.shape != (dimensions,):
                raise ValueError(
                    f"Embedding {custom_id} has shape {vector.shape}; "
                    f"expected ({dimensions},)."
                )
            if not np.isfinite(vector).all():
                raise ValueError(
                    f"Embedding {custom_id} contains non-finite values."
                )

            embeddings[custom_id] = vector.tolist()

    return embeddings


def _merge_cache(
    cache_path: Path,
    embeddings: dict[str, list[float]],
    model: str,
    dimensions: int,
) -> None:
    """Merge embeddings into a versioned Parquet cache atomically."""
    if not embeddings:
        return

    new_rows = pd.DataFrame(
        {
            "content_hash": list(embeddings),
            "model": model,
            "dimensions": dimensions,
            "embedding": list(embeddings.values()),
        }
    )

    if cache_path.exists():
        cache = pd.read_parquet(cache_path)
        cache = pd.concat([cache, new_rows], ignore_index=True)
    else:
        cache = new_rows

    cache = cache.drop_duplicates(
        subset=["content_hash", "model", "dimensions"],
        keep="last",
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(f".{cache_path.stem}.tmp.parquet")
    cache.to_parquet(temporary_path, index=False)
    temporary_path.replace(cache_path)


def collect_completed_batch(
    client: Any,
    manifest_path: str | Path,
    cache_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Collect the next completed chunk and update its manifest state."""
    manifest_path = Path(manifest_path)
    cache_path = Path(cache_path)
    manifest = refresh_batch_statuses(client, manifest_path)

    completed_chunk = None
    chunk_number = 0

    for chunk in manifest["chunks"]:
        if chunk.get("status") == "completed":
            completed_chunk = chunk
            chunk_number = int(chunk["chunk_number"])
            break

    if completed_chunk is None:
        raise RuntimeError("No completed, uncollected Batch chunk was found.")

    output_file_id = completed_chunk.get("output_file_id")
    if not output_file_id:
        raise RuntimeError("Completed chunk has no output_file_id.")

    run_directory = manifest_path.parent
    output_path = run_directory / f"batch_{chunk_number:04d}_output.jsonl"
    error_path = run_directory / f"batch_{chunk_number:04d}_errors.jsonl"
    missing_path = run_directory / f"batch_{chunk_number:04d}_missing_ids.json"

    _download_file(client, output_file_id, output_path)

    error_file_id = completed_chunk.get("error_file_id")
    if error_file_id:
        _download_file(client, error_file_id, error_path)

    expected_ids = _load_expected_ids(
        run_directory / completed_chunk["filename"]
    )
    embeddings = _parse_output(
        output_path,
        expected_ids,
        int(manifest["config"]["dimensions"]),
    )
    missing_ids = sorted(expected_ids - embeddings.keys())

    _merge_cache(
        cache_path,
        embeddings,
        model=manifest["config"]["model"],
        dimensions=int(manifest["config"]["dimensions"]),
    )

    if missing_ids:
        missing_path.write_text(
            json.dumps(missing_ids, indent=2),
            encoding="utf-8",
        )

    completed_chunk.update(
        {
            "status": (
                "collected_with_errors" if missing_ids else "collected"
            ),
            "output_path": output_path.name,
            "error_path": error_path.name if error_file_id else None,
            "missing_ids_path": missing_path.name if missing_ids else None,
            "cached_embedding_count": len(embeddings),
            "missing_request_count": len(missing_ids),
        }
    )
    save_manifest(manifest_path, manifest)

    return manifest, completed_chunk