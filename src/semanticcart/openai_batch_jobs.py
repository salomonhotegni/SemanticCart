"""Submit and monitor prepared OpenAI embedding Batch chunks."""

import json
from pathlib import Path
from typing import Any


ACTIVE_SERVER_STATUSES = {
    "validating",
    "in_progress",
    "finalizing",
    "cancelling",
}

FAILED_SERVER_STATUSES = {
    "failed",
    "expired",
    "cancelled",
}


def load_manifest(manifest_path: str | Path) -> dict:
    """Load and minimally validate an embedding workload manifest.

    Args:
        manifest_path: JSON manifest created during batch preparation.

    Returns:
        The parsed manifest dictionary.

    Raises:
        FileNotFoundError: If the manifest does not exist.
        ValueError: If the manifest schema is unsupported.
    """
    manifest_path = Path(manifest_path)

    with manifest_path.open("r", encoding="utf-8") as source:
        manifest = json.load(source)

    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported batch manifest schema.")

    if not isinstance(manifest.get("chunks"), list):
        raise ValueError("Batch manifest does not contain chunks.")

    return manifest


def save_manifest(
    manifest_path: str | Path,
    manifest: dict,
) -> None:
    """Atomically persist an updated workload manifest.

    Args:
        manifest_path: Destination manifest path.
        manifest: Updated manifest state.
    """
    manifest_path = Path(manifest_path)
    temporary_path = manifest_path.with_name(
        f".{manifest_path.name}.tmp"
    )

    with temporary_path.open("w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)

    temporary_path.replace(manifest_path)


def refresh_batch_statuses(
    client: Any,
    manifest_path: str | Path,
) -> dict:
    """Refresh server state for every submitted chunk.

    Locally collected chunks are not refreshed so their completed lifecycle
    state is preserved.

    Args:
        client: Authenticated OpenAI client.
        manifest_path: Workload manifest to update.

    Returns:
        The updated manifest.
    """
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)

    for chunk in manifest["chunks"]:
        batch_id = chunk.get("batch_id")

        if not batch_id or chunk.get("status") in {"collected", "collected_with_errors"}:
            continue

        batch = client.batches.retrieve(batch_id)

        chunk["server_status"] = batch.status
        chunk["output_file_id"] = batch.output_file_id
        chunk["error_file_id"] = batch.error_file_id

        if batch.status == "completed":
            chunk["status"] = "completed"
        elif batch.status in FAILED_SERVER_STATUSES:
            chunk["status"] = batch.status
        else:
            chunk["status"] = "submitted"

    save_manifest(manifest_path, manifest)
    return manifest


def submit_next_batch(
    client: Any,
    manifest_path: str | Path,
) -> tuple[dict, dict]:
    """Upload and submit the next prepared embedding chunk.

    The manifest records the uploaded file ID before creating the Batch job.
    This allows a retry after an upload failure without uploading the same
    JSONL file again. Only one chunk may be active at a time, and a completed
    chunk must be collected before the next one is submitted.

    Args:
        client: Authenticated OpenAI client.
        manifest_path: Prepared workload manifest.

    Returns:
        The updated manifest and submitted chunk.

    Raises:
        FileNotFoundError: If a referenced JSONL chunk is missing.
        RuntimeError: If another chunk is active, completed but uncollected,
            failed, or no prepared chunks remain.
    """
    manifest_path = Path(manifest_path)
    manifest = refresh_batch_statuses(
        client,
        manifest_path,
    )

    active_chunks = [
        chunk
        for chunk in manifest["chunks"]
        if chunk.get("server_status")
        in ACTIVE_SERVER_STATUSES
    ]

    if active_chunks:
        chunk = active_chunks[0]
        raise RuntimeError(
            f"Chunk {chunk['chunk_number']:04d} is still "
            f"{chunk['server_status']}."
        )

    completed_chunks = [
        chunk
        for chunk in manifest["chunks"]
        if chunk.get("status") == "completed"
    ]

    if completed_chunks:
        chunk = completed_chunks[0]
        raise RuntimeError(
            f"Chunk {chunk['chunk_number']:04d} must be "
            "collected before submitting another chunk."
        )

    failed_chunks = [
        chunk
        for chunk in manifest["chunks"]
        if chunk.get("status") in FAILED_SERVER_STATUSES
    ]

    if failed_chunks:
        chunk = failed_chunks[0]
        raise RuntimeError(
            f"Chunk {chunk['chunk_number']:04d} ended with "
            f"status {chunk['status']}."
        )

    chunk = next(
        (
            candidate
            for candidate in manifest["chunks"]
            if candidate.get("status")
            in {"prepared", "uploaded"}
        ),
        None,
    )

    if chunk is None:
        raise RuntimeError("No prepared chunks remain.")

    chunk_path = manifest_path.parent / chunk["filename"]

    if not chunk_path.exists():
        raise FileNotFoundError(
            f"Batch input file not found: {chunk_path}"
        )

    if not chunk.get("input_file_id"):
        with chunk_path.open("rb") as source:
            uploaded_file = client.files.create(
                file=source,
                purpose="batch",
            )

        chunk["input_file_id"] = uploaded_file.id
        chunk["status"] = "uploaded"
        save_manifest(manifest_path, manifest)

    batch = client.batches.create(
        input_file_id=chunk["input_file_id"],
        endpoint="/v1/embeddings",
        completion_window="24h",
        metadata={
            "workload_id": manifest["workload_id"],
            "chunk_number": str(chunk["chunk_number"]),
        },
    )

    chunk["batch_id"] = batch.id
    chunk["server_status"] = batch.status
    chunk["status"] = "submitted"
    chunk["output_file_id"] = batch.output_file_id
    chunk["error_file_id"] = batch.error_file_id

    save_manifest(manifest_path, manifest)
    return manifest, chunk