import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from semanticcart.openai_batch_jobs import (
    refresh_batch_statuses,
    save_manifest,
)
from semanticcart.openai_batch_results import collect_completed_batch


class FakeDownload:
    """Write fake OpenAI file content to a requested destination."""

    def __init__(self, content):
        self.content = content

    def write_to_file(self, destination):
        Path(destination).write_text(self.content, encoding="utf-8")


class FakeFiles:
    """Return predefined Batch output files."""

    def __init__(self, files):
        self.files = files

    def content(self, file_id):
        return FakeDownload(self.files[file_id])


class FakeBatches:
    """Return a completed Batch job without calling OpenAI."""

    def __init__(self, has_errors=False):
        self.has_errors = has_errors
        self.retrieve_calls = 0

    def retrieve(self, batch_id):
        assert batch_id == "batch-test"
        self.retrieve_calls += 1

        return SimpleNamespace(
            status="completed",
            output_file_id="file-output",
            error_file_id="file-errors" if self.has_errors else None,
        )


class FakeClient:
    """Expose fake files and Batch API resources."""

    def __init__(self, output, errors=None):
        files = {"file-output": output}

        if errors is not None:
            files["file-errors"] = errors

        self.files = FakeFiles(files)
        self.batches = FakeBatches(has_errors=errors is not None)


def batch_result(custom_id, embedding):
    """Build one successful Batch embeddings response."""
    return json.dumps(
        {
            "custom_id": custom_id,
            "response": {
                "status_code": 200,
                "body": {
                    "data": [{"embedding": embedding}],
                },
            },
        }
    )


def create_workload(tmp_path, custom_ids, dimensions=2):
    """Create a minimal submitted workload for collector tests."""
    run_directory = tmp_path / "workload"
    run_directory.mkdir()

    input_path = run_directory / "batch_0001.jsonl"
    input_path.write_text(
        "\n".join(
            json.dumps({"custom_id": custom_id})
            for custom_id in custom_ids
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "workload_id": "workload-test",
        "config": {
            "model": "text-embedding-3-small",
            "dimensions": dimensions,
        },
        "chunks": [
            {
                "chunk_number": 1,
                "filename": input_path.name,
                "status": "submitted",
                "batch_id": "batch-test",
                "output_file_id": None,
                "error_file_id": None,
            }
        ],
    }

    manifest_path = run_directory / "manifest.json"
    save_manifest(manifest_path, manifest)
    return manifest_path


def test_collects_unordered_embeddings_into_cache(tmp_path):
    manifest_path = create_workload(tmp_path, ["hash-a", "hash-b"])
    cache_path = tmp_path / "embedding_cache.parquet"

    output = "\n".join(
        [
            batch_result("hash-b", [0.3, 0.4]),
            batch_result("hash-a", [0.1, 0.2]),
        ]
    )
    client = FakeClient(output)

    _, chunk = collect_completed_batch(
        client,
        manifest_path,
        cache_path,
    )

    assert chunk["status"] == "collected"
    assert chunk["cached_embedding_count"] == 2
    assert chunk["missing_request_count"] == 0

    cache = pd.read_parquet(cache_path).sort_values("content_hash")

    assert cache["content_hash"].tolist() == ["hash-a", "hash-b"]
    assert np.allclose(
        np.vstack(cache["embedding"]),
        [[0.1, 0.2], [0.3, 0.4]],
    )

    refresh_batch_statuses(client, manifest_path)

    assert client.batches.retrieve_calls == 1


def test_records_missing_requests_and_successful_vectors(tmp_path):
    manifest_path = create_workload(tmp_path, ["hash-a", "hash-b"])
    cache_path = tmp_path / "embedding_cache.parquet"

    client = FakeClient(
        output=batch_result("hash-a", [0.1, 0.2]),
        errors=json.dumps(
            {
                "custom_id": "hash-b",
                "error": {"message": "Temporary request failure"},
            }
        ),
    )

    _, chunk = collect_completed_batch(
        client,
        manifest_path,
        cache_path,
    )

    assert chunk["status"] == "collected_with_errors"
    assert chunk["cached_embedding_count"] == 1
    assert chunk["missing_request_count"] == 1

    missing_path = manifest_path.parent / chunk["missing_ids_path"]

    assert json.loads(missing_path.read_text()) == ["hash-b"]
    assert (manifest_path.parent / chunk["error_path"]).exists()

    cache = pd.read_parquet(cache_path)
    assert cache["content_hash"].tolist() == ["hash-a"]


def test_rejects_incorrect_embedding_dimensions(tmp_path):
    manifest_path = create_workload(
        tmp_path,
        ["hash-a"],
        dimensions=2,
    )
    cache_path = tmp_path / "embedding_cache.parquet"
    client = FakeClient(batch_result("hash-a", [0.1]))

    with pytest.raises(ValueError, match="expected \\(2,\\)"):
        collect_completed_batch(
            client,
            manifest_path,
            cache_path,
        )

    assert not cache_path.exists()