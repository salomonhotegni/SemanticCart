from types import SimpleNamespace

import pytest

from semanticcart.openai_batch_jobs import (
    load_manifest,
    refresh_batch_statuses,
    save_manifest,
    submit_next_batch,
)


class FakeFiles:
    """Record fake OpenAI file uploads."""

    def __init__(self):
        self.create_calls = 0
        self.uploaded_purpose = None

    def create(self, file, purpose):
        self.create_calls += 1
        self.uploaded_purpose = purpose

        return SimpleNamespace(id="file-test")


class FakeBatches:
    """Provide configurable fake Batch API responses."""

    def __init__(self, retrieve_status="in_progress"):
        self.retrieve_status = retrieve_status
        self.create_calls = 0
        self.created_request = None

    def create(
        self,
        input_file_id,
        endpoint,
        completion_window,
        metadata,
    ):
        self.create_calls += 1
        self.created_request = {
            "input_file_id": input_file_id,
            "endpoint": endpoint,
            "completion_window": completion_window,
            "metadata": metadata,
        }

        return SimpleNamespace(
            id="batch-test",
            status="validating",
            output_file_id=None,
            error_file_id=None,
        )

    def retrieve(self, batch_id):
        assert batch_id == "batch-test"

        completed = self.retrieve_status == "completed"

        return SimpleNamespace(
            id=batch_id,
            status=self.retrieve_status,
            output_file_id=(
                "file-output" if completed else None
            ),
            error_file_id=None,
        )


class FakeClient:
    """Expose fake files and batches resources."""

    def __init__(self, retrieve_status="in_progress"):
        self.files = FakeFiles()
        self.batches = FakeBatches(retrieve_status)


def create_manifest(tmp_path, chunks):
    """Create a workload manifest and referenced input files."""
    run_directory = tmp_path / "workload"
    run_directory.mkdir()

    for chunk in chunks:
        filename = chunk.get("filename")

        if filename:
            (run_directory / filename).write_text(
                "{}\n",
                encoding="utf-8",
            )

    manifest_path = run_directory / "manifest.json"

    manifest = {
        "schema_version": 1,
        "workload_id": "workload-test",
        "chunks": chunks,
    }

    save_manifest(manifest_path, manifest)
    return manifest_path


def test_submit_uploads_and_records_batch_state(tmp_path):
    manifest_path = create_manifest(
        tmp_path,
        [
            {
                "chunk_number": 1,
                "filename": "batch_0001.jsonl",
                "status": "prepared",
                "input_file_id": None,
                "batch_id": None,
                "output_file_id": None,
                "error_file_id": None,
            }
        ],
    )

    client = FakeClient()

    _, submitted_chunk = submit_next_batch(
        client,
        manifest_path,
    )

    assert client.files.create_calls == 1
    assert client.files.uploaded_purpose == "batch"
    assert client.batches.create_calls == 1

    assert client.batches.created_request == {
        "input_file_id": "file-test",
        "endpoint": "/v1/embeddings",
        "completion_window": "24h",
        "metadata": {
            "workload_id": "workload-test",
            "chunk_number": "1",
        },
    }

    assert submitted_chunk["input_file_id"] == "file-test"
    assert submitted_chunk["batch_id"] == "batch-test"
    assert submitted_chunk["status"] == "submitted"
    assert submitted_chunk["server_status"] == "validating"

    persisted = load_manifest(manifest_path)
    assert persisted["chunks"][0] == submitted_chunk

    with pytest.raises(
        RuntimeError,
        match="still in_progress",
    ):
        submit_next_batch(client, manifest_path)

    assert client.files.create_calls == 1
    assert client.batches.create_calls == 1


def test_refresh_records_completed_output(tmp_path):
    manifest_path = create_manifest(
        tmp_path,
        [
            {
                "chunk_number": 1,
                "filename": "batch_0001.jsonl",
                "status": "submitted",
                "input_file_id": "file-test",
                "batch_id": "batch-test",
                "output_file_id": None,
                "error_file_id": None,
            },
            {
                "chunk_number": 2,
                "filename": "batch_0002.jsonl",
                "status": "prepared",
                "input_file_id": None,
                "batch_id": None,
                "output_file_id": None,
                "error_file_id": None,
            },
        ],
    )

    client = FakeClient(retrieve_status="completed")

    manifest = refresh_batch_statuses(
        client,
        manifest_path,
    )

    completed_chunk = manifest["chunks"][0]

    assert completed_chunk["status"] == "completed"
    assert completed_chunk["server_status"] == "completed"
    assert completed_chunk["output_file_id"] == "file-output"

    with pytest.raises(
        RuntimeError,
        match="must be collected",
    ):
        submit_next_batch(client, manifest_path)

    assert client.files.create_calls == 0
    assert client.batches.create_calls == 0


def test_failed_batch_blocks_later_submission(tmp_path):
    manifest_path = create_manifest(
        tmp_path,
        [
            {
                "chunk_number": 1,
                "filename": "batch_0001.jsonl",
                "status": "submitted",
                "input_file_id": "file-test",
                "batch_id": "batch-test",
                "output_file_id": None,
                "error_file_id": None,
            },
            {
                "chunk_number": 2,
                "filename": "batch_0002.jsonl",
                "status": "prepared",
                "input_file_id": None,
                "batch_id": None,
                "output_file_id": None,
                "error_file_id": None,
            },
        ],
    )

    client = FakeClient(retrieve_status="failed")

    with pytest.raises(
        RuntimeError,
        match="ended with status failed",
    ):
        submit_next_batch(client, manifest_path)

    assert client.files.create_calls == 0
    assert client.batches.create_calls == 0