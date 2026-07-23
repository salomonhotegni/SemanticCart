"""Submit prepared embedding batches and inspect their server state."""

import argparse
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from semanticcart.openai_batch_jobs import (
    refresh_batch_statuses,
    submit_next_batch,
)
from semanticcart.openai_batch_results import collect_completed_batch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse the requested operation and workload manifest."""
    parser = argparse.ArgumentParser(
        description="Manage OpenAI embedding Batch chunks."
    )
    parser.add_argument(
        "command",
        choices=["status", "submit-next", "collect"],
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data/artifacts/video_games_5core/openai_embeddings"
            / "embedding_cache.parquet"
        ),
        help="Versioned Parquet embedding cache.",
    )
    return parser.parse_args()


def print_status(manifest: dict) -> None:
    """Print local and server lifecycle state for every chunk."""
    print(f"Workload: {manifest['workload_id']}")
    print()

    for chunk in manifest["chunks"]:
        local_status = chunk.get("status", "-")
        server_status = chunk.get("server_status") or "-"
        batch_id = chunk.get("batch_id") or "-"

        print(
            f"Chunk {chunk['chunk_number']:04d}: "
            f"local={local_status} "
            f"server={server_status} "
            f"batch_id={batch_id}"
        )


def main() -> None:
    """Execute one manifest-management operation."""
    args = parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    client = OpenAI()

    if args.command == "status":
        manifest = refresh_batch_statuses(
            client,
            args.manifest,
        )
        print_status(manifest)
        return
    
    if args.command == "collect":
        manifest, collected_chunk = collect_completed_batch(
            client,
            args.manifest,
            args.cache,
        )

        print(
            f"Collected chunk "
            f"{collected_chunk['chunk_number']:04d}"
        )
        print(
            f"Cached vectors: "
            f"{collected_chunk['cached_embedding_count']:,}"
        )
        print(
            f"Missing requests: "
            f"{collected_chunk['missing_request_count']:,}"
        )
        print(f"Local status:   {collected_chunk['status']}")
        print(f"Cache:          {args.cache}")
        print()
        print_status(manifest)
        return

    manifest, submitted_chunk = submit_next_batch(
        client,
        args.manifest,
    )

    print(
        f"Submitted chunk "
        f"{submitted_chunk['chunk_number']:04d}"
    )
    print(
        f"Requests:     "
        f"{submitted_chunk['request_count']:,}"
    )
    print(
        f"Tokens:       "
        f"{submitted_chunk['token_count']:,}"
    )
    print(
        f"Input file:   "
        f"{submitted_chunk['input_file_id']}"
    )
    print(
        f"Batch ID:     "
        f"{submitted_chunk['batch_id']}"
    )
    print(
        f"Server status:"
        f" {submitted_chunk['server_status']}"
    )
    print()
    print_status(manifest)


if __name__ == "__main__":
    main()