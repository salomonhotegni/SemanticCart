"""Publish serving-bundle embeddings to PostgreSQL."""

import argparse
import os
from pathlib import Path
from time import perf_counter

from semanticcart.pgvector_catalog import (
    sync_serving_embeddings,
)
from semanticcart.serving import ServingBundle


DEFAULT_SERVING_ROOT = Path(
    "data/artifacts/video_games_5core/serving"
)


def parse_args() -> argparse.Namespace:
    """Parse database and serving-bundle options."""
    parser = argparse.ArgumentParser(
        description=(
            "Load the current SemanticCart serving embeddings "
            "into PostgreSQL."
        )
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv(
            "SEMANTICCART_DATABASE_URL"
        ),
        help=(
            "PostgreSQL URL. Defaults to "
            "SEMANTICCART_DATABASE_URL."
        ),
    )
    parser.add_argument(
        "--serving-root",
        type=Path,
        default=DEFAULT_SERVING_ROOT,
        help="Root containing versioned serving bundles.",
    )
    parser.add_argument(
        "--version",
        default=None,
        help=(
            "Optional immutable bundle version. "
            "Defaults to CURRENT."
        ),
    )
    parser.add_argument(
        "--skip-index-rebuild",
        action="store_true",
        help=(
            "Publish vectors without rebuilding the HNSW index."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Load, validate, and transactionally publish one snapshot."""
    args = parse_args()

    if not args.database_url:
        raise SystemExit(
            "Set SEMANTICCART_DATABASE_URL or pass "
            "--database-url."
        )

    print("Loading and validating the serving bundle...")
    load_started = perf_counter()

    bundle = ServingBundle.load(
        serving_root=args.serving_root,
        version=args.version,
        verify_checksums=True,
    )

    bundle_load_seconds = (
        perf_counter() - load_started
    )

    print(
        "Publishing "
        f"{len(bundle.item_index.item_ids):,} vectors..."
    )
    sync_started = perf_counter()

    result = sync_serving_embeddings(
        database_url=args.database_url,
        bundle=bundle,
        rebuild_index=(
            not args.skip_index_rebuild
        ),
    )

    sync_seconds = perf_counter() - sync_started

    print()
    print("pgvector catalogue publication complete")
    print(f"Model version:   {result.model_version}")
    print(f"Embedding model: {result.embedding_model}")
    print(f"Rows:            {result.rows:,}")
    print(f"Dimensions:      {result.dimensions:,}")
    print(
        "Vector norms:    "
        f"{result.minimum_norm:.6f} "
        f"to {result.maximum_norm:.6f}"
    )
    print(
        f"Replaced rows:   {result.replaced_rows:,}"
    )
    print(
        f"Index rebuilt:   {result.index_rebuilt}"
    )
    print(
        "Bundle loading:  "
        f"{bundle_load_seconds:.2f} seconds"
    )
    print(
        "Database sync:   "
        f"{sync_seconds:.2f} seconds"
    )


if __name__ == "__main__":
    main()