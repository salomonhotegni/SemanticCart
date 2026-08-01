"""Load and validate immutable SemanticCart serving bundles."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from semanticcart.collaborative import ALSRecommender
from semanticcart.dense_index import DenseItemIndex
from semanticcart.dense_profiles import DenseUserProfiles


CATALOG_COLUMNS = {
    "item_id",
    "title",
    "main_category",
    "categories",
    "store",
    "price",
    "image_url",
    "popularity",
}

RANKING_KEYS = {
    "model",
    "k",
    "candidate_k",
    "session_length",
    "session_weight",
    "diversity",
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one artifact."""
    digest = hashlib.sha256()

    with path.open("rb") as source:
        for block in iter(
            lambda: source.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def _safe_artifact_path(
    bundle_directory: Path,
    relative_path: str,
) -> Path:
    """Resolve one manifest path without allowing bundle escape."""
    bundle_resolved = bundle_directory.resolve()
    candidate = (
        bundle_directory / relative_path
    ).resolve()

    try:
        candidate.relative_to(bundle_resolved)
    except ValueError as error:
        raise ValueError(
            f"Artifact path escapes bundle: {relative_path}"
        ) from error

    return candidate


def _resolve_version(
    serving_root: Path,
    version: str | None,
) -> str:
    """Resolve an explicit version or the current-version pointer."""
    if version is None:
        current_path = serving_root / "CURRENT"

        if not current_path.exists():
            raise FileNotFoundError(
                f"Missing serving pointer: {current_path}"
            )

        version = current_path.read_text(
            encoding="utf-8"
        ).strip()

    if (
        not version
        or version in {".", ".."}
        or "/" in version
        or "\\" in version
        or Path(version).name != version
    ):
        raise ValueError(
            "Serving version must be one directory name."
        )

    return version


@dataclass
class ServingBundle:
    """Hold loaded models, metadata, and frozen ranking settings."""

    version: str
    directory: Path
    manifest: dict
    ranking_config: dict
    als: ALSRecommender
    item_index: DenseItemIndex
    profiles: DenseUserProfiles
    catalog: pd.DataFrame
    item_features: pd.DataFrame

    @classmethod
    def load(
        cls,
        serving_root: str | Path,
        version: str | None = None,
        verify_checksums: bool = True,
    ) -> "ServingBundle":
        """Load one current or explicitly versioned serving bundle.

        Args:
            serving_root: Directory containing CURRENT and version folders.
            version: Optional immutable model version to load.
            verify_checksums: Whether to hash every inventoried artifact.

        Returns:
            A fully validated runtime bundle.

        Raises:
            FileNotFoundError: If pointers, bundles, or artifacts are missing.
            ValueError: If manifests, checksums, mappings, or counts disagree.
        """
        serving_root = Path(serving_root)
        version = _resolve_version(
            serving_root,
            version,
        )
        bundle_directory = serving_root / version

        if not bundle_directory.is_dir():
            raise FileNotFoundError(
                f"Serving bundle not found: {bundle_directory}"
            )

        manifest_path = bundle_directory / "manifest.json"
        ranking_path = (
            bundle_directory / "ranking_config.json"
        )
        catalog_path = bundle_directory / "catalog.parquet"

        for path in (
            manifest_path,
            ranking_path,
            catalog_path,
        ):
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing serving artifact: {path}"
                )

        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        ranking_config = json.loads(
            ranking_path.read_text(encoding="utf-8")
        )

        if manifest.get("model_version") != version:
            raise ValueError(
                "Manifest version does not match bundle directory."
            )
        if manifest.get("ranking_config") != ranking_config:
            raise ValueError(
                "Ranking configuration does not match manifest."
            )

        missing_ranking_keys = (
            RANKING_KEYS - set(ranking_config)
        )

        if missing_ranking_keys:
            raise ValueError(
                "Missing ranking configuration keys: "
                f"{sorted(missing_ranking_keys)}"
            )

        if (
            ranking_config["k"] <= 0
            or ranking_config["candidate_k"]
            < ranking_config["k"]
            or ranking_config["session_length"] <= 0
            or not 0
            <= ranking_config["session_weight"]
            <= 1
        ):
            raise ValueError(
                "Frozen ranking configuration is invalid."
            )

        inventory = manifest.get("artifacts")

        if not isinstance(inventory, dict):
            raise ValueError(
                "Manifest artifact inventory is missing."
            )

        for relative_path, record in inventory.items():
            artifact_path = _safe_artifact_path(
                bundle_directory,
                relative_path,
            )

            if not artifact_path.is_file():
                raise FileNotFoundError(
                    f"Missing inventoried artifact: {artifact_path}"
                )

            if artifact_path.stat().st_size != record.get(
                "bytes"
            ):
                raise ValueError(
                    f"Artifact size mismatch: {relative_path}"
                )

            if (
                verify_checksums
                and sha256_file(artifact_path)
                != record.get("sha256")
            ):
                raise ValueError(
                    f"Artifact checksum mismatch: {relative_path}"
                )

        with threadpool_limits(
            limits=1,
            user_api="blas",
        ):
            als = ALSRecommender.load(
                bundle_directory / "als"
            )

        item_index = DenseItemIndex.load(
            bundle_directory / "dense_index"
        )
        profiles = DenseUserProfiles.load(
            item_index,
            bundle_directory / "recent_profiles",
        )

        catalog = pd.read_parquet(catalog_path)
        missing_catalog_columns = (
            CATALOG_COLUMNS - set(catalog.columns)
        )

        if missing_catalog_columns:
            raise ValueError(
                "Missing serving catalogue columns: "
                f"{sorted(missing_catalog_columns)}"
            )

        catalog = catalog[
            [
                "item_id",
                "title",
                "main_category",
                "categories",
                "store",
                "price",
                "image_url",
                "popularity",
            ]
        ].copy()
        catalog["item_id"] = catalog[
            "item_id"
        ].astype(str)

        if (
            catalog["item_id"].eq("").any()
            or catalog["item_id"].duplicated().any()
        ):
            raise ValueError(
                "Serving catalogue item IDs must be nonempty and unique."
            )

        catalog["popularity"] = pd.to_numeric(
            catalog["popularity"],
            errors="coerce",
        )

        popularity = catalog[
            "popularity"
        ].to_numpy()

        if (
            np.isnan(popularity).any()
            or not np.isfinite(popularity).all()
            or np.any(popularity < 0)
        ):
            raise ValueError(
                "Serving popularity must be finite and non-negative."
            )

        catalog_items = set(catalog["item_id"])
        indexed_items = set(item_index.item_ids)

        if catalog_items != indexed_items:
            raise ValueError(
                "Serving catalogue and dense index items differ."
            )
        if not set(als.item_ids).issubset(indexed_items):
            raise ValueError(
                "ALS items are not a subset of the semantic catalogue."
            )
        if set(als.user_ids) != set(profiles.user_ids):
            raise ValueError(
                "ALS and dense-profile users differ."
            )

        catalog = (
            catalog.set_index("item_id")
            .loc[item_index.item_ids]
            .reset_index()
        )

        item_features = catalog[
            [
                "item_id",
                "categories",
                "price",
                "popularity",
            ]
        ].copy()
        item_features["embedding"] = list(
            item_index.item_vectors
        )

        expected_counts = {
            "users": len(als.user_ids),
            "als_items": len(als.item_ids),
            "semantic_items": len(item_index.item_ids),
            "catalog_only_items": (
                len(item_index.item_ids)
                - len(als.item_ids)
            ),
        }

        for name, expected in expected_counts.items():
            if manifest.get(name) != expected:
                raise ValueError(
                    f"Manifest count mismatch for {name}."
                )

        return cls(
            version=version,
            directory=bundle_directory,
            manifest=manifest,
            ranking_config=ranking_config,
            als=als,
            item_index=item_index,
            profiles=profiles,
            catalog=catalog,
            item_features=item_features,
        )

    def model_info(self) -> dict:
        """Return API-safe model metadata."""
        return {
            "model_version": self.version,
            "model": self.ranking_config["model"],
            "dataset": self.manifest["dataset"],
            "created_at_utc": self.manifest[
                "created_at_utc"
            ],
            "fit_splits": self.manifest["fit_splits"],
            "users": self.manifest["users"],
            "als_items": self.manifest["als_items"],
            "semantic_items": self.manifest[
                "semantic_items"
            ],
            "catalog_only_items": self.manifest[
                "catalog_only_items"
            ],
            "embedding_config": self.manifest[
                "embedding_config"
            ],
            "ranking_config": self.ranking_config,
        }