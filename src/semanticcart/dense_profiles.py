"""Build recency-weighted user profiles from dense product vectors."""

from collections.abc import Iterable
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from scipy.sparse import (
    csr_matrix,
    load_npz,
    save_npz,
)

import faiss
import numpy as np
import pandas as pd

from semanticcart.dense_index import DenseItemIndex


@dataclass(frozen=True)
class DenseProfileConfig:
    """Configure dense user-profile construction.

    Attributes:
        recency_decay: Multiplicative weight per step into the past.
    """

    recency_decay: float = 0.85


class DenseUserProfiles:
    """Store normalized user vectors and their observed products."""

    def __init__(
        self,
        item_index: DenseItemIndex,
        user_ids: np.ndarray,
        user_profiles: np.ndarray,
        user_items: csr_matrix,
        config: DenseProfileConfig,
    ) -> None:
        self.item_index = item_index
        self.user_ids = user_ids
        self.user_profiles = user_profiles
        self.user_items = user_items
        self.config = config

        self.user_to_index = {
            user_id: index
            for index, user_id in enumerate(user_ids)
        }

    @classmethod
    def build(
        cls,
        item_index: DenseItemIndex,
        interactions: pd.DataFrame,
        config: DenseProfileConfig | None = None,
    ) -> "DenseUserProfiles":
        """Build recency-weighted profiles from chronological interactions.

        Only the latest event for each user-item pair contributes. Events for
        products outside the supplied item index are ignored.

        Args:
            item_index: Indexed products and normalized dense vectors.
            interactions: Events containing user_id, item_id, and timestamp.
            config: Optional recency-weighting configuration.

        Returns:
            Aligned user profiles and a sparse observed-item matrix.

        Raises:
            ValueError: If inputs are invalid or no events match the index.
        """
        config = config or DenseProfileConfig()
        required = {"user_id", "item_id", "timestamp"}
        missing = required - set(interactions.columns)

        if missing:
            raise ValueError(
                f"Missing interaction columns: {sorted(missing)}"
            )
        if not 0 < config.recency_decay <= 1:
            raise ValueError(
                "recency_decay must be between zero and one."
            )

        events = interactions[
            ["user_id", "item_id", "timestamp"]
        ].copy()

        events["timestamp"] = pd.to_numeric(
            events["timestamp"],
            errors="coerce",
        )
        events = events.dropna(
            subset=["user_id", "item_id", "timestamp"]
        )

        events["user_id"] = events["user_id"].astype(str)
        events["item_id"] = events["item_id"].astype(str)

        events = events.loc[
            events["item_id"].isin(item_index.item_to_index)
        ]

        events = (
            events.sort_values(
                ["user_id", "timestamp", "item_id"],
                ascending=[True, False, True],
            )
            .drop_duplicates(
                ["user_id", "item_id"],
                keep="first",
            )
            .reset_index(drop=True)
        )

        if events.empty:
            raise ValueError(
                "No interactions match the dense item index."
            )

        events["recency_position"] = (
            events.groupby("user_id").cumcount()
        )

        weights = np.power(
            config.recency_decay,
            events["recency_position"].to_numpy(),
        ).astype(np.float32)

        user_ids = np.sort(events["user_id"].unique())

        user_indices = pd.Categorical(
            events["user_id"],
            categories=user_ids,
        ).codes

        item_indices = (
            events["item_id"]
            .map(item_index.item_to_index)
            .to_numpy()
        )

        user_items = csr_matrix(
            (
                weights,
                (user_indices, item_indices),
            ),
            shape=(len(user_ids), len(item_index.item_ids)),
            dtype=np.float32,
        )

        user_profiles = np.asarray(
            user_items @ item_index.item_vectors,
            dtype=np.float32,
        )

        if np.any(np.linalg.norm(user_profiles, axis=1) == 0):
            raise ValueError(
                "Recency weighting produced a zero user profile."
            )

        user_profiles = np.ascontiguousarray(user_profiles)
        faiss.normalize_L2(user_profiles)

        return cls(
            item_index=item_index,
            user_ids=user_ids,
            user_profiles=user_profiles,
            user_items=user_items,
            config=config,
        )

    @classmethod
    def load(
        cls,
        item_index: DenseItemIndex,
        directory: str | Path,
    ) -> "DenseUserProfiles":
        """Load persisted profiles aligned to a dense item index.

        Args:
            item_index: Loaded item index used to validate item alignment.
            directory: Directory containing profiles, observed items,
                user mappings, and profile configuration.

        Returns:
            User profiles ready for semantic scoring and retrieval.

        Raises:
            FileNotFoundError: If a required artifact is missing.
            ValueError: If mappings, vectors, sparse data, or dimensions
                are inconsistent.
        """
        directory = Path(directory)
        paths = {
            "profiles": directory / "user_profiles.npy",
            "user_items": directory / "user_items.npz",
            "users": directory / "users.parquet",
            "config": directory / "profile_config.json",
        }

        missing = [
            path.name
            for path in paths.values()
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                f"Missing profile artifacts: {sorted(missing)}"
            )

        with paths["config"].open(
            encoding="utf-8",
        ) as source:
            config_values = json.load(source)

        try:
            config = DenseProfileConfig(**config_values)
        except TypeError as error:
            raise ValueError(
                "Invalid dense-profile configuration artifact."
            ) from error

        if not 0 < config.recency_decay <= 1:
            raise ValueError(
                "recency_decay must be between zero and one."
            )

        users = pd.read_parquet(paths["users"])
        required = {"user_index", "user_id"}
        missing_columns = required - set(users.columns)

        if missing_columns:
            raise ValueError(
                "Missing dense user mapping columns: "
                f"{sorted(missing_columns)}"
            )

        users["user_index"] = pd.to_numeric(
            users["user_index"],
            errors="coerce",
        )

        if users[["user_index", "user_id"]].isna().any().any():
            raise ValueError(
                "Dense user mapping values cannot be missing."
            )

        users = users.sort_values("user_index")
        expected_indexes = np.arange(len(users))

        if not np.array_equal(
            users["user_index"].to_numpy(),
            expected_indexes,
        ):
            raise ValueError(
                "Dense user indexes must be contiguous."
            )

        user_ids = users["user_id"].astype(str)

        if user_ids.eq("").any() or user_ids.duplicated().any():
            raise ValueError(
                "Dense user IDs must be nonempty and unique."
            )

        profiles = np.load(
            paths["profiles"],
            allow_pickle=False,
        ).astype(np.float32)

        expected_profile_shape = (
            len(user_ids),
            item_index.item_vectors.shape[1],
        )

        if profiles.shape != expected_profile_shape:
            raise ValueError(
                f"User profiles have shape {profiles.shape}; "
                f"expected {expected_profile_shape}."
            )
        if not np.isfinite(profiles).all():
            raise ValueError(
                "Persisted user profiles must be finite."
            )

        profile_norms = np.linalg.norm(
            profiles,
            axis=1,
        )

        if np.any(profile_norms == 0):
            raise ValueError(
                "Persisted user profiles cannot be zero."
            )
        if not np.allclose(
            profile_norms,
            1.0,
            atol=1e-4,
        ):
            raise ValueError(
                "Persisted user profiles must be normalized."
            )

        user_items = load_npz(
            paths["user_items"]
        ).tocsr().astype(np.float32)

        expected_item_shape = (
            len(user_ids),
            len(item_index.item_ids),
        )

        if user_items.shape != expected_item_shape:
            raise ValueError(
                f"Profile user-item matrix has shape "
                f"{user_items.shape}; expected "
                f"{expected_item_shape}."
            )
        if not np.isfinite(user_items.data).all():
            raise ValueError(
                "Persisted profile weights must be finite."
            )
        if np.any(user_items.data <= 0):
            raise ValueError(
                "Persisted profile weights must be positive."
            )

        return cls(
            item_index=item_index,
            user_ids=user_ids.to_numpy(),
            user_profiles=np.ascontiguousarray(profiles),
            user_items=user_items,
            config=config,
        )

    def indices_for(
        self,
        user_ids: Iterable[str],
    ) -> np.ndarray:
        """Resolve known user IDs into profile row positions."""
        requested = list(map(str, user_ids))
        unknown = [
            user_id
            for user_id in requested
            if user_id not in self.user_to_index
        ]

        if unknown:
            raise ValueError(f"Unknown users: {unknown[:5]}")

        return np.asarray(
            [self.user_to_index[user_id] for user_id in requested],
            dtype=np.int32,
        )

    def save(self, directory: str | Path) -> None:
        """Persist dense profiles, observed items, users, and configuration.

        Args:
            directory: Model artifact directory to create or update.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        np.save(
            directory / "user_profiles.npy",
            self.user_profiles,
            allow_pickle=False,
        )

        save_npz(
            directory / "user_items.npz",
            self.user_items,
        )

        pd.DataFrame(
            {
                "user_index": np.arange(
                    len(self.user_ids),
                    dtype=np.int32,
                ),
                "user_id": self.user_ids,
            }
        ).to_parquet(
            directory / "users.parquet",
            index=False,
        )

        with (directory / "profile_config.json").open(
            "w",
            encoding="utf-8",
        ) as output:
            json.dump(asdict(self.config), output, indent=2)