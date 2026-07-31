"""Train and serve an implicit-feedback ALS collaborative recommender."""

from collections.abc import Iterable
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import (
    csr_matrix,
    load_npz,
    save_npz,
)
from threadpoolctl import threadpool_limits


@dataclass(frozen=True)
class ALSConfig:
    """Configure implicit Alternating Least Squares training and retrieval.

    Attributes:
        factors: Number of latent dimensions for users and items.
        regularization: L2 regularization applied during factor updates.
        alpha: Confidence multiplier for observed binary interactions.
        iterations: Number of alternating optimization passes.
        random_state: Seed used to initialize latent factors.
        batch_size: Number of users passed to batched recommendation.
    """

    factors: int = 64
    regularization: float = 0.05
    alpha: float = 20.0
    iterations: int = 20
    random_state: int = 42
    batch_size: int = 1024


def _load_id_mapping(
    path: Path,
    index_column: str,
    id_column: str,
) -> np.ndarray:
    """Load and validate one contiguous model ID mapping."""
    frame = pd.read_parquet(path)
    required = {index_column, id_column}
    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"Missing mapping columns in {path.name}: "
            f"{sorted(missing)}"
        )

    frame[index_column] = pd.to_numeric(
        frame[index_column],
        errors="coerce",
    )

    if frame[[index_column, id_column]].isna().any().any():
        raise ValueError(
            f"Mapping values cannot be missing in {path.name}."
        )
    if frame[index_column].duplicated().any():
        raise ValueError(
            f"Mapping indexes must be unique in {path.name}."
        )

    frame = frame.sort_values(index_column)
    expected_indexes = np.arange(len(frame))

    if not np.array_equal(
        frame[index_column].to_numpy(),
        expected_indexes,
    ):
        raise ValueError(
            f"Mapping indexes must be contiguous in {path.name}."
        )

    identifiers = frame[id_column].astype(str)

    if identifiers.duplicated().any():
        raise ValueError(
            f"Mapping IDs must be unique in {path.name}."
        )

    return identifiers.to_numpy()


class ALSRecommender:
    """Recommend unseen products from implicit user-item latent factors.

    User and item arrays preserve the row and column alignment of user_items
    and the trained implicit model.
    """

    def __init__(
        self,
        model: AlternatingLeastSquares,
        user_items: csr_matrix,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        config: ALSConfig,
    ) -> None:
        self.model = model
        self.user_items = user_items
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.config = config

        self.user_to_index = {
            user_id: index for index, user_id in enumerate(user_ids)
        }

    @classmethod
    def fit(
        cls,
        interactions: pd.DataFrame,
        config: ALSConfig | None = None,
    ) -> "ALSRecommender":
        """Fit ALS to a binary sparse interaction matrix.

        Repeated user-item events are collapsed to one positive observation.
        BLAS is limited to one thread while implicit uses its native parallel
        training implementation.

        Args:
            interactions: Training events containing user_id and item_id.
            config: Optional ALS hyperparameters and retrieval batch size.

        Returns:
            A fitted recommender with ID mappings and the training matrix.

        Raises:
            ValueError: If a required interaction column is missing.
        """

        config = config or ALSConfig()
        required_columns = {"user_id", "item_id"}
        missing = required_columns - set(interactions.columns)

        if missing:
            raise ValueError(f"Missing interaction columns: {sorted(missing)}")

        user_values = interactions["user_id"].astype(str)
        item_values = interactions["item_id"].astype(str)

        user_ids = np.sort(user_values.unique())
        item_ids = np.sort(item_values.unique())

        user_indices = pd.Categorical(
            user_values,
            categories=user_ids,
        ).codes

        item_indices = pd.Categorical(
            item_values,
            categories=item_ids,
        ).codes

        user_items = csr_matrix(
            (
                np.ones(len(interactions), dtype=np.float32),
                (user_indices, item_indices),
            ),
            shape=(len(user_ids), len(item_ids)),
            dtype=np.float32,
        )

        user_items.sum_duplicates()
        user_items.data[:] = 1.0

        with threadpool_limits(limits=1, user_api="blas"):
            model = AlternatingLeastSquares(
                factors=config.factors,
                regularization=config.regularization,
                alpha=config.alpha,
                iterations=config.iterations,
                random_state=config.random_state,
            )

            model.fit(user_items, show_progress=True)

        return cls(
            model=model,
            user_items=user_items,
            user_ids=user_ids,
            item_ids=item_ids,
            config=config,
        )

    def recommend_for_users(
        self,
        user_ids: Iterable[str],
        k: int = 10,
    ) -> pd.DataFrame:
        """Generate batched Top-K collaborative recommendations.

        Args:
            user_ids: Known training users to score; duplicates are removed in
                input order.
            k: Maximum number of unseen products returned per user.

        Returns:
            Rows containing user_id, item_id, one-based rank, and
            collaborative_score.

        Raises:
            ValueError: If k is not positive or a user was unseen in training.
        """

        if k <= 0:
            raise ValueError("k must be greater than zero.")

        requested_users = list(dict.fromkeys(map(str, user_ids)))

        if not requested_users:
            return pd.DataFrame(
                columns=[
                    "user_id",
                    "item_id",
                    "rank",
                    "collaborative_score",
                ]
            )

        unknown_users = [
            user_id
            for user_id in requested_users
            if user_id not in self.user_to_index
        ]

        if unknown_users:
            raise ValueError(
                f"Unknown users: {unknown_users[:5]}"
            )

        recommendation_frames = []

        for start in range(0, len(requested_users), self.config.batch_size):
            batch_users = requested_users[
                start : start + self.config.batch_size
            ]

            batch_indices = np.asarray(
                [self.user_to_index[user_id] for user_id in batch_users],
                dtype=np.int32,
            )

            item_indices, scores = self.model.recommend(
                batch_indices,
                self.user_items[batch_indices],
                N=k,
                filter_already_liked_items=True,
            )

            item_indices = np.atleast_2d(item_indices)
            scores = np.atleast_2d(scores)
            returned_k = item_indices.shape[1]

            flat_items = item_indices.reshape(-1)
            valid = flat_items >= 0

            recommendation_frames.append(
                pd.DataFrame(
                    {
                        "user_id": np.repeat(
                            np.asarray(batch_users),
                            returned_k,
                        )[valid],
                        "item_id": self.item_ids[flat_items[valid]],
                        "rank": np.tile(
                            np.arange(1, returned_k + 1),
                            len(batch_users),
                        )[valid],
                        "collaborative_score": scores.reshape(-1)[valid],
                    }
                )
            )

        return pd.concat(recommendation_frames, ignore_index=True)

    @classmethod
    def load(
        cls,
        directory: str | Path,
    ) -> "ALSRecommender":
        """Load a complete persisted ALS recommender.

        Args:
            directory: Directory containing the model, sparse interaction
                matrix, ID mappings, and configuration.

        Returns:
            A recommender ready for batched unseen-item retrieval.

        Raises:
            FileNotFoundError: If any required artifact is missing.
            ValueError: If mappings, configuration, or shapes are invalid.
        """
        directory = Path(directory)
        paths = {
            "model": directory / "als_model.npz",
            "user_items": directory / "user_items.npz",
            "users": directory / "users.parquet",
            "items": directory / "items.parquet",
            "config": directory / "config.json",
        }

        missing = [
            path.name
            for path in paths.values()
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                f"Missing ALS artifacts: {sorted(missing)}"
            )

        with paths["config"].open(
            encoding="utf-8",
        ) as source:
            config_values = json.load(source)

        try:
            config = ALSConfig(**config_values)
        except TypeError as error:
            raise ValueError(
                "Invalid ALS configuration artifact."
            ) from error

        model = AlternatingLeastSquares.load(
            str(paths["model"])
        )
        user_items = load_npz(
            paths["user_items"]
        ).tocsr().astype(np.float32)

        user_ids = _load_id_mapping(
            paths["users"],
            "user_index",
            "user_id",
        )
        item_ids = _load_id_mapping(
            paths["items"],
            "item_index",
            "item_id",
        )

        expected_shape = (
            len(user_ids),
            len(item_ids),
        )

        if user_items.shape != expected_shape:
            raise ValueError(
                f"User-item matrix has shape {user_items.shape}; "
                f"expected {expected_shape}."
            )
        if model.user_factors.shape[0] != len(user_ids):
            raise ValueError(
                "User factors do not match the user mapping."
            )
        if model.item_factors.shape[0] != len(item_ids):
            raise ValueError(
                "Item factors do not match the item mapping."
            )

        return cls(
            model=model,
            user_items=user_items,
            user_ids=user_ids,
            item_ids=item_ids,
            config=config,
        )

    def save(self, directory: str | Path) -> None:
        """Persist the ALS model and row-to-ID mappings.

        Args:
            directory: Output directory for als_model.npz, users.parquet, and
                items.parquet.
        """

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        self.model.save(str(directory / "als_model.npz"))
        save_npz(
            directory / "user_items.npz",
            self.user_items,
        )

        with (
            directory / "config.json"
        ).open("w", encoding="utf-8") as output:
            json.dump(
                asdict(self.config),
                output,
                indent=2,
            )

        pd.DataFrame(
            {
                "user_index": np.arange(len(self.user_ids)),
                "user_id": self.user_ids,
            }
        ).to_parquet(directory / "users.parquet", index=False)

        pd.DataFrame(
            {
                "item_index": np.arange(len(self.item_ids)),
                "item_id": self.item_ids,
            }
        ).to_parquet(directory / "items.parquet", index=False)
