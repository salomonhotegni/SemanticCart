"""Train and serve an implicit-feedback ALS collaborative recommender."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
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

    def save(self, directory: str | Path) -> None:
        """Persist the ALS model and row-to-ID mappings.

        Args:
            directory: Output directory for als_model.npz, users.parquet, and
                items.parquet.
        """

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        self.model.save(str(directory / "als_model.npz"))

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
