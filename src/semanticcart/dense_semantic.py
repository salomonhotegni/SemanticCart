"""Recommend unseen products from dense user profiles and FAISS."""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from semanticcart.dense_profiles import DenseUserProfiles


@dataclass(frozen=True)
class DenseSemanticConfig:
    """Configure batched semantic candidate retrieval.

    Attributes:
        batch_size: Number of user profiles queried together.
        candidate_multiplier: Candidate pool size relative to requested Top-K.
    """

    batch_size: int = 512
    candidate_multiplier: int = 10


class DenseSemanticRecommender:
    """Retrieve and rank unseen products for established users."""

    RESULT_COLUMNS = [
        "user_id",
        "item_id",
        "rank",
        "semantic_score",
    ]
    
    SESSION_RESULT_COLUMNS = [
        "item_id",
        "rank",
        "semantic_score",
    ]

    def __init__(
        self,
        profiles: DenseUserProfiles,
        config: DenseSemanticConfig | None = None,
    ) -> None:
        self.profiles = profiles
        self.item_index = profiles.item_index
        self.config = config or DenseSemanticConfig()

        if self.config.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.config.candidate_multiplier <= 0:
            raise ValueError(
                "candidate_multiplier must be positive."
            )

    def recommend_for_users(
        self,
        user_ids: Iterable[str],
        k: int = 10,
    ) -> pd.DataFrame:
        """Recommend unseen products using batched HNSW retrieval.

        The candidate pool is large enough to replace products removed by
        seen-item filtering. Duplicate requested users are evaluated once.

        Args:
            user_ids: Established users represented in the profile store.
            k: Maximum number of recommendations returned per user.

        Returns:
            Rows containing user ID, item ID, one-based rank, and cosine score.

        Raises:
            ValueError: If k is invalid or a requested user is unknown.
        """
        if k <= 0:
            raise ValueError("k must be greater than zero.")

        requested_users = list(
            dict.fromkeys(map(str, user_ids))
        )

        if not requested_users:
            return pd.DataFrame(columns=self.RESULT_COLUMNS)

        user_indices = self.profiles.indices_for(requested_users)
        records = []

        for start in range(
            0,
            len(requested_users),
            self.config.batch_size,
        ):
            stop = min(
                start + self.config.batch_size,
                len(requested_users),
            )

            batch_users = requested_users[start:stop]
            batch_indices = user_indices[start:stop]

            batch_seen = self.profiles.user_items[
                batch_indices
            ].tocsr()

            maximum_seen = int(
                np.max(batch_seen.getnnz(axis=1))
            )

            candidate_count = min(
                len(self.item_index.item_ids),
                max(
                    k * self.config.candidate_multiplier,
                    k + maximum_seen,
                ),
            )

            scores, candidates = self.item_index.search(
                self.profiles.user_profiles[batch_indices],
                candidate_count,
            )

            for row, user_id in enumerate(batch_users):
                seen_start = batch_seen.indptr[row]
                seen_stop = batch_seen.indptr[row + 1]
                seen_items = set(
                    batch_seen.indices[
                        seen_start:seen_stop
                    ].tolist()
                )

                rank = 0

                for score, item_index in zip(
                    scores[row],
                    candidates[row],
                ):
                    item_index = int(item_index)

                    if item_index < 0 or item_index in seen_items:
                        continue

                    rank += 1
                    records.append(
                        {
                            "user_id": user_id,
                            "item_id": self.item_index.item_ids[
                                item_index
                            ],
                            "rank": rank,
                            "semantic_score": float(score),
                        }
                    )

                    if rank == k:
                        break

        return pd.DataFrame.from_records(
            records,
            columns=self.RESULT_COLUMNS,
        )
        
    def recommend_from_history(
        self,
        viewed_item_ids: Iterable[str],
        k: int = 10,
        recency_decay: float | None = None,
    ) -> pd.DataFrame:
        """Recommend products from a new user's current session history.

        The input history is ordered oldest to newest. Repeated products retain
        only their latest position, and all viewed products are excluded.

        Args:
            viewed_item_ids: Ordered product history for an anonymous session.
            k: Maximum number of unseen recommendations.
            recency_decay: Optional session-specific decay; defaults to the
                profile-store configuration.

        Returns:
            Ranked product IDs and cosine-similarity scores.

        Raises:
            ValueError: If history, k, decay, or product IDs are invalid.
        """
        if k <= 0:
            raise ValueError("k must be greater than zero.")

        history = list(map(str, viewed_item_ids))

        if not history:
            raise ValueError("At least one viewed product is required.")

        unique_history = list(
            dict.fromkeys(reversed(history))
        )
        unique_history.reverse()

        unknown_items = [
            item_id
            for item_id in unique_history
            if item_id not in self.item_index.item_to_index
        ]

        if unknown_items:
            raise ValueError(
                f"Unknown products: {unknown_items[:5]}"
            )

        decay = (
            self.profiles.config.recency_decay
            if recency_decay is None
            else recency_decay
        )

        if not 0 < decay <= 1:
            raise ValueError(
                "recency_decay must be between zero and one."
            )

        viewed_indices = np.asarray(
            [
                self.item_index.item_to_index[item_id]
                for item_id in unique_history
            ],
            dtype=np.int32,
        )

        viewed_vectors = self.item_index.item_vectors[
            viewed_indices
        ]

        ages = np.arange(
            len(viewed_indices) - 1,
            -1,
            -1,
        )
        weights = np.power(decay, ages)

        session_profile = np.average(
            viewed_vectors,
            axis=0,
            weights=weights,
        ).astype(np.float32)

        candidate_count = min(
            len(self.item_index.item_ids),
            max(
                k * self.config.candidate_multiplier,
                k + len(viewed_indices),
            ),
        )

        scores, candidates = self.item_index.search(
            session_profile,
            candidate_count,
        )

        seen_items = set(viewed_indices.tolist())
        records = []
        rank = 0

        for score, item_index in zip(
            scores[0],
            candidates[0],
        ):
            item_index = int(item_index)

            if item_index < 0 or item_index in seen_items:
                continue

            rank += 1
            records.append(
                {
                    "item_id": self.item_index.item_ids[item_index],
                    "rank": rank,
                    "semantic_score": float(score),
                }
            )

            if rank == k:
                break

        return pd.DataFrame.from_records(
            records,
            columns=self.SESSION_RESULT_COLUMNS,
        )