"""Train and serve a recency-weighted TF-IDF content recommender."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


@dataclass(frozen=True)
class TfidfConfig:
    """Configure TF-IDF vectorization, user profiles, and retrieval.

    Attributes:
        max_features: Maximum unigram and bigram vocabulary size.
        min_df: Minimum number of catalogue documents containing a term.
        recency_decay: Multiplicative weight per interaction step into the
            past, with the latest unique product receiving weight one.
        batch_size: Number of users scored against the catalogue at once.
    """

    max_features: int = 50_000
    min_df: int = 2
    recency_decay: float = 0.85
    batch_size: int = 256


class TfidfRecommender:
    """Recommend products from sparse text vectors and user history.

    Item vectors and user profiles are L2-normalized, so their dot product is
    cosine similarity. The stored user_items matrix contains recency weights
    and is also used to remove products observed during training.
    """

    def __init__(
        self,
        vectorizer: TfidfVectorizer,
        item_vectors: csr_matrix,
        user_profiles: csr_matrix,
        user_items: csr_matrix,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        config: TfidfConfig,
    ) -> None:
        self.vectorizer = vectorizer
        self.item_vectors = item_vectors
        self.user_profiles = user_profiles
        self.user_items = user_items
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.config = config

        self.user_to_index = {
            user_id: index
            for index, user_id in enumerate(user_ids)
        }

    @classmethod
    def fit(
        cls,
        catalog: pd.DataFrame,
        interactions: pd.DataFrame,
        config: TfidfConfig | None = None,
    ) -> "TfidfRecommender":
        """Fit product TF-IDF vectors and recency-weighted user profiles.

        Only the latest event for each user-item pair contributes to a profile.
        Events referencing products outside the supplied catalogue are ignored.

        Args:
            catalog: Candidate products containing item_id and catalog_text.
            interactions: Training events containing user_id, item_id, and
                timestamp.
            config: Optional vectorization and retrieval configuration.

        Returns:
            A fitted recommender with aligned sparse item and user matrices.

        Raises:
            ValueError: If required columns are missing, recency_decay is
                invalid, or no interactions match the catalogue.
        """

        config = config or TfidfConfig()

        catalog_required = {"item_id", "catalog_text"}
        interaction_required = {"user_id", "item_id", "timestamp"}

        missing_catalog = catalog_required - set(catalog.columns)
        missing_interactions = (
            interaction_required - set(interactions.columns)
        )

        if missing_catalog:
            raise ValueError(
                f"Missing catalogue columns: {sorted(missing_catalog)}"
            )

        if missing_interactions:
            raise ValueError(
                "Missing interaction columns: "
                f"{sorted(missing_interactions)}"
            )

        if not 0 < config.recency_decay <= 1:
            raise ValueError(
                "recency_decay must be between zero and one."
            )

        items = catalog[
            ["item_id", "catalog_text"]
        ].copy()

        items["item_id"] = items["item_id"].astype(str)
        items["catalog_text"] = (
            items["catalog_text"].fillna("").astype(str)
        )

        items = (
            items.drop_duplicates("item_id")
            .sort_values("item_id")
            .reset_index(drop=True)
        )

        item_ids = items["item_id"].to_numpy()
        item_to_index = {
            item_id: index
            for index, item_id in enumerate(item_ids)
        }

        vectorizer = TfidfVectorizer(
            strip_accents="unicode",
            stop_words="english",
            ngram_range=(1, 2),
            min_df=config.min_df,
            max_df=0.95,
            max_features=config.max_features,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )

        item_vectors = vectorizer.fit_transform(
            items["catalog_text"]
        ).tocsr()

        events = interactions[
            ["user_id", "item_id", "timestamp"]
        ].copy()

        events["user_id"] = events["user_id"].astype(str)
        events["item_id"] = events["item_id"].astype(str)
        events["timestamp"] = pd.to_numeric(
            events["timestamp"],
            errors="coerce",
        )

        events = events.dropna(subset=["timestamp"])
        events = events.loc[
            events["item_id"].isin(item_to_index)
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
                "No interactions match the catalogue."
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

        item_indices = events["item_id"].map(
            item_to_index
        ).to_numpy()

        user_items = csr_matrix(
            (
                weights,
                (user_indices, item_indices),
            ),
            shape=(len(user_ids), len(item_ids)),
            dtype=np.float32,
        )

        user_profiles = (user_items @ item_vectors).tocsr()
        user_profiles = normalize(
            user_profiles,
            norm="l2",
            axis=1,
            copy=False,
        )

        return cls(
            vectorizer=vectorizer,
            item_vectors=item_vectors,
            user_profiles=user_profiles,
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
        """Rank unseen products by batched cosine similarity.

        Dense score blocks are limited by config.batch_size, and partial
        selection avoids sorting the full catalogue for every user.

        Args:
            user_ids: Known training users to score; duplicates are removed in
                input order.
            k: Maximum number of unseen products returned per user.

        Returns:
            Rows containing user_id, item_id, one-based rank, and
            semantic_score.

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
                    "semantic_score",
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

        result_frames = []
        returned_k = min(k, len(self.item_ids))

        for start in range(
            0,
            len(requested_users),
            self.config.batch_size,
        ):
            batch_users = requested_users[
                start : start + self.config.batch_size
            ]

            batch_indices = np.asarray(
                [
                    self.user_to_index[user_id]
                    for user_id in batch_users
                ],
                dtype=np.int32,
            )

            scores = (
                self.user_profiles[batch_indices]
                @ self.item_vectors.T
            ).toarray()

            seen = self.user_items[batch_indices].tocoo()
            scores[seen.row, seen.col] = -np.inf

            top_items = np.argpartition(
                -scores,
                kth=returned_k - 1,
                axis=1,
            )[:, :returned_k]

            top_scores = np.take_along_axis(
                scores,
                top_items,
                axis=1,
            )

            order = np.argsort(
                -top_scores,
                axis=1,
                kind="stable",
            )

            top_items = np.take_along_axis(
                top_items,
                order,
                axis=1,
            )
            top_scores = np.take_along_axis(
                top_scores,
                order,
                axis=1,
            )

            flat_scores = top_scores.reshape(-1)
            valid = np.isfinite(flat_scores)

            result_frames.append(
                pd.DataFrame(
                    {
                        "user_id": np.repeat(
                            np.asarray(batch_users),
                            returned_k,
                        )[valid],
                        "item_id": self.item_ids[
                            top_items.reshape(-1)[valid]
                        ],
                        "rank": np.tile(
                            np.arange(1, returned_k + 1),
                            len(batch_users),
                        )[valid],
                        "semantic_score": flat_scores[valid],
                    }
                )
            )

        return pd.concat(result_frames, ignore_index=True)

    def save(self, directory: str | Path) -> None:
        """Persist the vectorizer, sparse matrices, and ID mappings.

        Args:
            directory: Output directory for the joblib vectorizer, NPZ sparse
                matrices, and Parquet user/item mappings.
        """

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            self.vectorizer,
            directory / "tfidf_vectorizer.joblib",
        )

        save_npz(
            directory / "item_vectors.npz",
            self.item_vectors,
        )
        save_npz(
            directory / "user_profiles.npz",
            self.user_profiles,
        )
        save_npz(
            directory / "user_items.npz",
            self.user_items,
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
