"""Serve returning-user, cold-start, and similar-product recommendations."""

from collections.abc import Iterable
from dataclasses import replace
from threading import RLock

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from semanticcart.diversity import (
    DiversityRerankConfig,
    rerank_diverse_candidates,
)
from semanticcart.hybrid import (
    HybridConfig,
    rerank_collaborative_candidates,
)
from semanticcart.serving import ServingBundle


RECOMMENDATION_COLUMNS = [
    "user_id",
    "item_id",
    "rank",
    "recommendation_score",
    "relevance_score",
    "collaborative_score",
    "semantic_score",
    "strategy",
    "title",
    "main_category",
    "categories",
    "store",
    "price",
    "image_url",
    "popularity",
]

SIMILAR_COLUMNS = [
    "item_id",
    "rank",
    "similarity_score",
    "title",
    "main_category",
    "categories",
    "store",
    "price",
    "image_url",
    "popularity",
]


class RecommendationService:
    """Generate recommendations from one immutable serving bundle."""

    def __init__(
        self,
        bundle: ServingBundle,
    ) -> None:
        self.bundle = bundle
        self.ranking_config = bundle.ranking_config
        self.output_k = int(
            self.ranking_config["k"]
        )
        self.candidate_k = int(
            self.ranking_config["candidate_k"]
        )
        self.session_weight = float(
            self.ranking_config["session_weight"]
        )
        self.diversity_config = DiversityRerankConfig(
            **self.ranking_config["diversity"]
        )
        self._lock = RLock()

        if self.diversity_config.k != self.output_k:
            raise ValueError(
                "Diversity K does not match ranking K."
            )

        self._catalog_metadata = bundle.catalog[
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
        self._item_features_by_id = (
            bundle.item_features.set_index(
                "item_id",
                drop=False,
            )
        )

        self._als_item_ids = {
            str(item_id)
            for item_id in bundle.als.item_ids
        }

    def _validate_k(self, k: int) -> int:
        """Validate an API recommendation depth."""
        if (
            isinstance(k, bool)
            or not isinstance(k, int)
            or not 1 <= k <= self.output_k
        ):
            raise ValueError(
                f"k must be between 1 and {self.output_k}."
            )

        return k

    def _normalize_history(
        self,
        item_ids: Iterable[str],
    ) -> list[str]:
        """Normalize history and retain each product's latest position."""
        history = [
            str(item_id).strip()
            for item_id in item_ids
        ]

        if any(not item_id for item_id in history):
            raise ValueError(
                "Session product IDs cannot be empty."
            )

        unique_history = list(
            dict.fromkeys(reversed(history))
        )
        unique_history.reverse()

        unknown = [
            item_id
            for item_id in unique_history
            if item_id
            not in self.bundle.item_index.item_to_index
        ]

        if unknown:
            raise ValueError(
                f"Unknown products: {unknown[:5]}"
            )

        return unique_history

    def _session_profile(
        self,
        history: list[str],
    ) -> np.ndarray:
        """Build a normalized recency-weighted session vector."""
        positions = np.asarray(
            [
                self.bundle.item_index.item_to_index[
                    item_id
                ]
                for item_id in history
            ],
            dtype=np.int32,
        )
        vectors = self.bundle.item_index.item_vectors[
            positions
        ]

        decay = self.bundle.profiles.config.recency_decay
        ages = np.arange(
            len(history) - 1,
            -1,
            -1,
        )
        weights = np.power(decay, ages)

        profile = np.average(
            vectors,
            axis=0,
            weights=weights,
        ).astype(np.float32)

        norm = np.linalg.norm(profile)

        if not np.isfinite(profile).all() or norm == 0:
            raise ValueError(
                "Session history produced an invalid profile."
            )

        return profile / norm

    def _known_user_profile(
        self,
        user_id: str,
    ) -> np.ndarray:
        """Return one persisted recent-session profile."""
        user_index = self.bundle.profiles.user_to_index[
            user_id
        ]
        return self.bundle.profiles.user_profiles[
            user_index
        ]

    def _semantic_candidate_scores(
        self,
        user_id: str,
        collaborative: pd.DataFrame,
        profile: np.ndarray,
    ) -> pd.DataFrame:
        """Score fixed ALS candidates against one dense profile."""
        positions = collaborative["item_id"].map(
            self.bundle.item_index.item_to_index
        )

        if positions.isna().any():
            raise ValueError(
                "ALS candidates are missing from the dense index."
            )

        vectors = self.bundle.item_index.item_vectors[
            positions.to_numpy(dtype=np.int32)
        ]
        scores = vectors @ profile

        semantic = collaborative[
            ["user_id", "item_id"]
        ].copy()
        semantic["semantic_score"] = scores

        semantic = semantic.sort_values(
            ["semantic_score", "item_id"],
            ascending=[False, True],
        )
        semantic["rank"] = np.arange(
            1,
            len(semantic) + 1,
        )

        return semantic[
            [
                "user_id",
                "item_id",
                "rank",
                "semantic_score",
            ]
        ]

    def _attach_metadata(
        self,
        recommendations: pd.DataFrame,
    ) -> pd.DataFrame:
        """Attach display metadata and return the stable schema."""
        work = recommendations.merge(
            self._catalog_metadata,
            on="item_id",
            how="left",
            validate="many_to_one",
        )

        for column in (
            "relevance_score",
            "collaborative_score",
            "semantic_score",
        ):
            if column not in work:
                work[column] = np.nan

        return work[
            RECOMMENDATION_COLUMNS
        ].reset_index(drop=True)

    def _candidate_item_features(
        self,
        candidates: pd.DataFrame,
    ) -> pd.DataFrame:
        """Select only features required by one candidate pool."""
        item_ids = (
            candidates["item_id"]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        missing = [
            item_id
            for item_id in item_ids
            if item_id
            not in self._item_features_by_id.index
        ]

        if missing:
            raise ValueError(
                f"Missing candidate features: {missing[:5]}"
            )

        return (
            self._item_features_by_id
            .loc[item_ids]
            .reset_index(drop=True)
        )

    def _popularity_fallback(
        self,
        user_id: str,
        k: int,
    ) -> pd.DataFrame:
        """Return deterministic global-popularity recommendations."""
        work = (
            self._catalog_metadata
            .sort_values(
                ["popularity", "item_id"],
                ascending=[False, True],
            )
            .head(k)
            .copy()
        )

        maximum = work["popularity"].max()
        scores = (
            work["popularity"] / maximum
            if maximum > 0
            else 0.0
        )

        work.insert(0, "user_id", user_id)
        work["rank"] = np.arange(1, len(work) + 1)
        work["recommendation_score"] = scores
        work["relevance_score"] = scores
        work["collaborative_score"] = np.nan
        work["semantic_score"] = np.nan
        work["strategy"] = "popularity_fallback"

        return work[
            RECOMMENDATION_COLUMNS
        ].reset_index(drop=True)

    def _semantic_from_profile(
        self,
        user_id: str,
        profile: np.ndarray,
        excluded_items: set[str],
        k: int,
        strategy: str,
    ) -> pd.DataFrame:
        """Retrieve and diversify semantic session candidates."""
        candidate_count = min(
            len(self.bundle.item_index.item_ids),
            max(
                self.candidate_k * 10,
                self.candidate_k
                + len(excluded_items),
            ),
        )

        with self._lock:
            scores, positions = (
                self.bundle.item_index.search(
                    profile,
                    candidate_count,
                )
            )

        records = []

        for score, position in zip(
            scores[0],
            positions[0],
        ):
            position = int(position)

            if position < 0:
                continue

            item_id = str(
                self.bundle.item_index.item_ids[
                    position
                ]
            )

            if item_id in excluded_items:
                continue

            records.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "rank": len(records) + 1,
                    "semantic_score": float(score),
                }
            )

            if len(records) == self.candidate_k:
                break

        candidates = pd.DataFrame.from_records(
            records,
            columns=[
                "user_id",
                "item_id",
                "rank",
                "semantic_score",
            ],
        )

        if candidates.empty:
            return pd.DataFrame(
                columns=RECOMMENDATION_COLUMNS
            )

        diverse = rerank_diverse_candidates(
            candidates,
            self._candidate_item_features(
                candidates
            ),
            config=replace(
                self.diversity_config,
                k=k,
            ),
            score_column="semantic_score",
        )

        output = diverse.rename(
            columns={
                "diversity_score": (
                    "recommendation_score"
                )
            }
        ).merge(
            candidates[
                ["item_id", "semantic_score"]
            ],
            on="item_id",
            how="left",
            validate="one_to_one",
        )
        output["collaborative_score"] = np.nan
        output["strategy"] = strategy

        return self._attach_metadata(output)

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        session_item_ids: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Recommend for a returning user or cold-start session.

        Known users combine ALS candidates with either their persisted recent
        profile or an explicitly supplied current-session profile. Unknown
        users use semantic session retrieval when history exists and global
        popularity otherwise.
        """
        user_id = str(user_id).strip()

        if not user_id:
            raise ValueError("user_id cannot be empty.")

        k = self._validate_k(k)
        raw_history = (
            []
            if session_item_ids is None
            else list(session_item_ids)
        )
        history = (
            self._normalize_history(raw_history)
            if raw_history
            else []
        )

        known_user = (
            user_id in self.bundle.als.user_to_index
        )

        if not known_user:
            if not history:
                return self._popularity_fallback(
                    user_id,
                    k,
                )

            return self._semantic_from_profile(
                user_id=user_id,
                profile=self._session_profile(history),
                excluded_items=set(history),
                k=k,
                strategy="anonymous_session_semantic",
            )

        profile = (
            self._session_profile(history)
            if history
            else self._known_user_profile(user_id)
        )

        als_user_index = (
            self.bundle.als.user_to_index[user_id]
        )
        seen_item_positions = (
            self.bundle.als.user_items[
                als_user_index
            ].indices
        )
        seen_items = {
            str(
                self.bundle.als.item_ids[
                    position
                ]
            )
            for position in seen_item_positions
        }
        excluded_items = seen_items | set(history)
        excluded_als_items = {
            item_id
            for item_id in excluded_items
            if item_id in self._als_item_ids
        }

        requested_candidates = min(
            len(self.bundle.als.item_ids),
            self.candidate_k
            + len(excluded_als_items),
        )

        with self._lock, threadpool_limits(
            limits=1,
            user_api="blas",
        ):
            collaborative = (
                self.bundle.als.recommend_for_users(
                    [user_id],
                    k=requested_candidates,
                )
            )

        collaborative = collaborative.loc[
            ~collaborative["item_id"].isin(
                excluded_items
            )
        ]

        collaborative = (
            collaborative.head(self.candidate_k)
            .copy()
            .reset_index(drop=True)
        )
        collaborative["rank"] = np.arange(
            1,
            len(collaborative) + 1,
        )

        if collaborative.empty:
            return self._semantic_from_profile(
                user_id=user_id,
                profile=profile,
                excluded_items=set(history),
                k=k,
                strategy="returning_user_semantic_fallback",
            )

        semantic = self._semantic_candidate_scores(
            user_id,
            collaborative,
            profile,
        )

        relevance = rerank_collaborative_candidates(
            collaborative,
            semantic,
            HybridConfig(
                semantic_weight=self.session_weight,
                k=self.candidate_k,
            ),
        )

        diverse = rerank_diverse_candidates(
            relevance,
            self._candidate_item_features(
                relevance
            ),
            config=replace(
                self.diversity_config,
                k=k,
            ),
            score_column="hybrid_score",
        )

        output = diverse.rename(
            columns={
                "diversity_score": (
                    "recommendation_score"
                )
            }
        ).merge(
            relevance[
                [
                    "item_id",
                    "collaborative_score",
                    "semantic_score",
                ]
            ],
            on="item_id",
            how="left",
            validate="one_to_one",
        )
        output["strategy"] = (
            "returning_user_current_session"
            if history
            else "returning_user_recent_profile"
        )

        return self._attach_metadata(output)

    def similar_products(
        self,
        item_id: str,
        k: int = 10,
    ) -> pd.DataFrame:
        """Retrieve semantically similar products excluding the query."""
        item_id = str(item_id).strip()

        if not item_id:
            raise ValueError("item_id cannot be empty.")

        k = self._validate_k(k)

        if (
            item_id
            not in self.bundle.item_index.item_to_index
        ):
            raise ValueError(
                f"Unknown product: {item_id}"
            )

        item_position = (
            self.bundle.item_index.item_to_index[
                item_id
            ]
        )
        query = self.bundle.item_index.item_vectors[
            item_position
        ]

        candidate_count = min(
            len(self.bundle.item_index.item_ids),
            max(k * 10, k + 1),
        )

        with self._lock:
            scores, positions = (
                self.bundle.item_index.search(
                    query,
                    candidate_count,
                )
            )

        records = []
        seen_items = {item_id}

        for score, position in zip(
            scores[0],
            positions[0],
        ):
            position = int(position)

            if position < 0:
                continue

            candidate_id = str(
                self.bundle.item_index.item_ids[
                    position
                ]
            )

            if candidate_id in seen_items:
                continue

            seen_items.add(candidate_id)
            records.append(
                {
                    "item_id": candidate_id,
                    "rank": len(records) + 1,
                    "similarity_score": float(score),
                }
            )

            if len(records) == k:
                break

        result = pd.DataFrame.from_records(
            records,
            columns=[
                "item_id",
                "rank",
                "similarity_score",
            ],
        ).merge(
            self._catalog_metadata,
            on="item_id",
            how="left",
            validate="one_to_one",
        )

        return result[
            SIMILAR_COLUMNS
        ].reset_index(drop=True)