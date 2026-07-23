"""Provide a global item-popularity recommendation baseline."""

from collections.abc import Iterable

import pandas as pd


class PopularityModel:
    """Recommend globally popular products that each user has not seen."""

    def __init__(self, ranking: pd.DataFrame) -> None:
        self.ranking = ranking.reset_index(drop=True)

    @classmethod
    def fit(cls, interactions: pd.DataFrame) -> "PopularityModel":
        """Count interactions and construct a deterministic item ranking.

        Args:
            interactions: Training events containing item_id.

        Returns:
            A model ranked by descending interaction count, with item_id used
            to break ties deterministically.

        Raises:
            ValueError: If item_id is missing.
        """

        if "item_id" not in interactions.columns:
            raise ValueError("Interactions must contain item_id.")

        ranking = (
            interactions.groupby("item_id")
            .size()
            .rename("popularity_score")
            .reset_index()
            .sort_values(
                ["popularity_score", "item_id"],
                ascending=[False, True],
            )
        )

        return cls(ranking)

    def recommend_for_users(
        self,
        user_ids: Iterable[str],
        seen_interactions: pd.DataFrame,
        k: int = 10,
    ) -> pd.DataFrame:
        """Return each user's Top-K globally popular unseen products.

        Args:
            user_ids: Users to score; duplicate IDs are removed in input order.
            seen_interactions: Historical user_id and item_id pairs used to
                filter already observed products.
            k: Maximum number of recommendations per user.

        Returns:
            Rows containing user_id, item_id, one-based rank, and
            popularity_score.

        Raises:
            ValueError: If k is not positive.
        """

        if k <= 0:
            raise ValueError("k must be greater than zero.")

        users = list(dict.fromkeys(user_ids))
        target_users = set(users)

        relevant_history = seen_interactions.loc[
            seen_interactions["user_id"].isin(target_users)
        ]
        seen_by_user = (
            relevant_history.groupby("user_id")["item_id"]
            .agg(set)
            .to_dict()
        )

        ranked_items = list(self.ranking.itertuples(index=False))
        rows = []

        for user_id in users:
            seen_items = seen_by_user.get(user_id, set())
            rank = 1

            for item in ranked_items:
                if item.item_id in seen_items:
                    continue

                rows.append(
                    {
                        "user_id": user_id,
                        "item_id": item.item_id,
                        "rank": rank,
                        "popularity_score": item.popularity_score,
                    }
                )
                rank += 1

                if rank > k:
                    break

        return pd.DataFrame(rows)
