"""
Evaluation metrics for ranking models.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RankingMetrics:
    recall_at_k: float
    ndcg_at_k: float
    mrr_at_k: float
    catalog_coverage: float


def evaluate_ranking(
    recommendations: pd.DataFrame,
    ground_truth: pd.DataFrame,
    catalog_size: int,
    k: int = 10,
) -> RankingMetrics:
    if k <= 0 or catalog_size <= 0:
        raise ValueError("k and catalog_size must be positive.")

    truth = ground_truth[
        ["user_id", "item_id"]
    ].drop_duplicates()

    recs = (
        recommendations.loc[
            recommendations["rank"] <= k,
            ["user_id", "item_id", "rank"],
        ]
        .sort_values("rank")
        .drop_duplicates(["user_id", "item_id"], keep="first")
    )

    user_index = pd.Index(
        truth["user_id"].unique(),
        name="user_id",
    )
    relevant_counts = (
        truth.groupby("user_id")
        .size()
        .reindex(user_index)
    )

    hits = recs.merge(
        truth,
        on=["user_id", "item_id"],
        how="inner",
    )

    hit_counts = (
        hits.groupby("user_id")
        .size()
        .reindex(user_index, fill_value=0)
    )
    recall = (hit_counts / relevant_counts).mean()

    hits["discount"] = 1.0 / np.log2(hits["rank"] + 1)
    dcg = (
        hits.groupby("user_id")["discount"]
        .sum()
        .reindex(user_index, fill_value=0.0)
    )

    ideal_dcg = relevant_counts.clip(upper=k).map(
        lambda count: np.sum(
            1.0 / np.log2(np.arange(2, count + 2))
        )
    )
    ndcg = (dcg / ideal_dcg).mean()

    first_hit_rank = hits.groupby("user_id")["rank"].min()
    reciprocal_rank = (
        (1.0 / first_hit_rank)
        .reindex(user_index, fill_value=0.0)
    )
    mrr = reciprocal_rank.mean()

    coverage = recs["item_id"].nunique() / catalog_size

    return RankingMetrics(
        recall_at_k=float(recall),
        ndcg_at_k=float(ndcg),
        mrr_at_k=float(mrr),
        catalog_coverage=float(coverage),
    )