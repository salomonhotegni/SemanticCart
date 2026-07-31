"""Tune diversity reranking on validation and evaluate frozen test settings."""

import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import pandas as pd

from semanticcart.cached_embeddings import (
    load_cached_catalog_embeddings,
)
from semanticcart.diversity import (
    DiversityRerankConfig,
    evaluate_diversity,
    rerank_diverse_candidates,
)
from semanticcart.embedding_cache import EmbeddingConfig
from semanticcart.evaluation import evaluate_ranking
from semanticcart.hybrid import (
    HybridConfig,
    rerank_collaborative_candidates,
)


DATASET = "video_games_5core"
DATA_DIR = Path("data/processed/amazon_video_games_5core")
ARTIFACT_ROOT = Path("data/artifacts") / DATASET
DIVERSITY_DIR = ARTIFACT_ROOT / "diversity"
CACHE_PATH = (
    ARTIFACT_ROOT
    / "openai_embeddings"
    / "embedding_cache.parquet"
)

CANDIDATE_PATHS = {
    split: DIVERSITY_DIR / f"{split}_candidates.parquet"
    for split in ("validation", "test")
}
RECOMMENDATION_PATHS = {
    split: DIVERSITY_DIR / f"{split}_recommendations.parquet"
    for split in ("validation", "test")
}

RESULTS_DIR = Path("results")
TUNING_PATH = RESULTS_DIR / f"{DATASET}_diversity_tuning.json"
TEST_PATH = RESULTS_DIR / f"{DATASET}_diversity_test.json"

K = 10
CANDIDATE_K = 25
TUNING_COHORT_USERS = 10_000
NDCG_RETENTION = 0.99

SESSION_WEIGHTS = tuple(
    step / 10
    for step in range(11)
)

DIVERSITY_CONFIGS = (
    DiversityRerankConfig(
        k=K,
        relevance_weight=1.0,
        novelty_weight=0.0,
    ),
) + tuple(
    DiversityRerankConfig(
        k=K,
        relevance_weight=relevance_weight,
        novelty_weight=novelty_weight,
        semantic_similarity_weight=0.70,
        category_similarity_weight=0.20,
        price_similarity_weight=0.10,
    )
    for relevance_weight in (0.98, 0.95, 0.92, 0.90, 0.85)
    for novelty_weight in (0.0, 0.025, 0.05)
)

EMBEDDING_CONFIG = EmbeddingConfig(
    model="text-embedding-3-small",
    dimensions=512,
)


def load_candidates(split: str) -> pd.DataFrame:
    """Load and validate one prepared Top-25 candidate pool."""
    path = CANDIDATE_PATHS[split]

    if not path.exists():
        raise FileNotFoundError(
            f"Missing candidate artifact: {path}"
        )

    candidates = pd.read_parquet(path)
    required = {
        "user_id",
        "item_id",
        "rank",
        "collaborative_score",
        "semantic_score",
    }
    missing = required - set(candidates.columns)

    if missing:
        raise ValueError(
            f"Missing candidate columns: {sorted(missing)}"
        )

    candidates["user_id"] = candidates[
        "user_id"
    ].astype(str)
    candidates["item_id"] = candidates[
        "item_id"
    ].astype(str)

    if candidates.duplicated(
        ["user_id", "item_id"]
    ).any():
        raise ValueError(
            f"{split} candidate pairs must be unique."
        )

    depths = candidates.groupby("user_id").size()

    if not depths.eq(CANDIDATE_K).all():
        raise ValueError(
            f"{split} users must have exactly "
            f"{CANDIDATE_K} candidates."
        )

    return candidates


def build_item_features(
    fit_events: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Attach embeddings and fit-period popularity to catalogue products."""
    fit_item_ids = pd.Index(
        fit_events["item_id"].unique()
    )
    fit_catalog = catalog.loc[
        catalog["item_id"].isin(fit_item_ids)
    ].copy()

    missing_items = fit_item_ids.difference(
        pd.Index(fit_catalog["item_id"])
    )

    if len(missing_items):
        raise ValueError(
            f"Missing metadata for {len(missing_items):,} "
            "fit products."
        )

    embedded = load_cached_catalog_embeddings(
        fit_catalog,
        CACHE_PATH,
        id_column="item_id",
        config=EMBEDDING_CONFIG,
    )

    popularity = fit_events.groupby(
        "item_id"
    ).size()

    features = embedded[
        [
            "item_id",
            "embedding",
            "categories",
            "price",
        ]
    ].copy()
    features["popularity"] = (
        features["item_id"]
        .map(popularity)
        .fillna(0)
    )

    return features


def rank_relevance_candidates(
    candidates: pd.DataFrame,
    session_weight: float,
) -> pd.DataFrame:
    """Blend ALS and recent-session scores across the Top-25 pool."""
    collaborative = candidates[
        [
            "user_id",
            "item_id",
            "rank",
            "collaborative_score",
        ]
    ]
    semantic = candidates[
        [
            "user_id",
            "item_id",
            "rank",
            "semantic_score",
        ]
    ]

    return rerank_collaborative_candidates(
        collaborative,
        semantic,
        HybridConfig(
            semantic_weight=session_weight,
            k=CANDIDATE_K,
        ),
    )


def evaluate_all(
    recommendations: pd.DataFrame,
    ground_truth: pd.DataFrame,
    item_features: pd.DataFrame,
) -> dict:
    """Evaluate ranking quality and beyond-accuracy metrics together."""
    ranking = evaluate_ranking(
        recommendations=recommendations,
        ground_truth=ground_truth,
        catalog_size=len(item_features),
        k=K,
    )
    diversity = evaluate_diversity(
        recommendations=recommendations,
        item_features=item_features,
        k=K,
    )

    return {
        **asdict(ranking),
        **asdict(diversity),
    }


def metric_deltas(
    new_metrics: dict,
    baseline_metrics: dict,
) -> dict:
    """Return absolute metric changes from a baseline."""
    return {
        name: new_metrics[name] - baseline_metrics[name]
        for name in baseline_metrics
    }


def select_tuning_users(
    user_ids: pd.Series,
) -> set[str]:
    """Select a deterministic hash-ordered validation cohort."""
    unique_users = pd.Series(
        user_ids.astype(str).unique(),
        name="user_id",
    )
    hashes = pd.util.hash_pandas_object(
        unique_users,
        index=False,
    )

    cohort = (
        pd.DataFrame(
            {
                "user_id": unique_users,
                "hash": hashes,
            }
        )
        .sort_values(["hash", "user_id"])
        .head(TUNING_COHORT_USERS)
    )

    return set(cohort["user_id"])


def tune_session_weight(
    candidates: pd.DataFrame,
    validation: pd.DataFrame,
    catalog_size: int,
) -> tuple[float, list[dict]]:
    """Tune recent-session blending on the full validation split."""
    trials = []

    print("Tuning session relevance across Top-25 candidates...")

    for session_weight in SESSION_WEIGHTS:
        start_time = perf_counter()
        recommendations = rank_relevance_candidates(
            candidates,
            session_weight,
        )
        ranking_seconds = perf_counter() - start_time

        metrics = evaluate_ranking(
            recommendations=recommendations,
            ground_truth=validation,
            catalog_size=catalog_size,
            k=K,
        )

        record = {
            "session_weight": session_weight,
            "ranking_seconds": ranking_seconds,
            **asdict(metrics),
        }
        trials.append(record)

        print(
            f"session_weight={session_weight:.1f} "
            f"Recall@{K}={metrics.recall_at_k:.6f} "
            f"NDCG@{K}={metrics.ndcg_at_k:.6f} "
            f"MRR@{K}={metrics.mrr_at_k:.6f}"
        )

        del recommendations

    best = max(
        trials,
        key=lambda record: (
            record["ndcg_at_k"],
            record["mrr_at_k"],
            -record["session_weight"],
        ),
    )

    return best["session_weight"], trials


def tune_diversity(
    relevance_candidates: pd.DataFrame,
    validation: pd.DataFrame,
    item_features: pd.DataFrame,
) -> tuple[DiversityRerankConfig, list[dict]]:
    """Tune MMR under a validation NDCG-retention constraint."""
    cohort_users = select_tuning_users(
        relevance_candidates["user_id"]
    )
    cohort_candidates = relevance_candidates.loc[
        relevance_candidates["user_id"].isin(
            cohort_users
        )
    ].copy()
    cohort_truth = validation.loc[
        validation["user_id"].isin(
            cohort_users
        )
    ].copy()

    print()
    print(
        f"Tuning {len(DIVERSITY_CONFIGS)} diversity configurations "
        f"on {len(cohort_users):,} validation users..."
    )

    trials = []

    for config in DIVERSITY_CONFIGS:
        start_time = perf_counter()
        recommendations = rerank_diverse_candidates(
            cohort_candidates,
            item_features,
            config=config,
            score_column="hybrid_score",
        )
        ranking_seconds = perf_counter() - start_time

        metrics = evaluate_all(
            recommendations,
            cohort_truth,
            item_features,
        )

        record = {
            "config": asdict(config),
            "ranking_seconds": ranking_seconds,
            **metrics,
        }
        trials.append((config, record))

        print(
            f"lambda={config.relevance_weight:.3f} "
            f"novelty={config.novelty_weight:.3f} "
            f"NDCG@{K}={metrics['ndcg_at_k']:.6f} "
            f"ILD={metrics['intra_list_diversity']:.6f} "
            f"category_variety={metrics['category_variety']:.6f}"
        )

        del recommendations

    baseline_record = trials[0][1]
    minimum_ndcg = (
        baseline_record["ndcg_at_k"]
        * NDCG_RETENTION
    )

    eligible = [
        (config, record)
        for config, record in trials
        if record["ndcg_at_k"] >= minimum_ndcg
    ]

    selected_config, _ = max(
        eligible,
        key=lambda pair: (
            pair[1]["intra_list_diversity"],
            pair[1]["category_variety"],
            pair[1]["novelty"],
            pair[1]["ndcg_at_k"],
            pair[0].relevance_weight,
            -pair[0].novelty_weight,
        ),
    )

    return (
        selected_config,
        [record for _, record in trials],
    )


def main() -> None:
    """Tune on validation, freeze settings, then rank test candidates."""
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    validation = pd.read_parquet(
        DATA_DIR / "validation.parquet"
    )
    catalog = pd.read_parquet(
        DATA_DIR / "catalog.parquet"
    )

    for frame in (train, validation):
        frame["user_id"] = frame["user_id"].astype(str)
        frame["item_id"] = frame["item_id"].astype(str)

    catalog["item_id"] = catalog["item_id"].astype(str)

    validation_candidates = load_candidates("validation")
    validation_features = build_item_features(
        train,
        catalog,
    )

    session_weight, session_trials = tune_session_weight(
        validation_candidates,
        validation,
        catalog_size=len(validation_features),
    )

    print()
    print(f"Selected session weight: {session_weight:.1f}")

    relevance_start = perf_counter()
    validation_relevance = rank_relevance_candidates(
        validation_candidates,
        session_weight,
    )
    validation_relevance_seconds = (
        perf_counter() - relevance_start
    )

    selected_config, diversity_trials = tune_diversity(
        validation_relevance,
        validation,
        validation_features,
    )

    print()
    print(
        "Selected diversity configuration: "
        f"lambda={selected_config.relevance_weight:.3f}, "
        f"novelty={selected_config.novelty_weight:.3f}"
    )

    validation_baseline = validation_relevance.loc[
        validation_relevance["rank"] <= K
    ].copy()
    validation_baseline_metrics = evaluate_all(
        validation_baseline,
        validation,
        validation_features,
    )

    validation_rerank_start = perf_counter()
    validation_recommendations = rerank_diverse_candidates(
        validation_relevance,
        validation_features,
        config=selected_config,
        score_column="hybrid_score",
    )
    validation_rerank_seconds = (
        perf_counter() - validation_rerank_start
    )
    validation_metrics = evaluate_all(
        validation_recommendations,
        validation,
        validation_features,
    )

    DIVERSITY_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    validation_recommendations.to_parquet(
        RECOMMENDATION_PATHS["validation"],
        index=False,
    )

    tuning_result = {
        "dataset": DATASET,
        "model": "returning_user_diversity_reranker",
        "selection_split": "validation",
        "test_used_for_tuning": False,
        "k": K,
        "candidate_k": CANDIDATE_K,
        "tuning_cohort_users": TUNING_COHORT_USERS,
        "tuning_cohort_selection": (
            "stable pandas hash order"
        ),
        "ndcg_retention_constraint": NDCG_RETENTION,
        "selected_session_weight": session_weight,
        "selected_diversity_config": asdict(
            selected_config
        ),
        "session_trials": session_trials,
        "diversity_trials": diversity_trials,
        "full_validation_baseline": (
            validation_baseline_metrics
        ),
        "full_validation_selected": validation_metrics,
        "full_validation_delta": metric_deltas(
            validation_metrics,
            validation_baseline_metrics,
        ),
        "full_validation_relevance_seconds": (
            validation_relevance_seconds
        ),
        "full_validation_rerank_seconds": (
            validation_rerank_seconds
        ),
    }

    with TUNING_PATH.open(
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(tuning_result, output, indent=2)

    del validation_candidates
    del validation_relevance
    del validation_recommendations
    del validation_features

    print()
    print("Validation tuning saved. Ranking frozen test candidates...")

    final_fit = pd.concat(
        [train, validation],
        ignore_index=True,
    )
    test_candidates = load_candidates("test")
    test_features = build_item_features(
        final_fit,
        catalog,
    )

    test_relevance_start = perf_counter()
    test_relevance = rank_relevance_candidates(
        test_candidates,
        session_weight,
    )
    test_relevance_seconds = (
        perf_counter() - test_relevance_start
    )

    test_baseline = test_relevance.loc[
        test_relevance["rank"] <= K
    ].copy()

    test_rerank_start = perf_counter()
    test_recommendations = rerank_diverse_candidates(
        test_relevance,
        test_features,
        config=selected_config,
        score_column="hybrid_score",
    )
    test_rerank_seconds = (
        perf_counter() - test_rerank_start
    )

    test_recommendations.to_parquet(
        RECOMMENDATION_PATHS["test"],
        index=False,
    )

    # Test labels are loaded only after frozen recommendations exist.
    test = pd.read_parquet(DATA_DIR / "test.parquet")
    test["user_id"] = test["user_id"].astype(str)
    test["item_id"] = test["item_id"].astype(str)

    test_baseline_metrics = evaluate_all(
        test_baseline,
        test,
        test_features,
    )
    test_metrics = evaluate_all(
        test_recommendations,
        test,
        test_features,
    )

    test_result = {
        "dataset": DATASET,
        "model": "returning_user_diversity_reranker",
        "evaluation_split": "test",
        "fit_splits": ["train", "validation"],
        "selection_split": "validation",
        "test_used_for_tuning": False,
        "test_ground_truth_loaded_after_ranking": True,
        "k": K,
        "candidate_k": CANDIDATE_K,
        "candidate_users": test_candidates[
            "user_id"
        ].nunique(),
        "candidate_rows": len(test_candidates),
        "selected_session_weight": session_weight,
        "selected_diversity_config": asdict(
            selected_config
        ),
        "relevance_ranking_seconds": (
            test_relevance_seconds
        ),
        "diversity_reranking_seconds": (
            test_rerank_seconds
        ),
        "baseline_metrics": test_baseline_metrics,
        "selected_metrics": test_metrics,
        "delta_vs_relevance_baseline": metric_deltas(
            test_metrics,
            test_baseline_metrics,
        ),
    }

    with TEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(test_result, output, indent=2)

    print()
    print("Frozen diversity test result")
    print(f"Session weight: {session_weight:.1f}")
    print(
        "Diversity config: "
        f"lambda={selected_config.relevance_weight:.3f}, "
        f"novelty={selected_config.novelty_weight:.3f}"
    )
    print(
        f"Baseline NDCG@{K}: "
        f"{test_baseline_metrics['ndcg_at_k']:.6f}"
    )
    print(
        f"Selected NDCG@{K}: "
        f"{test_metrics['ndcg_at_k']:.6f}"
    )
    print(
        f"Baseline ILD:     "
        f"{test_baseline_metrics['intra_list_diversity']:.6f}"
    )
    print(
        f"Selected ILD:     "
        f"{test_metrics['intra_list_diversity']:.6f}"
    )
    print(
        f"Category variety: "
        f"{test_metrics['category_variety']:.6f}"
    )
    print(f"Tuning: {TUNING_PATH}")
    print(f"Test:   {TEST_PATH}")


if __name__ == "__main__":
    main()