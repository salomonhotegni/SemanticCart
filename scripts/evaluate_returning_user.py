"""Tune and evaluate collaborative plus recent-session reranking."""

import json
from dataclasses import asdict
from math import isclose
from pathlib import Path
from time import perf_counter

import pandas as pd

from semanticcart.evaluation import evaluate_ranking
from semanticcart.hybrid import (
    HybridConfig,
    rerank_collaborative_candidates,
)


DATASET = "video_games_5core"
DATA_DIR = Path("data/processed/amazon_video_games_5core")
ARTIFACT_ROOT = Path("data/artifacts") / DATASET
RETURNING_DIR = ARTIFACT_ROOT / "returning_user"

ALS_CANDIDATE_PATHS = {
    "validation": (
        ARTIFACT_ROOT
        / "als"
        / "validation_recommendations.parquet"
    ),
    "test": (
        ARTIFACT_ROOT
        / "final"
        / "als"
        / "test_recommendations.parquet"
    ),
}
SESSION_SCORE_PATHS = {
    split: RETURNING_DIR / f"{split}_session_scores.parquet"
    for split in ALS_CANDIDATE_PATHS
}
OUTPUT_PATHS = {
    split: RETURNING_DIR / f"{split}_recommendations.parquet"
    for split in ALS_CANDIDATE_PATHS
}

ALS_VALIDATION_RESULTS = (
    Path("results") / f"{DATASET}_als.json"
)
ALS_TEST_RESULTS = (
    Path("results") / f"{DATASET}_als_test.json"
)
LONG_TERM_HYBRID_RESULTS = (
    Path("results") / f"{DATASET}_hybrid_test.json"
)
TUNING_PATH = (
    Path("results")
    / f"{DATASET}_returning_user_tuning.json"
)
TEST_RESULTS_PATH = (
    Path("results")
    / f"{DATASET}_returning_user_test.json"
)

K = 10
SESSION_LENGTH = 1
ALPHA_GRID = tuple(step / 10 for step in range(11))
METRIC_NAMES = (
    "recall_at_k",
    "ndcg_at_k",
    "mrr_at_k",
    "catalog_coverage",
)


def load_json(path: Path) -> dict:
    """Load one required JSON artifact."""
    if not path.exists():
        raise FileNotFoundError(f"Missing result: {path}")

    with path.open(encoding="utf-8") as source:
        return json.load(source)


def load_candidates(
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ALS candidates and direct recent-session scores."""
    collaborative = pd.read_parquet(
        ALS_CANDIDATE_PATHS[split]
    )
    session = pd.read_parquet(
        SESSION_SCORE_PATHS[split]
    )

    for frame in (collaborative, session):
        frame["user_id"] = frame["user_id"].astype(str)
        frame["item_id"] = frame["item_id"].astype(str)

    expected = (
        collaborative[["user_id", "item_id"]]
        .sort_values(["user_id", "item_id"])
        .reset_index(drop=True)
    )
    actual = (
        session[["user_id", "item_id"]]
        .sort_values(["user_id", "item_id"])
        .reset_index(drop=True)
    )

    if not expected.equals(actual):
        raise ValueError(
            f"{split} session scores do not match ALS candidates."
        )

    return collaborative, session


def evaluate_weight(
    collaborative: pd.DataFrame,
    session: pd.DataFrame,
    ground_truth: pd.DataFrame,
    catalog_size: int,
    session_weight: float,
) -> tuple[pd.DataFrame, dict]:
    """Rerank and evaluate one validation session weight."""
    start_time = perf_counter()

    recommendations = rerank_collaborative_candidates(
        collaborative,
        session,
        HybridConfig(
            semantic_weight=session_weight,
            k=K,
        ),
    )

    ranking_seconds = perf_counter() - start_time

    metrics = evaluate_ranking(
        recommendations=recommendations,
        ground_truth=ground_truth,
        catalog_size=catalog_size,
        k=K,
    )

    record = {
        "session_weight": session_weight,
        "ranking_seconds": ranking_seconds,
        "recommendation_rows": len(recommendations),
        **asdict(metrics),
    }

    return recommendations, record


def verify_als_endpoint(
    record: dict,
    source: dict,
) -> None:
    """Confirm zero session weight reproduces ALS exactly."""
    for metric_name in METRIC_NAMES:
        if not isclose(
            record[metric_name],
            source[metric_name],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"ALS endpoint mismatch for {metric_name}."
            )


def metric_subset(result: dict) -> dict:
    """Select ranking metrics from a source result."""
    return {
        name: result[name]
        for name in METRIC_NAMES
    }


def main() -> None:
    """Tune on validation, freeze weight, then evaluate test."""
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    validation = pd.read_parquet(
        DATA_DIR / "validation.parquet"
    )

    for frame in (train, validation):
        frame["user_id"] = frame["user_id"].astype(str)
        frame["item_id"] = frame["item_id"].astype(str)

    collaborative, session = load_candidates(
        "validation"
    )
    als_validation = load_json(
        ALS_VALIDATION_RESULTS
    )

    print("Tuning collaborative plus recent-session reranking...")
    print(f"Validation users: {validation['user_id'].nunique():,}")
    print(f"Candidates:       {len(collaborative):,}")
    print(f"Alpha values:     {len(ALPHA_GRID)}")
    print()

    trials = []

    for session_weight in ALPHA_GRID:
        recommendations, record = evaluate_weight(
            collaborative=collaborative,
            session=session,
            ground_truth=validation,
            catalog_size=train["item_id"].nunique(),
            session_weight=session_weight,
        )
        trials.append(record)

        print(
            f"weight={session_weight:.1f} "
            f"Recall@{K}={record['recall_at_k']:.6f} "
            f"NDCG@{K}={record['ndcg_at_k']:.6f} "
            f"MRR@{K}={record['mrr_at_k']:.6f} "
            f"time={record['ranking_seconds']:.2f}s"
        )

        del recommendations

    als_endpoint = next(
        record
        for record in trials
        if record["session_weight"] == 0.0
    )
    verify_als_endpoint(
        als_endpoint,
        als_validation,
    )

    best = max(
        trials,
        key=lambda record: (
            record["ndcg_at_k"],
            record["mrr_at_k"],
            -record["session_weight"],
        ),
    )
    selected_weight = best["session_weight"]

    validation_recommendations, validation_best = (
        evaluate_weight(
            collaborative=collaborative,
            session=session,
            ground_truth=validation,
            catalog_size=train["item_id"].nunique(),
            session_weight=selected_weight,
        )
    )

    RETURNING_DIR.mkdir(parents=True, exist_ok=True)
    TUNING_PATH.parent.mkdir(parents=True, exist_ok=True)

    validation_recommendations.to_parquet(
        OUTPUT_PATHS["validation"],
        index=False,
    )

    tuning_results = {
        "dataset": DATASET,
        "model": "als_recent_session_hybrid",
        "strategy": "conservative_direct_scoring",
        "feature": "one_item_recent_session_similarity",
        "session_length": SESSION_LENGTH,
        "selection_split": "validation",
        "selection_metric": "ndcg_at_k",
        "alpha_grid": list(ALPHA_GRID),
        "selected_session_weight": selected_weight,
        "candidate_rows": len(collaborative),
        "direct_score_coverage": 1.0,
        "als_source_metrics": metric_subset(
            als_validation
        ),
        "trials": trials,
        "best_validation_result": validation_best,
    }

    with TUNING_PATH.open("w", encoding="utf-8") as output:
        json.dump(tuning_results, output, indent=2)

    print()
    print(f"Selected session weight: {selected_weight:.1f}")
    print("Validation tuning saved. Preparing frozen test ranking...")

    # Test candidates are ranked before test ground truth is loaded.
    test_collaborative, test_session = load_candidates(
        "test"
    )

    test_ranking_start = perf_counter()
    test_recommendations = rerank_collaborative_candidates(
        test_collaborative,
        test_session,
        HybridConfig(
            semantic_weight=selected_weight,
            k=K,
        ),
    )
    test_ranking_seconds = (
        perf_counter() - test_ranking_start
    )

    test = pd.read_parquet(DATA_DIR / "test.parquet")
    test["user_id"] = test["user_id"].astype(str)
    test["item_id"] = test["item_id"].astype(str)

    final_fit = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    test_metrics = evaluate_ranking(
        recommendations=test_recommendations,
        ground_truth=test,
        catalog_size=final_fit["item_id"].nunique(),
        k=K,
    )

    als_test = load_json(ALS_TEST_RESULTS)
    long_term_hybrid = load_json(
        LONG_TERM_HYBRID_RESULTS
    )

    for metric_name in (
        "recall_at_k",
        "catalog_coverage",
    ):
        if not isclose(
            getattr(test_metrics, metric_name),
            als_test[metric_name],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"Returning-user candidate set changed {metric_name}."
            )

    test_recommendations.to_parquet(
        OUTPUT_PATHS["test"],
        index=False,
    )

    metric_values = asdict(test_metrics)
    delta_vs_als = {
        name: metric_values[name] - als_test[name]
        for name in METRIC_NAMES
    }
    delta_vs_long_term_hybrid = {
        name: (
            metric_values[name]
            - long_term_hybrid[name]
        )
        for name in METRIC_NAMES
    }

    test_results = {
        "dataset": DATASET,
        "model": "als_recent_session_hybrid",
        "strategy": "conservative_direct_scoring",
        "evaluation_split": "test",
        "fit_splits": ["train", "validation"],
        "selection_split": "validation",
        "selection_metric": "ndcg_at_k",
        "session_length": SESSION_LENGTH,
        "session_weight": selected_weight,
        "test_used_for_tuning": False,
        "test_ground_truth_loaded_after_ranking": True,
        "k": K,
        "fit_interactions": len(final_fit),
        "test_interactions": len(test),
        "candidate_users": test_collaborative[
            "user_id"
        ].nunique(),
        "candidate_rows": len(test_collaborative),
        "direct_score_coverage": 1.0,
        "ranking_seconds": test_ranking_seconds,
        "source_metrics": {
            "als": metric_subset(als_test),
            "long_term_semantic_hybrid": metric_subset(
                long_term_hybrid
            ),
        },
        "delta_vs_als": delta_vs_als,
        "delta_vs_long_term_hybrid": (
            delta_vs_long_term_hybrid
        ),
        **metric_values,
    }

    with TEST_RESULTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(test_results, output, indent=2)

    print()
    print("Frozen returning-user test result")
    print(f"Session weight: {selected_weight:.1f}")
    print(f"Recall@{K}:     {test_metrics.recall_at_k:.6f}")
    print(f"NDCG@{K}:       {test_metrics.ndcg_at_k:.6f}")
    print(f"MRR@{K}:        {test_metrics.mrr_at_k:.6f}")
    print(f"Coverage:      {test_metrics.catalog_coverage:.6f}")
    print(
        f"NDCG delta vs ALS: "
        f"{delta_vs_als['ndcg_at_k']:+.6f}"
    )
    print(
        f"NDCG delta vs long-term hybrid: "
        f"{delta_vs_long_term_hybrid['ndcg_at_k']:+.6f}"
    )
    print(f"Tuning: {TUNING_PATH}")
    print(f"Test:   {TEST_RESULTS_PATH}")


if __name__ == "__main__":
    main()