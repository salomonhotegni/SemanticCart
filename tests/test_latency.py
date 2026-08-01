import math

import pytest

from semanticcart.latency import (
    summarize_latencies,
)


def test_summarizes_latency_distribution() -> None:
    metrics = summarize_latencies(
        [1.0, 2.0, 3.0, 4.0],
        wall_seconds=0.02,
    )

    assert metrics.requests == 4
    assert metrics.wall_seconds == 0.02
    assert metrics.requests_per_second == 200.0
    assert metrics.mean_ms == 2.5
    assert metrics.minimum_ms == 1.0
    assert metrics.p50_ms == 2.5
    assert metrics.p95_ms == pytest.approx(3.85)
    assert metrics.p99_ms == pytest.approx(3.97)
    assert metrics.maximum_ms == 4.0
    assert metrics.standard_deviation_ms == pytest.approx(
        1.11803398875
    )


def test_accepts_latency_generator() -> None:
    metrics = summarize_latencies(
        (value for value in [2.0, 4.0]),
        wall_seconds=0.5,
    )

    assert metrics.requests == 2
    assert metrics.requests_per_second == 4.0


def test_converts_metrics_to_dictionary() -> None:
    metrics = summarize_latencies(
        [5.0],
        wall_seconds=0.1,
    )

    result = metrics.to_dict()

    assert result["requests"] == 1
    assert result["p50_ms"] == 5.0
    assert result["requests_per_second"] == 10.0


@pytest.mark.parametrize(
    "latencies",
    [
        [],
        [-1.0],
        [math.nan],
        [math.inf],
        [1.0, -0.1],
    ],
)
def test_rejects_invalid_latencies(
    latencies: list[float],
) -> None:
    with pytest.raises(ValueError):
        summarize_latencies(
            latencies,
            wall_seconds=1.0,
        )


@pytest.mark.parametrize(
    "wall_seconds",
    [
        0.0,
        -1.0,
        math.nan,
        math.inf,
        True,
    ],
)
def test_rejects_invalid_wall_duration(
    wall_seconds: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="wall_seconds",
    ):
        summarize_latencies(
            [1.0],
            wall_seconds=wall_seconds,
        )