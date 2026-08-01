"""Summarize measured online serving latencies."""

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class LatencyMetrics:
    """Store distribution and throughput measurements."""

    requests: int
    wall_seconds: float
    requests_per_second: float
    mean_ms: float
    standard_deviation_ms: float
    minimum_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    maximum_ms: float

    def to_dict(self) -> dict[str, int | float]:
        """Return JSON-serializable latency measurements."""
        return asdict(self)


def summarize_latencies(
    latencies_ms: Iterable[float],
    wall_seconds: float,
) -> LatencyMetrics:
    """Calculate latency percentiles and observed throughput.

    Args:
        latencies_ms: Individual end-to-end request durations.
        wall_seconds: Wall-clock duration of the measured request batch.

    Returns:
        Validated latency-distribution and throughput metrics.

    Raises:
        ValueError: If measurements are empty, negative, non-finite, or have
            an invalid wall-clock duration.
    """
    values = np.asarray(
        list(latencies_ms),
        dtype=np.float64,
    )

    if values.ndim != 1 or len(values) == 0:
        raise ValueError(
            "At least one latency measurement is required."
        )
    if (
        not np.isfinite(values).all()
        or np.any(values < 0)
    ):
        raise ValueError(
            "Latencies must be finite and non-negative."
        )
    if (
        isinstance(wall_seconds, bool)
        or not np.isfinite(wall_seconds)
        or wall_seconds <= 0
    ):
        raise ValueError(
            "wall_seconds must be finite and positive."
        )

    percentiles = np.percentile(
        values,
        [50, 95, 99],
    )

    return LatencyMetrics(
        requests=len(values),
        wall_seconds=float(wall_seconds),
        requests_per_second=float(
            len(values) / wall_seconds
        ),
        mean_ms=float(np.mean(values)),
        standard_deviation_ms=float(
            np.std(values)
        ),
        minimum_ms=float(np.min(values)),
        p50_ms=float(percentiles[0]),
        p95_ms=float(percentiles[1]),
        p99_ms=float(percentiles[2]),
        maximum_ms=float(np.max(values)),
    )