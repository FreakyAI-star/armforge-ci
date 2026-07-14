from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def percentile(values: Iterable[float], quantile: float) -> float:
    samples = np.asarray(list(values), dtype=np.float64)
    if samples.size == 0:
        raise ValueError("at least one sample is required")
    return float(np.percentile(samples, quantile))


def latency_summary(samples_ms: list[float], batch_size: int) -> dict[str, float | int]:
    mean_ms = float(np.mean(samples_ms))
    return {
        "runs": len(samples_ms),
        "mean_ms": round(mean_ms, 4),
        "p50_ms": round(percentile(samples_ms, 50), 4),
        "p95_ms": round(percentile(samples_ms, 95), 4),
        "min_ms": round(min(samples_ms), 4),
        "max_ms": round(max(samples_ms), 4),
        "throughput_items_per_second": round(batch_size * 1000.0 / mean_ms, 2),
    }


def compare_outputs(baseline: list[np.ndarray], optimized: list[np.ndarray]) -> dict[str, float]:
    if len(baseline) != len(optimized):
        raise ValueError("model output counts differ")
    base = np.concatenate([item.astype(np.float64).ravel() for item in baseline])
    opt = np.concatenate([item.astype(np.float64).ravel() for item in optimized])
    if base.shape != opt.shape:
        raise ValueError("model output shapes differ")
    denominator = float(np.linalg.norm(base) * np.linalg.norm(opt))
    cosine = float(np.dot(base, opt) / denominator) if denominator else 1.0
    difference = np.abs(base - opt)
    return {
        "cosine_similarity": round(cosine, 8),
        "mean_absolute_error": round(float(np.mean(difference)), 8),
        "max_absolute_error": round(float(np.max(difference)), 8),
        "finite": bool(math.isfinite(cosine) and np.all(np.isfinite(difference))),
    }


def percentage_change(before: float, after: float) -> float:
    if before == 0:
        return 0.0
    return round((before - after) / before * 100.0, 2)

