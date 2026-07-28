"""확률 경로 요약 지표."""
from __future__ import annotations

import numpy as np


def percentile_snapshot(values: np.ndarray) -> dict:
    p025, p10, p50, p90, p975 = np.percentile(values, [2.5, 10, 50, 90, 97.5])
    return {
        "p2_5": round(float(p025), 1), "p10": round(float(p10), 1),
        "p50": round(float(p50), 1), "p90": round(float(p90), 1),
        "p97_5": round(float(p975), 1),
    }


def lower_tail_mean(values: np.ndarray, fraction: float = 0.05) -> float:
    count = max(1, int(len(values) * fraction))
    return float(np.partition(values, count - 1)[:count].mean())


def probability(mask: np.ndarray) -> float:
    return round(float(np.mean(mask)), 6)
