"""시계열 out-of-fold 잔차를 이용한 Conformal 예측구간 보정."""
from __future__ import annotations

import math

import numpy as np


def conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if not len(scores):
        return 0.0
    rank = min(len(scores), math.ceil((len(scores) + 1) * float(coverage)))
    return float(np.partition(scores, rank - 1)[rank - 1])


def calibrate_intervals(y_true: np.ndarray, lower: np.ndarray,
                        upper: np.ndarray) -> dict:
    y_true, lower, upper = map(lambda x: np.asarray(x, dtype=float),
                               (y_true, lower, upper))
    scores = np.maximum(lower - y_true, y_true - upper)
    q80 = max(0.0, conformal_quantile(scores, .80))
    q95 = max(q80, conformal_quantile(scores, .95))
    before = float(np.mean((y_true >= lower) & (y_true <= upper)))
    cover80 = float(np.mean((y_true >= lower - q80) & (y_true <= upper + q80)))
    cover95 = float(np.mean((y_true >= lower - q95) & (y_true <= upper + q95)))
    return {
        "method": "rolling_oof_conformalized_quantile_regression",
        "sample_count": int(len(y_true)),
        "qhat_80_monthly_log_return": q80,
        "qhat_95_monthly_log_return": q95,
        "raw_interval_coverage": before,
        "empirical_coverage_80": cover80,
        "empirical_coverage_95": cover95,
    }
