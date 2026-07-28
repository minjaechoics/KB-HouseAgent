"""시뮬레이션 분포와 상관 충격 생성."""
from __future__ import annotations

import math

import numpy as np


def annual_interval_sigma(low: float, high: float, *, minimum: float = 0.03,
                          maximum: float = 0.25) -> float:
    """P10~P90으로 간주한 연간 구간을 표준편차로 환산한다."""
    width = max(float(high) - float(low), 0.0)
    sigma = width / (2 * 1.281551565545) if width else minimum
    return max(minimum, min(sigma, maximum))


def correlated_monthly_shocks(rng: np.random.Generator, paths: int) -> np.ndarray:
    """소득·물가·금융자산·주택가격 간 완만한 상관을 보존한 월 충격."""
    correlation = np.array([
        [1.00, 0.22, 0.18, 0.12],
        [0.22, 1.00, -0.08, 0.05],
        [0.18, -0.08, 1.00, 0.28],
        [0.12, 0.05, 0.28, 1.00],
    ])
    transform = np.linalg.cholesky(correlation)
    return rng.standard_normal((paths, 4)) @ transform.T


def monthly_log_mean(annual_rate: float) -> float:
    return math.log1p(max(float(annual_rate), -0.95)) / 12.0
