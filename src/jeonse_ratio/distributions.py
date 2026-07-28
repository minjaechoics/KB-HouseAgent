"""Quantile reconstruction and copula helpers."""
from __future__ import annotations

import re

import numpy as np
from scipy.stats import norm


def parse_quantiles(values: dict[str, float]) -> tuple[np.ndarray, np.ndarray, bool]:
    pairs = []
    for key, value in values.items():
        match = re.fullmatch(r"p(\d{1,3})", str(key).lower())
        if match:
            q = int(match.group(1)) / 100.0
            if 0 <= q <= 1 and np.isfinite(float(value)):
                pairs.append((q, max(0.0, float(value))))
    pairs.sort()
    if len(pairs) < 2:
        raise ValueError("분포 복원에는 두 개 이상의 분위수가 필요합니다.")
    quantiles = np.asarray([item[0] for item in pairs], dtype=float)
    raw = np.asarray([item[1] for item in pairs], dtype=float)
    crossing = bool(np.any(np.diff(raw) < 0))
    return quantiles, np.maximum.accumulate(raw), crossing


def inverse_cdf_sample(
    values: dict[str, float],
    uniforms: np.ndarray,
    *,
    lower_bound: float = 0.0,
    upper_multiplier_limit: float = 3.0,
) -> tuple[np.ndarray, bool]:
    """Piecewise-linear inverse CDF with bounded linear tails."""
    q, x, crossing = parse_quantiles(values)
    u = np.asarray(uniforms, dtype=float)
    sampled = np.interp(u, q, x)
    lower_slope = (x[1] - x[0]) / max(q[1] - q[0], 1e-12)
    upper_slope = (x[-1] - x[-2]) / max(q[-1] - q[-2], 1e-12)
    sampled = np.where(
        u < q[0], x[0] + (u - q[0]) * lower_slope, sampled)
    sampled = np.where(
        u > q[-1], x[-1] + (u - q[-1]) * upper_slope, sampled)
    upper = max(x[-1], x[-1] * float(upper_multiplier_limit))
    return np.clip(sampled, float(lower_bound), upper), crossing


def copula_uniforms(
    samples: int, rho: float, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if not -0.99 < float(rho) < 0.99:
        raise ValueError("copula rho는 -0.99와 0.99 사이여야 합니다.")
    z1 = rng.standard_normal(int(samples))
    z2 = float(rho) * z1 + np.sqrt(1.0 - float(rho) ** 2) * \
        rng.standard_normal(int(samples))
    return norm.cdf(z1), norm.cdf(z2)
