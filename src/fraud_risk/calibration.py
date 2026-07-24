"""Probability calibration, calibration metrics, and cost-based decisions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def _clip_probability(p) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)


class PlattCalibrator:
    def __init__(self):
        self.model = LogisticRegression(C=1e6, solver="lbfgs")

    def fit(self, probability, y):
        p = _clip_probability(probability)
        self.model.fit(np.log(p / (1 - p)).reshape(-1, 1), np.asarray(y, dtype=int))
        return self

    def transform(self, probability) -> np.ndarray:
        p = _clip_probability(probability)
        return self.model.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    def __init__(self):
        self.model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

    def fit(self, probability, y):
        self.model.fit(np.asarray(probability, dtype=float), np.asarray(y, dtype=int))
        return self

    def transform(self, probability) -> np.ndarray:
        return np.asarray(self.model.predict(np.asarray(probability, dtype=float)), dtype=float)


def fit_calibrator_with_temporal_selection(probability, y) -> tuple[object, dict]:
    """Select Platt vs isotonic on the later half, then refit on all calibration rows."""
    probability = np.asarray(probability, dtype=float)
    y = np.asarray(y, dtype=int)
    if len(y) < 100 or y.sum() < 10:
        model = PlattCalibrator().fit(probability, y)
        return model, {"method": "platt", "selection": "small-sample-safe-default"}

    cut = len(y) // 2
    candidates = {
        "platt": PlattCalibrator(),
        "isotonic": IsotonicCalibrator(),
    }
    scores: dict[str, float] = {}
    for name, candidate in candidates.items():
        candidate.fit(probability[:cut], y[:cut])
        scores[name] = float(brier_score_loss(y[cut:], candidate.transform(probability[cut:])))
    selected = min(scores, key=scores.get)
    final = candidates[selected].__class__().fit(probability, y)
    return final, {"method": selected, "selection_brier": scores}


def expected_calibration_error(y, probability, bins: int = 10) -> float:
    y = np.asarray(y, dtype=int)
    p = np.asarray(probability, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    total = max(len(y), 1)
    error = 0.0
    for i in range(bins):
        mask = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1)
        if mask.any():
            error += mask.sum() / total * abs(float(y[mask].mean() - p[mask].mean()))
    return float(error)


@dataclass(frozen=True)
class CostDecision:
    threshold: float
    expected_cost: float
    false_negatives: int
    false_positives: int


def choose_cost_threshold(
    y,
    probability,
    false_negative_cost: float = 20.0,
    false_positive_cost: float = 1.0,
) -> CostDecision:
    """Minimize empirical FN/FP cost on a held-out calibration period."""
    if false_negative_cost <= 0 or false_positive_cost <= 0:
        raise ValueError("costs must be positive")
    y = np.asarray(y, dtype=int)
    p = np.asarray(probability, dtype=float)
    candidates = np.unique(np.r_[0.0, p, 1.0])
    best = None
    for threshold in candidates:
        pred = p >= threshold
        fn = int(((y == 1) & ~pred).sum())
        fp = int(((y == 0) & pred).sum())
        cost = fn * false_negative_cost + fp * false_positive_cost
        item = CostDecision(float(threshold), float(cost / max(len(y), 1)), fn, fp)
        if best is None or (item.expected_cost, -item.threshold) < (best.expected_cost, -best.threshold):
            best = item
    return best

