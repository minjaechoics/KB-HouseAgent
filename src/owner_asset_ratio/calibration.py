from __future__ import annotations

import numpy as np

from .models import weighted_quantile


def population_quantile_loss(
    simulated_ratios: np.ndarray,
    survey_ratios: np.ndarray,
    survey_weights: np.ndarray,
    *,
    quantiles=(0.1, 0.5, 0.9),
) -> float:
    """Aggregate-distribution loss; never an individual owner loss."""
    simulation_q = np.quantile(simulated_ratios, quantiles)
    survey_q = weighted_quantile(survey_ratios, quantiles, survey_weights)
    return float(np.mean(np.abs(simulation_q - survey_q)))


def choose_k_scale(
    total_deposit: np.ndarray,
    property_value: np.ndarray,
    k_samples: np.ndarray,
    survey_ratios: np.ndarray,
    survey_weights: np.ndarray,
    *,
    grid: np.ndarray | None = None,
) -> dict:
    """Post-hoc population calibration of only the K-prior scale."""
    grid = np.asarray(
        grid if grid is not None else np.linspace(0.5, 2.0, 61),
        dtype=float)
    losses = []
    for scale in grid:
        ratio = total_deposit / np.maximum(
            property_value * (1.0 + scale * k_samples), 1e-6)
        losses.append(population_quantile_loss(
            ratio, survey_ratios, survey_weights))
    index = int(np.argmin(losses))
    return {
        "k_scale": float(grid[index]),
        "loss": float(losses[index]),
        "grid": grid.tolist(),
        "losses": [float(value) for value in losses],
        "calibration_kind": "population_post_hoc_not_individual_label",
    }
