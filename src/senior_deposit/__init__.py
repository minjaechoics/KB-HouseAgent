"""Probabilistic existing/senior tenant-deposit estimation.

The package explicitly separates conservative existing-deposit estimates from
unobserved legal seniority and never reports scenario priors as verified law.
"""

from .pipeline import SeniorDepositPipeline
from .service import SeniorDepositIntegrationService
from .simulation import infer_senior_deposit_distribution

__all__ = [
    "SeniorDepositPipeline",
    "SeniorDepositIntegrationService",
    "infer_senior_deposit_distribution",
]
