"""Probabilistic deposit-to-owner-assets ratio estimation.

This package never claims to observe an individual landlord's wealth.  It
combines four separately trained distributions and reports conditional model
uncertainty.
"""

from .pipeline import OwnerAssetRatioPipeline
from .service import OwnerAssetRatioIntegrationService
from .simulation import infer_ratio_distribution

__all__ = [
    "OwnerAssetRatioPipeline",
    "OwnerAssetRatioIntegrationService",
    "infer_ratio_distribution",
]
