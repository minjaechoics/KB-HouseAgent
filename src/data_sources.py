"""Canonical paths for downloaded source datasets.

This module keeps file naming stable for code.  The original downloaded files
live under ``data/downloaded``; generated artifacts stay under
``data/generated``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src import config


REAL_ESTATE_DIR = config.DATA_RAW / "real_estate"
RTMS_DIR = REAL_ESTATE_DIR
MARKET_INDICES_DIR = REAL_ESTATE_DIR / "market_indices"
FRAUD_STATS_DIR = REAL_ESTATE_DIR / "fraud_stats"
HUG_DIR = REAL_ESTATE_DIR / "hug"
SURVEY_DIR = REAL_ESTATE_DIR / "survey"
PORTAL_DIR = REAL_ESTATE_DIR / "portal"
SAFETY_DIR = config.DATA_RAW / "safety"
DUPLICATES_DIR = REAL_ESTATE_DIR / "duplicates"


@dataclass(frozen=True)
class DownloadedDatasets:
    """Stable aliases for commonly used source datasets."""

    rtms_sh_rent: Path = RTMS_DIR / "rtms_sh_rent.csv"
    rtms_sh_trade: Path = RTMS_DIR / "rtms_sh_trade.csv"
    rtms_apt_trade: Path = RTMS_DIR / "rtms_apt_trade.csv"
    rtms_officetel_rent: Path = RTMS_DIR / "rtms_offi_rent.csv"
    rtms_officetel_trade: Path = RTMS_DIR / "rtms_offi_trade.csv"
    rtms_silv_trade: Path = RTMS_DIR / "rtms_silv_trade.csv"
    rtms_nrg_trade: Path = RTMS_DIR / "rtms_nrg_trade.csv"

    hug_return_guarantee_issuance: Path = HUG_DIR / "hug_return_guarantee_issuance_20260331.csv"
    hug_return_guarantee_accidents_by_region: Path = HUG_DIR / "hug_return_guarantee_accidents_by_region_20250831.xlsx"
    hug_return_guarantee_detail_status: Path = HUG_DIR / "hug_return_guarantee_detail_status_20260331.csv"
    hug_subrogation_synthetic: Path = HUG_DIR / "hug_subrogation_synthetic_20250731.csv"

    kb_apt_jeonse_price_index_monthly: Path = MARKET_INDICES_DIR / "kb_apt_jeonse_price_index_monthly.csv"
    kb_apt_average_sale_to_jeonse_price_ratio_monthly: Path = MARKET_INDICES_DIR / "kb_apt_average_sale_to_jeonse_price_ratio_monthly.csv"
    kb_apt_jeonse_average_price_by_region_monthly: Path = MARKET_INDICES_DIR / "kb_apt_jeonse_average_price_by_region_monthly.csv"
    kb_apt_monthly_rent_price_index_monthly: Path = MARKET_INDICES_DIR / "kb_apt_monthly_rent_price_index_monthly.csv"
    kb_officetel_jeonse_price_monthly: Path = MARKET_INDICES_DIR / "kb_officetel_jeonse_price_monthly_202401_onward.csv"
    kb_officetel_monthly_rent_price_monthly: Path = MARKET_INDICES_DIR / "kb_officetel_monthly_rent_price_monthly_202401_onward.csv"

    gyeonggi_officetel_jeonse: Path = PORTAL_DIR / "gyeonggi_real_estate_portal_officetel_jeonse.csv"
    housing_survey_jeonse_deposit: Path = SURVEY_DIR / "housing_survey_jeonse_deposit_20260715214247.csv"


DATASETS = DownloadedDatasets()
