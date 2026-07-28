from __future__ import annotations

import pandas as pd

from src.owner_asset_ratio.data import (
    match_sales_to_buildings,
    normalize_household_survey,
)


MAPPING = {
    "year": "조사연도",
    "total_assets": "자산",
    "financial_assets": "금융자산",
    "real_estate_assets": "부동산",
    "rental_real_estate_assets": "거주외부동산",
    "owner_occupied_home_assets": "거주주택",
    "rental_deposit_liability": "임대보증금",
    "financial_debt": "금융부채",
    "survey_weight": "가중값",
    "region": "수도권여부",
}


def test_public_ahs_mapping_uses_row_year_and_g1_as_capital_region() -> None:
    raw = pd.DataFrame({
        "조사연도": [2024, 2025],
        "자산": [100_000, 120_000],
        "금융자산": [20_000, 25_000],
        "부동산": [75_000, 90_000],
        "거주외부동산": [30_000, 40_000],
        "거주주택": [45_000, 50_000],
        "임대보증금": [10_000, 15_000],
        "금융부채": [12_000, 14_000],
        "가중값": [1.5, 2.0],
        "수도권여부": ["G1", "G2"],
    })

    normalized = normalize_household_survey(raw, MAPPING)

    assert normalized["survey_year"].tolist() == [2024, 2025]
    assert normalized["capital_region"].tolist() == [True, False]
    assert normalized["K_other"].round(6).tolist() == [
        round((100_000 - 30_000) / 30_000, 6),
        round((120_000 - 40_000) / 40_000, 6),
    ]


def test_non_landlords_are_excluded_from_owner_prior() -> None:
    raw = pd.DataFrame({
        "조사연도": [2025, 2025],
        "자산": [100_000, 100_000],
        "금융자산": [20_000, 20_000],
        "부동산": [70_000, 70_000],
        "거주외부동산": [30_000, 0],
        "거주주택": [40_000, 70_000],
        "임대보증금": [10_000, 0],
        "금융부채": [5_000, 5_000],
        "가중값": [1.0, 1.0],
        "수도권여부": ["G1", "G1"],
    })

    normalized = normalize_household_survey(raw, MAPPING)

    assert len(normalized) == 1
    assert normalized.iloc[0]["rental_deposit_liability"] == 10_000


def test_masked_rtms_lot_matches_hub_address_ending_in_beonji() -> None:
    buildings = pd.DataFrame([{
        "building_id": "B1",
        "legal_dong": "우만동",
        "lot_number": "경기도 수원시 팔달구 우만동 412-3번지",
        "land_area": 100.0,
        "total_floor_area": 200.0,
    }])
    sales = pd.DataFrame([{
        "legal_dong": "우만동",
        "partial_lot_number": "4**",
        "land_area": 100.0,
        "total_floor_area": 200.0,
    }])

    matched = match_sales_to_buildings(sales, buildings)

    assert matched.iloc[0]["building_id"] == "B1"
    assert matched.iloc[0]["match_confidence"] == "high"
