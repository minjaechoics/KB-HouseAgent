"""Synthetic fixtures for pipeline smoke tests only.

No metric computed from these frames may be described as real-world
performance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_frames(seed: int = 20260728,
                          n_buildings: int = 900,
                          n_leases: int = 2400,
                          n_sales: int = 1200,
                          n_survey: int = 1500) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dongs = np.array(["인계동", "우만동", "지동", "매산로", "화서동", "고등동"])
    codes = np.array(["41115"] * len(dongs))
    valid_months = np.array([
        year * 100 + month
        for year in range(2023, 2027)
        for month in range(1, 13)
        if year * 100 + month <= 202607
    ])
    area = rng.lognormal(np.log(230), .42, n_buildings)
    units = np.clip(np.rint(area / rng.normal(38, 5, n_buildings)), 1, 30)
    buildings = pd.DataFrame({
        "building_id": [f"SYN-B-{i:05d}" for i in range(n_buildings)],
        "snapshot_month": rng.choice(valid_months, n_buildings),
        "sigungu_code": rng.choice(codes, n_buildings),
        "legal_dong_code": rng.choice(dongs, n_buildings),
        "legal_dong": rng.choice(dongs, n_buildings),
        "main_use_code": rng.choice(["다가구주택", "단독주택"], n_buildings,
                                    p=[.82, .18]),
        "structure_code": rng.choice(["철근콘크리트", "벽돌", "일반철골"], n_buildings),
        "land_area": area * rng.uniform(.45, .9, n_buildings),
        "total_floor_area": area,
        "residential_floor_area": area * rng.uniform(.7, 1, n_buildings),
        "parking_count": np.clip(units * rng.uniform(.15, .8, n_buildings), 0, None),
        "ground_floors": rng.integers(2, 6, n_buildings),
        "building_age": rng.integers(1, 45, n_buildings),
        "registered_units_observed": units.astype(int),
    })
    # Keep a genuine missing-unit subset so inference fallback is exercised.
    buildings.loc[rng.random(n_buildings) < .25,
                  "registered_units_observed"] = np.nan

    lease_area = rng.lognormal(np.log(42), .32, n_leases)
    lease_age = rng.integers(0, 45, n_leases)
    lease_dong = rng.choice(dongs, n_leases)
    monthly = np.where(rng.random(n_leases) < .45,
                       rng.integers(20, 90, n_leases), 0)
    dong_effect = pd.Series(lease_dong).map(
        {dong: factor for dong, factor in zip(
            dongs, np.linspace(.85, 1.2, len(dongs)))}
    ).to_numpy()
    deposit = (
        3000 + lease_area * 180 * dong_effect - monthly * 20
        - lease_age * 15 + rng.normal(0, 1200, n_leases)
    ).clip(300, None)
    lease_month = rng.choice(valid_months, n_leases)
    leases = pd.DataFrame({
        "contract_id": [f"SYN-L-{i:06d}" for i in range(n_leases)],
        "contract_year_month": lease_month,
        "sigungu_code": "41115",
        "legal_dong": lease_dong,
        "partial_lot_number": "",
        "housing_type": rng.choice(["다가구", "단독"], n_leases, p=[.85, .15]),
        "rental_area": lease_area,
        "built_year": lease_month // 100 - lease_age,
        "building_age": lease_age,
        "deposit": deposit,
        "monthly_rent": monthly,
        "contract_type": np.where(monthly > 0, "보증부월세", "전세"),
        "renewal_flag": "",
        "legal_dong_3m_deposit_median": pd.Series(deposit).rolling(
            50, min_periods=1).median().shift(1).bfill(),
        "legal_dong_12m_deposit_median": pd.Series(deposit).rolling(
            200, min_periods=1).median().shift(1).bfill(),
        "legal_dong_12m_deposit_growth": rng.normal(.02, .03, n_leases),
        "transaction_count_3m": rng.integers(10, 80, n_leases),
        "transaction_count_12m": rng.integers(60, 300, n_leases),
    })

    sale_land = rng.lognormal(np.log(160), .4, n_sales)
    sale_floor = sale_land * rng.uniform(1.2, 2.8, n_sales)
    sale_age = rng.integers(0, 45, n_sales)
    official_land = rng.normal(260, 65, n_sales).clip(80, None)
    sale_price = (
        sale_land * official_land * rng.normal(1.65, .2, n_sales)
        + sale_floor * 65 - sale_age * 120
        + rng.normal(0, 7000, n_sales)
    ).clip(5000, None)
    sales = pd.DataFrame({
        "sale_id": [f"SYN-S-{i:06d}" for i in range(n_sales)],
        "contract_year_month": rng.choice(valid_months, n_sales),
        "sigungu_code": "41115",
        "legal_dong": rng.choice(dongs, n_sales),
        "partial_lot_number": "",
        "housing_type": rng.choice(["다가구", "단독"], n_sales, p=[.8, .2]),
        "sale_price": sale_price,
        "land_area": sale_land,
        "total_floor_area": sale_floor,
        "residential_floor_area": sale_floor * rng.uniform(.7, 1, n_sales),
        "building_age": sale_age,
        "ground_floors": rng.integers(2, 6, n_sales),
        "parking_count": rng.integers(0, 10, n_sales),
        "structure_code": rng.choice(["철근콘크리트", "벽돌"], n_sales),
        "main_use_code": rng.choice(["다가구주택", "단독주택"], n_sales),
        "official_house_price": sale_price * rng.uniform(.55, .8, n_sales),
        "official_land_price": official_land,
        "nearby_sale_price_per_land_area": sale_price / sale_land,
        "nearby_sale_price_per_floor_area": sale_price / sale_floor,
        "match_confidence": "exact",
    })

    rental_assets = rng.lognormal(np.log(55_000), .85, n_survey)
    other_ratio = rng.lognormal(np.log(.7), .75, n_survey)
    other_assets = rental_assets * other_ratio
    total_assets = rental_assets + other_assets
    rental_deposit = rental_assets * rng.beta(2.3, 4.2, n_survey)
    financial_debt = total_assets * rng.beta(1.4, 8.5, n_survey)
    survey = pd.DataFrame({
        "total_assets": total_assets,
        "financial_assets": other_assets * rng.uniform(.2, .7, n_survey),
        "real_estate_assets": rental_assets + other_assets * .5,
        "rental_real_estate_assets": rental_assets,
        "owner_occupied_home_assets": other_assets * .4,
        "rental_deposit_liability": rental_deposit,
        "financial_debt": financial_debt,
        "survey_weight": rng.lognormal(0, .5, n_survey),
        "region": rng.choice(["경기", "서울", "충청", "부산"], n_survey,
                             p=[.38, .25, .2, .17]),
        "capital_region": False,
        "home_count": rng.integers(1, 5, n_survey),
        "survey_year": rng.choice([2023, 2024, 2025, 2026], n_survey,
                                  p=[.25, .35, .3, .1]),
    })
    survey["capital_region"] = survey["region"].isin(["서울", "경기", "인천"])
    survey["K_other"] = other_assets / rental_assets
    survey["R_survey"] = rental_deposit / total_assets
    survey["L_debt"] = financial_debt / total_assets
    return {
        "buildings": buildings, "leases": leases,
        "sales": sales, "survey": survey,
    }
