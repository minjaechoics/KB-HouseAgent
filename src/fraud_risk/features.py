"""
(1) 전세사기 위험도 — 피처 엔지니어링.

다가구주택 특수성을 반영한 피처를 '실험 세트'로 나눠 제공한다.
연구/실무 근거:
  - 안전식: (내보증금+선순위보증금+선순위근저당) <= 시세*낙찰가율
    https://brunch.co.kr/@b2fa784ba86f4b0/19
    https://biz.heraldcorp.com/article/10776093
  - 후순위 위험(다가구 등기 단일 → 전세대 경합):
    https://realestate.ehyun.co.kr/dangerous-jeonse-scam-legal-response
    https://www.lawtalk.co.kr/posts/113459

FEATURE_SETS: 실험 케이스별 피처 목록. train 스크립트에서 --feature_set 으로 선택.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from src import config


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """원본 매물 DataFrame -> 파생 피처 추가."""
    df = df.copy()
    market = df["market_price_manwon"].to_numpy()
    units = df["building_total_units"].to_numpy()
    deposit = df["deposit_manwon"].to_numpy()
    senior_dep = df["senior_deposit_sum_manwon"].to_numpy()
    senior_mtg = df["senior_mortgage_manwon"].to_numpy()
    rank = df["my_priority_rank"].to_numpy()

    recover = market * config.AUCTION_RECOVERY_RATIO
    unit_value = market / np.maximum(units, 1)

    # 핵심 위험 피처 --------------------------------------------------
    # 1) 부채비율(안전식): 내 보증금까지 포함해 낙찰가로 회수 가능한가
    df["debt_ratio"] = (deposit + senior_dep + senior_mtg) / np.maximum(recover, 1.0)
    # 2) 선순위담보비율: 내 앞의 채권만
    df["senior_ratio"] = (senior_dep + senior_mtg) / np.maximum(recover, 1.0)
    # 3) 전세가율(호실가치 대비)
    df["jeonse_ratio"] = deposit / np.maximum(unit_value, 1.0)
    # 4) 후순위 정도 (0=최선순위, 1=최후순위)
    df["rank_frac"] = (rank - 1) / np.maximum(units - 1, 1)
    # 5) LTV(근저당만)
    df["mortgage_ltv"] = senior_mtg / np.maximum(market, 1.0)
    # 6) 내 보증금 흡수 여력 (낙찰가에서 선순위 뺀 잔여 / 내 보증금)
    residual = np.maximum(recover - senior_dep - senior_mtg, 0.0)
    df["recovery_cushion"] = residual / np.maximum(deposit, 1.0)  # >1이면 안전
    # 7) 건물 규모(세대수 많을수록 경합 심함)
    df["log_units"] = np.log1p(units)
    # 8) 연식
    df["building_age_years"] = df["building_age_years"]
    # 9) 수도권 여부
    metro = {"서울", "경기", "인천"}
    df["is_metro"] = df["sido"].isin(metro).astype(int)

    return df


# 실험용 피처 세트 -------------------------------------------------------
FEATURE_SETS: dict[str, list[str]] = {
    # A) 최소 핵심 위험식만 (해석성 최고)
    "core": [
        "debt_ratio", "senior_ratio", "jeonse_ratio", "rank_frac",
    ],
    # B) 핵심 + 물건 속성
    "core_plus": [
        "debt_ratio", "senior_ratio", "jeonse_ratio", "rank_frac",
        "mortgage_ltv", "recovery_cushion", "log_units",
        "building_age_years", "is_metro",
    ],
    # C) 원자료 raw 위주 (모델이 스스로 비율 학습하게)
    "raw": [
        "deposit_manwon", "market_price_manwon", "senior_deposit_sum_manwon",
        "senior_mortgage_manwon", "my_priority_rank", "building_total_units",
        "building_age_years", "area_m2",
    ],
    # D) 전체 (파생 + raw)
    "full": [
        "debt_ratio", "senior_ratio", "jeonse_ratio", "rank_frac",
        "mortgage_ltv", "recovery_cushion", "log_units",
        "deposit_manwon", "market_price_manwon", "senior_deposit_sum_manwon",
        "senior_mortgage_manwon", "my_priority_rank", "building_total_units",
        "building_age_years", "area_m2", "is_metro",
    ],
}

LABEL_COL = "fraud_label"


def build_xy(df: pd.DataFrame, feature_set: str):
    """전세 물건만 필터 -> (X, y, feature_names)."""
    if feature_set not in FEATURE_SETS:
        raise KeyError(f"unknown feature_set '{feature_set}'. choices={list(FEATURE_SETS)}")
    feats = engineer(df)
    jeonse = feats[feats["lease_type"] == "전세"].dropna(subset=[LABEL_COL])
    cols = FEATURE_SETS[feature_set]
    X = jeonse[cols].astype(float).to_numpy()
    y = jeonse[LABEL_COL].astype(int).to_numpy()
    return X, y, cols
