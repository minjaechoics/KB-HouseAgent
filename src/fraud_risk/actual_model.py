"""Feature contract and the externally validated HF accident-risk model.

The published coefficients in this module were estimated from 453,122 actual
guarantee contracts.  They are used only as a transparent transfer model until
contract-level HUG/HF labels are supplied to :mod:`train_actual`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


HF_PAPER_DOI = "https://doi.org/10.52344/hfr.2025.9.2.47"
HF_SAMPLE_ACCIDENT_RATE = 22_601 / 453_122

# Table 10, full-sample binary logit model. Issuance-year controls were omitted
# from the published table, so the raw intercept is never used without a
# documented portfolio-prior adjustment.
HF_PUBLISHED_INTERCEPT = -40.347
HF_PUBLISHED_COEFFICIENTS: dict[str, float] = {
    "landlord_corporation": 1.286,
    "landlord_multi_home": 0.930,
    "registered_rental_business": -0.037,
    "tenant_youth": 0.299,
    "house_officetel": 0.836,
    "house_row_multifamily": 1.759,
    "house_detached_multihousehold": -0.205,
    "debt_60_70": 0.411,
    "debt_70_80": 0.952,
    "debt_80_90": 1.703,
    "debt_90_plus": 3.399,
    "log_deposit_won": 1.613,
    "valuation_expert": 1.577,
    "valuation_supplier": 0.920,
    "monthly_rent_contract": -1.264,
    "has_senior_claim": -0.737,
    "has_jeonse_loan": -1.362,
    "loan_to_deposit": 1.570,
    "sale_price_index_decline": 0.612,
    "mortgage_rate_change_pctp": 0.490,
    "region_incheon": 1.947,
    "region_gyeonggi": 1.273,
}
ACTUAL_FEATURE_NAMES = list(HF_PUBLISHED_COEFFICIENTS)


def _series(df: pd.DataFrame, names: tuple[str, ...], default: Any = 0) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series(default, index=df.index)


def _numeric(df: pd.DataFrame, names: tuple[str, ...], default: float = 0.0) -> pd.Series:
    return pd.to_numeric(_series(df, names, default), errors="coerce").fillna(default)


def _boolean(df: pd.DataFrame, names: tuple[str, ...], default: bool = False) -> pd.Series:
    raw = _series(df, names, default)
    if pd.api.types.is_bool_dtype(raw):
        return raw.fillna(default).astype(float)
    numeric = pd.to_numeric(raw, errors="coerce")
    text = raw.astype(str).str.strip().str.lower()
    positive = text.isin({"1", "true", "y", "yes", "예", "네", "해당", "사고", "법인"})
    return numeric.fillna(positive.astype(float)).clip(0, 1).astype(float)


def build_actual_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-free HF-aligned features from labels or listing rows."""
    result = pd.DataFrame(index=df.index)

    house_price = _numeric(df, ("house_price_manwon", "market_price_manwon", "주택가격_만원"))
    deposit = _numeric(df, ("deposit_manwon", "guarantee_amount_manwon", "보증금_만원"))
    senior_direct = _numeric(df, ("senior_claim_manwon", "선순위채권_만원"), np.nan)
    senior_deposit = _numeric(df, ("senior_deposit_sum_manwon",), 0.0)
    senior_mortgage = _numeric(df, ("senior_mortgage_manwon",), 0.0)
    senior = senior_direct.where(senior_direct.notna(), senior_deposit + senior_mortgage)
    debt_pct = (deposit + senior).div(house_price.clip(lower=1.0)).mul(100.0)

    ownership = _series(df, ("ownership_type", "registered_owner_type"), "").astype(str)
    result["landlord_corporation"] = np.maximum(
        _boolean(df, ("landlord_is_corporation",)), ownership.str.contains("법인").astype(float)
    )
    result["landlord_multi_home"] = _boolean(df, ("landlord_is_multi_home", "multi_home_landlord"))
    result["registered_rental_business"] = _boolean(
        df, ("registered_rental_business", "private_rental_housing")
    )
    result["tenant_youth"] = _boolean(df, ("tenant_is_youth", "tenant_youth"))

    house = _series(df, ("house_type", "property_type", "주택유형"), "").astype(str)
    result["house_officetel"] = house.str.contains("오피스텔").astype(float)
    result["house_row_multifamily"] = house.str.contains("연립|다세대|빌라").astype(float)
    result["house_detached_multihousehold"] = house.str.contains("단독|다가구").astype(float)

    result["debt_60_70"] = ((debt_pct >= 60) & (debt_pct < 70)).astype(float)
    result["debt_70_80"] = ((debt_pct >= 70) & (debt_pct < 80)).astype(float)
    result["debt_80_90"] = ((debt_pct >= 80) & (debt_pct < 90)).astype(float)
    result["debt_90_plus"] = (debt_pct >= 90).astype(float)
    result["log_deposit_won"] = np.log((deposit.clip(lower=1.0) * 10_000.0))

    valuation = _series(
        df, ("price_assessment_method", "valuation_method", "price_estimation_method"), ""
    ).astype(str)
    result["valuation_expert"] = valuation.str.contains("감정|공인중개|전문가").astype(float)
    result["valuation_supplier"] = valuation.str.contains("분양가|1년.?이내.?매매|공급자").astype(float)

    monthly_rent = _numeric(df, ("monthly_rent_manwon",), 0.0)
    result["monthly_rent_contract"] = np.maximum(
        _boolean(df, ("has_monthly_rent", "monthly_rent_contract")),
        (monthly_rent > 0).astype(float),
    )
    result["has_senior_claim"] = np.maximum(
        _boolean(df, ("has_senior_claim",)), (senior > 0).astype(float)
    )
    loan_amount = _numeric(df, ("jeonse_loan_manwon", "loan_amount_manwon"), 0.0)
    result["has_jeonse_loan"] = np.maximum(
        _boolean(df, ("has_jeonse_loan",)), (loan_amount > 0).astype(float)
    )
    result["loan_to_deposit"] = loan_amount.div(deposit.clip(lower=1.0)).clip(0, 2)

    price_change = _numeric(
        df, ("sale_price_index_change_pct", "sale_price_change_pct"), 0.0
    )
    result["sale_price_index_decline"] = np.maximum(
        _boolean(df, ("sale_price_index_decline",)), (price_change < 0).astype(float)
    )
    result["mortgage_rate_change_pctp"] = _numeric(
        df, ("mortgage_rate_change_pctp", "mortgage_rate_change_pct"), 0.0
    ).clip(-5, 10)

    sido = _series(df, ("sido", "region_sido", "시도"), "").astype(str)
    result["region_incheon"] = sido.str.contains("인천").astype(float)
    result["region_gyeonggi"] = sido.str.contains("경기").astype(float)
    return result[ACTUAL_FEATURE_NAMES].astype(float)


def stable_sigmoid(logit: np.ndarray) -> np.ndarray:
    logit = np.asarray(logit, dtype=float)
    out = np.empty_like(logit)
    positive = logit >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-logit[positive]))
    exp_x = np.exp(logit[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def published_hf_logits(df: pd.DataFrame) -> np.ndarray:
    features = build_actual_feature_frame(df)
    coef = np.array([HF_PUBLISHED_COEFFICIENTS[c] for c in ACTUAL_FEATURE_NAMES])
    return HF_PUBLISHED_INTERCEPT + features.to_numpy() @ coef


def solve_prior_logit_shift(logits: np.ndarray, target_rate: float) -> float:
    """Find an intercept shift whose mean predicted probability is target_rate."""
    if not 0 < target_rate < 1:
        raise ValueError("target_rate must be between 0 and 1")
    lo, hi = -30.0, 30.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if stable_sigmoid(logits + mid).mean() < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def cost_ratio_threshold(false_negative_cost: float, false_positive_cost: float) -> float:
    """Bayes threshold for calibrated probabilities and constant action costs."""
    if false_negative_cost <= 0 or false_positive_cost <= 0:
        raise ValueError("costs must be positive")
    return float(false_positive_cost / (false_positive_cost + false_negative_cost))


@dataclass(frozen=True)
class PublishedHFModel:
    prior_logit_shift: float
    decision_threshold: float

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return stable_sigmoid(published_hf_logits(df) + self.prior_logit_shift)

