"""Strict ingestion contract for contract-level actual HF/HUG accident labels."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ALIASES = {
    "guarantee_id": ("guarantee_id", "보증번호", "일련번호"),
    "guarantee_issue_date": ("guarantee_issue_date", "보증발급일", "보증시작일", "계약일"),
    "guarantee_end_date": ("guarantee_end_date", "보증종료일", "계약종료일"),
    "accident_label": ("accident_label", "보증사고여부", "사고여부", "대위변제여부"),
    "house_price_manwon": ("house_price_manwon", "주택가격_만원", "주택가액_만원"),
    "deposit_manwon": ("deposit_manwon", "보증금_만원", "임대보증금_만원"),
    "senior_claim_manwon": ("senior_claim_manwon", "선순위채권_만원"),
    "house_type": ("house_type", "주택유형", "주택구분"),
    "sido": ("sido", "시도", "지역"),
    "landlord_is_corporation": ("landlord_is_corporation", "법인임대인"),
    "landlord_is_multi_home": ("landlord_is_multi_home", "다주택자"),
    "registered_rental_business": ("registered_rental_business", "등록임대사업자"),
    "tenant_is_youth": ("tenant_is_youth", "청년임차인"),
    "price_assessment_method": ("price_assessment_method", "가격산정방식"),
    "monthly_rent_manwon": ("monthly_rent_manwon", "월세_만원"),
    "jeonse_loan_manwon": ("jeonse_loan_manwon", "전세대출금_만원"),
    "sale_price_index_change_pct": ("sale_price_index_change_pct", "매매가격지수변화율"),
    "mortgage_rate_change_pctp": ("mortgage_rate_change_pctp", "주담대금리차"),
    "loss_amount_manwon": ("loss_amount_manwon", "손실금액_만원", "대위변제금액_만원"),
}

REQUIRED_MODEL_COLUMNS = {
    "house_price_manwon", "deposit_manwon", "senior_claim_manwon", "house_type", "sido",
    "landlord_is_corporation", "landlord_is_multi_home", "registered_rental_business",
    "tenant_is_youth", "price_assessment_method", "monthly_rent_manwon",
    "jeonse_loan_manwon", "sale_price_index_change_pct", "mortgage_rate_change_pctp",
}


@dataclass(frozen=True)
class LabelAudit:
    source_sha256: str
    input_rows: int
    eligible_rows: int
    excluded_immature_rows: int
    positive_rows: int
    observation_cutoff: str
    provider: str


def _read_csv(path: Path) -> pd.DataFrame:
    last_error = None
    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def _rename_aliases(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for canonical, candidates in ALIASES.items():
        for candidate in candidates:
            if candidate in df.columns:
                rename[candidate] = canonical
                break
    return df.rename(columns=rename)


def load_actual_contract_labels(
    csv_path: str | Path,
    provenance_path: str | Path,
    *,
    min_rows: int = 100,
) -> tuple[pd.DataFrame, LabelAudit, dict]:
    """Load only actual, matured, contract-level labels with signed provenance metadata."""
    csv_path = Path(csv_path)
    provenance_path = Path(provenance_path)
    suspicious = f"{csv_path.name} {provenance_path.name}".lower()
    if "synthetic" in suspicious or "합성" in suspicious:
        raise ValueError("합성데이터는 실제 사고 라벨 학습에 사용할 수 없습니다.")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("data_classification") != "actual_contract_level":
        raise ValueError("provenance.data_classification must be 'actual_contract_level'")
    if provenance.get("contains_synthetic") is not False:
        raise ValueError("provenance.contains_synthetic must explicitly be false")
    if not provenance.get("provider") or not provenance.get("observation_cutoff"):
        raise ValueError("provenance requires provider and observation_cutoff")

    raw = _read_csv(csv_path)
    if "is_synthetic" in raw and raw["is_synthetic"].fillna(False).astype(bool).any():
        raise ValueError("is_synthetic=true rows are forbidden")
    df = _rename_aliases(raw)
    base_required = {"guarantee_id", "guarantee_issue_date", "guarantee_end_date", "accident_label"}
    missing = sorted((base_required | REQUIRED_MODEL_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"계약 단위 실제 라벨 필수 컬럼 누락: {missing}")
    if len(df) < min_rows:
        raise ValueError(f"actual label rows must be >= {min_rows}; got {len(df)}")
    if df["guarantee_id"].isna().any() or df["guarantee_id"].duplicated().any():
        raise ValueError("guarantee_id must be present and unique")

    df["guarantee_issue_date"] = pd.to_datetime(df["guarantee_issue_date"], errors="coerce")
    df["guarantee_end_date"] = pd.to_datetime(df["guarantee_end_date"], errors="coerce")
    cutoff = pd.Timestamp(provenance["observation_cutoff"])
    if df[["guarantee_issue_date", "guarantee_end_date"]].isna().any().any():
        raise ValueError("invalid guarantee issue/end dates")
    # HUG defines non-return accident after one month; negatives not observed for
    # at least 30 days after maturity are right-censored and must not be controls.
    mature = df["guarantee_end_date"] <= cutoff - pd.Timedelta(days=30)
    eligible = df.loc[mature].copy()
    eligible["accident_label"] = pd.to_numeric(eligible["accident_label"], errors="raise").astype(int)
    if not set(eligible["accident_label"].unique()).issubset({0, 1}):
        raise ValueError("accident_label must contain only 0/1")
    if eligible["accident_label"].nunique() != 2:
        raise ValueError("both accident and normal labels are required")

    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    expected = provenance.get("sha256")
    if expected and expected.lower() != digest:
        raise ValueError("source sha256 does not match provenance")
    audit = LabelAudit(
        source_sha256=digest,
        input_rows=len(df),
        eligible_rows=len(eligible),
        excluded_immature_rows=int((~mature).sum()),
        positive_rows=int(eligible["accident_label"].sum()),
        observation_cutoff=cutoff.date().isoformat(),
        provider=str(provenance["provider"]),
    )
    return eligible.sort_values("guarantee_issue_date").reset_index(drop=True), audit, provenance

