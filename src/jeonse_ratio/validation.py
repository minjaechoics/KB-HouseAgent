"""Cross-model schema, key, date and unit validation."""
from __future__ import annotations

from dataclasses import dataclass

from .adapters import DistributionContract


@dataclass(frozen=True)
class AlignmentPolicy:
    max_date_difference_days: int = 31
    require_same_currency: bool = True
    require_same_building_id: bool = True


def validate_alignment(
    deposit: DistributionContract,
    value: DistributionContract,
    policy: AlignmentPolicy = AlignmentPolicy(),
) -> dict:
    if policy.require_same_building_id and deposit.building_id != value.building_id:
        raise ValueError("building_id가 다른 모델 결과는 결합할 수 없습니다.")
    if policy.require_same_currency and deposit.currency != value.currency:
        raise ValueError("화폐가 다른 모델 결과는 결합할 수 없습니다.")
    if deposit.unit != value.unit:
        raise ValueError("adapter 이후 금액 단위가 일치하지 않습니다.")
    difference = abs((deposit.reference_date - value.reference_date).days)
    if difference > int(policy.max_date_difference_days):
        raise ValueError("기준일 차이가 허용 범위를 초과했습니다.")
    price_basis = value.metadata.get("price_basis")
    if price_basis not in {"market_value", "transaction_price_estimate"}:
        raise ValueError("시장가치가 아닌 가격은 전세가율 분모로 사용할 수 없습니다.")
    return {
        "building_id": deposit.building_id,
        "reference_date": max(
            deposit.reference_date, value.reference_date).isoformat(),
        "date_difference_days": difference,
        "currency": deposit.currency,
        "unit": deposit.unit,
    }
