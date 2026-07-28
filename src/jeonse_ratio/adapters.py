"""Adapters for the actual senior-deposit and property-value model outputs."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass(frozen=True)
class DistributionContract:
    building_id: str
    reference_date: date
    currency: str
    unit: str
    quality: str
    quantiles: dict[str, dict[str, float]]
    warnings: tuple[str, ...]
    metadata: dict[str, Any]


def _quality(value: Any) -> str:
    value = str(value or "low").lower()
    return value if value in {"high", "medium", "low"} else "low"


def _money_quantiles(values: dict, divisor: float) -> dict[str, float]:
    result = {}
    for key, value in (values or {}).items():
        if str(key).startswith("p") and value is not None:
            result[str(key).lower()] = max(0.0, float(value) / divisor)
    if not result:
        raise ValueError("금액 분위수 출력이 비어 있습니다.")
    return result


class DepositModelAdapter:
    """Normalize senior-model KRW summaries to internal 만원 distributions."""

    def adapt(self, result: dict) -> DistributionContract:
        if not result.get("available"):
            raise ValueError("임차보증금 모델 결과를 사용할 수 없습니다.")
        estimate = result.get("estimate") or {}
        building_id = str(
            estimate.get("building_id")
            or (result.get("match") or {}).get("building_id")
            or ""
        )
        if not building_id:
            raise ValueError("임차보증금 모델 building_id가 없습니다.")
        reference = estimate.get("reference_date")
        if not reference:
            raise ValueError("임차보증금 모델 reference_date가 없습니다.")
        total = _money_quantiles(
            estimate.get("estimated_total_deposit") or {}, 10_000)
        senior = _money_quantiles(
            estimate.get("estimated_senior_deposit") or {}, 10_000)
        upper = _money_quantiles(
            estimate.get("conservative_upper_deposit") or {}, 10_000)
        warnings = list(estimate.get("warnings") or [])
        warnings.append(
            "외부 출력에 원시 draw가 없어 분위수 piecewise-linear inverse CDF로 "
            "분포를 복원했습니다."
        )
        return DistributionContract(
            building_id=building_id,
            reference_date=date.fromisoformat(str(reference)),
            currency="KRW",
            unit="manwon",
            quality=_quality(estimate.get("data_quality")),
            quantiles={
                "total_deposit": total,
                "senior_deposit": senior,
                "conservative_upper_deposit": upper,
            },
            warnings=tuple(warnings),
            metadata={
                "model_name": estimate.get("model_name"),
                "model_version": estimate.get("model_version"),
                "model_mode": estimate.get("model_mode"),
                "source_unit": "KRW",
                "definition": (
                    "total=선택 호실을 제외한 기존 점유 호실 보증금 총합; "
                    "senior=그중 신규 임차인보다 우선한다고 가정한 총합; "
                    "upper=기존 점유 호실 전부를 선순위로 본 보수적 상한"
                ),
            },
        )


class PropertyValueModelAdapter:
    """Normalize market-value model output; owner total assets are not used."""

    def adapt(self, result: dict) -> DistributionContract:
        if not result.get("available"):
            raise ValueError("건물가치 모델 결과를 사용할 수 없습니다.")
        estimate = result.get("estimate") or {}
        values = estimate.get("estimated_property_value") or {}
        if not values:
            raise ValueError(
                "건물가치 모델에 estimated_property_value 출력이 없습니다.")
        building_id = str(
            estimate.get("building_id")
            or (result.get("match") or {}).get("building_id")
            or ""
        )
        reference = result.get("reference_date")
        if not building_id or not reference:
            raise ValueError("건물가치 모델 정렬 키가 없습니다.")
        warnings = list(estimate.get("warnings") or [])
        warnings.append(
            "건물가치 외부 출력의 분위수에서 시장가치 분포를 복원했습니다.")
        return DistributionContract(
            building_id=building_id,
            reference_date=date.fromisoformat(str(reference)),
            currency="KRW",
            unit="manwon",
            quality=_quality(estimate.get("data_quality")),
            quantiles={
                "property_value": _money_quantiles(values, 1.0),
            },
            warnings=tuple(warnings),
            metadata={
                "model_name": estimate.get("model_name"),
                "model_version": estimate.get("model_version"),
                "price_basis": "market_value",
                "source_unit": "manwon",
            },
        )
