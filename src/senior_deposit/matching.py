from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping


def normalize_korean_address(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = text.replace("번지", " ")
    text = re.sub(r"(?<=\d)\s*-\s*(?=\d)", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_masked_lot(value: object) -> bool:
    text = str(value or "")
    return "*" in text or "x" in text.lower()


@dataclass(frozen=True)
class MatchAssessment:
    confidence: str
    score: float
    usable_as_building_label: bool
    reasons: tuple[str, ...]


def assess_rtms_building_candidate(
    contract: Mapping[str, object],
    building: Mapping[str, object],
) -> MatchAssessment:
    """Score a candidate while preventing partial-lot label promotion."""
    reasons: list[str] = []
    score = 0.0
    contract_dong = str(contract.get("legal_dong") or "").strip()
    building_dong = str(building.get("legal_dong") or "").strip()
    if contract_dong and contract_dong == building_dong:
        score += .35
        reasons.append("legal_dong_match")

    contract_area = contract.get("rental_area")
    building_area = (
        building.get("residential_floor_area")
        or building.get("total_floor_area"))
    try:
        if contract_area and building_area:
            unit_count = max(
                1.0,
                float(
                    building.get("registered_units_observed")
                    or building.get("unit_count")
                    or 1),
            )
            expected_unit_area = float(building_area) / unit_count
            relative_error = abs(
                float(contract_area) - expected_unit_area
            ) / max(expected_unit_area, 1.0)
            if relative_error <= .2:
                score += .25
                reasons.append("unit_area_compatible")
    except (TypeError, ValueError):
        pass

    contract_year = contract.get("built_year")
    building_year = building.get("built_year")
    try:
        if contract_year and building_year and abs(
                int(float(contract_year)) - int(float(building_year))) <= 1:
            score += .15
            reasons.append("built_year_compatible")
    except (TypeError, ValueError):
        pass

    partial_lot = str(contract.get("partial_lot_number") or "")
    building_lot = normalize_korean_address(
        building.get("lot_number") or building.get("road_address"))
    masked = is_masked_lot(partial_lot)
    full_exact_lot = (
        bool(partial_lot)
        and not masked
        and normalize_korean_address(partial_lot) == building_lot
    )
    if full_exact_lot:
        score += .4
        reasons.append("unmasked_full_lot_match")
    elif masked:
        reasons.append("masked_partial_lot_not_exact")

    score = min(score, 1.0)
    usable = bool(full_exact_lot and score >= .75)
    if usable:
        confidence = "exact"
    elif score >= .6:
        confidence = "medium"
    elif score >= .35:
        confidence = "low"
    else:
        confidence = "unmatched"
    return MatchAssessment(
        confidence=confidence,
        score=round(score, 4),
        usable_as_building_label=usable,
        reasons=tuple(reasons),
    )
