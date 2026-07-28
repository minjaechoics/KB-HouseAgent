from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any


SUWON_SIGUNGU = {
    "41111": "장안구",
    "41113": "권선구",
    "41115": "팔달구",
    "41117": "영통구",
}

BUILDING_FIELDS = (
    "building_id", "management_register_pk", "sigungu_code",
    "legal_dong_code", "lot_number", "road_address", "latitude", "longitude",
    "main_use_code", "detailed_use", "structure_code", "land_area",
    "building_area", "total_floor_area", "residential_floor_area",
    "household_count", "family_count", "unit_count", "parking_count",
    "ground_floors", "underground_floors", "approval_date", "violation_flag",
)

LEASE_FIELDS = (
    "contract_id", "contract_year_month", "sigungu_code", "legal_dong",
    "partial_lot_number", "housing_type", "rental_area", "built_year",
    "deposit", "monthly_rent", "contract_type", "renewal_flag",
)

SALE_FIELDS = (
    "sale_id", "contract_year_month", "sigungu_code", "legal_dong",
    "partial_lot_number", "housing_type", "sale_price", "land_area",
    "total_floor_area", "built_year", "match_confidence",
)

SURVEY_MAPPING_FIELDS = (
    "total_assets", "financial_assets", "real_estate_assets",
    "rental_real_estate_assets", "owner_occupied_home_assets",
    "rental_deposit_liability", "financial_debt", "survey_weight", "region",
)


class SchemaValidationError(ValueError):
    """Raised when an official/raw input cannot be mapped without guessing."""


def require_columns(columns: list[str] | tuple[str, ...] | Any,
                    required: tuple[str, ...], source: str) -> None:
    present = set(columns)
    missing = [name for name in required if name not in present]
    if missing:
        raise SchemaValidationError(
            f"{source}: required columns missing: {', '.join(missing)}")


@dataclass
class BuildingEstimateInput:
    building_id: str
    legal_dong: str = ""
    legal_dong_code: str = ""
    sigungu_code: str = "41115"
    main_use_code: str = "다가구주택"
    structure_code: str = "unknown"
    land_area: float | None = None
    total_floor_area: float | None = None
    residential_floor_area: float | None = None
    unit_count: int | None = None
    family_count: int | None = None
    household_count: int | None = None
    parking_count: float | None = None
    ground_floors: int | None = None
    building_age: float | None = None
    built_year: int | None = None
    monthly_rent: float = 0.0
    contract_type: str = "전세"
    housing_type: str = "다가구"
    official_house_price: float | None = None
    official_land_price: float | None = None
    market_price: float | None = None
    observed_deposit: float | None = None
    source_kind: str = "unknown"
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> "BuildingEstimateInput":
        def number(*names: str) -> float | None:
            for name in names:
                value = row.get(name)
                if value not in (None, ""):
                    try:
                        value = float(value)
                        if value == value:
                            return value
                    except (TypeError, ValueError):
                        pass
            return None

        def integer(*names: str) -> int | None:
            value = number(*names)
            return int(value) if value is not None else None

        build_year = integer("built_year", "build_year")
        age = number("building_age", "building_age_years")
        if age is None and build_year is not None:
            age = max(0, date.today().year - build_year)
        return cls(
            building_id=str(row.get("building_id") or row.get("property_id") or ""),
            legal_dong=str(row.get("legal_dong") or row.get("dong") or ""),
            legal_dong_code=str(row.get("legal_dong_code") or ""),
            sigungu_code=str(row.get("sigungu_code") or row.get("lawd_cd") or "41115"),
            main_use_code=str(row.get("main_use_code") or row.get("building_use")
                              or row.get("house_type") or "다가구주택"),
            structure_code=str(row.get("structure_code") or
                               row.get("building_structure") or "unknown"),
            land_area=number("land_area", "land_area_m2"),
            total_floor_area=number(
                "total_floor_area", "building_total_area_m2", "contract_area_m2"),
            residential_floor_area=number(
                "residential_floor_area", "exclusive_area_m2", "area_m2"),
            unit_count=integer("unit_count", "building_total_units"),
            family_count=integer("family_count"),
            household_count=integer(
                "household_count", "building_total_households"),
            parking_count=number("parking_count", "parking_total"),
            ground_floors=integer("ground_floors", "total_floors"),
            building_age=age,
            built_year=build_year,
            monthly_rent=number("monthly_rent", "monthly_rent_manwon") or 0.0,
            contract_type=str(row.get("contract_type") or
                              row.get("transaction_type") or "전세"),
            housing_type=str(row.get("housing_type") or
                             row.get("house_type") or "다가구"),
            official_house_price=number(
                "official_house_price", "official_building_price_manwon"),
            official_land_price=number(
                "official_land_price", "official_land_price_manwon_m2"),
            market_price=number(
                "market_price", "market_price_manwon", "sale_price_manwon"),
            observed_deposit=number("deposit", "deposit_manwon"),
            source_kind=str(row.get("source_type") or row.get("source_kind") or "unknown"),
            extra=dict(row),
        )

    def observed_registered_units(self, max_valid: int = 100) -> int | None:
        for value in (self.unit_count, self.family_count, self.household_count):
            if value is not None and 1 <= int(value) <= max_valid:
                return int(value)
        return None
