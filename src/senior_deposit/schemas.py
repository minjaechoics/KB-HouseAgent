from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from src.owner_asset_ratio.schemas import (
    BuildingEstimateInput,
    SchemaValidationError,
    require_columns,
)


SENIOR_LABEL_FIELDS = (
    "building_id",
    "reference_date",
    "registered_units",
    "occupied_units",
    "target_room_excluded",
    "tenant_id_anonymized",
    "room_identifier_anonymized",
    "deposit",
    "monthly_rent",
    "lease_start_date",
    "lease_end_date",
    "move_in_date",
    "confirmed_date",
    "currently_occupied",
    "senior_to_target",
    "label_source",
    "label_confidence",
)

FORBIDDEN_PII_COLUMNS = {
    "name", "tenant_name", "resident_registration_number", "rrn",
    "phone", "phone_number", "email",
}


@dataclass
class SeniorDepositInput:
    building: BuildingEstimateInput
    reference_date: date
    target_rooms_excluded: int = 1

    @classmethod
    def from_mapping(
        cls,
        row: dict[str, Any],
        *,
        reference_date: str | date,
        target_rooms_excluded: int = 1,
    ) -> "SeniorDepositInput":
        parsed_date = (
            reference_date
            if isinstance(reference_date, date)
            else date.fromisoformat(str(reference_date))
        )
        excluded = int(target_rooms_excluded)
        if excluded < 0 or excluded > 10:
            raise ValueError("target_rooms_excluded must be between 0 and 10")
        return cls(
            building=BuildingEstimateInput.from_mapping(row),
            reference_date=parsed_date,
            target_rooms_excluded=excluded,
        )


def validate_senior_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate evidence labels without fabricating missing seniority."""
    forbidden = sorted(FORBIDDEN_PII_COLUMNS.intersection(frame.columns))
    if forbidden:
        raise SchemaValidationError(
            "senior labels contain direct personal identifiers: "
            + ", ".join(forbidden)
        )
    require_columns(
        list(frame.columns), SENIOR_LABEL_FIELDS, "senior-deposit labels")
    clean = frame.copy()
    for column in (
        "registered_units", "occupied_units", "deposit", "monthly_rent",
        "label_confidence",
    ):
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    if clean["building_id"].fillna("").astype(str).str.strip().eq("").any():
        raise SchemaValidationError("building_id must not be empty")
    if clean["reference_date"].map(
            lambda value: pd.to_datetime(value, errors="coerce")).isna().any():
        raise SchemaValidationError("reference_date contains invalid values")
    if clean["registered_units"].lt(1).any():
        raise SchemaValidationError("registered_units must be positive")
    if clean["occupied_units"].lt(0).any():
        raise SchemaValidationError("occupied_units must be non-negative")
    if (clean["occupied_units"] > clean["registered_units"]).any():
        raise SchemaValidationError(
            "occupied_units cannot exceed registered_units")
    if clean["deposit"].lt(0).any() or clean["monthly_rent"].lt(0).any():
        raise SchemaValidationError("deposit and monthly_rent cannot be negative")
    if (~clean["label_confidence"].between(0, 1)).any():
        raise SchemaValidationError("label_confidence must be in [0, 1]")

    senior = clean["senior_to_target"]
    observed_senior = senior.notna() & senior.astype(str).str.strip().ne("")
    allowed = {"0", "1", "0.0", "1.0", "false", "true", "False", "True"}
    if (~senior[observed_senior].astype(str).isin(allowed)).any():
        raise SchemaValidationError(
            "senior_to_target must be blank, 0/1, or true/false")
    missing_evidence = (
        observed_senior
        & clean["label_source"].fillna("").astype(str).str.strip().eq("")
    )
    if missing_evidence.any():
        raise SchemaValidationError(
            "senior_to_target requires a non-empty label_source")
    return clean
