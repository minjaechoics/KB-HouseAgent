from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import yaml

from .schemas import (
    BUILDING_FIELDS,
    LEASE_FIELDS,
    SALE_FIELDS,
    SUWON_SIGUNGU,
    SURVEY_MAPPING_FIELDS,
    SchemaValidationError,
    require_columns,
)


def _money(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).replace(",", "").strip()
    return float(text) if text else np.nan


def normalize_rtms_leases(frame: pd.DataFrame) -> pd.DataFrame:
    """Map the official SHRent schema without attempting building-level joins."""
    require_columns(
        frame.columns,
        ("lawd_cd", "deal_ym", "umdNm", "houseType", "totalFloorAr",
         "buildYear", "deposit", "monthlyRent"),
        "RTMS SHRent",
    )
    out = pd.DataFrame({
        "contract_id": [
            hashlib.sha256(
                "|".join(map(str, row)).encode("utf-8")).hexdigest()[:24]
            for row in frame[[
                "lawd_cd", "deal_ym", "umdNm", "totalFloorAr",
                "deposit", "monthlyRent",
            ]].itertuples(index=False, name=None)
        ],
        "contract_year_month": pd.to_numeric(frame["deal_ym"], errors="coerce"),
        "sigungu_code": frame["lawd_cd"].astype(str).str.zfill(5),
        "legal_dong": frame["umdNm"].fillna("").astype(str).str.strip(),
        "partial_lot_number": "",
        "housing_type": frame["houseType"].fillna("단독/다가구"),
        "rental_area": pd.to_numeric(frame["totalFloorAr"], errors="coerce"),
        "built_year": pd.to_numeric(frame["buildYear"], errors="coerce"),
        "deposit": frame["deposit"].map(_money),
        "monthly_rent": frame["monthlyRent"].map(_money),
        "contract_type": frame.get(
            "contractType", pd.Series("", index=frame.index)).fillna(""),
        "renewal_flag": frame.get(
            "useRRRight", pd.Series("", index=frame.index)).fillna(""),
    })
    inferred = np.where(
        out["monthly_rent"].fillna(0).gt(0), "보증부월세", "전세")
    out["contract_type"] = out["contract_type"].where(
        out["contract_type"].astype(str).str.len().gt(0), inferred)
    out["building_match_allowed"] = False
    out["match_confidence"] = "unmatched"
    out["source_kind"] = "official_rtms"
    return out


def normalize_rtms_sales(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        frame.columns,
        ("lawd_cd", "deal_ym", "umdNm", "houseType", "dealAmount",
         "plottageAr", "totalFloorAr", "buildYear"),
        "RTMS SHTrade",
    )
    partial = frame.get("jibun", pd.Series("", index=frame.index)).fillna("")
    out = pd.DataFrame({
        "sale_id": [
            hashlib.sha256(
                "|".join(map(str, row)).encode("utf-8")).hexdigest()[:24]
            for row in frame[[
                "lawd_cd", "deal_ym", "umdNm", "dealAmount",
                "plottageAr", "totalFloorAr",
            ]].itertuples(index=False, name=None)
        ],
        "contract_year_month": pd.to_numeric(frame["deal_ym"], errors="coerce"),
        "sigungu_code": frame["lawd_cd"].astype(str).str.zfill(5),
        "legal_dong": frame["umdNm"].fillna("").astype(str).str.strip(),
        "partial_lot_number": partial.astype(str),
        "housing_type": frame["houseType"].fillna("단독/다가구"),
        "sale_price": frame["dealAmount"].map(_money),
        "land_area": pd.to_numeric(frame["plottageAr"], errors="coerce"),
        "total_floor_area": pd.to_numeric(frame["totalFloorAr"], errors="coerce"),
        "built_year": pd.to_numeric(frame["buildYear"], errors="coerce"),
    })
    # RTMS masks single/multi-family lot numbers.  Never upgrade these rows to
    # exact/high confidence solely from the partial lot value.
    out["match_confidence"] = np.where(
        out["partial_lot_number"].str.contains(r"[*xX]", regex=True),
        "low", "medium")
    out["source_kind"] = "official_rtms"
    return out


def normalize_building_hub(frame: pd.DataFrame) -> pd.DataFrame:
    """Map Building HUB title rows to the canonical registry schema."""
    aliases = {
        "management_register_pk": ("mgmBldrgstPk", "management_register_pk"),
        "sigungu_code": ("sigunguCd", "sigungu_code"),
        "legal_dong_code": ("bjdongCd", "legal_dong_code"),
        "lot_number": ("platPlc", "lot_number"),
        "road_address": ("newPlatPlc", "road_address"),
        "main_use_code": ("mainPurpsCd", "main_use_code"),
        "detailed_use": ("etcPurps", "mainPurpsCdNm", "detailed_use"),
        "structure_code": ("strctCd", "strctCdNm", "structure_code"),
        "land_area": ("platArea", "land_area"),
        "building_area": ("archArea", "building_area"),
        "total_floor_area": ("totArea", "total_floor_area"),
        # Title endpoint does not expose a trustworthy residential-only area.
        # Populate this only after a separately validated floor/use aggregate.
        "residential_floor_area": ("residential_floor_area",),
        "household_count": ("hhldCnt", "household_count"),
        "family_count": ("fmlyCnt", "family_count"),
        "unit_count": ("hoCnt", "unit_count"),
        "parking_count": ("totPkngCnt", "parking_count"),
        "ground_floors": ("grndFlrCnt", "ground_floors"),
        "underground_floors": ("ugrndFlrCnt", "underground_floors"),
        "approval_date": ("useAprDay", "approval_date"),
        "violation_flag": ("violation_flag",),
        "latitude": ("latitude",),
        "longitude": ("longitude",),
    }

    def select(names: tuple[str, ...], default=np.nan) -> pd.Series:
        for name in names:
            if name in frame:
                return frame[name]
        return pd.Series(default, index=frame.index)

    out = pd.DataFrame({
        canonical: select(names)
        for canonical, names in aliases.items()
    })
    pk = out["management_register_pk"].fillna("").astype(str)
    out["building_id"] = [
        "HUB-" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]
        for value in pk
    ]
    out["legal_dong_code"] = (
        out["sigungu_code"].fillna("").astype(str).str.zfill(5)
        + out["legal_dong_code"].fillna("").astype(str).str.zfill(5)
    )
    out["source_kind"] = "official_building_hub"
    out = out.reindex(columns=[*BUILDING_FIELDS, "source_kind"])
    return out


def add_past_only_lease_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add monthly aggregates using strictly earlier contract months."""
    require_columns(frame.columns, LEASE_FIELDS, "normalized lease table")
    out = frame.copy()
    out["area_band"] = (
        pd.to_numeric(out["rental_area"], errors="coerce")
        .floordiv(10).mul(10).fillna(-1).astype(int)
    )
    monthly = (
        out.groupby(["legal_dong", "area_band", "contract_year_month"],
                    dropna=False)["deposit"]
        .agg(["median", "count"])
        .reset_index()
        .sort_values(["legal_dong", "area_band", "contract_year_month"])
    )
    pieces = []
    for _, group in monthly.groupby(["legal_dong", "area_band"], sort=False):
        group = group.copy()
        past = group["median"].shift(1)
        group["legal_dong_3m_deposit_median"] = (
            past.rolling(3, min_periods=1).median())
        group["legal_dong_12m_deposit_median"] = (
            past.rolling(12, min_periods=1).median())
        group["legal_dong_12m_deposit_growth"] = (
            group["legal_dong_12m_deposit_median"]
            .pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        )
        group["transaction_count_3m"] = (
            group["count"].shift(1).rolling(3, min_periods=1).sum())
        group["transaction_count_12m"] = (
            group["count"].shift(1).rolling(12, min_periods=1).sum())
        pieces.append(group)
    features = pd.concat(pieces, ignore_index=True) if pieces else monthly
    cols = [
        "legal_dong", "area_band", "contract_year_month",
        "legal_dong_3m_deposit_median",
        "legal_dong_12m_deposit_median",
        "legal_dong_12m_deposit_growth",
        "transaction_count_3m", "transaction_count_12m",
    ]
    out = out.merge(features[cols], how="left",
                    on=["legal_dong", "area_band", "contract_year_month"],
                    validate="many_to_one")
    out["building_age"] = (
        out["contract_year_month"].floordiv(100)
        - pd.to_numeric(out["built_year"], errors="coerce"))
    return out


def validate_buildings(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    require_columns(frame.columns, BUILDING_FIELDS, "building registry")
    out = frame.copy()
    numeric = [
        "land_area", "building_area", "total_floor_area",
        "residential_floor_area", "household_count", "family_count",
        "unit_count", "parking_count", "ground_floors", "underground_floors",
    ]
    for name in numeric:
        out[name] = pd.to_numeric(out[name], errors="coerce")
    invalid_units = {}
    for name in ("unit_count", "family_count", "household_count"):
        mask = out[name].notna() & ~out[name].between(1, 100)
        invalid_units[name] = int(mask.sum())
        out.loc[mask, name] = np.nan
    approval = pd.to_datetime(out["approval_date"], errors="coerce")
    out["building_age"] = (
        pd.Timestamp.today().year - approval.dt.year).clip(lower=0)
    registered = out["unit_count"].combine_first(
        out["family_count"]).combine_first(out["household_count"])
    out["registered_units_observed"] = registered
    denominator = registered.where(registered.gt(0))
    out["area_per_registered_unit"] = out["residential_floor_area"] / denominator
    out["parking_per_registered_unit"] = out["parking_count"] / denominator
    out["floor_area_ratio_observed"] = out["total_floor_area"] / out["land_area"]
    out["building_coverage_ratio_observed"] = out["building_area"] / out["land_area"]
    out["residential_area_ratio"] = (
        out["residential_floor_area"] / out["total_floor_area"])
    report = {
        "rows": len(out),
        "invalid_unit_values_set_missing": invalid_units,
        "missing_registered_units": int(registered.isna().sum()),
    }
    return out, report


def match_sales_to_buildings(
    sales: pd.DataFrame,
    buildings: pd.DataFrame,
) -> pd.DataFrame:
    """Conservative partial-lot matching with explicit confidence."""
    out = sales.copy()
    out["building_id"] = None
    out["match_confidence"] = "unmatched"

    def lot_token(value: object) -> str:
        text = str(value or "").strip()
        # Building HUB lot addresses commonly end in "번지", so anchoring the
        # number to the end of the string prevents every masked RTMS lot from
        # matching. The final numeric lot token is the main/sub lot pair.
        matches = re.findall(r"(\d+(?:-\d+)?)", text)
        return matches[-1] if matches else text

    building_rows = buildings.copy()
    building_rows["_lot"] = building_rows["lot_number"].map(lot_token)
    building_rows["_dong"] = (
        building_rows["legal_dong"].astype(str)
        if "legal_dong" in building_rows
        else building_rows["lot_number"].astype(str)
    )
    for index, sale in out.iterrows():
        partial = str(sale.get("partial_lot_number") or "").strip()
        if not partial:
            continue
        dong = str(sale.get("legal_dong") or "")
        candidates = building_rows[
            building_rows["_dong"].str.contains(
                re.escape(dong), regex=True, na=False)]
        masked = bool(re.search(r"[*xX]", partial))
        if masked:
            prefix = re.split(r"[*xX]", partial, maxsplit=1)[0]
            candidates = candidates[
                candidates["_lot"].str.startswith(prefix, na=False)]
        else:
            candidates = candidates[candidates["_lot"] == partial]
        if not len(candidates):
            continue

        land = float(sale.get("land_area") or np.nan)
        floor = float(sale.get("total_floor_area") or np.nan)
        score = pd.Series(0.0, index=candidates.index)
        if np.isfinite(land) and land > 0:
            score += (
                (candidates["land_area"] - land).abs()
                / land <= .05).astype(float)
        if np.isfinite(floor) and floor > 0:
            score += (
                (candidates["total_floor_area"] - floor).abs()
                / floor <= .05).astype(float)
        narrowed = candidates[score >= (2 if masked else 1)]
        if not masked and len(candidates) == 1:
            chosen, confidence = candidates.iloc[0], "exact"
        elif masked and len(narrowed) == 1:
            chosen, confidence = narrowed.iloc[0], "high"
        elif len(narrowed) == 1:
            chosen, confidence = narrowed.iloc[0], "high"
        elif len(candidates) == 1:
            chosen, confidence = candidates.iloc[0], "medium"
        else:
            out.at[index, "match_confidence"] = "low"
            continue
        out.at[index, "building_id"] = chosen["building_id"]
        out.at[index, "match_confidence"] = confidence
    return out


def load_survey_mapping(path: str | Path) -> dict[str, str]:
    mapping = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    missing = [key for key in SURVEY_MAPPING_FIELDS if key not in mapping]
    placeholders = [
        key for key in SURVEY_MAPPING_FIELDS
        if str(mapping.get(key, "")).upper() == "PLACEHOLDER"
    ]
    if missing or placeholders:
        raise SchemaValidationError(
            "survey schema mapping is incomplete; verify the year-specific "
            f"MDIS codebook. missing={missing}, placeholders={placeholders}")
    for key in list(mapping):
        if key not in SURVEY_MAPPING_FIELDS and key != "year":
            if str(mapping[key]).upper() == "PLACEHOLDER":
                mapping.pop(key)
    return mapping


def normalize_household_survey(frame: pd.DataFrame,
                               mapping: dict[str, str]) -> pd.DataFrame:
    required_actual = tuple(mapping[key] for key in SURVEY_MAPPING_FIELDS)
    require_columns(frame.columns, required_actual, "household survey microdata")
    out = pd.DataFrame({
        canonical: pd.to_numeric(frame[actual], errors="coerce")
        if canonical not in {"region"} else frame[actual].astype(str)
        for canonical, actual in mapping.items()
        if canonical != "year"
    })
    year_source = mapping.get("year")
    if isinstance(year_source, str) and year_source in frame.columns:
        out["survey_year"] = pd.to_numeric(
            frame[year_source], errors="coerce")
    else:
        out["survey_year"] = int(year_source or 0)
    out = out[
        out["rental_deposit_liability"].gt(0)
        & out["rental_real_estate_assets"].gt(0)
        & out["survey_weight"].gt(0)
    ].copy()
    eps = 1e-6
    out["K_other"] = (
        (out["total_assets"] - out["rental_real_estate_assets"])
        / (out["rental_real_estate_assets"] + eps)
    )
    negatives = out["K_other"].lt(0)
    if negatives.any():
        examples = out.loc[negatives, [
            "total_assets", "rental_real_estate_assets"]].head(5).to_dict("records")
        raise SchemaValidationError(
            "K_other is negative. Verify survey definitions/mapping before "
            f"training; values were not clipped. examples={examples}")
    out["R_survey"] = (
        out["rental_deposit_liability"] / (out["total_assets"] + eps))
    out["L_debt"] = out["financial_debt"] / (out["total_assets"] + eps)
    region = out["region"].astype(str).str.strip()
    out["capital_region"] = (
        region.isin({"G1", "1", "수도권", "서울", "경기", "인천"})
        | region.str.contains("서울|경기|인천|수도권", regex=True, na=False)
    )
    return out


class BuildingRegistryCollector:
    """Minimal official Building HUB collector with explicit key injection."""

    endpoint = (
        "https://apis.data.go.kr/1613000/"
        "BldRgstHubService/getBrTitleInfo"
    )
    floor_endpoint = (
        "https://apis.data.go.kr/1613000/"
        "BldRgstHubService/getBrFlrOulnInfo"
    )

    def __init__(self, service_key: str | None = None, timeout: int = 30):
        self.service_key = (
            service_key or os.environ.get("MOLIT_BUILDING_HUB_KEY", "")).strip()
        self.timeout = timeout

    def _collect_endpoint(self, endpoint: str,
                          legal_dong_codes: Iterable[str],
                          page_size: int) -> pd.DataFrame:
        if not self.service_key:
            raise RuntimeError(
                "MOLIT_BUILDING_HUB_KEY is required; no key is hardcoded.")
        rows: list[dict] = []
        for full_code in legal_dong_codes:
            sigungu, bjdong = str(full_code)[:5], str(full_code)[5:10]
            if sigungu not in SUWON_SIGUNGU:
                continue
            page = 1
            while True:
                response = requests.get(
                    endpoint,
                    params={
                        "serviceKey": self.service_key,
                        "sigunguCd": sigungu,
                        "bjdongCd": bjdong,
                        "numOfRows": page_size,
                        "pageNo": page,
                        "_type": "json",
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                body = response.json().get("response", {}).get("body", {})
                items = body.get("items", {}).get("item", [])
                if isinstance(items, dict):
                    items = [items]
                rows.extend(items)
                if page * page_size >= int(body.get("totalCount") or 0):
                    break
                page += 1
        return pd.DataFrame(rows)

    def collect(self, legal_dong_codes: Iterable[str],
                page_size: int = 1000) -> pd.DataFrame:
        codes = list(legal_dong_codes)
        titles = self._collect_endpoint(self.endpoint, codes, page_size)
        floors = self._collect_endpoint(self.floor_endpoint, codes, page_size)
        if len(titles) and len(floors) and {
            "mgmBldrgstPk", "area",
        }.issubset(floors.columns):
            use_name = floors.get(
                "mainPurpsCdNm", pd.Series("", index=floors.index)
            ).fillna("").astype(str)
            residential = floors[
                use_name.str.contains("주택|주거|다가구|단독", regex=True)
            ].copy()
            residential["_area"] = pd.to_numeric(
                residential["area"], errors="coerce")
            totals = residential.groupby(
                "mgmBldrgstPk")["_area"].sum(min_count=1)
            titles["residential_floor_area"] = (
                titles["mgmBldrgstPk"].map(totals))
        return titles


def write_provenance(path: Path, *, source_kind: str, rows: int,
                     extra: dict | None = None) -> None:
    payload = {
        "source_kind": source_kind,
        "rows": int(rows),
        "synthetic": source_kind.startswith("synthetic"),
        **(extra or {}),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")
