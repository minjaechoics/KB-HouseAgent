from __future__ import annotations

from datetime import date
import hashlib
from pathlib import Path
import threading
import time
from typing import Any

import pandas as pd

from src import config
from .matching import normalize_korean_address
from .pipeline import SeniorDepositPipeline


LEASE_TYPES = {"전세", "월세"}
RESIDENTIAL_WORDS = ("다가구", "다세대", "연립", "단독주택", "공동주택", "주택")


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


class SeniorDepositIntegrationService:
    """One safe integration path for reports and the standalone API.

    A prototype listing is never treated as an official building row.  The
    service requires an exact normalized road-address match and records which
    Building HUB row was selected.  Fuzzy address matching is intentionally
    excluded because it can attach another owner's building to a listing.
    """

    def __init__(
        self,
        artifact_path: Path | None = None,
        registry_path: Path | None = None,
        *,
        cache_ttl_seconds: int = 3600,
    ):
        self.artifact_path = artifact_path or (
            config.ROOT / "models" / "senior_deposit"
            / "senior_deposit_actual.joblib"
        )
        self.registry_path = registry_path or (
            config.ROOT / "data" / "processed"
            / "owner_asset_ratio" / "buildings.csv"
        )
        self.cache_ttl_seconds = cache_ttl_seconds
        self._pipeline: SeniorDepositPipeline | None = None
        self._registry: pd.DataFrame | None = None
        self._cache: dict[str, tuple[float, dict]] = {}
        self._lock = threading.RLock()

    @property
    def available(self) -> bool:
        return self.artifact_path.exists() and self.registry_path.exists()

    def _load_pipeline(self) -> SeniorDepositPipeline:
        with self._lock:
            if self._pipeline is None:
                self._pipeline = SeniorDepositPipeline.load(
                    self.artifact_path, allow_synthetic=False
                )
            return self._pipeline

    def _load_registry(self) -> pd.DataFrame:
        with self._lock:
            if self._registry is None:
                registry = pd.read_csv(self.registry_path, low_memory=False)
                registry["_normalized_road_address"] = registry[
                    "road_address"
                ].map(normalize_korean_address)
                self._registry = registry
            return self._registry

    @staticmethod
    def _building_score(row: pd.Series) -> tuple[float, float, float]:
        use = f"{_text(row.get('main_use_code'))} {_text(row.get('detailed_use'))}"
        residential = any(word in use for word in RESIDENTIAL_WORDS)
        multifamily = "다가구" in use
        observed_units = pd.to_numeric(
            pd.Series([row.get("registered_units_observed")]), errors="coerce"
        ).iloc[0]
        residential_area = pd.to_numeric(
            pd.Series([row.get("residential_floor_area")]), errors="coerce"
        ).iloc[0]
        total_area = pd.to_numeric(
            pd.Series([row.get("total_floor_area")]), errors="coerce"
        ).iloc[0]
        return (
            (100.0 if multifamily else 0.0)
            + (50.0 if residential else 0.0)
            + (30.0 if pd.notna(observed_units) and observed_units >= 1 else 0.0)
            + (20.0 if pd.notna(residential_area) and residential_area > 0 else 0.0),
            float(residential_area) if pd.notna(residential_area) else -1.0,
            float(total_area) if pd.notna(total_area) else -1.0,
        )

    def match_property(self, prop: dict) -> dict:
        address = normalize_korean_address(prop.get("road_address"))
        if not address:
            return {
                "matched": False,
                "method": "exact_normalized_road_address",
                "reason": "도로명주소가 없어 건축물대장과 안전하게 연결할 수 없습니다.",
            }
        registry = self._load_registry()
        candidates = registry[
            registry["_normalized_road_address"].astype(str).eq(address)
        ].copy()
        if candidates.empty:
            return {
                "matched": False,
                "method": "exact_normalized_road_address",
                "normalized_address": address,
                "reason": (
                    "같은 도로명주소의 건축물대장 원문을 찾지 못했습니다. "
                    "오매칭을 막기 위해 유사주소 추론은 하지 않습니다."
                ),
            }
        candidates["_selection_score"] = candidates.apply(
            lambda row: self._building_score(row), axis=1
        )
        candidates = candidates.sort_values(
            "_selection_score", ascending=False, kind="stable"
        )
        selected = candidates.iloc[0].drop(
            labels=["_normalized_road_address", "_selection_score"],
            errors="ignore",
        )
        use = f"{_text(selected.get('main_use_code'))} {_text(selected.get('detailed_use'))}"
        if not any(word in use for word in RESIDENTIAL_WORDS):
            return {
                "matched": False,
                "method": "exact_normalized_road_address",
                "normalized_address": address,
                "candidate_count": int(len(candidates)),
                "reason": "동일 주소는 찾았지만 주거용 건축물대장 행을 확인하지 못했습니다.",
            }
        return {
            "matched": True,
            "method": "exact_normalized_road_address",
            "confidence": "exact",
            "normalized_address": address,
            "candidate_count": int(len(candidates)),
            "building_id": _text(selected.get("building_id")),
            "building_use": _text(selected.get("detailed_use"))
            or _text(selected.get("main_use_code")),
            "row": selected.to_dict(),
        }

    @staticmethod
    def _seed(prop: dict, reference_date: str) -> int:
        material = (
            f"{prop.get('property_id') or ''}|"
            f"{prop.get('road_address') or ''}|{reference_date}"
        )
        return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:8], 16)

    def analyze_property(
        self,
        prop: dict,
        *,
        reference_date: str | None = None,
        samples: int = 5_000,
        seed: int | None = None,
        mode: str = "conservative",
        occupancy_scenario: str = "baseline",
        senior_probability: float | None = None,
        random_effect_sigma: float | None = None,
        target_rooms_excluded: int = 1,
    ) -> dict:
        transaction_type = _text(
            prop.get("transaction_type") or prop.get("lease_type")
        )
        if transaction_type not in LEASE_TYPES:
            return {
                "available": False,
                "status": "not_applicable",
                "applicability": "전세·월세 임대차 매물만 분석합니다.",
            }
        house_type = _text(
            prop.get("house_type") or prop.get("property_type")
        )
        if "다가구" not in house_type:
            return {
                "available": False,
                "status": "not_applicable",
                "applicability": "다가구주택 임대차 매물에만 적용합니다.",
            }
        if not self.available:
            return {
                "available": False,
                "status": "model_or_registry_unavailable",
                "applicability": (
                    "실제 건축물대장과 RTMS로 학습한 모델 파일을 확인할 수 없습니다."
                ),
            }
        matched = self.match_property(prop)
        if not matched.get("matched"):
            # A prototype listing may have a real road address that is absent
            # from the downloaded Building HUB title snapshot.  In that case
            # do not fuzzy-match another building.  Run the statistical model
            # on the listing's own disclosed attributes and mark the evidence
            # quality as low.
            matched = {
                "matched": True,
                "method": "listing_features_without_registry_match",
                "confidence": "low",
                "normalized_address": normalize_korean_address(
                    prop.get("road_address")
                ),
                "candidate_count": 0,
                "building_id": _text(prop.get("property_id")),
                "building_use": house_type,
                "registry_exact_match": False,
                "registry_match_reason": matched.get("reason"),
                "row": {
                    **prop,
                    "building_id": _text(prop.get("property_id")),
                    "main_use_code": "다가구주택",
                    "detailed_use": "다가구주택",
                    "registered_units_observed": prop.get(
                        "building_total_units"
                    ),
                    "total_floor_area": prop.get(
                        "building_total_area_m2"
                    ),
                    "residential_floor_area": prop.get("area_m2"),
                    "ground_floors": prop.get("total_floors"),
                },
            }
        public_match = {key: value for key, value in matched.items() if key != "row"}

        reference_date = reference_date or date.today().isoformat()
        date.fromisoformat(reference_date)
        seed = int(seed if seed is not None else self._seed(prop, reference_date))
        cache_key = "|".join(
            [
                str(matched["building_id"]),
                reference_date,
                str(int(samples)),
                str(seed),
                mode,
                occupancy_scenario,
                str(senior_probability),
                str(random_effect_sigma),
                str(target_rooms_excluded),
            ]
        )
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached and now - cached[0] < self.cache_ttl_seconds:
                result = dict(cached[1])
                result["cache_hit"] = True
                return result

        building = dict(matched["row"])
        building.update(
            {
                "monthly_rent": prop.get("monthly_rent_manwon") or 0,
                "contract_type": transaction_type,
                "housing_type": prop.get("house_type") or "다가구",
                "observed_deposit": prop.get("deposit_manwon"),
            }
        )
        estimate = self._load_pipeline().infer(
            building,
            reference_date=reference_date,
            samples=samples,
            seed=seed,
            mode=mode,
            occupancy_scenario=occupancy_scenario,
            senior_probability=senior_probability,
            random_effect_sigma=random_effect_sigma,
            target_rooms_excluded=target_rooms_excluded,
        )
        target_deposit_won = int(
            round(float(prop.get("deposit_manwon") or 0) * 10_000)
        )
        conservative_p95_won = int(
            (estimate.get("conservative_upper_deposit") or {}).get("p95") or 0
        )
        output = {
            "available": True,
            "status": (
                "estimated" if matched.get("registry_exact_match", True)
                else "estimated_from_listing_features"
            ),
            "applicability": (
                "선택 호실을 제외한 기존 임차인의 보증금 총합을 "
                "통계적으로 추정했습니다."
            ),
            "cache_hit": False,
            "match": public_match,
            "estimate": estimate,
            "decision_support": {
                "target_deposit_won": target_deposit_won,
                "existing_deposit_conservative_p95_won": conservative_p95_won,
                "official_verification_required": True,
                "risk_score_changed": False,
            },
        }
        with self._lock:
            self._cache[cache_key] = (now, output)
            if len(self._cache) > 128:
                oldest = min(self._cache, key=lambda key: self._cache[key][0])
                self._cache.pop(oldest, None)
        return output
