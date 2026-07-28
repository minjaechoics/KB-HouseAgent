"""Safe report integration for the landlord total-assets ratio model."""
from __future__ import annotations

from datetime import date
import hashlib
from pathlib import Path
import threading
import time
from typing import Any

from src import config
from .pipeline import OwnerAssetRatioPipeline


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


class OwnerAssetRatioIntegrationService:
    """Estimate a multifamily landlord-capacity ratio for a selected jeonse.

    The service deliberately reuses the senior-deposit service's exact
    normalized-road-address matcher.  It never fuzzy-matches a prototype
    listing to another owner's building and never presents the statistical
    household prior as the observed landlord's wealth.
    """

    def __init__(
        self,
        artifact_path: Path | None = None,
        registry_path: Path | None = None,
        *,
        cache_ttl_seconds: int = 3600,
    ):
        self.artifact_path = artifact_path or (
            config.ROOT / "models" / "owner_asset_ratio"
            / "owner_asset_ratio_actual.joblib"
        )
        self.registry_path = registry_path or (
            config.ROOT / "data" / "processed"
            / "owner_asset_ratio" / "buildings.csv"
        )
        # Imported lazily to avoid a package import cycle: the senior-deposit
        # pipeline itself reuses owner-asset model primitives.
        from src.senior_deposit.service import SeniorDepositIntegrationService
        self.matcher = SeniorDepositIntegrationService(
            registry_path=self.registry_path
        )
        self.cache_ttl_seconds = int(cache_ttl_seconds)
        self._pipeline: OwnerAssetRatioPipeline | None = None
        self._cache: dict[str, tuple[float, dict]] = {}
        self._lock = threading.RLock()

    @property
    def available(self) -> bool:
        return self.artifact_path.exists() and self.registry_path.exists()

    def _load_pipeline(self) -> OwnerAssetRatioPipeline:
        with self._lock:
            if self._pipeline is None:
                self._pipeline = OwnerAssetRatioPipeline.load(
                    self.artifact_path, allow_synthetic=False
                )
            return self._pipeline

    @staticmethod
    def _applicable(prop: dict) -> bool:
        transaction = _text(
            prop.get("transaction_type") or prop.get("lease_type")
        )
        house_type = _text(
            prop.get("house_type") or prop.get("property_type")
        )
        return transaction == "전세" and "다가구" in house_type

    @staticmethod
    def _seed(prop: dict, reference_date: str) -> int:
        material = (
            f"{prop.get('property_id') or ''}|"
            f"{prop.get('road_address') or ''}|{reference_date}|owner-assets"
        )
        return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:8], 16)

    def analyze_property(
        self,
        prop: dict,
        *,
        reference_date: str | None = None,
        samples: int = 20_000,
        seed: int | None = None,
        occupancy_scenario: str = "baseline",
    ) -> dict:
        if not self._applicable(prop):
            return {
                "available": False,
                "status": "not_applicable",
                "applicability": "다가구주택 전세 매물에만 적용합니다.",
            }
        if not self.available:
            return {
                "available": False,
                "status": "model_or_registry_unavailable",
                "applicability": (
                    "실제 건축물대장·RTMS·가계금융복지조사로 만든 "
                    "집주인 총자산 추정 모델을 확인할 수 없습니다."
                ),
            }

        matched = self.matcher.match_property(prop)
        if not matched.get("matched"):
            matched = {
                "matched": True,
                "method": "listing_features_without_registry_match",
                "confidence": "low",
                "normalized_address": _text(prop.get("road_address")),
                "candidate_count": 0,
                "building_id": _text(prop.get("property_id")),
                "building_use": _text(prop.get("house_type")),
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
        public_match = {
            key: value for key, value in matched.items() if key != "row"
        }

        building_use = _text(matched.get("building_use"))
        if "다가구" not in building_use:
            return {
                "available": False,
                "status": "matched_building_not_multifamily",
                "applicability": (
                    "정확 주소는 확인했지만 건축물대장의 주용도가 "
                    "다가구주택이 아닙니다."
                ),
                "match": public_match,
            }

        reference_date = reference_date or date.today().isoformat()
        date.fromisoformat(reference_date)
        seed = int(seed if seed is not None else self._seed(prop, reference_date))
        samples = max(1_000, min(100_000, int(samples)))
        cache_key = "|".join([
            str(matched.get("building_id") or ""),
            reference_date,
            str(samples),
            str(seed),
            occupancy_scenario,
            str(prop.get("deposit_manwon") or 0),
        ])
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached and now - cached[0] < self.cache_ttl_seconds:
                output = dict(cached[1])
                output["cache_hit"] = True
                return output

        building = dict(matched["row"])
        building.update({
            "monthly_rent": 0,
            "contract_type": "전세",
            "housing_type": "다가구",
            "observed_deposit": float(prop.get("deposit_manwon") or 0),
        })
        estimate = self._load_pipeline().infer(
            building,
            samples=samples,
            seed=seed,
            occupancy_scenario=occupancy_scenario,
        )
        ratio = estimate.get("target_deposit_to_total_assets_ratio") or {}
        ratio_p50 = float(ratio.get("p50") or 0)
        grade = (
            "위험" if ratio_p50 >= 0.8
            else "주의" if ratio_p50 >= 0.6
            else "낮음"
        )
        output = {
            "available": True,
            "reference_date": reference_date,
            "status": (
                "estimated" if matched.get("registry_exact_match", True)
                else "estimated_from_listing_features"
            ),
            "applicability": (
                "선택 전세보증금을 통계적으로 추정한 집주인 총자산 "
                "분포로 나눈 비율입니다."
            ),
            "cache_hit": False,
            "match": public_match,
            "estimate": estimate,
            "decision_support": {
                "formula": "선택 전세보증금 / 추정 집주인 총자산",
                "ratio_p50": round(ratio_p50, 4),
                "risk_score": round(max(0.0, min(1.0, ratio_p50)), 6),
                "risk_grade": grade,
                "warning_threshold": 0.6,
                "danger_threshold": 0.8,
                "probability_ratio_over_0_8": float(
                    estimate.get("probability_target_ratio_over_0_8") or 0
                ),
                "owner_assets_are_observed": False,
                "senior_deposit_added_to_ratio": False,
                "official_verification_required": True,
            },
        }
        with self._lock:
            self._cache[cache_key] = (now, output)
            if len(self._cache) > 128:
                oldest = min(
                    self._cache, key=lambda key: self._cache[key][0]
                )
                self._cache.pop(oldest, None)
        return output
