from __future__ import annotations

from collections import Counter
from dataclasses import asdict

import numpy as np
import pandas as pd

from .schemas import BuildingEstimateInput


MANDATORY_WARNING = (
    "이 값은 특정 집주인의 실제 금융·부동산 자산을 조회한 결과가 아니다. "
    "대상 건물의 추정 보증금·가치와 임대인 가구의 통계적 자산분포를 "
    "결합한 확률적 추정치다."
)


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    q = np.quantile(
        values[np.isfinite(values)], [0.05, 0.1, 0.5, 0.9, 0.95]
    )
    return {
        "p05": round(float(q[0]), 2),
        "p10": round(float(q[1]), 2),
        "p50": round(float(q[2]), 2),
        "p90": round(float(q[3]), 2),
        "p95": round(float(q[4]), 2),
    }


def _model_frame(building: BuildingEstimateInput) -> pd.DataFrame:
    row = asdict(building)
    row.update(building.extra)
    row.setdefault("legal_dong", building.legal_dong)
    row.setdefault("legal_dong_code", building.legal_dong_code)
    row.setdefault("main_use_code", building.main_use_code)
    row.setdefault("structure_code", building.structure_code)
    row.setdefault("land_area", building.land_area)
    row.setdefault("total_floor_area", building.total_floor_area)
    row.setdefault("residential_floor_area", building.residential_floor_area)
    row.setdefault("parking_count", building.parking_count)
    row.setdefault("ground_floors", building.ground_floors)
    row.setdefault("building_age", building.building_age)
    row.setdefault("rental_area", (
        building.residential_floor_area or building.total_floor_area or 50.0))
    row.setdefault("monthly_rent", building.monthly_rent)
    row.setdefault("contract_type", building.contract_type)
    row.setdefault("housing_type", building.housing_type)
    row.setdefault("official_house_price", building.official_house_price)
    row.setdefault("official_land_price", building.official_land_price)
    return pd.DataFrame([row])


def _quality(building: BuildingEstimateInput, *,
             registered_observed: bool, prior_fallback: str,
             local_market_count: int) -> str:
    sufficient_building = sum(
        value is not None for value in (
            building.land_area, building.total_floor_area,
            building.residential_floor_area, building.building_age,
            building.ground_floors, building.parking_count,
        )) >= 4
    if (registered_observed and sufficient_building
            and local_market_count >= 30
            and prior_fallback in {"exact_group", "relaxed_asset_band"}):
        return "high"
    if sufficient_building and local_market_count >= 10:
        return "medium"
    return "low"


def _ablation_contributions(
    n_rented: np.ndarray,
    mean_unit_deposit: np.ndarray,
    random_effect: np.ndarray,
    value: np.ndarray,
    k_other: np.ndarray,
) -> dict[str, float]:
    eps = 1e-6

    def ratio(n=n_rented, d=mean_unit_deposit, u=random_effect,
              v=value, k=k_other):
        total = n * d * np.exp(u)
        return total / np.maximum(v * (1.0 + k), eps)

    baseline_var = float(np.var(ratio()))
    if baseline_var <= eps:
        return {name: 0.0 for name in (
            "room_count", "occupancy", "deposit_distribution",
            "property_value", "owner_other_assets")}
    # Registered-room and occupancy uncertainty are jointly represented in
    # n_rented at this stage. Split their reduction equally unless the caller
    # has direct occupancy labels, which this first implementation does not.
    rented_reduction = max(
        0.0, baseline_var - float(np.var(ratio(
            n=np.full_like(n_rented, np.median(n_rented))))))
    raw = {
        "room_count": rented_reduction / 2.0,
        "occupancy": rented_reduction / 2.0,
        "deposit_distribution": max(
            0.0, baseline_var - float(np.var(ratio(
                d=np.full_like(mean_unit_deposit,
                               np.median(mean_unit_deposit)),
                u=np.zeros_like(random_effect))))),
        "property_value": max(
            0.0, baseline_var - float(np.var(ratio(
                v=np.full_like(value, np.median(value)))))),
        "owner_other_assets": max(
            0.0, baseline_var - float(np.var(ratio(
                k=np.full_like(k_other, np.median(k_other)))))),
    }
    total = sum(raw.values())
    if total <= eps:
        return {name: 0.0 for name in raw}
    return {name: round(value / total, 4) for name, value in raw.items()}


def infer_ratio_distribution(
    pipeline,
    building: BuildingEstimateInput | dict,
    *,
    samples: int = 20_000,
    seed: int = 20260728,
    occupancy_scenario: str = "baseline",
    random_effect_sigma: float | None = None,
) -> dict:
    """Combine four independent models in a reproducible Monte Carlo run."""
    if isinstance(building, dict):
        building = BuildingEstimateInput.from_mapping(building)
    samples = int(np.clip(samples, 1_000, 100_000))
    rng = np.random.default_rng(int(seed))
    frame = _model_frame(building)
    metadata = pipeline.metadata
    occupancy = metadata["occupancy"][occupancy_scenario]
    sigma = float(
        metadata.get("building_random_effect_sigma", 0.12)
        if random_effect_sigma is None else random_effect_sigma)

    observed_units = building.observed_registered_units()
    if observed_units is not None:
        n_registered = np.full(samples, observed_units, dtype=int)
        units_source = "observed_registry_priority"
    else:
        repeated = pd.concat([frame] * samples, ignore_index=True)
        n_registered = pipeline.unit_model.sample(repeated, rng)
        units_source = f"estimated:{pipeline.unit_model.selected_name}"
    n_registered = np.clip(n_registered, 1, 100)

    occupancy_rate = rng.beta(
        float(occupancy["alpha"]), float(occupancy["beta"]), samples)
    n_rented = rng.binomial(n_registered, occupancy_rate)

    residential_area = (
        building.residential_floor_area or building.total_floor_area or 50.0)
    unit_area = residential_area / np.maximum(n_registered, 1)
    deposit_frame = pd.concat([frame] * samples, ignore_index=True)
    deposit_frame["rental_area"] = unit_area
    mean_unit_deposit = pipeline.deposit_model.sample(
        deposit_frame, rng, size=samples)
    random_effect = rng.normal(0.0, sigma, samples)
    total_deposit = (
        n_rented * mean_unit_deposit * np.exp(random_effect)).clip(min=0)

    value_frame = pd.concat([frame] * samples, ignore_index=True)
    property_value = pipeline.value_model.sample(
        value_frame, rng, size=samples).clip(min=1e-6)

    k_other, financial_debt_ratio, fallbacks = pipeline.owner_prior.sample(
        property_value, rng)
    owner_total_assets = property_value * (1.0 + k_other)
    financial_debt = financial_debt_ratio * owner_total_assets
    owner_net_assets = owner_total_assets - financial_debt
    ratio = total_deposit / np.maximum(owner_total_assets, 1e-6)
    # The product's primary landlord-capacity indicator uses only the
    # selected tenant's deposit in the numerator.  Existing/senior tenant
    # deposits remain a separate estimate and are never silently merged into
    # this ratio.
    target_deposit = max(0.0, float(building.observed_deposit or 0.0))
    target_ratio = (
        np.full(samples, target_deposit, dtype=float)
        / np.maximum(owner_total_assets, 1e-6)
    )
    building_ratio = total_deposit / np.maximum(property_value, 1e-6)
    net_ratio = total_deposit / np.maximum(owner_net_assets, 1e-6)

    fallback_counts = Counter(fallbacks)
    dominant_fallback = fallback_counts.most_common(1)[0][0]
    local_count = int(building.extra.get("local_market_count") or 0)
    data_quality = _quality(
        building, registered_observed=observed_units is not None,
        prior_fallback=dominant_fallback, local_market_count=local_count)

    k_high = np.quantile(k_other, 0.9)
    scenarios = {
        "building_only_k0": _summary(building_ratio),
        "conditional_owner_prior": _summary(ratio),
        "high_asset_buffer": _summary(
            total_deposit / np.maximum(
                property_value * (1.0 + k_high), 1e-6)),
        "occupancy_scenario": occupancy_scenario,
        "building_random_effect_sigma": sigma,
    }
    warnings = [MANDATORY_WARNING]
    if metadata.get("data_kind") != "actual":
        warnings.append(
            "현재 artifact는 합성 smoke 검증용이다. 실제 수원시 추정치로 "
            "사용하거나 성능을 실제 성능으로 보고하면 안 된다.")
    if observed_units is None:
        warnings.append("공부상 호실 수가 없어 확률모델로 추정했다.")
    if dominant_fallback not in {"exact_group", "relaxed_asset_band"}:
        warnings.append(f"소유자 기타자산 prior가 {dominant_fallback}로 fallback했다.")
    if local_count < 10:
        warnings.append("수원시 법정동 최근 임대차·매매 표본이 부족하다.")

    return {
        "model_name": "four_component_owner_asset_ratio_v1",
        "model_version": metadata.get("model_version"),
        "data_kind": metadata.get("data_kind"),
        "building_id": building.building_id,
        "samples": samples,
        "seed": int(seed),
        "estimated_total_deposit": _summary(total_deposit),
        # Public integration contract. Values are in 만원 and represent the
        # sampled market value of the selected building, not owner net worth.
        "estimated_property_value": _summary(property_value),
        "estimated_owner_total_assets": _summary(owner_total_assets),
        "deposit_to_total_assets_ratio": _summary(ratio),
        "target_deposit": round(target_deposit, 2),
        "target_deposit_to_total_assets_ratio": _summary(target_ratio),
        "probability_target_ratio_over_0_6": round(
            float(np.mean(target_ratio > 0.6)), 6),
        "probability_target_ratio_over_0_8": round(
            float(np.mean(target_ratio > 0.8)), 6),
        "probability_target_ratio_over_1_0": round(
            float(np.mean(target_ratio > 1.0)), 6),
        "probability_ratio_over_0_6": round(float(np.mean(ratio > 0.6)), 6),
        "probability_ratio_over_0_8": round(float(np.mean(ratio > 0.8)), 6),
        "probability_ratio_over_1_0": round(float(np.mean(ratio > 1.0)), 6),
        "building_only_ratio": _summary(building_ratio),
        "net_asset_ratio_auxiliary": _summary(net_ratio),
        "data_quality": data_quality,
        "assumptions": [
            f"registered units source={units_source}",
            (
                "occupancy Beta("
                f"{occupancy['alpha']},{occupancy['beta']})"
            ),
            f"shared-building log random effect sigma={sigma}",
            (
                "K is a conditional landlord-population prior, not the "
                "observed owner's other assets"
            ),
            "the target building approximates the owner's rental real estate",
            (
                "primary product risk ratio = selected jeonse deposit / "
                "estimated owner total assets"
            ),
            (
                "estimated existing tenant deposits are reported separately "
                "and are not added to the primary ratio numerator"
            ),
        ],
        "warnings": warnings,
        "sensitivity_scenarios": scenarios,
        "owner_prior_fallback_counts": dict(fallback_counts),
        "uncertainty_contribution": _ablation_contributions(
            n_rented.astype(float), mean_unit_deposit, random_effect,
            property_value, k_other),
        "provenance": metadata.get("provenance", {}),
    }
