from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from src.owner_asset_ratio.schemas import BuildingEstimateInput
from .schemas import SeniorDepositInput


MANDATORY_WARNING = (
    "이 결과는 전입세대확인서, 확정일자 정보 또는 개별 임대차계약을 직접 "
    "확인한 법적 확정값이 아니다. 건축물대장, 지역 임대차 실거래 분포, "
    "점유율 및 선순위 확률을 결합한 통계적 추정치다. 실제 계약 전에는 "
    "기존 임차보증금 현황과 공식 확인자료를 별도로 요청해야 한다."
)


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
    row.setdefault("monthly_rent", building.monthly_rent)
    row.setdefault("contract_type", building.contract_type)
    row.setdefault("housing_type", building.housing_type)
    return pd.DataFrame([row])


def _summary_won(values_manwon: np.ndarray, levels: tuple[float, ...]) -> dict:
    values = np.asarray(values_manwon, dtype=float)
    finite = values[np.isfinite(values)]
    quantiles = np.quantile(finite, levels)
    return {
        f"p{int(round(level * 100))}": int(round(float(value) * 10_000))
        for level, value in zip(levels, quantiles)
    }


def _summary_count(values: np.ndarray) -> dict[str, int]:
    q = np.quantile(np.asarray(values, dtype=float), [.1, .5, .9])
    return {
        "p10": int(round(float(q[0]))),
        "p50": int(round(float(q[1]))),
        "p90": int(round(float(q[2]))),
    }


def _sample_piecewise_matrix(
    quantile_values: np.ndarray,
    quantiles: np.ndarray,
    draws: np.ndarray,
) -> np.ndarray:
    """Vectorized piecewise-linear inverse CDF with flat tails."""
    quantiles = np.asarray(quantiles, dtype=float)
    row_count, unit_count = draws.shape
    if unit_count == 0:
        return np.zeros((row_count, 0), dtype=float)
    index = np.searchsorted(quantiles, draws, side="right") - 1
    index = np.clip(index, 0, len(quantiles) - 2)
    rows = np.arange(row_count)[:, None]
    lower_value = quantile_values[rows, index]
    upper_value = quantile_values[rows, index + 1]
    lower_q = quantiles[index]
    upper_q = quantiles[index + 1]
    weight = (draws - lower_q) / np.maximum(upper_q - lower_q, 1e-12)
    sampled = lower_value + np.clip(weight, 0, 1) * (
        upper_value - lower_value)
    sampled = np.where(
        draws <= quantiles[0], quantile_values[:, :1], sampled)
    sampled = np.where(
        draws >= quantiles[-1], quantile_values[:, -1:], sampled)
    return np.clip(sampled, 0, None)


def _attach_past_market_features(
    pipeline,
    deposit_frame: pd.DataFrame,
    *,
    legal_dong: str,
    reference_date,
) -> tuple[pd.DataFrame, str]:
    """Attach only market aggregates strictly before the reference month."""
    history = getattr(pipeline, "market_reference", None)
    if history is None or len(history) == 0 or not legal_dong:
        return deposit_frame, "training_median_fallback"
    reference_month = reference_date.year * 100 + reference_date.month
    month = pd.to_numeric(
        history["contract_year_month"], errors="coerce")
    candidates = history[
        history["legal_dong"].astype(str).eq(str(legal_dong))
        & month.lt(reference_month)
    ].copy()
    if candidates.empty:
        return deposit_frame, "training_median_fallback"
    latest_month = int(pd.to_numeric(
        candidates["contract_year_month"], errors="coerce").max())
    candidates = candidates[
        pd.to_numeric(
            candidates["contract_year_month"], errors="coerce"
        ).eq(latest_month)
    ].copy()
    candidates["_area"] = pd.to_numeric(
        candidates["rental_area"], errors="coerce")
    feature_columns = (
        "legal_dong_3m_deposit_median",
        "legal_dong_12m_deposit_median",
        "legal_dong_12m_deposit_growth",
        "transaction_count_3m",
        "transaction_count_12m",
    )
    result = deposit_frame.copy()
    areas = pd.to_numeric(result["rental_area"], errors="coerce")
    for area in areas.dropna().unique():
        distance = (candidates["_area"] - float(area)).abs()
        selected = candidates.loc[distance.idxmin()]
        mask = areas.eq(area)
        for column in feature_columns:
            if column in selected:
                result.loc[mask, column] = selected[column]
    return result, f"legal_dong_asof:{latest_month}"


def _quality(
    building: BuildingEstimateInput,
    *,
    observed_units: bool,
    local_market_count: int,
    has_occupancy_labels: bool,
    has_seniority_labels: bool,
) -> str:
    sufficient = sum(value is not None for value in (
        building.land_area,
        building.total_floor_area,
        building.residential_floor_area,
        building.building_age,
        building.ground_floors,
        building.parking_count,
    )) >= 4
    if (observed_units and sufficient and local_market_count >= 30
            and has_occupancy_labels and has_seniority_labels):
        return "high"
    if observed_units and sufficient and local_market_count >= 10:
        return "medium"
    return "low"


def _uncertainty_contribution(
    occupied: np.ndarray,
    deposit_matrix: np.ndarray,
    occupied_mask: np.ndarray,
    senior_mask: np.ndarray,
    random_effect: np.ndarray,
    senior_probability: float,
) -> dict[str, float]:
    eps = 1e-9
    adjusted = deposit_matrix * np.exp(random_effect[:, None])
    baseline = np.sum(adjusted * occupied_mask * senior_mask, axis=1)
    baseline_var = float(np.var(baseline))
    names = (
        "unit_count", "occupancy", "unit_deposit",
        "within_building_correlation", "seniority",
    )
    if baseline_var <= eps:
        return {name: 0.0 for name in names}
    fixed_occupied = np.full_like(occupied, int(round(np.median(occupied))))
    columns = np.arange(deposit_matrix.shape[1])[None, :]
    fixed_occupied_mask = columns < fixed_occupied[:, None]
    occupied_reduction = max(
        0.0,
        baseline_var - float(np.var(np.sum(
            adjusted * fixed_occupied_mask * senior_mask, axis=1))),
    )
    fixed_deposit = np.full_like(
        deposit_matrix, float(np.median(deposit_matrix)))
    raw = {
        "unit_count": occupied_reduction / 2.0,
        "occupancy": occupied_reduction / 2.0,
        "unit_deposit": max(
            0.0,
            baseline_var - float(np.var(np.sum(
                fixed_deposit * np.exp(random_effect[:, None])
                * occupied_mask * senior_mask,
                axis=1,
            ))),
        ),
        "within_building_correlation": max(
            0.0,
            baseline_var - float(np.var(np.sum(
                deposit_matrix * occupied_mask * senior_mask, axis=1))),
        ),
        "seniority": max(
            0.0,
            baseline_var - float(np.var(
                np.sum(adjusted * occupied_mask, axis=1)
                * senior_probability)),
        ),
    }
    total = sum(raw.values())
    if total <= eps:
        return {name: 0.0 for name in names}
    return {name: round(value / total, 4) for name, value in raw.items()}


def infer_senior_deposit_distribution(
    pipeline,
    building: SeniorDepositInput | BuildingEstimateInput | dict,
    *,
    reference_date: str | None = None,
    samples: int = 20_000,
    seed: int = 20260728,
    mode: str = "conservative",
    occupancy_scenario: str = "baseline",
    senior_probability: float | None = None,
    random_effect_sigma: float | None = None,
    target_rooms_excluded: int = 1,
) -> dict:
    if isinstance(building, SeniorDepositInput):
        request = building
    elif isinstance(building, BuildingEstimateInput):
        request = SeniorDepositInput.from_mapping(
            asdict(building),
            reference_date=reference_date or str(pd.Timestamp.today().date()),
            target_rooms_excluded=target_rooms_excluded,
        )
    else:
        request = SeniorDepositInput.from_mapping(
            building,
            reference_date=reference_date or str(pd.Timestamp.today().date()),
            target_rooms_excluded=target_rooms_excluded,
        )
    if mode not in {"conservative", "probabilistic", "scenario"}:
        raise ValueError(
            "mode must be conservative, probabilistic, or scenario")
    samples = int(np.clip(samples, 1_000, 100_000))
    rng = np.random.default_rng(int(seed))
    metadata = pipeline.metadata
    building_input = request.building
    frame = _model_frame(building_input)

    observed = building_input.observed_registered_units()
    if observed is not None:
        registered = np.full(samples, observed, dtype=int)
        units_source = "observed_registry_priority"
    else:
        repeated = pd.concat([frame] * samples, ignore_index=True)
        registered = pipeline.unit_model.sample(repeated, rng)
        units_source = f"estimated:{pipeline.unit_model.selected_name}"
    registered = np.clip(registered, 1, 100)
    available_other = np.maximum(
        registered - request.target_rooms_excluded, 0)

    occupancy = metadata["occupancy_priors"][occupancy_scenario]
    occupancy_rate = rng.beta(
        float(occupancy["alpha"]), float(occupancy["beta"]), samples)
    occupied_other = rng.binomial(available_other, occupancy_rate)
    max_other = int(np.max(available_other))

    residential_area = (
        building_input.residential_floor_area
        or building_input.total_floor_area
        or 50.0
    )
    unit_area = residential_area / np.maximum(registered, 1)
    deposit_frame = pd.concat([frame] * samples, ignore_index=True)
    deposit_frame["rental_area"] = unit_area
    deposit_frame, market_feature_source = _attach_past_market_features(
        pipeline,
        deposit_frame,
        legal_dong=building_input.legal_dong,
        reference_date=request.reference_date,
    )
    quantile_values = pipeline.deposit_model.predict_quantiles(deposit_frame)
    uniform = rng.uniform(0.0, 1.0, (samples, max_other))
    deposit_matrix = _sample_piecewise_matrix(
        quantile_values,
        np.asarray(pipeline.deposit_model.quantiles, dtype=float),
        uniform,
    )
    columns = np.arange(max_other)[None, :]
    occupied_mask = columns < occupied_other[:, None]

    sigma = float(
        metadata.get("within_building_sigma", .12)
        if random_effect_sigma is None else random_effect_sigma)
    standard_effect = rng.normal(0.0, 1.0, samples)
    random_effect = standard_effect * sigma
    adjusted_deposit = deposit_matrix * np.exp(random_effect[:, None])
    upper = np.sum(adjusted_deposit * occupied_mask, axis=1)

    warnings = [MANDATORY_WARNING]
    has_seniority_model = bool(metadata.get("seniority_model_trained"))
    if mode == "conservative":
        probability = 1.0
    elif mode == "probabilistic" and has_seniority_model:
        probability = float(
            pipeline.predict_seniority_probability(
                deposit_frame, request.reference_date))
    else:
        if senior_probability is None:
            probability = float(
                metadata.get("baseline_senior_probability", .9))
        else:
            probability = float(senior_probability)
        if mode == "probabilistic" and not has_seniority_model:
            warnings.append(
                "실제 선순위 라벨이 없어 probabilistic 모드를 명시적 "
                "scenario prior로 fallback했다.")
    if not 0 <= probability <= 1:
        raise ValueError("senior_probability must be in [0, 1]")
    senior_mask = rng.uniform(0.0, 1.0, (samples, max_other)) < probability
    baseline = np.sum(
        adjusted_deposit * occupied_mask * senior_mask, axis=1)

    thresholds = (300_000_000, 500_000_000, 700_000_000, 1_000_000_000)
    threshold_probability = {
        str(value): round(float(np.mean(baseline * 10_000 > value)), 6)
        for value in thresholds
    }
    local_count = int(building_input.extra.get("local_market_count") or 0)
    data_quality = _quality(
        building_input,
        observed_units=observed is not None,
        local_market_count=local_count,
        has_occupancy_labels=bool(metadata.get("occupancy_model_trained")),
        has_seniority_labels=has_seniority_model,
    )
    if observed is None:
        warnings.append("공부상 호실 수가 없어 Model A로 추정했다.")
    if not metadata.get("occupancy_model_trained"):
        warnings.append(
            "현재 점유 호실 실제 라벨이 없어 Beta-Binomial 시나리오 prior를 사용했다.")
    if not has_seniority_model:
        warnings.append(
            "전입일·확정일자 기반 실제 선순위 라벨이 없어 선순위 분류기를 학습하지 않았다.")
    if local_count < 10:
        warnings.append("해당 법정동의 최근 유사 임대차 표본이 부족하다.")
    if market_feature_source == "training_median_fallback":
        warnings.append(
            "기준일 이전 동일 법정동 시계열을 찾지 못해 학습 중앙값으로 fallback했다.")

    correlations = {}
    for label, scenario_sigma in (
        ("independent", 0.0), ("moderate", .12), ("high", .25)):
        scenario_total = np.sum(
            deposit_matrix
            * np.exp((standard_effect * scenario_sigma)[:, None])
            * occupied_mask,
            axis=1,
        )
        correlations[label] = {
            "sigma": scenario_sigma,
            **_summary_won(scenario_total, (.5, .9, .95)),
        }

    return {
        "model_name": "senior_deposit_mvp_v1",
        "model_version": metadata.get("model_version"),
        "reference_date": request.reference_date.isoformat(),
        "building_id": building_input.building_id,
        "samples": samples,
        "seed": int(seed),
        "market_feature_source": market_feature_source,
        "registered_units": {
            "observed": observed,
            **_summary_count(registered),
        },
        "occupied_other_units": _summary_count(occupied_other),
        "estimated_total_deposit": _summary_won(
            upper, (.05, .1, .5, .9, .95)),
        "estimated_senior_deposit": _summary_won(
            baseline, (.05, .1, .5, .9, .95)),
        "conservative_upper_deposit": _summary_won(
            upper, (.05, .1, .5, .9, .95)),
        "probability_senior_deposit_over": threshold_probability,
        "data_quality": data_quality,
        "model_mode": metadata.get("model_mode", "scenario_only"),
        "scenario": {
            "mode": mode,
            "senior_probability": probability,
            "occupancy": occupancy_scenario,
            "target_rooms_excluded": request.target_rooms_excluded,
            "within_building_sigma": sigma,
        },
        "within_building_correlation_scenarios": correlations,
        "uncertainty_contribution": _uncertainty_contribution(
            occupied_other,
            deposit_matrix,
            occupied_mask,
            senior_mask,
            random_effect,
            probability,
        ),
        "assumptions": [
            f"registered units source={units_source}",
            (
                "occupied other units use Beta-Binomial mixture Beta("
                f"{occupancy['alpha']},{occupancy['beta']})"
            ),
            "the prospective target room is excluded from existing tenants",
            "RTMS partial lots are used for conditional distributions, not exact joins",
            f"market features source={market_feature_source}",
            f"seniority probability={probability:.4f}",
        ],
        "warnings": warnings,
        "provenance": metadata.get("provenance", {}),
    }
