from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.owner_asset_ratio.data import add_past_only_lease_features
from src.owner_asset_ratio.models import weighted_quantile
from src.owner_asset_ratio.schemas import BuildingEstimateInput
from src.owner_asset_ratio.simulation import infer_ratio_distribution
from src.owner_asset_ratio.pipeline import OwnerAssetRatioPipeline
from src.owner_asset_ratio.calibration import choose_k_scale


class FakeUnitModel:
    selected_name = "fake_nb"

    def __init__(self, value: int = 5, fail_if_called: bool = False):
        self.value = value
        self.fail_if_called = fail_if_called
        self.calls = 0

    def sample(self, frame, rng):
        self.calls += 1
        if self.fail_if_called:
            raise AssertionError("observed units must not be overwritten")
        return np.full(len(frame), self.value, dtype=int)


class FakeDistribution:
    def __init__(self, value: float):
        self.value = value

    def sample(self, frame, rng, size=None):
        size = size or len(frame)
        return np.full(size, self.value, dtype=float)

    def predict_quantiles(self, frame):
        # Deliberately represents a valid monotonic distribution.
        return np.tile(
            np.array([self.value * .7, self.value * .85, self.value,
                      self.value * 1.15, self.value * 1.3]),
            (len(frame), 1),
        )


class FakePrior:
    def __init__(self, k: float = 0.5, debt: float = 0.1):
        self.k = k
        self.debt = debt

    def sample(self, building_value, rng):
        n = len(building_value)
        return (
            np.full(n, self.k),
            np.full(n, self.debt),
            ["exact_group"] * n,
        )


class FakePipeline:
    def __init__(self, units=5, deposit=1000.0, value=10_000.0, k=.5,
                 fail_units=False):
        self.unit_model = FakeUnitModel(units, fail_units)
        self.deposit_model = FakeDistribution(deposit)
        self.value_model = FakeDistribution(value)
        self.owner_prior = FakePrior(k)
        self.metadata = {
            "model_version": "test",
            "data_kind": "actual",
            "occupancy": {
                "low": {"alpha": 1e9, "beta": 1.0},
                "baseline": {"alpha": 1e9, "beta": 1.0},
                "high": {"alpha": 1e9, "beta": 1.0},
            },
            "building_random_effect_sigma": 0.0,
            "provenance": {"fixture": "deterministic"},
        }


def building(**updates):
    values = {
        "building_id": "B1", "legal_dong": "인계동",
        "residential_floor_area": 200.0, "unit_count": 5,
        "land_area": 120.0, "total_floor_area": 220.0,
        "building_age": 15, "ground_floors": 4,
        "parking_count": 3, "extra": {"local_market_count": 50},
    }
    values.update(updates)
    return BuildingEstimateInput(**values)


def run(pipeline, row=None, seed=11):
    return infer_ratio_distribution(
        pipeline, row or building(), samples=4000, seed=seed)


def test_room_count_is_never_negative():
    result = run(FakePipeline(units=-20), building(
        unit_count=None, family_count=None, household_count=None))
    assert result["estimated_total_deposit"]["p10"] >= 0


def test_deposit_sample_is_never_negative():
    result = run(FakePipeline(deposit=-500))
    assert result["estimated_total_deposit"]["p10"] >= 0


def test_property_value_sample_is_never_negative():
    result = run(FakePipeline(value=-100))
    assert result["estimated_owner_total_assets"]["p10"] >= 0


def test_predicted_quantiles_are_monotonic():
    values = FakeDistribution(100).predict_quantiles(pd.DataFrame([{}]))
    assert np.all(np.diff(values, axis=1) >= 0)


def test_same_seed_reproduces_identical_result():
    first = run(FakePipeline(), seed=123)
    second = run(FakePipeline(), seed=123)
    assert first == second


def test_higher_k_reduces_ratio():
    low = run(FakePipeline(k=0.0))
    high = run(FakePipeline(k=2.0))
    assert (
        high["deposit_to_total_assets_ratio"]["p50"]
        < low["deposit_to_total_assets_ratio"]["p50"]
    )


def test_higher_total_deposit_increases_ratio():
    low = run(FakePipeline(deposit=500))
    high = run(FakePipeline(deposit=1500))
    assert (
        high["deposit_to_total_assets_ratio"]["p50"]
        > low["deposit_to_total_assets_ratio"]["p50"]
    )


def test_target_deposit_ratio_is_separate_from_existing_tenant_deposits():
    selected = building(observed_deposit=4_500.0)
    low_existing = run(FakePipeline(deposit=500), selected)
    high_existing = run(FakePipeline(deposit=1_500), selected)

    assert low_existing["target_deposit"] == 4_500.0
    assert (
        low_existing["target_deposit_to_total_assets_ratio"]
        == high_existing["target_deposit_to_total_assets_ratio"]
    )
    assert low_existing["target_deposit_to_total_assets_ratio"]["p50"] == .3
    assert (
        low_existing["deposit_to_total_assets_ratio"]["p50"]
        != high_existing["deposit_to_total_assets_ratio"]["p50"]
    )


def test_higher_property_value_reduces_ratio():
    low = run(FakePipeline(value=5000))
    high = run(FakePipeline(value=20_000))
    assert (
        high["deposit_to_total_assets_ratio"]["p50"]
        < low["deposit_to_total_assets_ratio"]["p50"]
    )


def test_k_zero_equals_building_ratio():
    result = run(FakePipeline(k=0.0))
    assert (
        result["deposit_to_total_assets_ratio"]
        == result["building_only_ratio"]
    )


def test_observed_unit_count_is_not_overwritten():
    pipeline = FakePipeline(fail_units=True)
    run(pipeline, building(unit_count=7))
    assert pipeline.unit_model.calls == 0


def test_future_deposit_does_not_enter_past_feature():
    frame = pd.DataFrame({
        "contract_id": ["a", "b"],
        "contract_year_month": [202501, 202502],
        "sigungu_code": ["41115", "41115"],
        "legal_dong": ["인계동", "인계동"],
        "partial_lot_number": ["", ""],
        "housing_type": ["다가구", "다가구"],
        "rental_area": [40.0, 40.0],
        "built_year": [2010, 2010],
        "deposit": [100.0, 10_000.0],
        "monthly_rent": [0.0, 0.0],
        "contract_type": ["전세", "전세"],
        "renewal_flag": ["", ""],
    })
    featured = add_past_only_lease_features(frame)
    assert np.isnan(featured.loc[0, "legal_dong_3m_deposit_median"])
    assert featured.loc[1, "legal_dong_3m_deposit_median"] == 100.0


def test_survey_weights_change_distribution():
    values = np.array([1.0, 100.0])
    unweighted = weighted_quantile(values, [0.5])[0]
    weighted = weighted_quantile(
        values, [0.5], np.array([1000.0, 1.0]))[0]
    assert weighted != unweighted
    assert weighted < unweighted


def test_output_contains_required_warning_and_uncertainty():
    result = run(FakePipeline())
    assert "특정 집주인의 실제" in result["warnings"][0]
    assert set(result["uncertainty_contribution"]) == {
        "room_count", "occupancy", "deposit_distribution",
        "property_value", "owner_other_assets",
    }


def test_synthetic_artifact_is_rejected_for_real_inference():
    with pytest.raises(RuntimeError, match="synthetic smoke artifact"):
        OwnerAssetRatioPipeline.load(
            "models/owner_asset_ratio/"
            "owner_asset_ratio_synthetic_smoke.joblib")


def test_smoke_artifact_has_conformal_and_spatial_holdout_metadata():
    model = OwnerAssetRatioPipeline.load(
        "models/owner_asset_ratio/"
        "owner_asset_ratio_synthetic_smoke.joblib",
        allow_synthetic=True,
    )
    for name in ("deposit", "property_value"):
        metrics = model.metadata["validation"][name]
        assert metrics["crossing_rate_after"] == 0
        assert metrics["coverage_50"] >= .5
        assert metrics["coverage_80"] >= .8
        assert metrics["backend"] == "lightgbm_quantile"
    holdout = model.metadata["splits"]["spatial_holdout"]
    assert holdout["lease_legal_dongs"]
    assert holdout["sale_legal_dongs"]


def test_population_calibration_changes_only_k_scale():
    result = choose_k_scale(
        total_deposit=np.full(100, 80.0),
        property_value=np.full(100, 100.0),
        k_samples=np.full(100, 1.0),
        survey_ratios=np.full(100, .3),
        survey_weights=np.ones(100),
    )
    assert result["calibration_kind"] == (
        "population_post_hoc_not_individual_label")
    assert result["k_scale"] > 1.0
