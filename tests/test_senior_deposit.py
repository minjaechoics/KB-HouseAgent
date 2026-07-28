from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.owner_asset_ratio.data import add_past_only_lease_features
from src.owner_asset_ratio.schemas import SchemaValidationError
from src.senior_deposit.schemas import validate_senior_labels
from src.senior_deposit.matching import assess_rtms_building_candidate
from src.senior_deposit.simulation import (
    _attach_past_market_features,
    infer_senior_deposit_distribution,
)


class FakeUnitModel:
    selected_name = "fake_nb"

    def sample(self, frame, rng):
        return rng.negative_binomial(5, .5, len(frame)) + 1


class FakeDepositModel:
    quantiles = np.asarray([.05, .1, .25, .5, .75, .9, .95])

    def __init__(self, scale=1.0):
        self.scale = scale

    def predict_quantiles(self, frame):
        base = np.asarray(
            [1000, 1200, 1600, 2200, 3000, 4200, 5200],
            dtype=float,
        ) * self.scale
        return np.repeat(base[None, :], len(frame), axis=0)


class FakePipeline:
    def __init__(self, deposit_scale=1.0):
        self.unit_model = FakeUnitModel()
        self.deposit_model = FakeDepositModel(deposit_scale)
        self.metadata = {
            "model_version": "test",
            "model_mode": "scenario_only",
            "occupancy_priors": {
                "low": {"alpha": 7.0, "beta": 3.0},
                "baseline": {"alpha": 18.0, "beta": 2.0},
                "high": {"alpha": 38.0, "beta": 2.0},
            },
            "baseline_senior_probability": .9,
            "within_building_sigma": .12,
            "occupancy_model_trained": False,
            "seniority_model_trained": False,
            "provenance": {"lease_rtms": "actual"},
        }


def building(**updates):
    value = {
        "building_id": "B1",
        "unit_count": 10,
        "land_area": 180,
        "total_floor_area": 420,
        "residential_floor_area": 360,
        "building_age": 20,
        "ground_floors": 4,
        "parking_count": 5,
        "legal_dong": "매탄동",
        "local_market_count": 100,
    }
    value.update(updates)
    return value


def infer(**kwargs):
    return infer_senior_deposit_distribution(
        kwargs.pop("pipeline", FakePipeline()),
        kwargs.pop("building", building()),
        reference_date="2026-07-28",
        samples=kwargs.pop("samples", 4000),
        seed=kwargs.pop("seed", 28),
        **kwargs,
    )


def test_registered_and_occupied_counts_are_bounded():
    result = infer()
    assert result["registered_units"]["p10"] >= 0
    assert result["occupied_other_units"]["p90"] <= 9


def test_deposit_outputs_are_non_negative_and_ordered():
    result = infer()
    senior = result["estimated_senior_deposit"]
    assert 0 <= senior["p10"] <= senior["p50"] <= senior["p90"] <= senior["p95"]


def test_senior_never_exceeds_total_at_reported_quantiles():
    result = infer(mode="scenario", senior_probability=.6)
    senior = result["estimated_senior_deposit"]
    upper = result["conservative_upper_deposit"]
    assert senior["p50"] <= upper["p50"]
    assert senior["p90"] <= upper["p90"]
    assert senior["p95"] <= upper["p95"]


def test_conservative_equals_existing_other_deposit():
    result = infer(mode="conservative")
    assert result["estimated_senior_deposit"]["p50"] == (
        result["conservative_upper_deposit"]["p50"])
    assert result["estimated_senior_deposit"]["p90"] == (
        result["conservative_upper_deposit"]["p90"])


def test_higher_seniority_probability_is_monotonic_for_same_seed():
    low = infer(mode="scenario", senior_probability=.2)
    high = infer(mode="scenario", senior_probability=.8)
    assert high["estimated_senior_deposit"]["p50"] >= (
        low["estimated_senior_deposit"]["p50"])


def test_higher_occupancy_prior_increases_distribution():
    low = infer(mode="conservative", occupancy_scenario="low", samples=10_000)
    high = infer(
        mode="conservative", occupancy_scenario="high", samples=10_000)
    assert high["estimated_senior_deposit"]["p50"] >= (
        low["estimated_senior_deposit"]["p50"])


def test_higher_unit_deposit_increases_distribution():
    low = infer(pipeline=FakePipeline(.5), mode="conservative")
    high = infer(pipeline=FakePipeline(2.0), mode="conservative")
    assert high["estimated_senior_deposit"]["p50"] > (
        low["estimated_senior_deposit"]["p50"])


def test_same_seed_is_reproducible():
    assert infer(seed=777) == infer(seed=777)


def test_observed_unit_count_is_not_overwritten():
    result = infer(building=building(unit_count=7))
    assert result["registered_units"] == {
        "observed": 7, "p10": 7, "p50": 7, "p90": 7}


def test_target_room_is_excluded():
    result = infer(
        building=building(unit_count=2),
        occupancy_scenario="high",
        mode="conservative",
    )
    assert result["occupied_other_units"]["p90"] <= 1
    assert result["scenario"]["target_rooms_excluded"] == 1


def test_probability_one_matches_upper_and_zero_is_zero():
    one = infer(mode="scenario", senior_probability=1)
    zero = infer(mode="scenario", senior_probability=0)
    assert one["estimated_senior_deposit"]["p90"] == (
        one["conservative_upper_deposit"]["p90"])
    assert zero["estimated_senior_deposit"]["p95"] == 0


def test_probabilistic_mode_falls_back_and_warns_without_labels():
    result = infer(mode="probabilistic")
    assert result["model_mode"] == "scenario_only"
    assert any("fallback" in warning for warning in result["warnings"])


def test_past_only_features_do_not_see_future_contracts():
    frame = pd.DataFrame({
        "contract_id": ["C1", "C2", "C3"],
        "contract_year_month": [202401, 202402, 202403],
        "sigungu_code": ["41117"] * 3,
        "legal_dong": ["매탄동"] * 3,
        "partial_lot_number": ["1**"] * 3,
        "housing_type": ["다가구"] * 3,
        "deposit": [1000, 2000, 1_000_000],
        "rental_area": [50] * 3,
        "built_year": [2000] * 3,
        "monthly_rent": [0] * 3,
        "contract_type": ["전세"] * 3,
        "renewal_flag": [False] * 3,
    })
    featured = add_past_only_lease_features(frame)
    february = featured.loc[
        featured["contract_year_month"].eq(202402)].iloc[0]
    assert february["legal_dong_12m_deposit_median"] == 1000


def test_verified_label_schema_rejects_pii_and_invalid_seniority():
    valid = pd.DataFrame([{
        "building_id": "B1",
        "reference_date": "2026-07-28",
        "registered_units": 10,
        "occupied_units": 8,
        "target_room_excluded": True,
        "tenant_id_anonymized": "T-1",
        "room_identifier_anonymized": "R-1",
        "deposit": 5000,
        "monthly_rent": 0,
        "lease_start_date": "2024-01-01",
        "lease_end_date": "2026-01-01",
        "move_in_date": "2024-01-01",
        "confirmed_date": "2024-01-02",
        "currently_occupied": True,
        "senior_to_target": 1,
        "label_source": "official_confirmed_record",
        "label_confidence": 1.0,
    }])
    assert len(validate_senior_labels(valid)) == 1
    with pytest.raises(SchemaValidationError):
        validate_senior_labels(valid.assign(tenant_name="홍길동"))
    with pytest.raises(SchemaValidationError):
        validate_senior_labels(valid.assign(senior_to_target="maybe"))


def test_masked_rtms_lot_is_never_promoted_to_building_label():
    assessment = assess_rtms_building_candidate(
        {
            "legal_dong": "매탄동",
            "partial_lot_number": "12**",
            "rental_area": 40,
            "built_year": 2005,
        },
        {
            "legal_dong": "매탄동",
            "lot_number": "123-4번지",
            "residential_floor_area": 400,
            "registered_units_observed": 10,
            "built_year": 2005,
        },
    )
    assert assessment.usable_as_building_label is False
    assert assessment.confidence != "exact"
    assert "masked_partial_lot_not_exact" in assessment.reasons


def test_reference_date_selects_only_strictly_past_market_features():
    pipeline = FakePipeline()
    pipeline.market_reference = pd.DataFrame({
        "contract_year_month": [202602, 202608],
        "legal_dong": ["매탄동", "매탄동"],
        "rental_area": [40, 40],
        "legal_dong_3m_deposit_median": [2000, 999999],
        "legal_dong_12m_deposit_median": [1800, 999999],
        "legal_dong_12m_deposit_growth": [.1, 99],
        "transaction_count_3m": [10, 999],
        "transaction_count_12m": [40, 999],
    })
    frame, source = _attach_past_market_features(
        pipeline,
        pd.DataFrame({"rental_area": [40.0]}),
        legal_dong="매탄동",
        reference_date=pd.Timestamp("2026-07-28").date(),
    )
    assert source == "legal_dong_asof:202602"
    assert frame.iloc[0]["legal_dong_3m_deposit_median"] == 2000
