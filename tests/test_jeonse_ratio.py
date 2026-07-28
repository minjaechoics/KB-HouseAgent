from datetime import date
import sqlite3

import numpy as np
import pytest

from src.db.build_db import build_guarantee_db
from src.jeonse_ratio.adapters import (
    DepositModelAdapter, DistributionContract, PropertyValueModelAdapter,
)
from src.jeonse_ratio.distributions import (
    copula_uniforms, inverse_cdf_sample, parse_quantiles,
)
from src.jeonse_ratio.engine import JeonseRatioEngine
from src.jeonse_ratio.service import JeonseRatioIntegrationService
from src.jeonse_ratio.validation import AlignmentPolicy, validate_alignment
from src.tools.guarantee_tool import GuaranteeProductTool


def contract(kind="deposit", *, building="B1", day=date(2026, 7, 28)):
    if kind == "deposit":
        quantiles = {
            "total_deposit": {"p05": 4000, "p50": 5000, "p95": 6000},
            "senior_deposit": {"p05": 2000, "p50": 3000, "p95": 4000},
            "conservative_upper_deposit": {
                "p05": 3000, "p50": 4000, "p95": 5000,
            },
        }
        metadata = {"model_mode": "scenario_only"}
    else:
        quantiles = {
            "property_value": {"p05": 9000, "p50": 10000, "p95": 11000}
        }
        metadata = {"price_basis": "market_value"}
    return DistributionContract(
        building_id=building, reference_date=day, currency="KRW",
        unit="manwon", quality="high", quantiles=quantiles,
        warnings=(), metadata=metadata,
    )


def run(**kwargs):
    return JeonseRatioEngine().calculate(
        contract(), contract("value"),
        my_deposit_manwon=2000, samples=4000, seed=7, **kwargs,
    )


def test_01_parse_quantiles_repairs_crossing():
    _, values, crossing = parse_quantiles(
        {"p10": 10, "p50": 9, "p90": 12}
    )
    assert crossing and np.all(np.diff(values) >= 0)


def test_02_inverse_cdf_has_bounded_nonnegative_tails():
    values, _ = inverse_cdf_sample(
        {"p10": 10, "p90": 20}, np.array([0.0, 1.0])
    )
    assert values[0] >= 0 and values[1] <= 60


def test_03_copula_rejects_invalid_rho():
    with pytest.raises(ValueError):
        copula_uniforms(10, 1.0, np.random.default_rng(1))


def test_04_copula_positive_dependence():
    left, right = copula_uniforms(5000, .6, np.random.default_rng(1))
    assert np.corrcoef(left, right)[0, 1] > .5


def test_05_alignment_rejects_building_mismatch():
    with pytest.raises(ValueError):
        validate_alignment(contract(), contract("value", building="B2"))


def test_06_alignment_rejects_stale_reference_date():
    stale = contract("value", day=date(2026, 5, 1))
    with pytest.raises(ValueError):
        validate_alignment(contract(), stale)


def test_07_alignment_rejects_wrong_price_basis():
    value = contract("value")
    value = DistributionContract(
        **{**value.__dict__, "metadata": {"price_basis": "asking_price"}}
    )
    with pytest.raises(ValueError):
        validate_alignment(contract(), value)


def test_08_ratio_output_has_four_definitions():
    ratios = run()["ratios"]
    assert set(ratios) == {
        "all_deposit_ratio", "senior_deposit_ratio",
        "post_contract_ratio", "conservative_post_contract_ratio",
    }


def test_09_post_contract_is_primary_metric():
    assert run()["risk"]["primary_metric"] == "post_contract_ratio"


def test_10_ratio_is_not_clamped_at_one():
    result = JeonseRatioEngine().calculate(
        contract(), contract("value"), my_deposit_manwon=9000,
        samples=2000, seed=2,
    )
    assert result["ratios"]["post_contract_ratio"]["p50"] > 1


def test_11_threshold_probabilities_are_monotone():
    values = run()["threshold_probabilities"]
    assert (
        values["post_contract_over_0_6"]
        >= values["post_contract_over_0_8"]
        >= values["post_contract_over_1_0"]
    )


def test_12_stress_haircut_increases_ratio():
    result = run()
    assert result["stress"]["0.2"]["p50"] > \
        result["ratios"]["post_contract_ratio"]["p50"]


def test_13_seed_is_reproducible():
    assert run()["ratios"] == run()["ratios"]


def test_14_dependence_sensitivity_contains_four_scenarios():
    assert len(run()["dependence"]["sensitivity"]) == 4


def test_15_quantile_only_inputs_downgrade_quality():
    assert run()["data_quality"] != "high"


def test_16_triangular_user_deposit_is_supported():
    result = JeonseRatioEngine().calculate(
        contract(), contract("value"),
        my_deposit_distribution=(1500, 2000, 2500),
        samples=2000, seed=3,
    )
    assert result["ratios"]["post_contract_ratio"]["p50"] > 0


def test_17_deposit_adapter_converts_won_to_manwon():
    result = {
        "available": True,
        "match": {"building_id": "B1"},
        "estimate": {
            "reference_date": "2026-07-28", "building_id": "B1",
            "data_quality": "medium",
            "estimated_total_deposit": {"p10": 100_000_000, "p90": 200_000_000},
            "estimated_senior_deposit": {"p10": 50_000_000, "p90": 100_000_000},
            "conservative_upper_deposit": {
                "p10": 80_000_000, "p90": 150_000_000,
            },
        },
    }
    assert DepositModelAdapter().adapt(
        result
    ).quantiles["total_deposit"]["p10"] == 10000


def test_18_property_adapter_uses_market_value_not_owner_assets():
    result = {
        "available": True, "reference_date": "2026-07-28",
        "estimate": {
            "building_id": "B1", "data_quality": "medium",
            "estimated_property_value": {"p10": 10000, "p90": 20000},
            "estimated_owner_total_assets": {"p10": 99999, "p90": 99999},
        },
    }
    adapted = PropertyValueModelAdapter().adapt(result)
    assert adapted.quantiles["property_value"]["p10"] == 10000


def test_19_service_is_not_applicable_to_sale():
    result = JeonseRatioIntegrationService().calculate(
        {"transaction_type": "매매", "house_type": "다가구주택"},
        {}, {},
    )
    assert result["status"] == "not_applicable"


def test_20_guarantee_precheck_keeps_legal_result_conditional(tmp_path):
    db = tmp_path / "test.db"
    with sqlite3.connect(db) as conn:
        build_guarantee_db(conn)
    tool = GuaranteeProductTool(db)
    result = tool.evaluate(
        {
            "transaction_type": "전세", "sido": "경기",
            "gugun": "수원시 팔달구", "deposit_manwon": 10000,
        },
        {},
        {
            "ratios": {
                "post_contract_ratio": {"p50": .6},
                "conservative_post_contract_ratio": {"p90": .8},
            }
        },
    )
    assert len(result["products"]) == 3
    assert result["preferential_repayment"]["legally_guaranteed"] is False
