import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.fraud_risk.actual_model import (
    HF_SAMPLE_ACCIDENT_RATE,
    cost_ratio_threshold,
    published_hf_logits,
    solve_prior_logit_shift,
    stable_sigmoid,
)
from src.fraud_risk.calibration import choose_cost_threshold
from src.fraud_risk.bootstrap_published_hf import create_metadata
from src.fraud_risk.real_labels import load_actual_contract_labels


def _properties():
    return pd.DataFrame([
        {
            "market_price_manwon": 30_000,
            "deposit_manwon": 15_000,
            "senior_deposit_sum_manwon": 0,
            "senior_mortgage_manwon": 0,
            "house_type": "아파트",
            "sido": "서울",
            "ownership_type": "개인",
        },
        {
            "market_price_manwon": 20_000,
            "deposit_manwon": 18_000,
            "senior_deposit_sum_manwon": 2_000,
            "senior_mortgage_manwon": 1_000,
            "house_type": "다세대주택",
            "sido": "인천",
            "ownership_type": "법인",
        },
    ])


def test_prior_shift_hits_reference_incidence():
    logits = published_hf_logits(_properties())
    shift = solve_prior_logit_shift(logits, HF_SAMPLE_ACCIDENT_RATE)
    assert stable_sigmoid(logits + shift).mean() == pytest.approx(HF_SAMPLE_ACCIDENT_RATE, abs=1e-9)


def test_cost_threshold_formula_and_empirical_selector():
    assert cost_ratio_threshold(20, 1) == pytest.approx(1 / 21)
    decision = choose_cost_threshold(
        [0, 0, 1, 1], [0.01, 0.10, 0.20, 0.90],
        false_negative_cost=20, false_positive_cost=1,
    )
    assert 0 <= decision.threshold <= 0.20
    assert decision.false_negatives == 0


def test_actual_loader_rejects_synthetic_filename(tmp_path):
    csv_path = tmp_path / "hug_synthetic.csv"
    csv_path.write_text("guarantee_id,accident_label\n1,0\n", encoding="utf-8")
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({
        "data_classification": "actual_contract_level",
        "contains_synthetic": False,
        "provider": "HUG",
        "observation_cutoff": "2025-12-31",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="합성데이터"):
        load_actual_contract_labels(csv_path, provenance)


def test_actual_loader_rejects_aggregate_schema(tmp_path):
    csv_path = tmp_path / "aggregate.csv"
    csv_path.write_text("지역,사고건수,사고율\n서울,10,1.2\n", encoding="utf-8")
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({
        "data_classification": "actual_contract_level",
        "contains_synthetic": False,
        "provider": "HUG",
        "observation_cutoff": "2025-12-31",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="필수 컬럼 누락"):
        load_actual_contract_labels(csv_path, provenance, min_rows=1)


def test_bootstrap_can_calibrate_from_serving_database(tmp_path):
    db = tmp_path / "serving.db"
    frame = _properties()
    frame["lease_type"] = "전세"
    with sqlite3.connect(db) as connection:
        frame.to_sql("properties", connection, index=False)
    metadata = create_metadata(None, HF_SAMPLE_ACCIDENT_RATE, 20, 1, database_path=db)
    assert metadata["prior_calibration"]["portfolio_rows"] == 2
    assert metadata["prior_calibration"]["portfolio_source"] == str(db)
    assert metadata["prior_calibration"]["portfolio_mean_after"] == pytest.approx(
        HF_SAMPLE_ACCIDENT_RATE
    )
