from pathlib import Path

import pandas as pd

from src.report.budget import simulate
from src.report.service import json_safe
from src.tools.finance_tool import FinanceTool
from src.tools.public_facility_cache import PublicFacilityCache


ROOT = Path(__file__).parents[1]


def test_mobile_landing_profile_and_loading_contract():
    gui = (ROOT / "src" / "server" / "gui.html").read_text(encoding="utf-8")
    for token in (
        "똘똘한최", "syncSpouseIncome", "spouseIncome", "minorChildren",
        "field-warning", "loader-balloon", "KB 금융상품", "runnerTravel",
        "AI가 다시 짚은 핵심",
        "assetLoanAmount", "finance-choice>input", "금융상품 미적용",
        "renderFundingVerdict", "simulation-blocked", "eligibilityChips",
    ):
        assert token in gui


def test_finance_eligibility_exposes_compact_condition_checks():
    row = {
        "name": "테스트 주거대출", "age_min": 19, "age_max": 34,
        "income_limit_manwon": 5000, "requires_korean_national": 1,
        "max_home_count": 0, "requires_income_proof": 1,
    }
    checked = FinanceTool.annotate_eligibility(
        row,
        {"age": 29, "monthly_income_manwon": 300,
         "is_korean_national": True, "home_ownership_count": 0,
         "has_income_proof": True},
    )
    labels = {item["label"]: item["status"]
              for item in checked["eligibility_checks"]}
    assert labels["나이"] == "passed"
    assert labels["연소득"] == "passed"
    assert labels["국적"] == "passed"
    assert labels["주택보유"] == "passed"


def test_single_childless_profile_excludes_kb_multi_child_product():
    profile = {
        "age": 29, "marital_status": "single", "minor_children_count": 0,
        "employment_type": "employee", "home_ownership_count": 0,
        "household_role": "prospective_head", "is_korean_national": True,
        "has_income_proof": True, "contract_deposit_paid_5pct": True,
    }
    rows = FinanceTool().search(
        user_income_manwon=300, user_age=29, user_profile=profile, limit=200)
    assert "KB-AE460567BDE87F" not in {str(row.get("program_id")) for row in rows}


def test_finance_none_and_rent_hide_house_price_scenario():
    result = simulate(
        {"age": 29, "monthly_income_manwon": 300,
         "total_asset_manwon": 3000, "monthly_living_cost_manwon": 100},
        {"transaction_type": "전세", "deposit_manwon": 8000,
         "maintenance_fee_manwon": 8},
        {"annual_growth_rate": 0.02, "annual_low": 0, "annual_high": 0.04},
        [{"program_id": "loan", "name": "청년 전세 대출", "category": "전세대출",
          "product_kind": "대출", "rate_pct": 3.0, "max_amount_manwon": 5000}],
        {"selected_finance_program_id": "__none__"},
    )
    assert result["funding"]["chosen_program_id"] is None
    assert result["funding"]["known_product_loan_manwon"] == 0
    assert result["scenario_metadata"]["price_affects_user_asset"] is False


def test_official_public_facility_spatial_cache(tmp_path):
    path = tmp_path / "facilities.csv"
    pd.DataFrame([
        {"category": "pharmacy", "name": "공식약국", "address": "테스트",
         "lat": 37.5, "lng": 127.0, "subcategory": "약국",
         "source_url": "https://www.data.go.kr/"},
    ]).to_csv(path, index=False, encoding="utf-8-sig")
    found = PublicFacilityCache(path).nearby("pharmacy", 37.5, 127.0, 300)
    assert found["count"] == 1
    assert found["source"] == "official_public_csv"


def test_report_json_boundary_replaces_nested_non_finite_values():
    payload = {
        "plain": 1.0,
        "nested": [float("nan"), {"positive_inf": float("inf")}],
    }
    safe = json_safe(payload)
    assert safe["plain"] == 1.0
    assert safe["nested"] == [None, {"positive_inf": None}]
