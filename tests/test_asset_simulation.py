from pathlib import Path

from src.report.budget import _compatible_loan, simulate
from src.report.lifestyle import estimate_monthly_lifestyle


class MapFixture:
    def geocode(self, query):
        return {"ok": True, "lat": 37.5, "lng": 127.0,
                "address": f"{query} 공식 주소", "source": "fixture"}

    def travel_time(self, start, goal, mode):
        return {"minutes": 30, "distance_km": 10, "fare_krw": 1500,
                "source": "tmap_transit", "estimated": False}


def test_itemized_lifestyle_uses_routes_and_not_legacy_living_cost():
    result = estimate_monthly_lifestyle(
        {"monthly_living_cost_manwon": 120},
        {"lat": 37.4, "lng": 127.0},
        {"use_itemized_budget": True, "transport_mode": "transit",
         "transit_taxi_ratio_pct": 0,
         "destinations": [{"query": "학교", "visits_per_month": 20}],
         "daily_food_krw": 10000, "telecom_monthly_krw": 50000},
        MapFixture(),
    )
    assert result["route_api_calls"] == 1
    assert result["breakdown_krw"]["transport"] == 60000
    assert result["breakdown_krw"]["food"] == 304000
    assert result["effective_monthly_living_cost_krw"] == 414000


def test_premium_tmap_lifestyle_keeps_more_than_five_destinations():
    destinations = [
        {"query": f"목적지 {index}", "visits_per_month": 1}
        for index in range(7)
    ]
    result = estimate_monthly_lifestyle(
        {}, {"lat": 37.4, "lng": 127.0},
        {"use_itemized_budget": True, "transport_mode": "transit",
         "destinations": destinations},
        MapFixture(),
    )
    assert result["route_api_calls"] == 7
    assert len(result["destinations"]) == 7
    assert result["route_api_call_limit"] is None
    assert result["route_api_call_policy"] == "tmap_transit_unlimited"


def test_selected_finance_product_and_age_axis_change_simulation():
    programs = [
        {"program_id": "cheap", "name": "주택 구입 대출", "category": "구입대출",
         "product_kind": "대출", "rate_pct": 2.0, "max_amount_manwon": 10000},
        {"program_id": "chosen", "name": "주택 매매 대출", "category": "구입대출",
         "product_kind": "대출", "rate_pct": 3.0, "max_amount_manwon": 10000},
    ]
    result = simulate(
        {"age": 30, "monthly_income_manwon": 400, "total_asset_manwon": 5000,
         "monthly_living_cost_manwon": 100},
        {"transaction_type": "매매", "sale_price_manwon": 10000,
         "maintenance_fee_manwon": 10},
        {"annual_growth_rate": 0.02, "annual_low": 0, "annual_high": 0.04},
        programs,
        {"simulation_end_age": 35, "selected_finance_program_id": "chosen",
         "lifestyle": {"effective_monthly_living_cost_manwon": 80}},
    )
    assert result["funding"]["chosen_program_id"] == "chosen"
    assert result["scenarios"]["base"][0]["age"] == 30
    assert result["scenarios"]["base"][-1]["age"] == 35
    assert result["scenarios"]["base"][-1]["house_price_index"] > 100


def test_gui_exposes_asset_simulation_inputs_and_separate_charts():
    gui = (Path(__file__).parents[1] / "src" / "server" / "gui.html").read_text(
        encoding="utf-8")
    for token in ("자산 시뮬레이션", "assetAddDestination", "assetTaxiRatio",
                  "assetAddSubscription", "assetOtherInsurance", "assetDailyFood",
                  "assetTelecom", "assetInternet", "assetLeisure",
                  "house_price_index", "현금성 금융자산"):
        assert token in gui


def test_gui_separates_condition_builder_from_decision_advisor():
    gui = (Path(__file__).parents[1] / "src" / "server" / "gui.html").read_text(
        encoding="utf-8")
    for token in (
        "＋ 조건 추가", "✨ AI 추천·상담", "/api/advisor/chat",
        "전세가 좋을까 월세가 좋을까?", "advisorResultMarkup",
    ):
        assert token in gui


def test_children_and_inheritance_are_visible_cashflow_events():
    current_year = __import__("datetime").date.today().year
    result = simulate(
        {"age": 30, "monthly_income_manwon": 500,
         "total_asset_manwon": 30000, "monthly_living_cost_manwon": 100,
         "children_plans": [{"birth_year": current_year}],
         "expected_inheritance_manwon": 2000, "expected_inheritance_age": 32},
        {"transaction_type": "매매", "sale_price_manwon": 5000,
         "maintenance_fee_manwon": 10},
        {"annual_growth_rate": 0.01, "annual_low": 0, "annual_high": 0.02},
        [], {"simulation_end_age": 33},
    )
    path = result["scenarios"]["base"]
    assert path[1]["annual_child_cost"] > 0
    assert path[2]["inheritance_inflow"] == 2000
    assert "예상 증여·상속 유입" in path[2]["event_labels"]
    assert result["scenario_metadata"]["family_cost_method"]


def test_finance_comparison_does_not_hide_applicable_products_after_eight():
    programs = [{
        "program_id": f"p{i}", "name": f"청년 주택구입 대출 {i}",
        "category": "주택구입대출", "product_kind": "대출",
        "rate_pct": 3 + i / 100, "max_amount_manwon": 10000,
        "eligibility_status": "preliminarily_eligible",
    } for i in range(12)]
    result = simulate(
        {"age": 30, "monthly_income_manwon": 600,
         "total_asset_manwon": 5000, "monthly_living_cost_manwon": 100},
        {"transaction_type": "매매", "sale_price_manwon": 10000,
         "maintenance_fee_manwon": 10},
        {"annual_growth_rate": 0.01, "annual_low": 0, "annual_high": 0.02},
        programs, {"simulation_end_age": 35},
    )
    assert len(result["finance_comparison"]["options"]) == 12


def test_needs_review_program_is_still_usable_for_funding():
    """"구매가능" 필터(harness._classify_transaction_finance)는 자격요건
    미입력으로 eligibility_status가 needs_review인 상품도 "가입한다고
    가정하면 조달 가능"으로 취급한다. 상세 리포트의 simulate()가 이를
    preliminarily_eligible만 인정해 제외하면, 같은 매물이 필터에서는
    구매가능으로 나오고 상세페이지에서는 조달불가로 나오는 모순이 생긴다."""
    programs = [{
        "program_id": "p1", "name": "청년 전세자금대출",
        "category": "전세자금대출", "product_kind": "대출",
        "rate_pct": 3.0, "max_amount_manwon": 20000,
        "eligibility_status": "needs_review",
    }]
    result = simulate(
        {"age": 30, "monthly_income_manwon": 400,
         "total_asset_manwon": 3000, "monthly_living_cost_manwon": 100},
        {"transaction_type": "전세", "deposit_manwon": 10000},
        {"annual_growth_rate": 0.01, "annual_low": 0, "annual_high": 0.02},
        programs, {"simulation_end_age": 32},
    )
    assert result["funding"]["eligible_product_count"] == 1
    assert result["funding"]["feasible_with_known_products"] is True
    assert result["funding"]["known_product_loan_manwon"] > 0


def test_not_eligible_program_is_still_excluded_from_funding():
    programs = [{
        "program_id": "p1", "name": "청년 전세자금대출",
        "category": "전세자금대출", "product_kind": "대출",
        "rate_pct": 3.0, "max_amount_manwon": 20000,
        "eligibility_status": "not_eligible",
    }]
    result = simulate(
        {"age": 30, "monthly_income_manwon": 400,
         "total_asset_manwon": 3000, "monthly_living_cost_manwon": 100},
        {"transaction_type": "전세", "deposit_manwon": 10000},
        {"annual_growth_rate": 0.01, "annual_low": 0, "annual_high": 0.02},
        programs, {"simulation_end_age": 32},
    )
    assert result["funding"]["eligible_product_count"] == 0
    assert result["funding"]["known_product_loan_manwon"] == 0


def test_finance_comparison_amortizes_purchase_loan():
    result = simulate(
        {"age": 30, "monthly_income_manwon": 400, "total_asset_manwon": 5000,
         "monthly_living_cost_manwon": 100},
        {"transaction_type": "매매", "sale_price_manwon": 10000,
         "maintenance_fee_manwon": 10},
        {"annual_growth_rate": 0.02, "annual_low": 0, "annual_high": 0.04},
        [{"program_id": "purchase", "name": "청년 주택 구입 대출",
          "category": "구입대출", "product_kind": "대출", "rate_pct": 3.0,
          "max_amount_manwon": 10000, "loan_period_text": "30년"}],
        {"simulation_end_age": 35},
    )
    comparison = result["finance_comparison"]
    option = comparison["options"][0]
    assert comparison["baseline"]["program_id"] is None
    assert option["loan_amount_manwon"] > 0
    assert option["total_principal_repaid_manwon"] > 0
    assert option["total_interest_manwon"] > 0
    assert option["path"][-1]["debt"] < option["path"][0]["debt"]
    assert option["final_debt_manwon"] == option["path"][-1]["debt"]


def test_finance_comparison_repays_rental_principal_from_returned_deposit():
    result = simulate(
        {"age": 25, "monthly_income_manwon": 300, "total_asset_manwon": 2000,
         "monthly_living_cost_manwon": 90},
        {"transaction_type": "전세", "deposit_manwon": 8000,
         "maintenance_fee_manwon": 8},
        {"annual_growth_rate": 0.01, "annual_low": 0, "annual_high": 0.02},
        [{"program_id": "rent", "name": "청년 전세자금 대출",
          "category": "전세대출", "product_kind": "대출", "rate_pct": 2.0,
          "max_amount_manwon": 10000, "loan_period_text": "2년"}],
        {"simulation_end_age": 27},
    )
    option = result["finance_comparison"]["options"][0]
    assert option["path"][0]["debt"] > 0
    assert option["path"][-1]["debt"] == 0
    assert option["path"][-1]["deposit_asset"] == 0
    assert option["final_debt_manwon"] == 0
    assert option["payoff_age"] == 27


def test_gui_exposes_landing_finance_comparison():
    gui = (Path(__file__).parents[1] / "src" / "server" / "gui.html").read_text(
        encoding="utf-8")
    for token in ("landingView", "landingStart", "youth-home-hero-v1.png",
                  "financeComparison", "data-finance-series",
                  "금융상품별 자산 변화 비교", "최종 확인"):
        assert token in gui


def test_gui_has_no_debug_trace_panel():
    """실서비스 화면에는 RAG DEBUG 버튼·트레이스 패널을 노출하지 않는다."""
    gui = (Path(__file__).parents[1] / "src" / "server" / "gui.html").read_text(
        encoding="utf-8")
    for token in ("debugBtn", "debugPanel", "debugClose", "debugText",
                  "debugTraceText", "sanitizeTrace", "RAG DEBUG"):
        assert token not in gui


def test_purchase_finance_recognizes_mortgage_but_not_subscription_account():
    assert _compatible_loan(
        {"name": "KB 주택담보대출", "category": "담보대출", "product_kind": "대출"},
        "매매",
    )
    assert not _compatible_loan(
        {"name": "청년주택드림청약통장", "category": "청약·연계대출",
         "product_kind": "청약,대출"},
        "매매",
    )


def test_auto_finance_prefers_fully_funded_product_over_cheaper_shortfall():
    result = simulate(
        {"age": 30, "monthly_income_manwon": 500, "total_asset_manwon": 5000,
         "monthly_living_cost_manwon": 100},
        {"transaction_type": "매매", "sale_price_manwon": 10000,
         "maintenance_fee_manwon": 10},
        {"annual_growth_rate": 0.02, "annual_low": 0, "annual_high": 0.04},
        [
            {"program_id": "cheap-short", "name": "저금리 주택구입대출",
             "category": "주택구입대출", "product_kind": "대출",
             "rate_pct": 1.0, "max_amount_manwon": 1000},
            {"program_id": "funded", "name": "KB 주택담보대출",
             "category": "담보대출", "product_kind": "대출",
             "rate_pct": 4.0, "max_amount_manwon": 10000,
             "loan_period_text": "30년"},
        ],
        {},
    )
    assert result["funding"]["chosen_program_id"] == "funded"
    assert result["funding"]["simulation_valid"] is True
    assert result["funding"]["verdict_code"] == "financeable"


def test_unfunded_purchase_stops_asset_simulation_contract():
    result = simulate(
        {"age": 30, "monthly_income_manwon": 300, "total_asset_manwon": 2000,
         "monthly_living_cost_manwon": 100},
        {"transaction_type": "매매", "sale_price_manwon": 30000,
         "maintenance_fee_manwon": 10},
        {"annual_growth_rate": 0.02, "annual_low": 0, "annual_high": 0.04},
        [],
        {},
    )
    assert result["funding"]["simulation_valid"] is False
    assert result["funding"]["verdict_code"] == "no_eligible_finance"
    assert "불가" in result["funding"]["verdict_title"]


def test_cash_purchase_does_not_auto_apply_unused_finance_product():
    result = simulate(
        {"age": 30, "monthly_income_manwon": 400, "total_asset_manwon": 20000,
         "monthly_living_cost_manwon": 100},
        {"transaction_type": "매매", "sale_price_manwon": 5000,
         "maintenance_fee_manwon": 10},
        {"annual_growth_rate": 0.02, "annual_low": 0, "annual_high": 0.04},
        [{"program_id": "mortgage", "name": "KB 주택담보대출",
          "category": "담보대출", "product_kind": "대출", "rate_pct": 4.0,
          "max_amount_manwon": 10000, "eligibility_status": "preliminarily_eligible"}],
        {},
    )
    assert result["funding"]["verdict_code"] == "cash_possible"
    assert result["funding"]["chosen_program_id"] is None
    assert result["funding"]["known_product_loan_manwon"] == 0


def test_facility_counts_show_the_backend_radius_in_every_tile():
    gui = (Path(__file__).parents[1] / "src" / "server" / "gui.html").read_text(
        encoding="utf-8")
    assert "data?.radius_m" in gui
    assert 'class="radius-badge"' in gui
    assert 'class="facility-radius">${radiusText}' in gui
