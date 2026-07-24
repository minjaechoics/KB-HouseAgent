"""선택 매물 주거·자산 리포트 회귀 테스트."""
import json
from pathlib import Path

from src import config
from src.market_forecast import HousePriceForecaster
from src.report.budget import simulate
from src.report.service import PropertyReportService
from src.fraud_risk.infer import FraudRiskScorer


def _user():
    return {"age": 29, "monthly_income_manwon": 300,
            "total_asset_manwon": 6000, "monthly_living_cost_manwon": 120}


def test_trained_house_price_model_has_holdout_metadata():
    result = HousePriceForecaster().forecast({
        "house_type": "아파트", "legal_dong_code": "1111010100",
        "sido": "서울특별시", "gugun": "종로구",
    })
    assert result["model_version"] == "rtms_calendar_3month_gbdt_news_v3"
    assert result["training"]["rows"] > 0
    assert result["training"]["inference_feature_month_max"] >= \
        result["training"]["training_target_month_max"]
    assert result["annual_low"] <= result["annual_high"]
    assert -0.15 <= result["annual_growth_rate"] <= 0.15


def test_renter_does_not_receive_landlord_house_price_appreciation():
    forecast = {"annual_growth_rate": 0.08, "annual_low": 0.02, "annual_high": 0.12}
    result = simulate(_user(), {
        "transaction_type": "전세", "deposit_manwon": 5000,
        "maintenance_fee_manwon": 10,
    }, forecast, [], {"horizon_years": 3})
    assert all(row["property_value"] == 0
               for row in result["scenarios"]["optimistic"])
    assert all(row["deposit_asset"] == 5000
               for row in result["scenarios"]["base"])


def test_unfunded_contract_capital_is_not_counted_as_net_worth():
    user = {"total_asset_manwon": 1000, "monthly_income_manwon": 250,
            "monthly_living_cost_manwon": 100}
    prop = {"transaction_type": "전세", "deposit_manwon": 10000,
            "maintenance_fee_manwon": 10}
    forecast = {"annual_growth_rate": 0.03, "annual_low": 0.0,
                "annual_high": 0.06}
    result = simulate(user, prop, forecast, [], {"horizon_years": 1})
    funding = result["funding"]
    initial = result["scenarios"]["base"][0]

    assert funding["funding_gap_manwon"] > 0
    assert initial["unfunded_gap"] == funding["funding_gap_manwon"]
    assert initial["net_worth"] <= user["total_asset_manwon"]


def test_negative_house_forecast_explains_positive_total_net_worth():
    user = {"total_asset_manwon": 20000, "monthly_income_manwon": 500,
            "monthly_living_cost_manwon": 100}
    prop = {"transaction_type": "매매", "sale_price_manwon": 10000,
            "maintenance_fee_manwon": 5}
    forecast = {"annual_growth_rate": -0.02, "annual_low": -0.05,
                "annual_high": -0.01}
    result = simulate(user, prop, forecast, [], {"horizon_years": 2})
    meta = result["scenario_metadata"]

    assert meta["labels"]["optimistic"] == "집값 상방 경로"
    assert "저축" in meta["explanation"]
    assert meta["growth_rates"]["optimistic"] < 0


def test_report_never_republishes_sex_offender_or_synthetic_owner_identity():
    service = PropertyReportService()
    import sqlite3
    con = sqlite3.connect(config.DB_PATH)
    property_id = con.execute(
        "SELECT property_id FROM properties LIMIT 1").fetchone()[0]
    con.close()
    result = service.build(_user(), property_id, {"horizon_years": 2})
    restricted = result["restricted_checks"]
    assert restricted["sex_offender"]["count"] is None
    assert restricted["sex_offender"]["status"] == "manual_identity_verification_required"
    assert result["contract_safety"]["tax_arrears"]["synthetic_field_ignored"] is True
    assert len(result["budget"]["scenarios"]["base"]) == 3


def test_gui_has_mobile_report_tabs_and_simulator_controls():
    gui = (Path(__file__).parents[1] / "src" / "server" / "gui.html").read_text(
        encoding="utf-8")
    for value in (
        'id="reportView"', "주거·자산 진단 리포트", "예산/자산", "대출/지원",
        "범죄/치안", "생활/편의", "계약안전", "reportRecalc",
        "/api/properties/report", "성범죄자 알림e",
    ):
        assert value in gui
    for value in ('data-tab="final"', "renderFinalAssessment",
                  "forecastGauge", "newsDonut", "childPlanList"):
        assert value in gui


class StructuredRiskLLM:
    supports_agentic_calls = True

    def analyze_json(self, **kwargs):
        assert kwargs["operation"] == "report.risk_explanation"
        evidence = json.loads(kwargs["user"])
        assert evidence["model_drivers"]
        return {
            "headline": "주의 구간의 핵심 근거",
            "summary": "모델에 실제 입력된 주택유형과 부채비율을 함께 해석했습니다.",
            "factors": [{"tone": "risk", "label": "부채비율",
                         "detail": "공개모형에서 점수를 높이는 방향입니다."}],
            "next_checks": ["보증 가입 가능 여부를 확인하세요."],
            "limitations": "계약 안전을 확정하는 값은 아닙니다.",
        }


def test_llm_explains_jeonse_risk_from_model_evidence_only():
    import sqlite3
    connection = sqlite3.connect(config.DB_PATH)
    connection.row_factory = sqlite3.Row
    row = connection.execute(
        "SELECT * FROM properties WHERE transaction_type='전세' "
        "AND fraud_score IS NOT NULL LIMIT 1"
    ).fetchone()
    connection.close()
    assert row is not None

    service = object.__new__(PropertyReportService)
    service.llm = StructuredRiskLLM()
    service.risk_scorer = FraudRiskScorer()
    result = service._risk_explanation(dict(row))

    assert result["strategy"] == "llm_structured"
    assert result["model_evidence"]["score"] >= 0
    assert result["factors"][0]["label"] == "부채비율"
    assert result["next_checks"]


def test_gui_renders_llm_long_text_and_risk_explanation_safely():
    gui = (Path(__file__).parents[1] / "src" / "server" / "gui.html").read_text(
        encoding="utf-8")
    for token in ("richLlmText", "richInline", "renderRiskExplanation",
                  "renderContractRisk", "LLM 근거 설명", "llm-rich-line"):
        assert token in gui
    assert "role==='ai'?richLlmText(text):esc(text)" in gui
