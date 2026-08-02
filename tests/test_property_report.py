"""선택 매물 주거·자산 리포트 회귀 테스트."""
import json
from pathlib import Path

from src import config
from src.market_forecast import HousePriceForecaster
from src.report.budget import simulate
from src.report.service import PropertyReportService
from src.fraud_risk.infer import FraudRiskScorer
from src.senior_deposit import SeniorDepositIntegrationService


def _user():
    return {"age": 29, "monthly_income_manwon": 300,
            "total_asset_manwon": 6000, "monthly_living_cost_manwon": 120}


def test_trained_house_price_model_has_holdout_metadata():
    result = HousePriceForecaster().forecast({
        "house_type": "아파트", "legal_dong_code": "1111010100",
        "sido": "서울특별시", "gugun": "종로구",
    })
    assert result["model_version"] == "rtms_walkforward_conformal_v4"
    assert result["news_numeric_effect_applied"] is False
    assert result["annual_low_95"] <= result["annual_growth_rate"] <= result["annual_high_95"]
    assert result["training"]["walk_forward"]["selected_base_model"] in {
        "seasonal_naive", "ridge", "hist_gbdt", "lightgbm"
    }
    assert result["training"]["conformal"]["empirical_coverage_95"] >= .90
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


def test_gui_defaults_to_auto_selected_finance_program_on_first_load():
    """첫 로드 시 selected_finance_program_id를 '__none__'으로 강제하면 백엔드가
    program=None으로 시뮬레이션해 대출을 아예 안 쓴 것으로 계산한다. '구매가능'
    필터는 최적 대출 상품 가입을 가정해 예산을 산정하므로, 상세페이지도 첫
    로드부터 최적 상품을 자동 선택(null → 백엔드 auto-pick)해야 필터와 일치한다."""
    gui = (Path(__file__).parents[1] / "src" / "server" / "gui.html").read_text(
        encoding="utf-8")
    assert "selected_finance_program_id:null" in gui
    assert (
        "defaultAssetInputs=function(){return {...defaultAssetInputsBase(),"
        "selected_finance_program_id:'__none__'}}"
    ) not in gui


def test_gui_finance_program_selection_does_not_resubmit_stale_zero_loan():
    """대출액 입력칸은 화면에 보여주는 자동 추천값을 항상 표시하지만, 이 값을
    매번 collectAssetInputs()가 '사용자가 명시적으로 입력한 값'으로 재제출하면
    안 된다(0으로 렌더된 순간 그 0이 영구 고정돼 대출 미적용과 동일해진다).
    사용자가 실제로 입력칸을 편집했을 때만 state.assetInputs에 반영돼야 한다."""
    gui = (Path(__file__).parents[1] / "src" / "server" / "gui.html").read_text(
        encoding="utf-8")
    assert (
        "value.requested_loan_amount_manwon="
        "state.assetInputs?.requested_loan_amount_manwon??null"
    ) in gui
    assert "value.requested_loan_amount_manwon=loanInputValue()??state.assetInputs" \
        not in gui


def test_gui_auto_fills_zero_minor_children_for_single_marital_status():
    """미혼(single) 선택 시 미성년 자녀 수를 비워두면 finance_tool.py가 이를
    None으로 받아 '미성년 자녀 수 미입력' 확인 필요 플래그를 띄운다. 미혼이면
    자녀 수 0을 자동으로 채워 넣어 불필요한 확인 요청을 피한다."""
    gui = (Path(__file__).parents[1] / "src" / "server" / "gui.html").read_text(
        encoding="utf-8")
    assert "syncMinorChildrenDefault" in gui
    assert "maritalStatus').value==='single'" in gui


def test_gui_finance_series_toggle_preserves_accordion_state_without_full_reload():
    """금융상품 체크박스를 눌러 재계산할 때 화면 전체를 로딩 애니메이션으로
    덮어쓰면 재렌더 후 <details> 아코디언이 전부 닫힌 상태로 리셋된다. 로딩
    화면 대신 가벼운 busy 표시를 쓰고, 재렌더 전후로 아코디언 열림 상태를
    캡처/복원해야 한다."""
    gui = (Path(__file__).parents[1] / "src" / "server" / "gui.html").read_text(
        encoding="utf-8")
    for token in ("captureDetailsState", "restoreDetailsState",
                  "restoreDetailsState(__openDetails);bindReportUi()",
                  "btn.classList.add('busy')"):
        assert token in gui
    assert (
        "current.selected_finance_program_id='__none__';state.assetInputs=current;"
        "state.financeSelectionInFlight=true;$('reportContent').innerHTML="
        "reportLoadingMarkup();"
    ) not in gui


def test_gui_has_exactly_one_renderjeonseprimaryrisk_definition():
    """gui.html이 세션 내내 반복해 온 '뒤에 있는 재할당(name=function(){})이
    앞선 function 선언을 조용히 덮어써 죽은 코드를 만드는' 패턴에 다시
    당했었다 — 이전 두 차례 수정이 실제로는 뒤에 있는 진짜 활성 정의가 아니라
    앞의 죽은 function 선언에 적용돼 배포해도 반영되지 않았다. 죽은 선언을
    제거해 정의가 정확히 하나만 남았는지 회귀 검증한다."""
    gui = (Path(__file__).parents[1] / "src" / "server" / "gui.html").read_text(
        encoding="utf-8")
    assert gui.count("renderJeonsePrimaryRisk") == 2  # 정의 1 + 호출부 1
    assert "function renderJeonsePrimaryRisk(p,cs){" not in gui
    assert "renderJeonsePrimaryRisk=function(p,cs){" in gui


def _jeonse_primary_risk_card_source() -> str:
    gui = (Path(__file__).parents[1] / "src" / "server" / "gui.html").read_text(
        encoding="utf-8")
    start = gui.find("renderJeonsePrimaryRisk=function")
    end = gui.find("};", start) + 2
    return gui[start:end]


def test_gui_jeonse_ratio_card_leads_with_plain_language_summary():
    """P10/P50/P90·80%초과·100%초과·보수추정 6개 숫자를 처음부터 한꺼번에
    보여주면 이 지표를 모르는 사용자가 이해하기 어렵다. 기본으로는 등급
    헤드라인과 '매물 100곳 중 약 X곳' 같은 한 문장 요약만 보여준다."""
    card = _jeonse_primary_risk_card_source()
    assert "overFullPct=Math.round(Number(th.post_contract_over_1_0||0)*1000)/10" \
        in card
    assert "꼴로 건물을 팔아도 보증금을 온전히 못 돌려받는 상황이 시뮬레이션됐어요" \
        in card


def test_gui_jeonse_ratio_card_drops_the_six_quantile_probability_boxes():
    """사용자가 P10/P50/P90(낮은·중앙·높은 추정)과 80%·100% 초과 확률,
    보수 추정까지 6개 박스를 아예 없애 달라고 요청했다."""
    card = _jeonse_primary_risk_card_source()
    for token in ("낮은 추정", "중앙 추정", "높은 추정", "80% 초과",
                  "100% 초과", "보수 추정", 'class="ratio-range"',
                  'class="status-grid contract-status"'):
        assert token not in card


def test_gui_guarantee_exposure_updates_when_a_product_is_selected():
    """반환보증 상품을 선택해도 '반환위험 노출액'이라는 라벨 자체가 통째로
    '보증 대상액'으로 바뀌어 버려서, 사용자 입장에서는 같은 지표가 안 바뀌는
    것처럼 보였다. 선택 여부와 무관하게 '반환위험 노출액' 라벨은 항상 보이고
    값이 (전체 보증금 - 보증 대상액)으로 실제로 줄어들어야 한다."""
    gui = (Path(__file__).parents[1] / "src" / "server" / "gui.html").read_text(
        encoding="utf-8")
    assert (
        "exposure=Math.max(0,Number(review.uninsured_exposure_manwon||0)-covered)"
    ) in gui
    assert (
        '<div class="coverage-summary">반환위험 노출액 <b>${reportMan(exposure)}</b>'
    ) in gui
    assert "가입 심사 통과 가정 · 보증 대상액 <b>${reportMan(covered)}</b>`:`미가입 가정" \
        not in gui


def test_gui_facility_tile_does_not_overflow_grid_on_narrow_mobile_screens():
    """police/fire_station 캡션이 '반경 300m 내 0곳 · 최근접 도보 약 42.1분'처럼
    길어지면서, 그리드 아이템(및 그 자식들)의 기본 min-width:auto 때문에 실제
    390px 모바일 화면에서 3번째 컬럼이 화면 밖으로 밀려나는 버그가 있었다
    (Chrome CDP로 390px 뷰포트 렌더링을 재현해 확인). 그리드 아이템과 그 자식
    모두에 min-width:0을 줘야 콘텐츠가 줄바꿈되고 넘치지 않는다."""
    gui = (Path(__file__).parents[1] / "src" / "server" / "gui.html").read_text(
        encoding="utf-8")
    assert (
        ".facility-tile{min-height:92px;border-radius:15px;background:#f6f8fa;"
        "padding:12px 9px;display:grid;align-content:center;text-align:center;"
        "min-width:0}"
    ) in gui
    assert ".facility-tile>*{min-width:0}" in gui
    assert "max-width:100%;margin:5px auto 0" in gui


def test_gui_facility_tile_leads_with_walk_minutes_for_sparse_facilities():
    """경찰서/소방서는 300m 반경 안에 있는 경우가 드물어 반경 집계 건수만
    보여주면 도보 15분 이내 조건으로 찾은 매물인데도 '0건'만 크게 보여
    모순처럼 보인다. police/fire_station은 최근접 도보 시간을 대표 수치로
    앞세우고, 반경 집계 건수는 보조 설명으로 내려야 한다."""
    gui = (Path(__file__).parents[1] / "src" / "server" / "gui.html").read_text(
        encoding="utf-8")
    assert "isNearestType=key==='police'||key==='fire_station'" in gui
    assert "headline=isNearestType&&walk!=null?`도보 약 ${walk}분`" in gui


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


class StructuredMetricLLM:
    supports_agentic_calls = True

    def analyze_json(self, **kwargs):
        assert kwargs["operation"] == "report.metric_explanations"
        facts = json.loads(kwargs["user"])["facts"]
        return {
            "items": [{
                "id": item["id"],
                "explanation": "현재 선택의 의미를 비교해서 읽어야 하는 값입니다.",
                "tone": item["tone"],
            } for item in facts]
        }


def test_metric_explanations_use_one_grounded_llm_batch():
    service = PropertyReportService(llm=StructuredMetricLLM())
    report = {
        "budget": {"funding": {
            "required_capital_manwon": 10000,
            "cash_used_manwon": 4000,
            "initial_cash_shortfall_manwon": 6000,
            "monthly_budget_shortfall_manwon": 20,
        }},
        "forecast": {
            "annual_growth_rate": .02,
            "price_history": {
                "latest_price_manwon": 12000,
                "change_period": .03,
            },
        },
        "probabilistic_simulation": {
            "horizon_years": 10,
            "base": {
                "terminal_net_worth": {
                    "p10": 3000, "p50": 9000, "p90": 18000,
                },
                "cash_depletion_probability": .1,
                "repayment_distress_probability": .2,
                "cvar_5_terminal_change_manwon": -2500,
            },
        },
        "contract_safety": {},
        "safety": {"safety_score": 65},
        "convenience": {"convenience_score": 72},
        "final_assessment": {"score": 61},
    }
    result = service.explain_metrics(report)

    assert result["strategy"] == "llm_structured"
    assert len(result["items"]) >= 10
    assert {item["section"] for item in result["items"]} >= {
        "budget", "assets", "safety", "living", "final",
    }
    assert all(not any(ch.isdigit() for ch in item["explanation"])
               for item in result["items"])


def test_gui_loads_metric_explanations_after_main_report():
    gui = (Path(__file__).parents[1] / "src" / "server" / "gui.html").read_text(
        encoding="utf-8")
    for token in (
        "/api/properties/report/metric-explanations",
        "loadMetricExplanations",
        "injectMetricExplanations",
        "AI 숫자 해설",
        "metric-ai-panel",
    ):
        assert token in gui


def test_async_news_assessment_enables_llm_after_fast_report():
    calls = []

    class NewsTool:
        def assess(self, *args, **kwargs):
            calls.append((args, kwargs))
            return {
                "judge_strategy": "llm_structured",
                "overall_assessment": {
                    "label": "neutral",
                    "summary": "관련 기사와 실거래 추세를 함께 확인했습니다.",
                },
            }

    service = PropertyReportService.__new__(PropertyReportService)
    service.forecaster = type(
        "Forecaster", (), {"news_tool": NewsTool()}
    )()
    result = service.explain_news({
        "property": {
            "sido": "경기도", "gugun": "수원시 팔달구",
            "house_type": "다가구주택", "building_name": "테스트주택",
        },
        "forecast": {
            "time_series_annual_growth_rate": .01,
            "annual_low": -.02, "annual_high": .04,
            "price_history": {"change_1m": .01, "change_period": .03},
        },
        "regional_market": {},
    })

    assert calls[0][1]["use_llm"] is True
    assert result["news"]["judge_strategy"] == "llm_structured"
    assert result["news"]["ai_judgement_completed"] is True


def test_gui_runs_independent_news_and_metric_llms_in_parallel():
    gui = (Path(__file__).parents[1] / "src/server/gui.html").read_text(
        encoding="utf-8")
    for token in (
        "/api/properties/report/news-assessment",
        "Promise.allSettled([loadNewsAssessment(),loadMetricExplanations()])",
        "지역 뉴스를 정밀 분석하고 있어요",
        "ai_judgement_completed",
    ):
        assert token in gui


def test_report_ui_hides_internal_model_names_and_exposes_calculation_details():
    root = Path(__file__).parents[1]
    gui = (root / "src/server/gui.html").read_text(encoding="utf-8")
    service = (root / "src/report/service.py").read_text(encoding="utf-8")

    for token in (
        "어떻게 구하나요?", "calc-disclosure", "calc-formula",
        "metric-ai-head", "publicUiText",
    ):
        assert token in gui
    for internal_name in (
        "hf_actual", "published_logit", "prior_calibration",
    ):
        assert internal_name not in gui
    assert "모델명, 테이블명, 보정기법 이름은 절대 노출하지 않는다" in service


def test_senior_deposit_integration_marks_non_registry_input_low_confidence():
    import sqlite3

    connection = sqlite3.connect(config.DB_PATH)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        "SELECT * FROM properties WHERE transaction_type IN ('전세','월세')"
    ).fetchall()
    connection.close()
    service = SeniorDepositIntegrationService()
    matched = next(
        (dict(row) for row in rows
         if service.match_property(dict(row)).get("matched")),
        None,
    )
    assert matched is not None
    result = service.analyze_property(
        matched, reference_date="2026-07-28", samples=1_000, seed=42)

    assert result["available"] is True
    assert result["match"]["method"] == "exact_normalized_road_address"
    assert result["match"]["confidence"] == "exact"
    assert result["estimate"]["model_mode"] == "scenario_only"
    assert result["decision_support"]["risk_score_changed"] is False
    assert result["decision_support"]["target_deposit_won"] >= 0
    assert result["decision_support"][
        "existing_deposit_conservative_p95_won"] >= 0
    assert "combined_deposit_exposure_p95_won" not in result["decision_support"]

    unmatched = dict(matched)
    unmatched["road_address"] = "경기도 수원시 팔달구 존재하지않는로 99999"
    missing = service.analyze_property(
        unmatched, reference_date="2026-07-28", samples=1_000, seed=42)
    assert missing["available"] is True
    assert missing["status"] == "estimated_from_listing_features"
    assert missing["match"]["method"] == (
        "listing_features_without_registry_match")
    assert missing["match"]["confidence"] == "low"
    assert missing["match"]["registry_exact_match"] is False


def test_gui_and_docker_include_integrated_senior_deposit_evidence():
    root = Path(__file__).parents[1]
    gui = (root / "src" / "server" / "gui.html").read_text(encoding="utf-8")
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    for value in (
        "renderSeniorDeposit", "기존 임차보증금 추정",
        "기존 임차보증금 · 보수 P95", "선택 매물의 내 보증금",
        "건축HUB 정확주소", "어떻게 추정되었나요?",
        "금융상품 미적용", "누적 순자산 (만원)", "나이 (세)",
        "youth-loader-stage", "KB 금융상품",
    ):
        assert value in gui
    assert "data/processed/owner_asset_ratio/buildings.csv" in dockerfile


def test_final_assessment_never_calls_dangerous_contract_safe():
    service = PropertyReportService.__new__(PropertyReportService)
    service.llm = None
    result = service._final_assessment(
        {"transaction_type": "전세", "fraud_score": .92},
        {"funding": {
            "simulation_valid": True,
            "verdict_title": "자금조달 가능",
            "verdict_message": "예산 범위입니다.",
        }},
        {
            "annual_growth_rate": .03,
            "price_history": {"available": True},
            "news": {"relevant_count": 1},
            "market_assessment": {"label": "positive"},
        },
        {"safety_score": 90, "grade": "안전"},
        {"convenience_score": 90, "grade": "우수"},
        {
            "risk_explanation": {"model_evidence": {
                "score": .92,
                "grade": "위험",
                "formula": "선택 전세보증금 / 추정 집주인 총자산",
                "method": (
                    "target_jeonse_deposit/"
                    "estimated_owner_total_assets"
                ),
                "property_facts": {},
            }},
            "senior_deposit": {"available": False},
        },
    )

    assert result["recommendation"] == "avoid"
    assert "위험" in result["summary"]
    assert "우수" not in result["summary"]
    assert "낮" not in result["summary"]
