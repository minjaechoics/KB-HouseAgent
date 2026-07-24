"""통합 테스트: 도구 → Text2SQL → 추천 → Agent harness."""
import os
import tempfile
from pathlib import Path

# 테스트 중 실제 유료 LLM 호출은 하지 않는다. 운영 기본값은 config의 OpenAI다.
os.environ["JEONSE_LLM"] = "mock"
os.environ["MAP_LIVE_DISABLED"] = "1"

import pandas as pd
from src.tools.map_tool import MapTool
from src.tools.finance_tool import FinanceTool
from src.tools.property_db_tool import PropertyDBTool
from src.agent.harness import JeonseAgent


def test_map_tool_fallback():
    t = MapTool()
    r = t.travel_time((37.48, 126.95), (37.525, 126.926), "transit")
    assert r["minutes"] > 0 and r["distance_km"] > 0


def test_text2sql_safety():
    t = PropertyDBTool()
    assert t.validate_sql("DROP TABLE properties")[0] is False
    assert t.validate_sql("SELECT * FROM properties; DELETE FROM x")[0] is False
    assert t.validate_sql("SELECT sido FROM properties LIMIT 3")[0] is True


def test_text2sql_query():
    t = PropertyDBTool()
    rows = t.query(dict(lease_type="전세", sido="서울",
                        order_by="fraud_score ASC", limit=5))
    assert len(rows) > 0
    for r in rows:
        assert r["lease_type"] == "전세"


def test_finance_tool():
    t = FinanceTool()
    res = t.search(product_kind="지원", user_income_manwon=300)
    assert len(res) > 0
    assert all("지원" in row["product_kind"] for row in res)


def test_agent_clarification():
    agent = JeonseAgent("rule")
    user = dict(user_id="U", age=29, monthly_income_manwon=280,
                total_asset_manwon=5000, monthly_living_cost_manwon=110,
                income_decile=5, preferred_sido=None)
    s = agent.new_session(user)
    res = agent.handle(s, "좋은 집 살고 싶어")   # 원하지만 불명확 → clarify
    assert res["status"] == "clarify"


def test_agent_commute_and_safety():
    agent = JeonseAgent("rule")
    user = dict(user_id="U", age=29, monthly_income_manwon=320,
                total_asset_manwon=30000, monthly_living_cost_manwon=120,
                income_decile=6, preferred_sido="서울")
    s = agent.new_session(user)
    r1 = agent.handle(s, "IFC몰 근처 안전한 전세")
    assert r1["status"] == "confirm"
    r2 = agent.handle(s, "응")
    assert r2["status"] == "recommendation"
    # 안전 선호(hard) → 추천된 전세 매물 위험도가 낮아야
    for recs in r2["groups"].values():
        for rec in recs:
            if rec["fraud_score"] is not None:
                assert rec["fraud_score"] <= 0.5


def test_agent_finance_intent():
    agent = JeonseAgent("rule")
    user = dict(user_id="U", age=29, monthly_income_manwon=280,
                total_asset_manwon=4000, monthly_living_cost_manwon=110,
                income_decile=5, preferred_sido="서울")
    session = agent.new_session(user)
    res = agent.handle(session, "대출 받아서 더 좋은 집 갈 수 있을까?")
    assert res["status"] == "qa" and res["qa_type"] == "finance"
    assert len(res["programs"]) > 0


def test_financed_jeonse_goal_runs_multi_step_rag_and_ranking():
    agent = JeonseAgent("rule")
    user = dict(user_id="GOAL", age=22, monthly_income_manwon=250,
                total_asset_manwon=6000, monthly_living_cost_manwon=100,
                income_decile=5, preferred_sido=None)
    session = agent.new_session(user)

    result = agent.handle(
        session,
        "금융상품 도움을 받아서라도 최대한 전세가 비싼 집에 살고 싶어. "
        "그 방안을 알려줘. 추천해주고",
    )

    assert result["status"] == "recommendation"
    assert result["recommendation_mode"] == "financed_jeonse_goal"
    tools = [step["tool"] for step in result["agent_trace"]["tools"]]
    assert tools == [
        "finance_text2sql", "financing_budget_calculator",
        "property_text2sql", "goal_ranker",
    ]
    assert result["agent_trace"]["planner"]["qa_args"]["finance_mode"] == "eligibility"
    assert len(result["agent_trace"]["workflow"]["steps"]) == 3

    budget = result["financing_plan"]["estimated_max_deposit_manwon"]
    recommendations = [row for group in result["groups"].values() for row in group]
    deposits = [row["deposit_manwon"] for row in recommendations]
    assert deposits == sorted(deposits, reverse=True)
    assert all(deposit <= budget for deposit in deposits)
    assert all(program.get("goal_role") for program in result["finance_programs"])


def test_finance_goal_budget_excludes_unrelated_loan_products():
    from types import SimpleNamespace
    from src.agent.harness import _classify_goal_finance

    affordability = SimpleNamespace(
        recommended_jeonse_deposit_manwon=5000,
        max_monthly_housing_manwon=50,
    )
    programs = [
        {"program_id": "S", "name": "청약 연계 주택구입대출",
         "category": "청약·연계대출", "product_kind": "청약,대출",
         "max_amount_manwon": 40000, "rate_pct": 2.4},
        {"program_id": "J", "name": "청년 전세자금대출",
         "category": "전세대출", "product_kind": "대출",
         "max_amount_manwon": 20000, "rate_pct": 2.0},
        {"program_id": "F", "name": "전세보증금반환보증 보증료 지원",
         "category": "보증료지원", "product_kind": "지원",
         "max_amount_manwon": 40, "rate_pct": None},
    ]

    relevant, plan = _classify_goal_finance(programs, affordability)

    assert [p["program_id"] for p in relevant] == ["J", "F"]
    assert plan["selected_program_id"] == "J"
    assert plan["direct_loan_limit_manwon"] == 20000
    assert plan["estimated_max_deposit_manwon"] == 25000


def test_compound_preferred_region_is_normalized_and_kept():
    agent = JeonseAgent("rule")
    user = dict(user_id="REGION", age=22, monthly_income_manwon=100,
                total_asset_manwon=800, monthly_living_cost_manwon=70,
                income_decile=10, preferred_sido="대전시 유성구")
    session = agent.new_session(user)
    assert session["user"]["preferred_sido"] == "대전"
    assert session["user"]["preferred_gugun"] == "유성구"
    result = agent.handle(session, "아무거나")
    assert result["status"] == "recommendation"
    recommendations = [row for group in result["groups"].values() for row in group]
    assert recommendations
    assert all(row["sido"] == "대전" and row["gugun"] == "유성구"
               for row in recommendations)


def test_two_stage_confirm_flow():
    agent = JeonseAgent("rule")
    user = dict(user_id="U", age=29, monthly_income_manwon=320,
                total_asset_manwon=9000, monthly_living_cost_manwon=120,
                income_decile=6, preferred_sido="서울")
    s = agent.new_session(user)
    r1 = agent.handle(s, "서울 안전한 전세 5천 이내")
    assert r1["status"] == "confirm"       # 바로 추천 X, 확인 먼저
    r2 = agent.handle(s, "응")
    assert r2["status"] == "recommendation"
    assert 0 in r2["groups"]                # 완전 만족 그룹 존재


def test_anything_goes_no_clarify():
    agent = JeonseAgent("rule")
    user = dict(user_id="U", age=29, monthly_income_manwon=300,
                total_asset_manwon=6000, monthly_living_cost_manwon=120,
                income_decile=5, preferred_sido=None)
    s = agent.new_session(user)
    r = agent.handle(s, "그냥 아무거나 추천해줘")
    # 아무거나 → clarify가 아니라 바로 추천
    assert r["status"] == "recommendation"


def test_vague_triggers_clarify():
    agent = JeonseAgent("rule")
    user = dict(user_id="U", age=29, monthly_income_manwon=300,
                total_asset_manwon=6000, monthly_living_cost_manwon=120,
                income_decile=5, preferred_sido=None)
    s = agent.new_session(user)
    r = agent.handle(s, "음 뭐 그냥")
    assert r["status"] == "clarify"


def test_missing_condition_groups():
    agent = JeonseAgent("rule")
    user = dict(user_id="U", age=29, monthly_income_manwon=320,
                total_asset_manwon=9000, monthly_living_cost_manwon=120,
                income_decile=6, preferred_sido="서울")
    s = agent.new_session(user)
    agent.handle(s, "서울 전세 관리비 5만원 이하 신축")
    r = agent.handle(s, "응")
    assert r["status"] == "recommendation"
    # 누락 조건이 표시되는지(타협 그룹)
    all_recs = [rec for recs in r["groups"].values() for rec in recs]
    assert any("missing_conditions" in rec for rec in all_recs)


def test_safety_tool():
    from src.tools.safety_tool import SafetyTool
    # data/downloaded/safety의 첫 CCTV 좌표: 실제 원본 컬럼/CP949 인코딩까지 검증한다.
    r = SafetyTool().assess(37.5855, 126.9707)
    assert r["safety_score"] is None or 0 <= r["safety_score"] <= 100
    assert set(r["counts"]) >= {"cctv", "police", "fire_station"}
    assert r["counts"]["cctv"] >= 1
    assert r["sources"]["cctv"] == "raw"
    assert r["raw_data"]["emergency_bell"]["status"] == "ready"
    assert r["raw_data"]["police"]["raw_rows"] > 0
    assert r["raw_data"]["fire_station"]["raw_rows"] > 0
    assert not any(str(source).startswith("mock")
                   for source in r["sources"].values())


def test_convenience_tool():
    from src.tools.convenience_tool import ConvenienceTool
    r = ConvenienceTool().assess(37.4784, 126.9516)
    assert r["convenience_score"] is None or 0 <= r["convenience_score"] <= 100
    assert "convenience_24h" in r["counts"]
    assert not any(str(source).startswith("mock")
                   for source in r["sources"].values())


def test_safemap_convenience_api_cache():
    from src.tools.convenience_tool import (
        SafeMapConvenienceClient, _web_mercator_to_wgs84,
    )

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"body": {"totalCount": 1, "items": {"item": [{
                "objt_id": "1", "fclty_nm": "테스트 편의점",
                "rn_adres": "서울 테스트로 1", "x": "14137575.33",
                "y": "4518366.51", "data_yr": "2025",
            }]}}}

    class FakeSession:
        def get(self, *args, **kwargs):
            assert kwargs["params"]["returnType"] == "json"
            return FakeResponse()

    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "stores.csv"
        client = SafeMapConvenienceClient("test-key", cache, FakeSession())
        df = client.refresh(page_size=100)
        lat, lng = _web_mercator_to_wgs84(14137575.33, 4518366.51)
        assert len(df) == 1 and cache.exists()
        assert client.count_nearby(lat, lng, 100) == 1


def test_advisory_tools():
    from src.tools.advisory_tools import contract_checklist, lease_compare, cost_breakdown
    assert len(contract_checklist("전세", True)["recommended_special_terms"]) > 0
    assert "cheaper_option_if_loan" in lease_compare(20000, 2000, 70)
    assert cost_breakdown(2000, 70, 8, 60)["total_monthly_real_cost"] > 0


def test_codef_tool_fallback():
    from src.tools.codef_tool import CodefClient
    reg = CodefClient().real_estate_register("서울 관악구 …")
    assert "senior_mortgage_manwon" in reg
    assert reg["source"].startswith("codef_")
    accounts = CodefClient().personal_accounts()
    assert accounts["total_asset_manwon"] > 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ✓ {name}")
    print("OK: integration tests passed")
