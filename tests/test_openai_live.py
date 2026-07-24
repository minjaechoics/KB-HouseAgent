"""실제 OpenAI Planner/Text-to-SQL/합성까지 확인하는 선택적 live 테스트.

CMD::
    set RUN_LIVE_OPENAI_TEST=1 && set JEONSE_LLM=api && py -3 -m tests.test_openai_live
"""
from __future__ import annotations

import os

from src.agent.harness import JeonseAgent
from src.agent.llm import APILLM


def _enabled() -> bool:
    return os.environ.get("RUN_LIVE_OPENAI_TEST") == "1"


def _user(user_id: str, income: float = 320) -> dict:
    return {
        "user_id": user_id, "age": 29,
        "monthly_income_manwon": income, "total_asset_manwon": 8000,
        "monthly_living_cost_manwon": 120, "income_decile": 6,
        "preferred_sido": None,
    }


def test_openai_live_plan():
    if not _enabled():
        print("SKIP: RUN_LIVE_OPENAI_TEST=1일 때만 실제 OpenAI API를 호출합니다.")
        return
    agent = JeonseAgent("rule")
    assert isinstance(agent.llm, APILLM)
    session = agent.new_session(_user("LIVE-PROPERTY"))
    planned = agent.handle(
        session, "서울에서 안전한 전세를 보증금 5천만원 이내로 찾아줘")
    assert planned.get("reason") == "api" and planned["status"] == "confirm"
    slots = session["pending_slots"]
    assert slots.get("lease_type") == "전세"
    assert slots.get("region_sido") == "서울"
    assert float(slots.get("max_deposit_manwon", 0)) == 5000.0

    executed = agent.handle(session, "yes")
    assert executed["status"] in {"recommendation", "no_result"}
    sql_steps = [t for t in executed["agent_trace"]["tools"]
                 if t.get("tool") == "property_text2sql"]
    assert sql_steps and (sql_steps[0].get("final_sql") or sql_steps[0].get("fallback"))
    assert executed.get("answer")
    print({"test": "property", "status": executed["status"],
           "sql_strategy": sql_steps[0].get("strategy"),
           "sql_fallback": sql_steps[0].get("fallback"),
           "synthesis_ok": executed["agent_trace"].get("synthesis", {}).get("ok")})


def test_openai_live_finance_rate_where():
    if not _enabled():
        return
    agent = JeonseAgent("rule")
    assert isinstance(agent.llm, APILLM)
    session = agent.new_session(_user("LIVE-FINANCE", income=300))
    response = agent.handle(
        session, "금리 2% 미만 대출로 받고 싶은데 나한테 해당하는거 있나?")
    sql_step = next(t for t in response["agent_trace"]["tools"]
                    if t.get("tool") == "finance_text2sql")
    assert sql_step["fallback"] is False
    sql = sql_step["final_sql"].lower()
    assert "where" in sql and "rate_pct" in sql and "product_kind" in sql
    assert all(float(p["rate_pct"]) < 2 and "대출" in p["product_kind"]
               for p in response["programs"])
    print({"test": "finance", "programs": len(response["programs"]),
           "sql": sql_step["final_sql"], "fallback": sql_step["fallback"]})


def test_openai_live_finance_catalog_semantics():
    if not _enabled():
        return
    agent = JeonseAgent("rule")
    session = agent.new_session(_user("LIVE-FINANCE-CATALOG", income=9999))
    response = agent.handle(session, "금융지원책 뭐가 있지")
    sql_step = next(t for t in response["agent_trace"]["tools"]
                    if t.get("tool") == "finance_text2sql")
    assert response["finance_mode"] == "catalog"
    assert len(response["programs"]) == 6
    low_sql = sql_step["final_sql"].lower()
    where_clause = low_sql.split("where", 1)[1] if "where" in low_sql else ""
    assert "income_limit_manwon" not in where_clause
    assert sql_step["fallback"] is False
    print({"test": "finance_catalog", "programs": len(response["programs"]),
           "sql": sql_step["final_sql"]})


def test_openai_live_financed_jeonse_goal():
    if not _enabled():
        return
    agent = JeonseAgent("rule")
    session = agent.new_session(_user("LIVE-FINANCED-GOAL", income=250))
    response = agent.handle(
        session,
        "금융상품 도움을 받아서라도 최대한 전세가 비싼 집에 살고 싶어. "
        "그 방안을 알려줘. 추천해주고",
    )
    assert response["recommendation_mode"] == "financed_jeonse_goal"
    assert response.get("answer")
    tools = response["agent_trace"]["tools"]
    assert [step["tool"] for step in tools] == [
        "finance_text2sql", "financing_budget_calculator",
        "property_text2sql", "goal_ranker",
    ]
    assert tools[0]["fallback"] is False
    assert tools[2]["fallback"] is False
    assert response["agent_trace"]["synthesis"]["ok"] is True
    print({"test": "financed_goal",
           "budget": response["financing_plan"]["estimated_max_deposit_manwon"],
           "properties": sum(len(v) for v in response["groups"].values()),
           "finance_sql_fallback": tools[0]["fallback"],
           "property_sql_fallback": tools[2]["fallback"]})


if __name__ == "__main__":
    test_openai_live_plan()
    test_openai_live_finance_rate_where()
    test_openai_live_finance_catalog_semantics()
    test_openai_live_financed_jeonse_goal()
    print("OK: OpenAI live integration tests passed")
