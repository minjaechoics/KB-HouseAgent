"""Agentic 계획/Text-to-SQL/재시도/폴백 회귀 테스트."""
from __future__ import annotations

import os

os.environ.setdefault("JEONSE_LLM", "mock")

from src.agent.llm import APILLM, BaseLLM
from src.agent.planner import Plan, Planner
from src.agent.reliability import RetryPolicy, call_with_retry
from src.agent.text2sql import Text2SQLPipeline
from src.tools.finance_tool import FinanceTool
from src.tools.property_db_tool import PropertyDBTool


class RepairingSQLLLM(BaseLLM):
    supports_agentic_calls = True

    def __init__(self, final_sql: str):
        super().__init__()
        self.calls = 0
        self.final_sql = final_sql

    def plan(self, text: str, has_prior_region: bool = False) -> Plan:
        return Plan()

    def generate_sql(self, request: str, schema: str,
                     previous_error: str | None = None) -> dict | None:
        self.calls += 1
        if self.calls == 1:
            return {"sql": "DROP TABLE properties", "purpose": "invalid"}
        assert previous_error and "SELECT" in previous_error
        return {"sql": self.final_sql, "purpose": "repaired"}


def test_rule_plan_supports_sale_and_house_type():
    plan = Planner().plan("서울 아파트를 5억 이하로 매매하고 싶어")
    assert plan.slots["lease_type"] == "매매"
    assert plan.slots["transaction_type"] == "매매"
    assert plan.slots["property_type"] == "아파트"
    assert plan.slots["max_sale_price_manwon"] == 50000
    assert any(c["tool"] == "property_search" for c in plan.tool_calls)


def test_rule_plan_routes_decision_support_questions():
    planner = Planner()
    assert planner.plan(
        "수원에서 내 예산과 대출로 제일 좋은 집은?"
    ).intent == "goal_best_affordable"
    assert planner.plan(
        "여기 말고 예산 맞는 다른 동네도 있을까?"
    ).intent == "goal_alternative_areas"
    assert planner.plan(
        "전세가 좋을까 월세가 좋을까?"
    ).intent == "qa_lease_compare"
    assert planner.plan(
        "이 동네 집값 앞으로 오를까 내릴까?"
    ).intent == "qa_market"
    assert planner.plan(
        "지금 사는 게 나을까 1~2년 기다리는 게 나을까?"
    ).intent == "qa_buy_or_wait"


def test_sql_validator_rejects_prompt_injection_shapes():
    tool = PropertyDBTool()
    bad = [
        "SELECT * FROM properties; DROP TABLE properties",
        "SELECT * FROM properties -- ignore limits",
        "SELECT load_extension('evil') FROM properties",
        "SELECT * FROM sqlite_master",
        "PRAGMA table_info(properties)",
    ]
    assert all(not tool.validate_sql(sql)[0] for sql in bad)


def test_schema_prompt_is_live_and_target_scoped():
    tool = PropertyDBTool()
    property_schema = tool.schema_prompt({"properties"})
    assert "테이블 properties" in property_schema
    assert "finance_programs" not in property_schema
    assert "transaction_type" in property_schema
    finance_schema = tool.schema_prompt({"finance_programs"})
    assert "테이블 finance_programs" in finance_schema
    assert "properties" not in finance_schema
    assert "product_kind" in finance_schema


def test_text2sql_repairs_then_executes():
    db = PropertyDBTool()
    sql = db.build_query({"lease_type": "전세", "limit": 3})
    llm = RepairingSQLLLM(sql)
    pipeline = Text2SQLPipeline(llm, db, FinanceTool())
    rows, trace = pipeline.search_properties("전세 세 건", {"lease_type": "전세"}, 3)
    assert len(rows) == 3
    assert llm.calls == 2
    assert trace["fallback"] is False
    assert trace["attempts"][0]["ok"] is False
    assert trace["attempts"][1]["ok"] is True


def test_text2sql_rule_fallback_is_auditable():
    from src.agent.llm import MockLLM
    db = PropertyDBTool()
    pipeline = Text2SQLPipeline(MockLLM(), db, FinanceTool())
    rows, trace = pipeline.search_properties("서울 전세", {
        "lease_type": "전세", "sido": "서울",
    }, 2)
    assert len(rows) <= 2
    assert trace["fallback"] is True
    assert trace["strategy"] == "deterministic_slots"
    assert trace["final_sql"].startswith("SELECT")


def test_llm_select_star_with_zero_rows_is_not_false_fallback():
    """정상적인 0건 SELECT *를 필수 컬럼 누락으로 오판하지 않아야 한다."""
    sql = (
        "SELECT * FROM finance_programs "
        "WHERE product_kind LIKE '%대출%' AND rate_pct < 0 LIMIT 10"
    )
    llm = RepairingSQLLLM(sql)
    pipeline = Text2SQLPipeline(llm, PropertyDBTool(), FinanceTool())

    rows, trace = pipeline.search_finance(
        "금리 0% 미만 대출", {"age": 29, "monthly_income_manwon": 300},
        max_rate_pct=0, product_kind="대출",
    )

    assert rows == []
    assert trace["fallback"] is False
    assert trace["validation"] == "passed_readonly_execution"


def test_retry_pipeline_on_429():
    class RateLimit(Exception):
        status_code = 429

    state = {"n": 0}

    def flaky():
        state["n"] += 1
        if state["n"] < 3:
            raise RateLimit("slow down")
        return "ok"

    result, events = call_with_retry(
        "test.retry", flaky,
        policy=RetryPolicy(max_attempts=3, base_delay_seconds=0,
                           max_delay_seconds=0, jitter_ratio=0),
        sleep=lambda _: None,
    )
    assert result == "ok" and len(events) == 3
    assert [e.ok for e in events] == [False, False, True]


def test_finance_fallback_trace_preserves_rate_and_product_filters():
    from src.agent.llm import MockLLM
    pipeline = Text2SQLPipeline(MockLLM(), PropertyDBTool(), FinanceTool())
    rows, trace = pipeline.search_finance(
        "금리 2% 미만 대출", {"age": 29, "monthly_income_manwon": 300},
        max_rate_pct=2, product_kind="대출")
    assert trace["strategy"] == "parameterized_finance_query"
    assert trace["fallback"] is True
    assert "rate_pct < ?" in trace["final_sql"]
    assert "product_kind LIKE ?" in trace["final_sql"]
    assert all(float(row["rate_pct"]) < 2 and "대출" in row["product_kind"] for row in rows)
    assert trace["row_count"] == len(rows)
    # 첨부 정책 중 대출 연계 최저금리는 2.4%이므로 올바른 결과는 0건이다.
    assert rows == []


def test_finance_natural_language_modes_are_semantic():
    planner = Planner()
    catalog = planner.plan("금융지원책 뭐가 있지")
    assert catalog.intent == "qa_finance"
    assert catalog.qa_args["finance_mode"] == "catalog"
    assert "product_kind" not in catalog.qa_args

    eligible = planner.plan("내 조건으로 받을 수 있는 금융상품이 뭐야?")
    assert eligible.intent == "qa_finance"
    assert eligible.qa_args["finance_mode"] == "eligibility"


def test_api_plan_preserves_explicit_finance_constraints():
    """LLM 계획이 빼먹은 명시적 금리·상품 조건은 안전하게 복원한다."""
    llm = object.__new__(APILLM)
    llm.provider = "test"
    llm.model = "test-model"
    llm.fallback = Planner()
    llm.last_trace = []
    llm._request_json = lambda **_: {
        "intent": "qa_finance", "action": "proceed", "clarify_message": None,
        "slots": {}, "tool_calls": [],
        "qa_args": {"finance_mode": "catalog"},
    }

    plan = llm.plan("금리 2% 미만 대출로 받고 싶은데 나한테 해당하는 거 있나?")

    assert plan.qa_args["max_rate_pct"] == 2.0
    assert plan.qa_args["product_kind"] == "대출"
    assert plan.qa_args["finance_mode"] == "eligibility"
    repair = plan.metadata["semantic_repair"]
    assert repair and "max_rate_pct" in repair[0]["fields"]


def test_api_plan_repairs_compound_finance_goal_misclassification():
    """LLM이 단순 금융 QA로 보더라도 명시적 목표는 다단계 계획으로 복원한다."""
    llm = object.__new__(APILLM)
    llm.provider = "test"
    llm.model = "test-model"
    llm.fallback = Planner()
    llm.last_trace = []
    llm._request_json = lambda **_: {
        "intent": "qa_finance", "action": "proceed", "clarify_message": None,
        "slots": {}, "tool_calls": [{"tool": "finance_search"}],
        "qa_args": {"finance_mode": "catalog"},
    }

    plan = llm.plan("금융상품을 활용해 최대한 비싼 전세를 찾아서 추천해줘")

    assert plan.intent == "goal_financed_jeonse"
    assert plan.qa_args["finance_mode"] == "eligibility"
    assert [call["tool"] for call in plan.tool_calls] == [
        "finance_search", "property_search",
    ]


def test_api_plan_repairs_best_affordable_goal_misclassification():
    llm = object.__new__(APILLM)
    llm.provider = "test"
    llm.model = "test-model"
    llm.fallback = Planner()
    llm.last_trace = []
    llm._request_json = lambda **_: {
        "intent": "recommend", "action": "confirm", "clarify_message": None,
        "slots": {}, "tool_calls": [{"tool": "property_search"}],
        "qa_args": {"finance_mode": None},
    }
    plan = llm.plan("수원에서 내 예산과 대출로 제일 좋은 집은?")
    assert plan.intent == "goal_best_affordable"
    assert plan.action == "proceed"
    assert [call["tool"] for call in plan.tool_calls] == [
        "finance_search", "property_search",
    ]


def test_api_planner_and_synthesis_receive_full_advisor_history():
    llm = object.__new__(APILLM)
    llm.provider = "test"
    llm.model = "test-model"
    llm.fallback = Planner()
    llm.last_trace = []
    captured = {}

    def fake_plan_request(**kwargs):
        captured["plan_user"] = kwargs["user"]
        return {
            "intent": "recommend", "action": "confirm",
            "clarify_message": None,
            "slots": {"lease_type": "월세", "transaction_type": "월세",
                      "sort_by": "price_asc"},
            "tool_calls": [{"tool": "property_search"}],
            "qa_args": {},
        }

    llm._request_json = fake_plan_request
    llm._request_text = lambda **kwargs: (
        captured.update(synthesis_user=kwargs["user"]) or "후속 답변"
    )
    history = [
        {"role": "user", "text": "월세만 찾아줘"},
        {"role": "assistant", "text": "월세 후보를 찾았어요."},
    ]
    plan = llm.plan(
        "그중 보증금이 가장 낮은 곳", conversation_history=history)
    answer = llm.synthesize(
        "그중 보증금이 가장 낮은 곳",
        {"status": "recommendation"}, conversation_history=history)

    assert plan.slots["sort_by"] == "price_asc"
    assert "<conversation_history>" in captured["plan_user"]
    assert "월세만 찾아줘" in captured["plan_user"]
    assert "<latest_user_message>그중 보증금이 가장 낮은 곳" in captured["plan_user"]
    assert "월세 후보를 찾았어요." in captured["synthesis_user"]
    assert answer == "후속 답변"


def test_finance_catalog_does_not_apply_personal_income_filter():
    from src.agent.llm import MockLLM
    pipeline = Text2SQLPipeline(MockLLM(), PropertyDBTool(), FinanceTool())
    rows, trace = pipeline.search_finance(
        "금융지원책 뭐가 있지", {"age": 29, "monthly_income_manwon": 9999},
        finance_mode="catalog")
    assert len(rows) == 10  # 정책+은행상품 통합 DB의 기본 조회 상한
    assert any(row["name"] == "청년주택드림청약통장" for row in rows)
    assert "income_limit_manwon" not in trace["final_sql"]
    assert trace["input_filters"]["finance_mode"] == "catalog"


def test_finance_followup_uses_conversation_context():
    import os
    os.environ["JEONSE_LLM"] = "mock"
    from src.agent.harness import JeonseAgent
    user = {"user_id": "FOLLOW", "age": 29, "monthly_income_manwon": 300,
            "total_asset_manwon": 6000, "monthly_living_cost_manwon": 120,
            "income_decile": 5, "preferred_sido": None}
    agent = JeonseAgent("rule")
    session = agent.new_session(user)
    first = agent.handle(session, "금융지원책 뭐가 있지")
    assert len(first["programs"]) == 10
    assert any(p["name"] == "청년주택드림청약통장" for p in first["programs"])
    second = agent.handle(session, "그중 대출만 보여줘")
    assert second["programs"]
    assert all("대출" in str(p.get("category") or "") or
               "대출" in str(p.get("product_kind") or "")
               for p in second["programs"])
    assert second["agent_trace"]["planner"]["llm"].get("conversation_context_used")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ✓ {name}")
    print("OK: agentic pipeline tests passed")
