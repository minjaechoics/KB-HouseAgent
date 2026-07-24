"""불확실성 질문→제안→승인→도구/SQL 초안 계약 회귀 테스트."""
from __future__ import annotations

from src.agent.llm import BaseLLM, MockLLM, _repair_condition_decision
from src.agent.prompts import (
    CONDITION_DECISION_JSON_SCHEMA, CONDITION_DIALOGUE_SYSTEM_PROMPT,
)
from src.agent.text2sql import Text2SQLPipeline
from src.server.property_search import atoms_from_slots
from src.tools.finance_tool import FinanceTool
from src.tools.map_tool import MapTool
from src.tools.property_db_tool import PropertyDBTool


class SQLPreviewLLM(BaseLLM):
    supports_agentic_calls = True

    def plan(self, text: str, has_prior_region: bool = False):
        return MockLLM().plan(text, has_prior_region)

    def generate_sql(self, request: str, schema: str,
                     previous_error: str | None = None) -> dict:
        return {
            "sql": (
                "SELECT property_id, is_synthetic, synthetic_notice, sido, gugun, "
                "lat, lng, lease_type, deposit_manwon, monthly_rent_manwon, "
                "maintenance_fee_manwon, market_price_manwon, my_priority_rank, "
                "building_total_units, fraud_score FROM properties "
                "WHERE transaction_type = '월세' AND monthly_rent_manwon <= 60 "
                "LIMIT 500"
            ),
            "purpose": "확인된 월세 상한 조건을 사전 컴파일",
        }


def test_prompt_has_explicit_uncertainty_and_confirmation_contract():
    assert "ask_clarification" in CONDITION_DIALOGUE_SYSTEM_PROMPT
    assert "ask_confirmation" in CONDITION_DIALOGUE_SYSTEM_PROMPT
    assert "사용자 버튼 승인 전에는" in CONDITION_DIALOGUE_SYSTEM_PROMPT
    assert "조건 추가" in CONDITION_DIALOGUE_SYSTEM_PROMPT
    assert "아주대" in CONDITION_DIALOGUE_SYSTEM_PROMPT
    assert CONDITION_DECISION_JSON_SCHEMA["additionalProperties"] is False


def test_api_semantic_guard_normalizes_alias_and_proposes_missing_time():
    raw = {
        "decision": "ask_clarification",
        "message": "시간을 알려주세요.",
        "goal_summary": "아주대 주변",
        "known_facts": ["목적지 아주대", "대중교통"],
        "uncertainties": [
            {"field": "max_commute_min", "description": "시간", "blocking": True},
        ],
        "slots": {"workplace_landmark": "아주대", "commute_mode": "transit"},
        "proposed_defaults": [], "tool_plan": [], "confidence": 0.8,
        "decision_reason": "시간이 없음",
    }
    repaired, repairs = _repair_condition_decision(
        raw, text="대중교통으로 가까운 곳",
        context={"known_slots": {"workplace_landmark": "아주대"}},
    )
    assert repaired["decision"] == "ask_confirmation"
    assert repaired["slots"]["workplace_landmark"] == "아주대학교"
    assert repaired["slots"]["max_commute_min"] == 20.0
    assert "proposed_missing_commute_time_default" in repairs
    assert set(CONDITION_DECISION_JSON_SCHEMA["required"]) == set(
        CONDITION_DECISION_JSON_SCHEMA["properties"]
    )


def test_landmark_only_requires_question_then_default_confirmation():
    llm = MockLLM()
    first = llm.plan_condition_dialogue(
        "아주대", {"state": "idle", "known_slots": {}, "proposed_slots": {}}
    )
    assert first["decision"] == "ask_clarification"
    assert first["slots"]["workplace_landmark"] == "아주대학교"
    assert {item["field"] for item in first["uncertainties"]} >= {
        "commute_mode", "max_commute_min",
    }

    second = llm.plan_condition_dialogue("대중교통으로 가까운 곳", {
        "state": "awaiting_clarification",
        "known_slots": first["slots"], "proposed_slots": {},
    })
    assert second["decision"] == "ask_confirmation"
    assert second["slots"]["commute_mode"] == "transit"
    assert second["slots"]["max_commute_min"] == 20
    assert second["proposed_defaults"][0]["field"] == "max_commute_min"

    third = llm.plan_condition_dialogue("응", {
        "state": "awaiting_confirmation", "known_slots": second["slots"],
        "proposed_slots": second["slots"],
    })
    assert third["decision"] == "ask_confirmation"
    assert third["tool_plan"] == []
    assert "조건 추가" in third["message"]


def test_confirmed_landmark_builds_map_atom_with_selected_mode():
    atoms, notes = atoms_from_slots({
        "workplace_landmark": "아주대학교", "commute_mode": "transit",
        "max_commute_min": 20,
    }, "아주대 대중교통", MapTool())
    commute = next(atom for atom in atoms if atom["field"] == "commute_minutes")
    assert commute["mode"] == "transit"
    assert commute["value"] == 20
    assert commute["geocode_source"] == "audited_landmark_catalog"
    assert notes


def test_confirmed_slots_are_compiled_by_text2sql_without_execution():
    llm = SQLPreviewLLM()
    pipeline = Text2SQLPipeline(llm, PropertyDBTool(), FinanceTool())
    trace = pipeline.compile_property_filter(
        "월세 60만원 이하", {"transaction_type": "월세",
                            "max_monthly_rent_manwon": 60},
    )
    assert trace["strategy"] == "llm_text2sql"
    assert trace["validation"] == "passed_readonly_validation_not_executed"
    assert "monthly_rent_manwon <= 60" in trace["final_sql"]
    assert trace["row_count"] == 0


def test_safety_language_becomes_sort_only_never_a_where_threshold():
    decision = MockLLM().plan_condition_dialogue(
        "안전한 전세부터 보고 싶어", {"state": "idle", "known_slots": {}},
    )
    assert decision["decision"] == "ask_confirmation"
    assert decision["slots"]["sort_by"] == "risk_asc"
    assert "max_fraud_score" not in decision["slots"]

    repaired, repairs = _repair_condition_decision({
        "decision": "ask_confirmation", "message": "안전 조건을 적용할까요?",
        "slots": {"max_fraud_score": 0.05, "safety_is_hard": True},
    }, text="위험이 낮은 집", context={})
    assert repaired["slots"] == {"sort_by": "risk_asc"}
    assert "removed_risk_where_threshold" in repairs


if __name__ == "__main__":
    test_prompt_has_explicit_uncertainty_and_confirmation_contract()
    test_api_semantic_guard_normalizes_alias_and_proposes_missing_time()
    test_landmark_only_requires_question_then_default_confirmation()
    test_confirmed_landmark_builds_map_atom_with_selected_mode()
    test_confirmed_slots_are_compiled_by_text2sql_without_execution()
    print("OK: condition agentic workflow tests passed")
