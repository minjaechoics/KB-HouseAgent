"""운영과 같은 실제 API LLM 조건 협상 3턴 스모크 테스트(비용 발생)."""
from src.server.app import (
    ChatIn, ConditionConfirmIn, SessionCreate, _agent, confirm_conditions,
    create_session, draft_conditions,
)


def main() -> None:
    session_id = create_session(SessionCreate(
        age=22, monthly_income_manwon=250, total_asset_manwon=3000,
        monthly_living_cost_manwon=100, income_decile=4,
    ))["session_id"]
    first = draft_conditions(ChatIn(session_id=session_id, text="아주대"))
    second = draft_conditions(ChatIn(
        session_id=session_id, text="대중교통으로 가까운 곳",
    ))
    third = confirm_conditions(ConditionConfirmIn(session_id=session_id))
    print("TURN1", first["status"], first["message"],
          first.get("trace", {}).get("planner", {}).get("strategy"))
    print("TURN2", second["status"], second["message"],
          second.get("trace", {}).get("planner", {}).get("strategy"))
    print("TURN3", third["status"], "atoms", len(third.get("active_atoms", [])),
          "stage", third.get("trace", {}).get("stage"))
    print("TOOLS", [(item.get("tool"), item.get("status"))
                    for item in third.get("trace", {}).get("tools", [])])
    sql_trace = third.get("trace", {}).get("text2sql", {})
    print("SQL", sql_trace.get("strategy"), sql_trace.get("validation"),
          "fallback", sql_trace.get("fallback"))
    print("LLM", type(_agent.llm).__name__, getattr(_agent.llm, "model", None))
    assert first["status"] == "ask_clarification", first
    assert second["status"] == "ask_confirmation", second
    assert third["status"] == "applied" and len(third["active_atoms"]) == 1, third
    assert str(sql_trace.get("validation", "")).startswith("passed"), sql_trace


if __name__ == "__main__":
    main()
