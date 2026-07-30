from __future__ import annotations

import time

from src.agent.llm import BaseLLM
from src.agent.planner import Plan
from src.experiments.cases import CASES
from src.experiments.console import _safe_console_text, format_elapsed
from src.experiments.pipeline import (
    build_ground_truth,
    run_case,
    score_result,
)
from src.experiments.runner import (
    _assert_live_result_valid,
    _effective_workers,
    _execute_speed_cases,
    _observed_peak_concurrency,
)


class _AnswerKeyLLM(BaseLLM):
    """Test double only: isolates retrieval and scoring from paid API calls."""

    supports_agentic_calls = True

    def __init__(self):
        super().__init__()
        self.condition_calls = []
        self.one_shot_calls = 0

    def _slots(self, text):
        return next(case.expected_slots for case in CASES if case.query in text)

    def plan(self, text: str, has_prior_region: bool = False,
             conversation_history=None) -> Plan:
        return Plan(intent="recommend", slots=dict(self._slots(text)),
                    tool_calls=[], action="confirm", reason="test double")

    def plan_condition_dialogue(self, text: str, context: dict) -> dict:
        self.condition_calls.append((text, context))
        return {
            "decision": "ask_confirmation",
            "message": "조건을 추가할까요?",
            "goal_summary": "테스트",
            "known_facts": [],
            "uncertainties": [],
            "slots": dict(self._slots(text)),
            "proposed_defaults": [],
            "tool_plan": [],
            "confidence": 1.0,
            "decision_reason": "test double",
        }

    def analyze_json(self, *, operation, system, user, schema, schema_name,
                     max_tokens=1200):
        self.one_shot_calls += 1
        return {
            "intent": "recommend",
            "action": "confirm",
            "clarify_message": None,
            "slots": dict(self._slots(user)),
            "tool_calls": [],
            "qa_args": {},
        }


class _LiveMarker:
    supports_agentic_calls = True


def test_shared_benchmark_has_exactly_50_unique_complex_queries():
    assert len(CASES) == 50
    assert len({case.case_id for case in CASES}) == 50
    assert len({case.query for case in CASES}) == 50
    assert all("조건" in case.query or "규칙" in case.query or "논리식" in case.query
               for case in CASES)


def test_all_ground_truth_queries_have_real_database_matches():
    assert all(build_ground_truth(case)["matching_property_count"] > 0
               for case in CASES)


def test_optimized_pipeline_is_grounded_for_answer_key_slots():
    case = CASES[0]
    truth = build_ground_truth(case)
    run = run_case(case, "optimized", _AnswerKeyLLM())
    score = score_result(case, run, truth)
    assert run["error"] is None
    assert run["rag_trace"]["retrieval"]["condition_scheduler"]
    assert score["correct"] is True


def test_naive_pipeline_uses_one_serial_sql_and_can_be_scored():
    case = CASES[7]
    truth = build_ground_truth(case)
    run = run_case(case, "naive", _AnswerKeyLLM())
    trace = run["rag_trace"]["retrieval"]
    assert trace["atomic_processing"] is False
    assert trace["parallel_processing"] is False
    assert trace["dependency_scheduling"] is False
    assert score_result(case, run, truth)["correct"] is True


def test_naive_uses_a_distinct_whole_prompt_instead_of_agentic_entrypoint():
    case = CASES[3]
    llm = _AnswerKeyLLM()
    optimized = run_case(case, "optimized", llm)
    naive = run_case(case, "naive", llm)
    assert optimized["error"] is None
    assert naive["error"] is None
    assert len(llm.condition_calls) == 1
    assert llm.one_shot_calls == 1
    assert optimized["rag_trace"]["planner_prompt_policy"] == (
        "identical_production_agentic_prompt"
    )
    assert naive["rag_trace"]["planner_prompt_policy"] == (
        "naive_whole_prompt_single_pass"
    )
    assert naive["llm_output"]["_trace"]["llm_call_count"] == 1


def test_missing_condition_is_not_counted_as_correct():
    case = CASES[0]
    truth = build_ground_truth(case)
    run = run_case(case, "naive", _AnswerKeyLLM())
    run["parsed_slots"].pop("max_monthly_rent_manwon")
    score = score_result(case, run, truth)
    assert score["correct"] is False
    assert "condition_extraction_mismatch" in score["reasons"]


def test_elapsed_format_has_millisecond_precision():
    assert format_elapsed(3661.2349) == "01:01:01:234"


def test_cp949_console_replaces_unsupported_math_symbol(monkeypatch):
    class _Stdout:
        encoding = "cp949"

    monkeypatch.setattr("src.experiments.console.sys.stdout", _Stdout())
    rendered = _safe_console_text("논리식 ¬A")
    assert rendered.startswith("논리식 ")
    assert rendered.encode("cp949")


def test_optimized_queries_really_overlap_but_naive_is_forced_serial():
    llm = _AnswerKeyLLM()
    started = time.perf_counter()
    optimized = [
        result for _, result in _execute_speed_cases(
            CASES[:4], "optimized", llm, 4, started)
    ]
    assert _effective_workers("optimized", 4, 4) == 4
    assert _observed_peak_concurrency(optimized) >= 2

    started = time.perf_counter()
    naive = [
        result for _, result in _execute_speed_cases(
            CASES[:3], "naive", llm, 1, started)
    ]
    assert _effective_workers("naive", 99, 3) == 1
    assert _observed_peak_concurrency(naive) == 1


def test_live_experiment_rejects_silent_rule_fallback():
    result = {
        "error": None,
        "llm_output": {"fallback": True, "fallback_error": "401 invalid_api_key"},
    }
    try:
        _assert_live_result_valid(result, _LiveMarker())
    except RuntimeError as exc:
        assert "invalid live experiment result" in str(exc)
    else:
        raise AssertionError("live rule fallback must invalidate the experiment")
