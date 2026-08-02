from __future__ import annotations

from src.experiments.advisor_cases import ADVISOR_CASES, category_counts
from src.experiments.advisor_hallucination import (
    _answer_conclusion_grounding,
    _answer_numeric_grounding,
    _stratified_cases,
)
from src.experiments.naive_baseline import (
    NAIVE_WHOLE_PROMPT_SYSTEM_PROMPT,
    NaiveWholePromptLLM,
)
from src.agent.llm import BaseLLM
from src.agent.planner import Plan


class _OneShotDelegate(BaseLLM):
    supports_agentic_calls = True

    def __init__(self):
        super().__init__()
        self.calls = []

    def plan(self, text, has_prior_region=False, conversation_history=None):
        raise AssertionError("NAIVE live baseline must call analyze_json directly")

    def analyze_json(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "intent": "qa_lease_compare", "action": "proceed",
            "clarify_message": None,
            "slots": {}, "tool_calls": [], "qa_args": {},
        }


def test_advisor_benchmark_has_100_unique_queries_and_required_distribution():
    assert len(ADVISOR_CASES) == 100
    assert len({case.query for case in ADVISOR_CASES}) == 100
    assert category_counts() == {
        "condition_dialogue": 15,
        "best_affordable": 10,
        "lease_compare": 8,
        "market_outlook": 7,
        "buy_or_wait": 5,
        "alternative_areas": 5,
        "condition_new_atoms": 16,
        "qa_finance": 12,
        "qa_safety": 8,
        "qa_convenience": 6,
        "qa_affordability": 8,
    }
    queries = {case.query for case in ADVISOR_CASES}
    assert "특별한 투자처는 없고 이 집에서 3년 정도 살 계획인데, 전세가 좋을까 월세가 좋을까?" in queries
    assert "수원에서 내 예산과 대출로 제일 좋은 집이 뭐야?" in queries
    assert "이 동네 집값 앞으로 오를까 내릴까?" in queries
    assert "지금 사는 게 나을까, 1~2년 기다리는 게 나을까?" in queries
    assert "여기 말고 예산 맞는 다른 동네도 있을까?" in queries


def test_small_run_is_stratified_across_all_six_advisor_categories():
    selected = _stratified_cases(6)
    assert len({case.category for case in selected}) == 6


def test_unsupported_number_in_final_llm_answer_is_detected():
    case = next(case for case in ADVISOR_CASES
                if case.category == "market_outlook")
    result = {
        "answer": "근거에는 없는 999999만원 상승을 예상합니다.",
        "market_outlook": {"annual_growth_rate": 0.03},
        "agent_trace": {
            "synthesis": {"strategy": "llm_grounded", "ok": True},
        },
    }
    grounding = _answer_numeric_grounding(case, result, live=True)
    assert grounding["passed"] is False
    assert grounding["unsupported_numbers"] == [999999.0]


def test_visible_market_conclusion_must_match_structured_direction():
    case = next(case for case in ADVISOR_CASES
                if case.category == "market_outlook")
    result = {
        "answer": "이 지역 가격은 앞으로 하락할 가능성이 가장 큽니다.",
        "market_outlook": {"direction": "상승"},
    }
    grounding = _answer_conclusion_grounding(case, result, live=True)
    assert grounding["passed"] is False
    assert grounding["expected"] == "상승"


def test_naive_advisor_uses_exactly_one_flat_whole_prompt_call():
    delegate = _OneShotDelegate()
    naive = NaiveWholePromptLLM(delegate, fixed_context={"age": 29})
    plan = naive.plan("전세가 좋을까 월세가 좋을까?")
    assert plan.intent == "qa_lease_compare"
    assert len(delegate.calls) == 1
    assert delegate.calls[0]["system"] == NAIVE_WHOLE_PROMPT_SYSTEM_PROMPT
    assert "전세가 좋을까 월세가 좋을까?" in delegate.calls[0]["user"]
    assert naive.supports_agentic_calls is False
