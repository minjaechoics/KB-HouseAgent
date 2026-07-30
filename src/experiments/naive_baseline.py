"""Single-pass whole-prompt baseline used by the console experiments.

The baseline intentionally does not reuse the production orchestration prompt.
It sends the complete user message to one flat structured-output request, then
hands the extracted plan to deterministic serial tools.  It never performs
atomic prompt decomposition, dependency scheduling, Text-to-SQL generation, or
LLM answer synthesis after that first request.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from src.agent.llm import BaseLLM
from src.agent.planner import Plan
from src.agent.prompts import PLAN_JSON_SCHEMA


NAIVE_WHOLE_PROMPT_SYSTEM_PROMPT = """너는 단순 비교 기준선(baseline) 추출기다.
사용자의 질문 전체를 한 번에 읽고, 추가 질문이나 단계별 추론 없이 검색·상담 계획을
평면적인 JSON 하나로 즉시 추출하라. 조건을 원자 단위 프롬프트로 나누지 말고,
의존성 그래프·스케줄러·도구 재계획·재검색을 사용하지 않는다.

출력 필드의 의미:
- intent: 사용자의 주된 요청 하나
- action: proceed, clarify, confirm 중 하나
- slots: 질문에 명시된 매물 조건만 추출
- tool_calls: 필요해 보이는 도구 이름만 평면 배열로 기록
- qa_args: 금융·Q&A에 직접 명시된 인자

금액은 만원, 면적은 ㎡, 시간은 분이다. 매매/전세/월세와 지역·주택유형·가격·
면적·연식·정렬조건을 사용자가 말한 그대로 보존한다. 사실, 조건, 매물, 금융상품을
만들어내지 않는다. 설명이나 Markdown 없이 주어진 스키마의 JSON 객체 하나만 반환하라."""


def naive_prompt_fingerprint() -> str:
    payload = (
        NAIVE_WHOLE_PROMPT_SYSTEM_PROMPT
        + json.dumps(PLAN_JSON_SCHEMA, ensure_ascii=False, sort_keys=True)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _plan_payload(plan: Plan) -> dict[str, Any]:
    return {
        "intent": plan.intent,
        "action": plan.action,
        "clarify_message": plan.clarify_message,
        "slots": copy.deepcopy(plan.slots),
        "tool_calls": copy.deepcopy(plan.tool_calls),
        "qa_args": copy.deepcopy(plan.qa_args),
    }


def _normalize_plan(data: dict[str, Any], metadata: dict[str, Any]) -> Plan:
    allowed_intents = set(PLAN_JSON_SCHEMA["properties"]["intent"]["enum"])
    allowed_actions = set(PLAN_JSON_SCHEMA["properties"]["action"]["enum"])
    intent = str(data.get("intent") or "vague")
    action = str(data.get("action") or "clarify")
    if intent not in allowed_intents:
        intent = "vague"
    if action not in allowed_actions:
        action = "clarify"
    slots = {
        key: value for key, value in (data.get("slots") or {}).items()
        if value is not None
    }
    return Plan(
        intent=intent,
        action=action,
        clarify_message=data.get("clarify_message"),
        slots=slots,
        tool_calls=list(data.get("tool_calls") or []),
        qa_args={
            key: value for key, value in (data.get("qa_args") or {}).items()
            if value is not None
        },
        reason="naive_whole_prompt_single_pass",
        metadata=metadata,
    )


class NaiveWholePromptLLM(BaseLLM):
    """Use one delegate LLM call for planning and no later LLM calls.

    ``supports_agentic_calls`` deliberately stays ``False``.  Production
    Text-to-SQL therefore uses its deterministic fallback after the one flat
    extraction request, rather than quietly adding more LLM stages to the
    NAIVE arm's internal orchestration.

    ``synthesize`` is still exposed as a passthrough to the delegate so the
    hallucination benchmark can give this arm one final-answer synthesis call
    matching the production arm's, and grade both arms' user-facing answers
    with the same grounding checks. See ``advisor_hallucination._run_one``.
    """

    supports_agentic_calls = False

    def __init__(self, delegate: BaseLLM,
                 fixed_context: dict[str, Any] | None = None):
        super().__init__()
        self.delegate = delegate
        self.fixed_context = copy.deepcopy(fixed_context or {})
        self.provider = getattr(delegate, "provider", "local")
        self.model = getattr(delegate, "model", "rule")
        self.api_backed = bool(getattr(delegate, "supports_agentic_calls", False))

    def plan(self, text: str, has_prior_region: bool = False,
             conversation_history: list[dict] | None = None) -> Plan:
        user_payload = {
            "context": self.fixed_context,
            "has_prior_region": bool(has_prior_region),
            "conversation_history": conversation_history or [],
            "user_prompt": text,
        }
        if self.api_backed:
            data = self.delegate.analyze_json(
                operation="experiment.naive_whole_prompt",
                system=NAIVE_WHOLE_PROMPT_SYSTEM_PROMPT,
                user=json.dumps(user_payload, ensure_ascii=False, default=str),
                schema=PLAN_JSON_SCHEMA,
                schema_name="naive_whole_prompt_plan",
                max_tokens=900,
            )
            if not isinstance(data, dict):
                raise RuntimeError("NAIVE 단일 추출 LLM이 유효한 JSON을 반환하지 않았습니다")
            delegate_trace = list(getattr(self.delegate, "last_trace", []))
            metadata = {
                "strategy": "naive_whole_prompt_single_pass",
                "fallback": False,
                "llm_call_count": 1,
                "prompt_fingerprint": naive_prompt_fingerprint(),
                "attempts": delegate_trace,
            }
        else:
            # Offline mode is only a structural smoke test.  The rule planner
            # still receives the complete message once and is never followed by
            # another LLM stage.
            fallback_plan = self.delegate.plan(
                text, has_prior_region=has_prior_region,
                conversation_history=conversation_history,
            )
            data = _plan_payload(fallback_plan)
            metadata = {
                "strategy": "naive_whole_prompt_offline_rule",
                "fallback": False,
                "llm_call_count": 0,
                "prompt_fingerprint": naive_prompt_fingerprint(),
                "attempts": list(getattr(self.delegate, "last_trace", [])),
            }
        self.last_trace = [metadata]
        return _normalize_plan(data, metadata)

    def synthesize(self, user_text: str, result: dict,
                   conversation_history: list[dict] | None = None) -> str | None:
        if not self.api_backed:
            return None
        answer = self.delegate.synthesize(
            user_text, result, conversation_history=conversation_history)
        self.last_trace = list(getattr(self.delegate, "last_trace", []))
        return answer

    def plan_condition_dialogue(self, text: str, context: dict) -> dict:
        plan = self.plan(
            text,
            conversation_history=[{
                "role": "application_context",
                "content": context,
            }],
        )
        decision = (
            "ask_clarification" if plan.action == "clarify"
            else "ask_confirmation"
        )
        return {
            "decision": decision,
            "message": plan.clarify_message or "추출한 조건을 적용할까요?",
            "goal_summary": "사용자 전체 문장의 단일 패스 조건 추출",
            "known_facts": [],
            "uncertainties": [],
            "slots": copy.deepcopy(plan.slots),
            "proposed_defaults": [],
            "tool_plan": [],
            "confidence": 0.5,
            "decision_reason": "naive_whole_prompt_single_pass",
            "_trace": copy.deepcopy(plan.metadata),
        }


def naive_decision(llm: BaseLLM, text: str,
                   *, context: dict[str, Any] | None = None,
                   conversation_history: list[dict] | None = None) -> dict[str, Any]:
    """Return a serializable one-shot decision for the search benchmark."""
    baseline = NaiveWholePromptLLM(llm, fixed_context=context)
    plan = baseline.plan(text, conversation_history=conversation_history)
    payload = _plan_payload(plan)
    payload["_trace"] = copy.deepcopy(plan.metadata)
    return payload
