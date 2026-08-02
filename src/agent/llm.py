"""Agentic LLM 계층: 계획, Text-to-SQL, 근거 기반 응답 합성.

모든 API 호출은 Structured Output(가능한 provider)과 명시적 재시도를 사용한다.
각 단계 실패는 서비스 전체 오류가 아니라 규칙 계획/결정론 SQL/템플릿 응답으로
격리되며, 어떤 폴백이 발생했는지 ``last_trace``에 남긴다.
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
from abc import ABC, abstractmethod
from typing import Any

from src import config
from src.agent.planner import Plan, Planner, parse_confirmation
from src.agent.prompts import (
    AGENT_SYSTEM_PROMPT, CONDITION_DECISION_JSON_SCHEMA,
    CONDITION_DIALOGUE_SYSTEM_PROMPT, PLAN_JSON_SCHEMA, SQL_JSON_SCHEMA,
    SQL_SYSTEM_PROMPT, SYNTHESIS_SYSTEM_PROMPT,
)
from src.agent.reliability import (
    RetryExhaustedError, RetryPolicy, call_with_retry, is_transient_error,
)


class BaseLLM(ABC):
    supports_agentic_calls = False

    def __init__(self):
        # A single APILLM instance is shared by FastAPI requests.  Thread-local
        # traces prevent concurrent prompts from overwriting each other's audit.
        self._trace_local = threading.local()
        self.last_trace = []

    @property
    def last_trace(self) -> list[dict]:
        trace_local = getattr(self, "_trace_local", None)
        return getattr(trace_local, "value", [])

    @last_trace.setter
    def last_trace(self, value: list[dict]) -> None:
        if not hasattr(self, "_trace_local"):
            self._trace_local = threading.local()
        self._trace_local.value = value

    @abstractmethod
    def plan(self, text: str, has_prior_region: bool = False,
             conversation_history: list[dict] | None = None) -> Plan: ...

    def plan_condition_dialogue(self, text: str, context: dict) -> dict:
        return _fallback_condition_decision(text, context, self.plan)

    def generate_sql(self, request: str, schema: str,
                     previous_error: str | None = None) -> dict | None:
        return None

    def synthesize(self, user_text: str, result: dict,
                   conversation_history: list[dict] | None = None) -> str | None:
        return None

    def analyze_json(self, *, operation: str, system: str, user: str,
                     schema: dict, schema_name: str,
                     max_tokens: int = 1200) -> dict | None:
        """도메인 도구가 쓰는 구조화 분석 확장점.

        Mock/로컬 구현은 지원하지 않으면 ``None``을 반환하고, API 구현만
        기존 재시도·모델 폴백·JSON Schema 검증 파이프라인을 재사용한다.
        """
        return None

    def web_search(self, query: str, purpose: str = "") -> dict | None:
        """외부 사실 확인이 필요한 도구 호출. 지원하지 않는 provider는 None."""
        return None


class MockLLM(BaseLLM):
    """오프라인 결정론적 폴백. 외부 API 없이 전체 파이프라인을 유지한다."""
    def __init__(self):
        super().__init__()
        self.planner = Planner()

    def plan(self, text: str, has_prior_region: bool = False,
             conversation_history: list[dict] | None = None) -> Plan:
        plan = self.planner.plan(text, has_prior_region)
        plan.reason = plan.reason or "rule"
        plan.metadata = {"strategy": "rule", "fallback": False, "attempts": []}
        self.last_trace = [plan.metadata]
        return plan

    def plan_condition_dialogue(self, text: str, context: dict) -> dict:
        decision = _fallback_condition_decision(text, context, self.plan)
        decision["_trace"] = {"strategy": "rule_condition_dialogue", "fallback": False}
        return decision


class QwenLLM(BaseLLM):
    """선택적 로컬 Qwen 계획기. 실패 시 규칙 플래너로 복구한다."""
    SYSTEM = AGENT_SYSTEM_PROMPT

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto")
        self.model_name = model_name
        self.fallback = Planner()

    def plan(self, text: str, has_prior_region: bool = False,
             conversation_history: list[dict] | None = None) -> Plan:
        try:
            history = json.dumps(
                conversation_history or [], ensure_ascii=False, default=str)
            user = (
                f"<conversation_history>{history}</conversation_history>\n"
                f"<latest_user_message>{text}</latest_user_message>"
                if conversation_history else text
            )
            messages = [{"role": "system", "content": self.SYSTEM},
                        {"role": "user", "content": user}]
            prompt = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tok([prompt], return_tensors="pt").to(self.model.device)
            out = self.model.generate(**inputs, max_new_tokens=500, do_sample=False)
            raw = self.tok.decode(out[0][inputs.input_ids.shape[1]:],
                                  skip_special_tokens=True)
            data = _extract_json(raw)
            plan = _plan_from_data(data, "qwen")
            plan.metadata = {"strategy": "qwen", "model": self.model_name,
                             "fallback": False, "attempts": 1}
            return plan
        except Exception as exc:
            plan = self.fallback.plan(text, has_prior_region)
            plan.reason = "qwen_fallback_rule"
            plan.metadata = {"strategy": "rule", "fallback": True,
                             "error": f"{type(exc).__name__}: {exc}"[:500]}
            return plan


class APILLM(BaseLLM):
    """API LLM. 한 모델이 계획/SQL/답변을 맡되 단계별 프롬프트는 격리한다."""
    supports_agentic_calls = True
    SYSTEM = AGENT_SYSTEM_PROMPT  # 기존 코드/테스트 호환용 공개 이름

    def __init__(self, api_key: str | None = None):
        super().__init__()
        self.provider = os.environ.get("LLM_PROVIDER", "openai").lower()
        if self.provider == "anthropic":
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            default_model = "claude-3-5-sonnet-latest"
        else:
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or config.OPENAI_API_KEY
            default_model = config.OPENAI_MODEL
        self.model = os.environ.get("LLM_MODEL", default_model)
        self.fallback_model = os.environ.get(
            "LLM_FALLBACK_MODEL", getattr(config, "OPENAI_FALLBACK_MODEL", "gpt-4o-mini"))
        self.fallback = Planner()
        self.retry_policy = RetryPolicy(
            max_attempts=max(1, int(os.environ.get("LLM_MAX_ATTEMPTS", "3"))),
            base_delay_seconds=max(0.0, float(os.environ.get("LLM_RETRY_BASE_SECONDS", "0.35"))),
            max_delay_seconds=max(0.0, float(os.environ.get("LLM_RETRY_MAX_SECONDS", "2.0"))),
        )
        self._concurrency = threading.BoundedSemaphore(
            max(1, int(getattr(config, "LLM_MAX_CONCURRENCY", 6)))
        )
        if not self.api_key:
            raise RuntimeError("API 키가 없습니다. config.py 또는 provider 환경설정을 확인하세요.")

        if self.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        else:
            import openai
            # SDK 내부 재시도와 애플리케이션 재시도가 겹치지 않게 한곳에서 제어한다.
            self._client = openai.OpenAI(
                api_key=self.api_key, base_url=config.OPENAI_BASE_URL,
                timeout=float(os.environ.get("LLM_TIMEOUT_SECONDS", "30")), max_retries=0,
            )

    def _call_in_slot(self, function):
        with self._concurrency:
            return function()

    def _request_json(self, *, operation: str, system: str, user: str,
                      schema: dict, schema_name: str, max_tokens: int = 800) -> dict:
        model_used = self.model

        def invoke() -> dict:
            def request():
                if self.provider == "anthropic":
                    schema_text = json.dumps(schema, ensure_ascii=False)
                    msg = self._client.messages.create(
                        model=model_used, max_tokens=max_tokens,
                        system=system + "\nJSON Schema:\n" + schema_text,
                        messages=[{"role": "user", "content": user}],
                    )
                    return "".join(
                        b.text for b in msg.content
                        if getattr(b, "type", "") == "text"
                    )
                response = self._client.responses.create(
                    model=model_used, instructions=system, input=user,
                    max_output_tokens=max_tokens,
                    text={"format": {"type": "json_schema", "name": schema_name,
                                     "strict": True, "schema": schema}},
                )
                return response.output_text

            raw = self._call_in_slot(request)
            return _extract_json(raw)

        retry_if = lambda exc: is_transient_error(exc) or isinstance(
            exc, (ValueError, json.JSONDecodeError, KeyError, TypeError))
        try:
            value, events = call_with_retry(
                operation, invoke, policy=self.retry_policy, retry_if=retry_if)
            self.last_trace = [e.to_dict() for e in events]
            self.last_trace.append({"operation": operation, "model": model_used,
                                    "provider": self.provider, "fallback_model": False})
            return value
        except RetryExhaustedError as first:
            events = [e.to_dict() for e in first.events]
            status = getattr(first.cause, "status_code", None)
            can_switch = (self.provider == "openai" and self.fallback_model
                          and self.fallback_model != self.model
                          and status not in {400, 401, 403})
            if not can_switch:
                self.last_trace = events
                raise
            model_used = self.fallback_model
            try:
                value, second_events = call_with_retry(
                    operation + ":fallback_model", invoke,
                    policy=RetryPolicy(max_attempts=1, base_delay_seconds=0),
                    retry_if=retry_if,
                )
                self.last_trace = events + [e.to_dict() for e in second_events]
                self.last_trace.append({"operation": operation, "model": model_used,
                                        "provider": self.provider, "fallback_model": True})
                return value
            except RetryExhaustedError as second:
                self.last_trace = events + [e.to_dict() for e in second.events]
                raise second from first

    def _request_text(self, *, operation: str, system: str, user: str,
                      max_tokens: int = 700) -> str:
        def invoke() -> str:
            def request():
                if self.provider == "anthropic":
                    msg = self._client.messages.create(
                        model=self.model, max_tokens=max_tokens, system=system,
                        messages=[{"role": "user", "content": user}],
                    )
                    return "".join(
                        b.text for b in msg.content
                        if getattr(b, "type", "") == "text"
                    ).strip()
                response = self._client.responses.create(
                    model=self.model, instructions=system, input=user,
                    max_output_tokens=max_tokens,
                )
                if not response.output_text.strip():
                    raise ValueError("LLM이 빈 응답을 반환했습니다")
                return response.output_text.strip()

            return self._call_in_slot(request)

        value, events = call_with_retry(
            operation, invoke, policy=self.retry_policy,
            retry_if=lambda e: is_transient_error(e) or isinstance(e, ValueError),
        )
        self.last_trace = [e.to_dict() for e in events]
        return value

    def analyze_json(self, *, operation: str, system: str, user: str,
                     schema: dict, schema_name: str,
                     max_tokens: int = 1200) -> dict | None:
        return self._request_json(
            operation=operation, system=system, user=user, schema=schema,
            schema_name=schema_name, max_tokens=max_tokens,
        )

    def web_search(self, query: str, purpose: str = "") -> dict | None:
        """OpenAI Responses 웹 검색을 호출하고 출처 URL까지 감사 가능하게 반환한다."""
        if self.provider != "openai":
            return None

        def invoke():
            return self._call_in_slot(
                lambda: self._client.responses.create(
                    model=self.model,
                    instructions=(
                        "당신은 주택 검색 에이전트의 웹 검색 도구다. 공식 기관·공식 사이트를 "
                        "우선하고, 확인되지 않은 좌표나 주소를 만들지 마라. 장소 확인 요청이면 "
                        "마지막 줄을 반드시 'MAP_QUERY: 도로명주소 또는 정식 장소명' 형식으로 써라. "
                        "찾지 못하면 'MAP_QUERY: NOT_FOUND'라고 써라."
                    ),
                    input=f"목적: {purpose or '외부 사실 확인'}\n검색 요청: {query}",
                    max_output_tokens=700,
                    tools=[{"type": "web_search", "search_context_size": "low"}],
                    tool_choice="required",
                    include=["web_search_call.action.sources"],
                )
            )

        try:
            response, events = call_with_retry(
                "llm.web_search", invoke, policy=self.retry_policy,
                retry_if=is_transient_error,
            )
        except Exception as exc:
            self.last_trace = [{"operation": "llm.web_search", "error":
                                f"{type(exc).__name__}: {exc}"[:500]}]
            return None
        payload = response.model_dump() if hasattr(response, "model_dump") else {}
        sources: list[dict] = []
        seen: set[str] = set()

        def walk(value):
            if isinstance(value, dict):
                url = value.get("url")
                if isinstance(url, str) and url.startswith(("http://", "https://")) \
                        and url not in seen:
                    seen.add(url)
                    sources.append({"title": value.get("title") or value.get("name") or url,
                                    "url": url})
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)

        walk(payload.get("output") or [])
        answer = str(getattr(response, "output_text", "") or "").strip()
        match = re.search(r"(?im)^MAP_QUERY:\s*(.+?)\s*$", answer)
        map_query = match.group(1).strip() if match else None
        if not map_query:
            # 웹 검색 모델이 마지막 줄 형식을 생략해도, 출처가 딸린 응답에서
            # 명시적으로 서술한 한국 도로명주소만 보수적으로 회수한다. 이 값은
            # 곧바로 조건이 되지 않고 NAVER 지오코딩 재검증을 다시 통과해야 한다.
            address_match = re.search(
                r"주소(?:는|\s*[:：])\s*([^\n。.]*(?:로|길)\s*\d+(?:-\d+)?)",
                answer,
            )
            if address_match:
                map_query = re.sub(r"[\[\]()*_`]", "", address_match.group(1)).strip()
        if map_query == "NOT_FOUND":
            map_query = None
        self.last_trace = [e.to_dict() for e in events] + [{
            "operation": "llm.web_search", "model": self.model,
            "source_count": len(sources), "fallback": False,
        }]
        return {"answer": answer, "map_query": map_query,
                "sources": sources[:10], "trace": list(self.last_trace)}

    def plan(self, text: str, has_prior_region: bool = False,
             conversation_history: list[dict] | None = None) -> Plan:
        try:
            history_payload = json.dumps(
                conversation_history or [], ensure_ascii=False, default=str)
            planner_user = (
                f"<conversation_history>{history_payload}</conversation_history>\n"
                f"<latest_user_message>{text}</latest_user_message>"
                if conversation_history else text
            )
            data = self._request_json(
                operation="llm.plan", system=AGENT_SYSTEM_PROMPT,
                user=(planner_user + (
                    "\n이전 지역 조건이 존재한다." if has_prior_region else "")),
                schema=PLAN_JSON_SCHEMA, schema_name="housing_agent_plan", max_tokens=900,
            )
            plan = _plan_from_data(data, "api")
            # 명백한 Q&A를 vague로 분류한 경우에만 결정론 감지기로 의도를 교정한다.
            # API 호출 자체는 성공했으므로 이후 Text-to-SQL/합성은 계속 실제 LLM이 맡는다.
            semantic_repairs = []
            rule_plan = self.fallback.plan(text, has_prior_region)
            compound_goals = {
                "goal_financed_jeonse", "goal_best_affordable",
                "goal_alternative_areas",
            }
            if (rule_plan.intent in compound_goals
                    and plan.intent != rule_plan.intent):
                semantic_repairs.append({
                    "from": plan.intent, "to": rule_plan.intent,
                    "reason": "명시적인 금융 활용 전세 최대화 복합 목표 보존",
                })
                plan.intent = rule_plan.intent
                plan.action = rule_plan.action
                plan.clarify_message = None
                plan.slots = rule_plan.slots
                plan.qa_args = rule_plan.qa_args
                plan.tool_calls = rule_plan.tool_calls
            elif plan.intent == "vague" and rule_plan.intent.startswith("qa_"):
                semantic_repairs.append({
                    "from": "vague", "to": rule_plan.intent,
                    "reason": "명시적 Q&A 키워드와 LLM 의도 불일치",
                })
                plan.intent = rule_plan.intent
                plan.action = rule_plan.action
                plan.clarify_message = None
                plan.qa_args = rule_plan.qa_args
                plan.tool_calls = rule_plan.tool_calls
            elif (plan.intent == rule_plan.intent
                  and (plan.intent.startswith("qa_")
                       or plan.intent in compound_goals)):
                # 정규식으로 확실히 읽을 수 있는 수치·상품종류 같은 명시 조건은
                # LLM이 누락/변형하더라도 보존한다. 자유로운 의미 판단은 LLM에
                # 맡기되 사용자가 직접 말한 WHERE 조건은 조용히 사라지지 않게 한다.
                repaired_args = {
                    key: value for key, value in rule_plan.qa_args.items()
                    if value is not None and plan.qa_args.get(key) != value
                }
                if repaired_args:
                    semantic_repairs.append({
                        "fields": sorted(repaired_args),
                        "reason": "명시적 자연어 제약을 결정론 파서로 보존",
                    })
                    plan.qa_args.update(repaired_args)
                if plan.intent in compound_goals:
                    plan.slots.update(rule_plan.slots)
            # 위험도 요청은 WHERE 상한이 아니라 정렬 의도로만 보존한다.
            if rule_plan.slots.get("sort_by") == "risk_asc":
                if plan.slots.get("sort_by") != "risk_asc":
                    semantic_repairs.append({
                        "field": "sort_by",
                        "to": "risk_asc",
                        "reason": "안전 선호를 위험도 낮은순 정렬로 보존",
                    })
                plan.slots.pop("max_fraud_score", None)
                plan.slots.pop("safety_is_hard", None)
                plan.slots["sort_by"] = "risk_asc"
            if rule_plan.slots.get("sort_by") in {"price_asc", "price_desc"}:
                requested_sort = rule_plan.slots["sort_by"]
                if plan.slots.get("sort_by") != requested_sort:
                    semantic_repairs.append({
                        "field": "sort_by",
                        "to": requested_sort,
                        "reason": "명시적인 최저·최고 가격 정렬 요청 보존",
                    })
                plan.slots["sort_by"] = requested_sort
            plan.metadata = {"strategy": "api", "provider": self.provider,
                              "model": self.model, "fallback": False,
                              "attempts": list(self.last_trace),
                              "semantic_repair": semantic_repairs or None}
            return plan
        except Exception as exc:
            # 계획 실패는 가장 안전한 결정론 규칙으로 복구한다. 오류는 추적에 보존한다.
            attempts = list(self.last_trace)
            plan = self.fallback.plan(text, has_prior_region)
            plan.reason = "api_fallback_rule"
            plan.metadata = {"strategy": "rule", "provider": self.provider,
                             "model": self.model, "fallback": True,
                             "error": f"{type(exc).__name__}: {exc}"[:500],
                             "attempts": attempts}
            return plan

    def plan_condition_dialogue(self, text: str, context: dict) -> dict:
        context_payload = json.dumps(context, ensure_ascii=False, default=str)
        if len(context_payload) > 9000:
            context_payload = context_payload[-9000:]
        user = (
            f"<conversation_context>{context_payload}</conversation_context>\n"
            f"<latest_user_message>{text}</latest_user_message>"
        )
        try:
            decision = self._request_json(
                operation="llm.condition_dialogue",
                system=CONDITION_DIALOGUE_SYSTEM_PROMPT,
                user=user,
                schema=CONDITION_DECISION_JSON_SCHEMA,
                schema_name="housing_condition_dialogue_decision",
                max_tokens=1200,
            )
            _validate_condition_decision(decision)
            decision, semantic_repairs = _repair_condition_decision(
                decision, text=text, context=context,
            )
            _validate_condition_decision(decision)
            decision["slots"] = _compact(decision.get("slots") or {})
            decision["_trace"] = {
                "strategy": "api_structured_condition_dialogue",
                "provider": self.provider, "model": self.model,
                "fallback": False, "attempts": list(self.last_trace),
                "semantic_repair": semantic_repairs or None,
            }
            return decision
        except Exception as exc:
            attempts = list(self.last_trace)
            decision = _fallback_condition_decision(text, context, self.fallback.plan)
            decision["_trace"] = {
                "strategy": "rule_condition_dialogue", "provider": self.provider,
                "model": self.model, "fallback": True,
                "error": f"{type(exc).__name__}: {exc}"[:500],
                "attempts": attempts,
            }
            return decision

    def generate_sql(self, request: str, schema: str,
                     previous_error: str | None = None) -> dict | None:
        repair = ""
        if previous_error:
            repair = f"\n이전 SQL 오류(수정 필요): {previous_error[:800]}"
        return self._request_json(
            operation="llm.text2sql", system=SQL_SYSTEM_PROMPT,
            user=f"허용 스키마:\n{schema}\n\n요청:\n{request}{repair}",
            schema=SQL_JSON_SCHEMA, schema_name="housing_text_to_sql", max_tokens=1000,
        )

    def synthesize(self, user_text: str, result: dict,
                   conversation_history: list[dict] | None = None) -> str | None:
        # 추적 전체나 대량 원시 행은 토큰/민감정보를 줄이기 위해 제외한다.
        grounded = {k: v for k, v in result.items()
                    if k not in {"agent_trace", "answer"}}
        grounded = _annotate_manwon_strings(grounded)
        payload = json.dumps(grounded, ensure_ascii=False, default=str)
        if len(payload) > 14000:
            payload = payload[:14000] + "...(일부 생략)"
        history_payload = json.dumps(
            conversation_history or [], ensure_ascii=False, default=str)
        history_block = (
            f"이전 AI 추천·상담 대화 전체(JSON):\n{history_payload}\n\n"
            if conversation_history else ""
        )
        try:
            return self._request_text(
                operation="llm.synthesize", system=SYNTHESIS_SYSTEM_PROMPT,
                user=(
                    f"{history_block}현재 사용자 질문: {user_text}\n\n"
                    f"도구 실행 결과 JSON:\n{payload}"
                ),
                max_tokens=700,
            )
        except Exception:
            return None


def _format_manwon(value: float) -> str:
    """만원 단위 숫자를 '3억 2,727만원'류 자연어 표기로 변환한다.

    LLM이 억/만원 환산을 직접 문장으로 쓰다 보면 10배 축소나 자릿수 누락이
    반복적으로 발생한다(예: 136923 -> "1억 3,692만원"). 계산 자체를 코드가
    맡고 LLM은 결과 문자열을 그대로 인용하게 해 이 오류를 원천 차단한다.
    """
    if value < 0:
        return "-" + _format_manwon(-value)
    eok, remainder = divmod(value, 10000)
    eok = int(eok)
    man = int(remainder)
    won = round((remainder - man) * 10000)
    if won >= 10000:
        man += 1
        won -= 10000
    segments = []
    if eok:
        segments.append(f"{eok}억")
    if won:
        segments.append(f"{man:,}만 {won:,}원")
    elif man or not segments:
        segments.append(f"{man:,}만원")
    return " ".join(segments)


def _annotate_manwon_strings(value: Any) -> Any:
    """결과 JSON의 큰 숫자 옆에 '<필드명>_formatted' 사전 계산 문자열을 붙인다.

    금액 필드 이름(...manwon)이거나 값 자체가 커서(>=1000) 만원 단위
    금액으로 추정되는 숫자만 대상으로 한다(통근분·나이·점수 등 다른 수치는
    이 도메인에서 1000을 넘지 않으므로 오탐 위험이 낮다).
    """
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, sub_value in value.items():
            out[key] = _annotate_manwon_strings(sub_value)
            is_number = (isinstance(sub_value, (int, float))
                         and not isinstance(sub_value, bool)
                         and math.isfinite(float(sub_value)))
            if is_number:
                looks_like_manwon = (
                    isinstance(key, str) and key.lower().endswith("manwon")
                ) or abs(float(sub_value)) >= 1000
                if looks_like_manwon:
                    out[f"{key}_formatted"] = _format_manwon(float(sub_value))
        return out
    if isinstance(value, list):
        return [_annotate_manwon_strings(item) for item in value]
    return value


def _extract_json(raw: str) -> dict:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("빈 JSON 응답")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(raw[start:end + 1])
    if not isinstance(data, dict):
        raise TypeError("JSON 객체가 필요합니다")
    return data


def _compact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _compact(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_compact(v) for v in value]
    return value


def _condition_decision_template(**overrides) -> dict:
    base = {
        "decision": "ask_clarification", "message": "원하는 조건을 조금 더 알려주세요.",
        "goal_summary": "주택 검색 조건 추가", "known_facts": [],
        "uncertainties": [], "slots": {}, "proposed_defaults": [],
        "tool_plan": [], "confidence": 0.5,
        "decision_reason": "검색 결과를 크게 바꾸는 조건이 불명확함",
    }
    base.update(overrides)
    return base


def _fallback_condition_decision(text: str, context: dict, plan_fn) -> dict:
    """API 장애 때도 질문→제안→확인 계약을 보존하는 결정론 상태 결정기."""
    utterance = str(text or "").strip()
    state = str(context.get("state") or "idle")
    known = dict(context.get("known_slots") or {})
    proposed = dict(context.get("proposed_slots") or {})
    confirmation = parse_confirmation(utterance)
    if state == "awaiting_confirmation" and confirmation == "yes":
        slots = proposed or known
        return _condition_decision_template(
            decision="ask_confirmation",
            message="채팅은 조건 수정에 사용됩니다. 아래 ‘조건 추가’ 버튼을 눌러 승인해 주세요.",
            known_facts=["직전 조건 제안이 준비됨"], uncertainties=[],
            slots=slots, tool_plan=[], confidence=1.0,
            decision_reason="최종 승인은 채팅 문구가 아니라 UI 버튼으로만 받음",
        )
    if state == "awaiting_confirmation" and confirmation == "no":
        return _condition_decision_template(
            decision="cancel", message="알겠습니다. 이 조건 제안은 취소할게요.",
            goal_summary="조건 추가 취소", known_facts=[], uncertainties=[],
            slots={}, confidence=1.0, decision_reason="사용자가 취소함",
        )

    try:
        parsed = plan_fn(utterance, bool(known))
        parsed_slots = {
            key.lstrip("_"): value for key, value in (parsed.slots or {}).items()
            if value is not None
        }
    except Exception:
        parsed_slots = {}
    slots = {**known, **parsed_slots}

    if "아주대" in utterance:
        slots["workplace_landmark"] = "아주대학교"
    elif "카이스트" in utterance.lower() or "kaist" in utterance.lower():
        slots["workplace_landmark"] = "카이스트"
    else:
        landmark_match = re.search(r"([가-힣A-Za-z0-9]{2,20}(?:대학교|대학|역))", utterance)
        if landmark_match and landmark_match.group(1) not in {"대중교통"}:
            slots["workplace_landmark"] = landmark_match.group(1)

    if re.search(r"대중교통|버스|지하철", utterance):
        slots["commute_mode"] = "transit"
    elif re.search(r"도보|걸어서|걷", utterance):
        slots["commute_mode"] = "walking"
    elif re.search(r"자동차|자차|운전|차로", utterance):
        slots["commute_mode"] = "driving"
    minute_match = re.search(r"(\d+(?:\.\d+)?)\s*분", utterance)
    if minute_match:
        slots["max_commute_min"] = float(minute_match.group(1))

    landmark = slots.get("workplace_landmark")
    mode = slots.get("commute_mode")
    minutes = slots.get("max_commute_min")
    if landmark and not mode:
        return _condition_decision_template(
            decision="ask_clarification",
            message=(f"{landmark} 주변을 어떤 이동 기준으로 찾을까요? "
                     "대중교통·도보·자동차 중 하나와 원하는 시간이 있으면 같이 알려주세요."),
            goal_summary=f"{landmark} 주변 주택 탐색",
            known_facts=[f"목적지 후보: {landmark}"],
            uncertainties=[
                {"field": "commute_mode", "description": "주변의 이동 기준", "blocking": True},
                {"field": "max_commute_min", "description": "허용 이동시간", "blocking": True},
            ],
            slots=slots, confidence=0.88,
            decision_reason="랜드마크만으로는 주변의 의미를 단일 WHERE/지도 조건으로 확정할 수 없음",
        )
    proposed_defaults = []
    if landmark and mode and minutes is None:
        default_minutes = {"transit": 20.0, "walking": 15.0, "driving": 20.0}[mode]
        slots["max_commute_min"] = default_minutes
        proposed_defaults.append({
            "field": "max_commute_min", "value": str(int(default_minutes)),
            "reason": "사용자가 시간을 지정하지 않아 탐색 시작값으로 제안",
        })
        minutes = default_minutes

    if slots:
        facts = []
        if landmark:
            mode_label = {"transit": "대중교통 예상", "walking": "도보 예상", "driving": "자동차 예상"}.get(mode, "이동 예상")
            facts.append(f"{landmark} {mode_label} {int(minutes)}분 이내")
        transaction = slots.get("transaction_type") or slots.get("lease_type")
        if transaction:
            facts.append(f"{transaction} 거래")
        if slots.get("max_monthly_rent_manwon") is not None:
            facts.append(f"월세 {slots['max_monthly_rent_manwon']:g}만원 이하")
        if slots.get("sort_by") == "risk_asc":
            facts.append("전세 위험도 낮은순 정렬")
        summary = ", ".join(facts) or "입력한 주택 조건"
        return _condition_decision_template(
            decision="ask_confirmation", message=f"{summary}를 검색 조건으로 추가할까요?",
            goal_summary=summary, known_facts=facts, uncertainties=[], slots=slots,
            proposed_defaults=proposed_defaults, confidence=0.9,
            decision_reason="실행 가능한 조건을 구성했으며 도구 호출 전 사용자 승인이 필요함",
        )
    return _condition_decision_template(
        message="지역·랜드마크·거래유형·가격 중 어떤 조건을 추가하고 싶은지 알려주세요.",
        uncertainties=[{"field": "search_goal", "description": "추가할 검색 조건", "blocking": True}],
        slots={}, confidence=0.3, decision_reason="실행 가능한 검색 제약을 확인하지 못함",
    )


def _validate_condition_decision(decision: dict) -> None:
    if not isinstance(decision, dict):
        raise TypeError("조건 대화 결정은 JSON 객체여야 합니다")
    allowed = {"ask_clarification", "ask_confirmation", "cancel"}
    if decision.get("decision") not in allowed:
        raise ValueError("지원하지 않는 조건 대화 decision")
    if not isinstance(decision.get("slots"), dict):
        raise TypeError("조건 대화 slots 객체가 필요합니다")
    if decision["decision"].startswith("ask_") and not str(decision.get("message") or "").strip():
        raise ValueError("질문/확인 메시지가 비어 있습니다")
def _repair_condition_decision(decision: dict, *, text: str,
                               context: dict) -> tuple[dict, list[str]]:
    """Enforce the prompt's negotiation contract at the application boundary.

    The model still performs the semantic interpretation.  This guard only
    normalizes audited aliases and prevents a tool call from being reached with
    an incomplete landmark/time pair.  Repairs are exposed in the debug trace.
    """
    repaired = dict(decision)
    slots = {
        **(context.get("known_slots") or {}),
        **(repaired.get("slots") or {}),
    }
    repairs: list[str] = []

    # 과거/모델 출력의 위험도 임계값을 방어적으로 제거한다. 위험도는 정렬 전용이다.
    if slots.pop("max_fraud_score", None) is not None:
        repairs.append("removed_risk_where_threshold")
    if slots.pop("safety_is_hard", None) is not None:
        repairs.append("removed_risk_hard_filter_flag")
    if re.search(r"안전|안심|위험(?:도|이)?\s*(?:낮|적|없)|사기\s*없", str(text or "")):
        if slots.get("sort_by") != "risk_asc":
            repairs.append("converted_safety_preference_to_sort")
        slots["sort_by"] = "risk_asc"

    aliases = {
        "아주대": "아주대학교",
        "카이스트": "카이스트",
        "kaist": "카이스트",
    }
    landmark = str(slots.get("workplace_landmark") or "").strip()
    normalized = aliases.get(landmark.lower(), landmark)
    if normalized and normalized != landmark:
        slots["workplace_landmark"] = normalized
        repairs.append("normalized_audited_landmark_alias")

    landmark = slots.get("workplace_landmark")
    mode = slots.get("commute_mode")
    minutes = slots.get("max_commute_min")

    # The prompt explicitly defines these as proposed defaults.  A model may
    # occasionally ask for the time again; convert that drift into the agreed
    # confirmation step, never directly into a tool call.
    if landmark and mode in {"transit", "walking", "driving"} and minutes is None:
        default_minutes = {"transit": 20.0, "walking": 15.0, "driving": 20.0}[mode]
        slots["max_commute_min"] = default_minutes
        mode_label = {"transit": "대중교통 예상", "walking": "도보 예상",
                      "driving": "자동차 예상"}[mode]
        repaired.update({
            "decision": "ask_confirmation",
            "message": (f"{landmark}를 목적지로 {mode_label} "
                        f"{int(default_minutes)}분 이내를 조건으로 추가할까요?"),
            "uncertainties": [],
            "proposed_defaults": [{
                "field": "max_commute_min",
                "value": str(int(default_minutes)),
                "reason": "사용자가 시간을 지정하지 않아 검색 시작값으로 제안",
            }],
            "tool_plan": [],
            "decision_reason": "이동수단은 확인됐고 시간은 없어 기본값을 제안한 뒤 승인 대기",
        })
        repairs.append("proposed_missing_commute_time_default")

    # A landmark without a mode is still ambiguous, regardless of an overly
    # eager model decision.  No tool plan may cross this gate.
    elif landmark and not mode:
        repaired.update({
            "decision": "ask_clarification",
            "message": (f"{landmark} 주변을 어떤 기준으로 찾을까요? "
                        "대중교통, 도보, 자동차 중 하나를 알려주세요."),
            "tool_plan": [],
            "decision_reason": "랜드마크만으로는 주변의 이동 기준을 확정할 수 없음",
        })
        repairs.append("blocked_ambiguous_landmark_only")

    repaired["slots"] = _compact(slots)
    return repaired, repairs


def _plan_from_data(data: dict, reason: str) -> Plan:
    intent = data.get("intent", "vague")
    action = data.get("action", "clarify")
    allowed_intents = set(PLAN_JSON_SCHEMA["properties"]["intent"]["enum"])
    if intent not in allowed_intents or action not in {"proceed", "clarify", "confirm"}:
        raise ValueError("지원하지 않는 intent/action")
    slots = _compact(data.get("slots") or {})
    qa_args = _compact(data.get("qa_args") or {})
    calls = []
    for item in data.get("tool_calls") or []:
        if not isinstance(item, dict) or not item.get("tool"):
            continue
        args = {k: v for k, v in item.items() if k != "tool" and v is not None}
        calls.append({"tool": item["tool"], "args": args})
    # 모델이 DB 호출 표시를 빠뜨려도 실행 의도에 맞춰 보정한다.
    if intent == "recommend" and not any(c["tool"] == "property_search" for c in calls):
        calls.append({"tool": "property_search", "args": {}})
    if intent == "qa_finance" and not any(c["tool"] == "finance_search" for c in calls):
        calls.append({"tool": "finance_search", "args": {}})
    if intent in {
        "goal_financed_jeonse", "goal_best_affordable",
        "goal_alternative_areas",
    }:
        if not any(c["tool"] == "finance_search" for c in calls):
            calls.append({"tool": "finance_search", "args": {}})
        if not any(c["tool"] == "property_search" for c in calls):
            calls.append({"tool": "property_search", "args": {}})
    landmark = slots.pop("workplace_landmark", None)
    if landmark and not any(c["tool"] == "map_regions_within" for c in calls):
        calls.append({"tool": "map_regions_within", "args": {
            "landmark": landmark, "minutes": slots.get("max_commute_min", 30)}})
    return Plan(
        intent=intent, slots=slots, tool_calls=calls, action=action,
        clarify_message=data.get("clarify_message"), qa_args=qa_args, reason=reason,
    )


def get_llm() -> BaseLLM:
    choice = os.environ.get("JEONSE_LLM", "api").lower()
    # "openai" is kept as a user-facing alias because older runbooks used it.
    # Previously that documented value fell through to MockLLM silently.
    if choice in {"api", "openai"}:
        try:
            return APILLM()
        except Exception as exc:
            # 사용자가 명시적으로 api를 선택했는데 조용히 Mock으로 바뀌면 실제 LLM
            # 장애를 정상 동작으로 오해하게 된다. 명시적 opt-in일 때만 초기화 폴백한다.
            if os.environ.get("LLM_ALLOW_INIT_FALLBACK") == "1":
                if os.environ.get("JEONSE_VERBOSE"):
                    print(f"  [llm] API 초기화 실패({type(exc).__name__}) → MockLLM 폴백")
                return MockLLM()
            raise RuntimeError(
                f"JEONSE_LLM=api 초기화 실패. Mock으로 전환하지 않았습니다: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    if choice == "qwen":
        return QwenLLM()
    if choice == "auto":
        try:
            return APILLM()
        except Exception:
            try:
                return QwenLLM()
            except Exception:
                return MockLLM()
    return MockLLM()
