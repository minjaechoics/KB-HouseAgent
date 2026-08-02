"""End-to-end hallucination benchmark for the production AWS advisor path."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sqlite3
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

from src import config
from src.agent.harness import JeonseAgent
from src.agent.prompts import AGENT_SYSTEM_PROMPT, PLAN_JSON_SCHEMA
from src.experiments.advisor_cases import ADVISOR_CASES, AdvisorCase
from src.experiments.console import FixedHeaderConsole, format_elapsed, wait_for_key
from src.experiments.pipeline import database_fingerprint, json_text
from src.experiments.naive_baseline import (
    NaiveWholePromptLLM,
    naive_prompt_fingerprint,
)
from src.experiments.runner import (
    _choose_mode,
    _create_llm,
    _effective_workers,
    _observed_peak_concurrency,
    _preflight_live_llm,
    _save_report,
)


DEFAULT_USER = {
    "user_id": "HALLUCINATION-BENCHMARK",
    "age": 29,
    "monthly_income_manwon": 600,
    "annual_income_manwon": 7200,
    "total_asset_manwon": 50000,
    "assets_manwon": 50000,
    "monthly_living_cost_manwon": 100,
    "income_decile": 5,
    "preferred_sido": "경기",
    "preferred_gugun": "수원시 팔달구",
    "preferences": {"mode": "balanced", "approved": True},
    # qa_finance 케이스(신용등급/차량/개인사업자 자격 확인)를 실제로
    # 검증하기 위한 프로필. 기존 카테고리는 이 필드들을 쓰지 않으므로
    # 추가해도 기존 점수에 영향을 주지 않는다.
    "employment_type": "employee",
    "employment_months": 36,
    "household_role": "head",
    "home_ownership_count": 0,
    "marital_status": "single",
    "is_korean_national": True,
    "has_income_proof": True,
    "contract_deposit_paid_5pct": True,
    "credit_grade": 3,
    "vehicle_purchase_type": "used_purchase",
    "vehicle_price_manwon": 8000,
}

SELECTED_REPORT = {
    "property": {
        "property_id": "oracle-selected-sale-001",
        "transaction_type": "매매",
        "lease_type": "매매",
        "sale_price_manwon": 30000.0,
        "asking_price_manwon": 30000.0,
        "sido": "경기",
        "gugun": "수원시 팔달구",
        "dong": "인계동",
    },
    "forecast": {
        "annual_growth_rate": 0.03,
        "annual_low": -0.01,
        "annual_high": 0.06,
        "model_version": "human_oracle_fixture_v1",
        "price_history": {
            "available": True,
            "series": [
                {"yyyymm": "202601", "median_price_manwon": 28800.0},
                {"yyyymm": "202604", "median_price_manwon": 29400.0},
                {"yyyymm": "202607", "median_price_manwon": 30000.0},
            ],
        },
        "news": {
            "relevant_headlines": [
                {"title": "수원 도심 교통 개선 계획", "sentiment": "positive"},
            ],
        },
        "market_assessment": "실거래 추세는 완만한 상승, 예측구간에는 하락 가능성 포함",
    },
}


def _prompt_fingerprint() -> str:
    payload = (
        AGENT_SYSTEM_PROMPT
        + json.dumps(PLAN_JSON_SCHEMA, sort_keys=True, ensure_ascii=False)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _stratified_cases(limit: int) -> list[AdvisorCase]:
    target = max(1, min(int(limit), len(ADVISOR_CASES)))
    buckets: dict[str, list[AdvisorCase]] = {}
    for case in ADVISOR_CASES:
        buckets.setdefault(case.category, []).append(case)
    result: list[AdvisorCase] = []
    while len(result) < target and any(buckets.values()):
        for category in sorted(buckets):
            if buckets[category] and len(result) < target:
                result.append(buckets[category].pop(0))
    return result


def _ground_truth(case: AdvisorCase) -> dict[str, Any]:
    truth: dict[str, Any] = {
        "case_id": case.case_id,
        "category": case.category,
        "expected_intent": case.expected_intent,
        "expected_status": case.expected_status,
        "expected_qa_type": case.expected_qa_type,
        "expected_recommendation_mode": case.expected_mode,
        "required_tools": list(case.required_tools),
        "rationale": case.rationale,
    }
    if case.expected_slots:
        truth["expected_slots"] = case.expected_slots
    if case.context_kind in {"selected_sale", "selected_area"}:
        truth["selected_report"] = copy.deepcopy(SELECTED_REPORT)
    return truth


def _session_for(agent: JeonseAgent, case: AdvisorCase) -> dict[str, Any]:
    session = agent.new_session(copy.deepcopy(DEFAULT_USER))
    if case.context_kind in {"selected_sale", "selected_area"}:
        session["last_property_report"] = copy.deepcopy(SELECTED_REPORT)
        session["last_recommended_properties"] = [
            copy.deepcopy(SELECTED_REPORT["property"])
        ]
    return session


def _history_for(case: AdvisorCase) -> list[dict[str, Any]]:
    if case.context_kind == "selected_area":
        return [
            {"role": "user", "text": "인계동에서 집을 보고 있어"},
            {"role": "assistant", "text": "인계동 후보를 확인했어요.",
             "slots": {"region_sido": "경기", "region_gugun": ["수원시 팔달구"]}},
        ]
    return []


def _subsequence(required: tuple[str, ...], actual: list[str]) -> bool:
    cursor = iter(actual)
    return all(any(item == wanted for item in cursor) for wanted in required)


def _canonical(field: str, value: Any) -> Any:
    if field in {"transaction_type", "lease_type"}:
        return str(value or "").strip()
    if field == "property_type":
        aliases = {
            "다가구": "다가구주택", "다세대": "다세대주택",
            "단독": "단독주택", "연립": "연립주택",
        }
        text = str(value or "").strip()
        return aliases.get(text, text)
    if field == "region_sido":
        return str(value or "").replace("경기도", "경기").strip()
    if field == "region_gugun":
        values = value if isinstance(value, (list, tuple, set)) else [value]
        return sorted(str(item).strip() for item in values if item)
    return value


def _slot_check(expected: dict[str, Any], actual: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for field, wanted in expected.items():
        got = actual.get(field)
        if field == "transaction_type" and got is None:
            got = actual.get("lease_type")
        if isinstance(wanted, (int, float)) and not isinstance(wanted, bool):
            try:
                equal = math.isclose(float(got), float(wanted), abs_tol=1e-6)
            except (TypeError, ValueError):
                equal = False
        else:
            equal = _canonical(field, got) == _canonical(field, wanted)
        if not equal:
            errors.append(f"{field}: expected={wanted!r}, actual={got!r}")
    return not errors, errors


def _groups_zero(result: dict[str, Any]) -> list[dict[str, Any]]:
    groups = result.get("groups") or {}
    return list(groups.get(0) or groups.get("0") or [])


def _all_hard_constraints_true(rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    for row in rows:
        constraints = row.get("hard_constraints")
        if not isinstance(constraints, dict) or not constraints:
            return False
        if not all(bool(value) for value in constraints.values()):
            return False
    return True


def _property_dongs(property_ids: list[str]) -> dict[str, str | None]:
    if not property_ids:
        return {}
    placeholders = ",".join("?" for _ in property_ids)
    uri = config.DB_PATH.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=5) as connection:
        rows = connection.execute(
            f"SELECT property_id, dong FROM properties WHERE property_id IN ({placeholders})",
            property_ids,
        ).fetchall()
    return {str(property_id): dong for property_id, dong in rows}


def _collect_numbers(value: Any, output: list[float]) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        output.append(float(value))
        return
    if isinstance(value, str):
        # eligibility_text/loan_limit_text 같은 원문 필드에 "수도권 7억원
        # 이하", "납입액의 95% 이내"처럼 근거 있는 수치가 텍스트로만 박혀
        # 있는 경우가 많다. 억/만원 결합 표기까지 함께 파싱해 이런 값도
        # "알려진 근거"로 인정한다(그렇지 않으면 원문을 그대로 인용한
        # 정확한 답변이 오탐으로 실패 처리된다).
        output.extend(_extract_amounts(value))
        return
    if isinstance(value, dict):
        output.append(float(len(value)))
        for item in value.values():
            _collect_numbers(item, output)
        return
    if isinstance(value, (list, tuple)):
        output.append(float(len(value)))
        for item in value:
            _collect_numbers(item, output)


def _extract_amounts(text: str) -> list[float]:
    """만원 단위로 결합한 숫자와 개별 토큰을 함께 반환한다.

    한국어는 큰 금액을 "3억 2,727만원"처럼 억/만 단위로 쪼개 쓴다. 단순
    자릿수 토큰 정규식은 이를 "3"과 "2727"로 따로 뽑아 실제 값(32727)과
    둘 다 불일치 판정하는 오탐을 만든다("3.5억원"→35000만원도 동일). 억
    표기를 만원 단위로 결합한 값을 우선 추가하고, 그 외 개별 토큰도 남겨
    다른 단위(나이·개월·퍼센트 등)의 정당한 숫자를 놓치지 않는다.
    """
    # URL의 쿼리스트링·프로그램ID에 박힌 숫자(예: LN20000030, page=C103429)는
    # 사실 주장이 아니므로 추출 전에 링크 전체를 제거한다.
    text = re.sub(r"https?://\S+", " ", text)
    # "24시간"은 편의점 카테고리를 가리키는 고정 관용구지 근거 대조가
    # 필요한 수치 주장이 아니므로 제거한다.
    text = re.sub(r"24\s*시간", " ", text)
    # "2026년"처럼 연도로 쓰인 4자리 숫자는 사실 주장이 아니라 날짜
    # 표기이므로 제거한다(가격 등 만원 단위 수치와 우연히 자릿수가 겹칠 수 있음).
    text = re.sub(r"\d{4}\s*년", " ", text)

    combined: list[float] = []
    consumed: list[tuple[int, int]] = []
    # "3억 2,727만 3,000원"처럼 억+만 뒤에 나머지 원 단위까지 붙는 3단 표기도
    # 하나의 값으로 결합한다(코드가 미리 계산해 붙이는 _formatted 문자열이
    # 이 3단 형태를 쓴다). 마지막 "원" 그룹이 없으면 기존 억+만 표기와 동일하다.
    for match in re.finditer(
        r"(\d[\d,]*(?:\.\d+)?)\s*억\s*(?:(\d[\d,]*(?:\.\d+)?)\s*(천)?\s*만"
        r"(?:\s*(\d[\d,]*(?:\.\d+)?)\s*원)?)?",
        text,
    ):
        eok = float(match.group(1).replace(",", ""))
        man_raw = float(match.group(2).replace(",", "")) if match.group(2) else 0.0
        man = man_raw * 1000 if match.group(3) else man_raw  # "5천만" = 5*1000만
        won = float(match.group(4).replace(",", "")) if match.group(4) else 0.0
        combined.append(eok * 10000 + man + won / 10000.0)
        consumed.append(match.span())
    # "억" 없이 단독으로 쓰는 "3천만원"(=3*1000만원) 표기도 결합한다.
    for match in re.finditer(r"(\d[\d,]*(?:\.\d+)?)\s*천\s*만", text):
        if any(start <= match.start() < end for start, end in consumed):
            continue
        combined.append(float(match.group(1).replace(",", "")) * 1000)
        consumed.append(match.span())
    # "81만2,000원"처럼 만원 단위 뒤 나머지를 원 단위로 이어 쓰는 표기를
    # 만원 기준값(81 + 2000/10000 = 81.2)으로 결합한다.
    for match in re.finditer(
        r"(\d[\d,]*(?:\.\d+)?)\s*만\s*(\d[\d,]*(?:\.\d+)?)\s*원", text
    ):
        if any(start <= match.start() < end for start, end in consumed):
            continue
        man = float(match.group(1).replace(",", ""))
        won = float(match.group(2).replace(",", ""))
        combined.append(man + won / 10000.0)
        consumed.append(match.span())
    # 억/천만/만원 표기에 이미 포함된 구간의 개별 숫자 토큰(예: "2억8,800만"의
    # "2","8800")은 결합값(28800)과 별개의 독립 주장으로 다시 세면 안 되므로 제외한다.
    tokens: list[float] = []
    for token_match in re.finditer(r"\d[\d,]*(?:\.\d+)?", text):
        if any(start <= token_match.start() < end for start, end in consumed):
            continue
        tokens.append(float(token_match.group(0).replace(",", "")))
    return combined + tokens


def _answer_numeric_grounding(case: AdvisorCase, result: dict[str, Any],
                              live: bool) -> dict[str, Any]:
    answer = str(result.get("answer") or result.get("message") or "")
    synthesis = (result.get("agent_trace") or {}).get("synthesis") or {}
    if not answer:
        return {
            "passed": not live,
            "answer_present": False,
            "unsupported_numbers": [],
            "reason": "mock structured response" if not live else "missing live synthesis",
        }
    evidence = {key: value for key, value in result.items()
                if key not in {"answer", "message"}}
    numbers: list[float] = []
    _collect_numbers(evidence, numbers)
    numbers.extend(_extract_amounts(case.query))
    variants = list(numbers)
    variants.extend(number * 100 for number in numbers if abs(number) <= 1)
    claimed = _extract_amounts(answer)
    unsupported = []
    for claim in claimed:
        # abs_tol=0.5: 자연어는 12.6분을 "약 13분"처럼 정수로 반올림해
        # 말하는 게 정상이다(오차 0.5 이내). 자릿수를 통째로 빠뜨리거나
        # 10배 단위를 잘못 읽는 실제 오류는 이 허용오차보다 훨씬 크므로
        # 여전히 잡아낸다.
        if not any(math.isclose(claim, value, rel_tol=0.005, abs_tol=0.5)
                   for value in variants):
            unsupported.append(claim)
    synthesis_ok = case.category in {"condition_dialogue", "condition_new_atoms"} or (not live) or (
        synthesis.get("strategy") == "llm_grounded" and synthesis.get("ok") is True
    )
    return {
        "passed": synthesis_ok and not unsupported,
        "answer_present": True,
        "synthesis_grounded": synthesis_ok,
        "unsupported_numbers": unsupported,
    }


def _answer_conclusion_grounding(case: AdvisorCase, result: dict[str, Any],
                                 live: bool) -> dict[str, Any]:
    """Check that the visible answer states the structured decision, not its opposite."""
    if not live:
        return {"passed": True, "reason": "mock structured response"}
    answer = str(result.get("answer") or "")
    if case.category == "lease_compare":
        expected = str((result.get("lease_monte_carlo") or {}).get("preferred") or "")
        return {"passed": bool(expected and expected in answer), "expected": expected}
    if case.category == "market_outlook":
        expected = str((result.get("market_outlook") or {}).get("direction") or "")
        aliases = {
            "상승": ("상승", "오를", "오름"),
            "하락": ("하락", "내릴", "내림"),
            "보합": ("보합", "횡보"),
        }.get(expected, (expected,))
        return {"passed": bool(expected and any(token in answer for token in aliases)),
                "expected": expected}
    if case.category == "buy_or_wait":
        expected = str((result.get("buy_or_wait") or {}).get("recommendation") or "")
        tokens = ("지금", "바로", "매수") if expected == "buy_now" else ("대기", "기다")
        return {"passed": bool(expected and any(token in answer for token in tokens)),
                "expected": expected}
    return {"passed": True, "reason": "structured contract is authoritative"}


def score_advisor_result(case: AdvisorCase, result: dict[str, Any],
                         truth: dict[str, Any], *, live: bool) -> dict[str, Any]:
    trace = result.get("agent_trace") or {}
    planner = trace.get("planner") or {}
    tools = [str(item.get("tool")) for item in trace.get("tools") or []]
    planner_meta = planner.get("llm") or {}
    checks: dict[str, bool] = {
        "pipeline_completed": not bool(result.get("_benchmark_error")),
        "intent_exact": planner.get("intent") == case.expected_intent,
        "status_exact": result.get("status") == case.expected_status,
        "required_tools_grounded": _subsequence(case.required_tools, tools),
        "no_planner_rule_fallback": planner_meta.get("fallback") is not True,
    }
    details: dict[str, Any] = {
        "actual_intent": planner.get("intent"),
        "actual_status": result.get("status"),
        "actual_tools": tools,
    }
    if case.expected_qa_type:
        checks["qa_type_exact"] = result.get("qa_type") == case.expected_qa_type
    if case.expected_mode:
        checks["recommendation_mode_exact"] = (
            result.get("recommendation_mode") == case.expected_mode)

    if case.category in {"condition_dialogue", "condition_new_atoms"}:
        slot_ok, slot_errors = _slot_check(
            case.expected_slots or {}, planner.get("slots") or {})
        checks.update({
            "condition_slots_exact": slot_ok,
            "approval_required_before_execution": (
                result.get("status") == "ask_confirmation"),
            "no_premature_property_search": (
                not tools and result.get("property_search_executed") is False),
        })
        details.update(slot_errors=slot_errors)

    elif case.category in {"best_affordable", "alternative_areas"}:
        rows = _groups_zero(result)
        checks.update({
            "optimization_succeeded": (
                (result.get("optimization") or {}).get("status") == "ok"),
            "pareto_recommendations_nonempty": bool(rows),
            "hard_constraints_satisfied": _all_hard_constraints_true(rows),
        })
        if case.category == "alternative_areas":
            ids = [str(row.get("property_id")) for row in rows]
            dongs = _property_dongs(ids)
            checks["current_dong_excluded"] = (
                result.get("excluded_dong") == "인계동"
                and bool(ids)
                and all(dongs.get(property_id) != "인계동" for property_id in ids)
            )
            details["recommended_property_dongs"] = dongs

    elif case.category == "lease_compare":
        comparison = result.get("lease_monte_carlo") or {}
        scenarios = comparison.get("scenarios") or {}
        quantiles_ok = set(scenarios) == {"전세", "월세"}
        if quantiles_ok:
            for scenario in scenarios.values():
                distribution = scenario.get("terminal_net_worth") or {}
                try:
                    quantiles_ok = quantiles_ok and (
                        float(distribution["p10"]) <= float(distribution["p50"])
                        <= float(distribution["p90"])
                    )
                except (KeyError, TypeError, ValueError):
                    quantiles_ok = False
        expected_preferred = None
        expected_gap = None
        if set(scenarios) == {"전세", "월세"}:
            p50 = {key: float(value["terminal_net_worth"]["p50"])
                   for key, value in scenarios.items()}
            expected_preferred = max(p50, key=p50.get)
            expected_gap = abs(p50["전세"] - p50["월세"])
        horizon = comparison.get("horizon_years")
        checks.update({
            "monte_carlo_contract": (
                comparison.get("path_count_per_option") == 3000
                # horizon_years는 이제 사용자가 밝힌 거주 예정 기간을 반영해
                # 동적으로 정해진다(과거처럼 항상 10년 고정이 아님).
                and isinstance(horizon, (int, float)) and 1 <= horizon <= 30),
            "quantile_order_valid": quantiles_ok,
            "preferred_matches_p50": comparison.get("preferred") == expected_preferred,
            "reported_gap_matches_p50": (
                expected_gap is not None and math.isclose(
                    float(comparison.get("p50_gap_manwon") or 0), expected_gap,
                    abs_tol=0.2)),
        })

    elif case.category == "market_outlook":
        market = result.get("market_outlook") or {}
        checks.update({
            "selected_property_preserved": (
                (market.get("property") or {}).get("property_id")
                == SELECTED_REPORT["property"]["property_id"]),
            "forecast_value_grounded": math.isclose(
                float(market.get("annual_growth_rate") or 0), 0.03, abs_tol=1e-9),
            "forecast_interval_grounded": (
                math.isclose(float(market.get("annual_low") or 0), -0.01, abs_tol=1e-9)
                and math.isclose(float(market.get("annual_high") or 0), 0.06, abs_tol=1e-9)),
            "direction_consistent": market.get("direction") == "상승",
            "time_series_evidence_present": bool(
                (market.get("price_history") or {}).get("available")),
        })

    elif case.category == "buy_or_wait":
        analysis = result.get("buy_or_wait") or {}
        horizons = analysis.get("horizons") or []
        arithmetic_ok = len(horizons) == 2
        for row in horizons:
            years = int(row.get("years") or 0)
            projected = 30000.0 * (1.03 ** years)
            extra = projected - 30000.0 + float(
                row.get("estimated_wait_housing_cost_manwon") or 0)
            arithmetic_ok = arithmetic_ok and math.isclose(
                float(row.get("projected_price_manwon") or 0), projected,
                abs_tol=0.11) and math.isclose(
                float(row.get("extra_required_vs_buy_now_manwon") or 0), extra,
                abs_tol=0.21)
        expected_recommendation = None
        if horizons:
            expected_recommendation = (
                "buy_now" if float(horizons[-1].get(
                    "extra_required_vs_buy_now_manwon") or 0) > 0 else "wait")
        checks.update({
            "buy_wait_status_ok": analysis.get("status") == "ok",
            "one_two_year_horizons": [row.get("years") for row in horizons] == [1, 2],
            "buy_wait_arithmetic_consistent": arithmetic_ok,
            "buy_wait_conclusion_consistent": (
                analysis.get("recommendation") == expected_recommendation),
        })

    elif case.category == "qa_affordability":
        from src.preference.affordability import compute_affordability
        oracle = compute_affordability(DEFAULT_USER).model_dump()
        actual = result.get("affordability") or {}
        numeric_fields = (
            "max_monthly_housing_manwon", "recommended_jeonse_deposit_manwon",
            "recommended_monthly_deposit_manwon", "recommended_monthly_rent_manwon",
            "rir_at_recommended",
        )
        affordability_matches_oracle = bool(actual)
        for field in numeric_fields:
            try:
                affordability_matches_oracle = affordability_matches_oracle and math.isclose(
                    float(actual.get(field)), float(oracle.get(field)),
                    rel_tol=0.005, abs_tol=0.11)
            except (TypeError, ValueError):
                affordability_matches_oracle = False
        checks["affordability_matches_deterministic_oracle"] = affordability_matches_oracle
        details["affordability_oracle"] = oracle

    answer_grounding = _answer_numeric_grounding(case, result, live)
    conclusion_grounding = _answer_conclusion_grounding(case, result, live)
    checks["final_answer_numerically_grounded"] = answer_grounding["passed"]
    checks["final_answer_conclusion_grounded"] = conclusion_grounding["passed"]
    details["answer_grounding"] = answer_grounding
    details["conclusion_grounding"] = conclusion_grounding
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "correct": not failed,
        "checks": checks,
        "passed_checks": sum(checks.values()),
        "total_checks": len(checks),
        "component_accuracy": sum(checks.values()) / len(checks),
        "failed_checks": failed,
        "details": details,
    }


def _condition_context(case: AdvisorCase) -> dict[str, Any]:
    return {
        "state": "idle",
        "known_slots": {},
        "proposed_slots": {},
        "last_question": None,
        "active_conditions": "",
        "initial_universe": "경기 · 수원시 팔달구",
        "condition_scope_policy": (
            "AI 조건은 initial_universe의 교집합 안에서만 후보를 줄이며 "
            "초기 조건을 완화하거나 대체할 수 없음"
        ),
        "recent_dialogue": [{"role": "user", "text": case.query}],
        "approval_channel": "ui_condition_add_button_only",
        "chat_messages_are_condition_edits": True,
    }


def _run_condition_dialogue(llm: Any, case: AdvisorCase) -> dict[str, Any]:
    decision = llm.plan_condition_dialogue(case.query, _condition_context(case))
    decision_name = decision.get("decision")
    if decision_name == "ready_to_draft":
        decision_name = "ask_confirmation"
    slots = {key: value for key, value in (decision.get("slots") or {}).items()
             if value is not None and not str(key).startswith("_")}
    return {
        "status": decision_name,
        "message": decision.get("message") or "조건을 조금 더 알려주세요.",
        "condition_decision": {
            key: value for key, value in decision.items() if key != "_trace"
        },
        "property_search_executed": False,
        "agent_trace": {
            "planner": {
                "intent": "condition_dialogue",
                "action": decision_name,
                "slots": slots,
                "llm": decision.get("_trace") or {},
            },
            "tools": [],
            "fallbacks": [],
        },
    }


def _run_one(agent: JeonseAgent, llm: Any, case: AdvisorCase, truth: dict[str, Any],
             live: bool, batch_started: float, mode: str) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        if case.category in {"condition_dialogue", "condition_new_atoms"}:
            result = _run_condition_dialogue(llm, case)
        else:
            history = _history_for(case) or None
            result = agent.handle(
                _session_for(agent, case), case.query, direct_recommend=True,
                conversation_history=history,
            )
            # NAIVE's internal pipeline never calls synthesize (its LLM's
            # supports_agentic_calls stays False so production stays honest
            # about "no atomic decomposition, no later LLM stages").  Give it
            # one matching final-answer synthesis call here so both arms'
            # user-facing text go through the same grounding checks instead
            # of NAIVE being silently exempted.
            if live and mode == "naive" and not result.get("answer"):
                answer = llm.synthesize(case.query, result,
                                        conversation_history=history)
                trace = result.setdefault("agent_trace", {})
                if answer:
                    result["answer"] = answer
                    trace["synthesis"] = {
                        "strategy": "llm_grounded", "ok": True,
                        "attempts": list(getattr(llm, "last_trace", [])),
                    }
                else:
                    trace["synthesis"] = {"strategy": "template", "ok": False}
                    trace.setdefault("fallbacks", []).append(
                        "최종 문장 합성 실패: 구조화 응답 사용")
    except Exception as exc:
        result = {
            "status": "error",
            "message": "benchmark execution failed",
            "_benchmark_error": f"{type(exc).__name__}: {exc}"[:1000],
            "agent_trace": {"planner": {}, "tools": [], "fallbacks": []},
        }
    score = score_advisor_result(case, result, truth, live=live)
    completed = time.perf_counter()
    return {
        "case_id": case.case_id,
        "category": case.category,
        "query": case.query,
        "result": result,
        "ground_truth": truth,
        "score": score,
        "elapsed_seconds": completed - started,
        "batch_timing": {
            "started_offset_seconds": started - batch_started,
            "completed_offset_seconds": completed - batch_started,
            "worker_elapsed_seconds": completed - started,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="실제 AWS 상담 경로의 혼합 50문항 근거 정확도 평가",
    )
    parser.add_argument("--workers", type=int, default=config.LLM_MAX_CONCURRENCY,
                        help="현재 시스템의 동시 상담 질의 수(기본 6; NAIVE는 직렬)")
    parser.add_argument(
        "--mode", choices=["optimized", "naive"],
        help=("optimized=Atomic·병렬·스케줄링 운영 경로, "
              "naive=사용자 질문 전체를 한 번에 추출하는 직렬 기준선"),
    )
    parser.add_argument("--limit", type=int, default=50,
                        help="평가 문항 수(기본 50, 작은 값은 유형별 round-robin)")
    parser.add_argument("--mock", action="store_true",
                        help="API 비용 없는 구조·채점 점검")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--output-dir", default="reports/experiments")
    return parser


def run_hallucination(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    mode = _choose_mode(args.mode)
    if args.mock:
        os.environ["JEONSE_LLM"] = "mock"
    cases = _stratified_cases(args.limit)
    selected_distribution: dict[str, int] = {}
    for case in cases:
        selected_distribution[case.category] = (
            selected_distribution.get(case.category, 0) + 1)
    workers = _effective_workers(mode, args.workers, len(cases))
    llm = _create_llm(args.no_wait)
    if llm is None:
        return 2
    try:
        _preflight_live_llm(llm)
    except Exception as exc:
        print(f"OpenAI 사전 점검 실패: {type(exc).__name__}: {exc}")
        return 2
    delegate_live = bool(getattr(llm, "supports_agentic_calls", False))
    runtime_llm = (
        llm if mode == "optimized"
        else NaiveWholePromptLLM(llm, fixed_context={"user_profile": DEFAULT_USER})
    )
    # Both arms get exactly one final-answer synthesis call so the numeric/
    # conclusion grounding checks measure hallucination in the user-facing
    # answer on equal footing.  NAIVE's internal pipeline still only uses the
    # LLM once for extraction (no atomic decomposition, no text2sql LLM
    # stage); the benchmark adds a matching synthesize() call in _run_one
    # instead of NAIVE being exempt from grounding checks entirely.
    require_live_synthesis = delegate_live
    agent = JeonseAgent("rule")
    agent.llm = runtime_llm
    agent.text2sql.llm = runtime_llm
    truths = {case.case_id: _ground_truth(case) for case in cases}

    correct = 0
    completed = 0
    started = time.perf_counter()
    console = FixedHeaderConsole(
        lambda: f"정답률: {correct:02d}/{len(cases):02d}  "
                f"({(100 * correct / len(cases)):05.1f}%) | "
                f"{format_elapsed(time.perf_counter() - started)}"
    )
    console.start()
    rows: list[dict[str, Any]] = []
    try:
        console.log(
            "현재 Agentic 시스템" if mode == "optimized"
            else "NAIVE 전체 프롬프트 단일 추출 기준선"
        )
        console.log("혼합 의도 50문항 | 결정론 정답지 | LLM 자기채점 없음")
        console.log(f"workers={workers} | model={getattr(llm, 'model', 'rule')}")
        console.log(f"구성={selected_distribution}")
        with ThreadPoolExecutor(max_workers=workers,
                                thread_name_prefix="advisor-hallucination") as executor:
            futures = {
                executor.submit(
                    _run_one, agent, runtime_llm, case, truths[case.case_id],
                    require_live_synthesis, started, mode
                ): case
                for case in cases
            }
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                completed += 1
                correct += int(row["score"]["correct"])
                console.log("\n" + "=" * 78)
                console.log(
                    f"[{completed:02d}/{len(cases):02d}] {row['case_id']} | "
                    f"{row['category']} | {'PASS' if row['score']['correct'] else 'FAIL'}"
                )
                console.log("[INPUT]")
                console.log(row["query"])
                console.log("\n[FINAL LLM ANSWER]")
                console.log(str(row["result"].get("answer") or row["result"].get("message") or ""))
                console.log("\n[RAW STRUCTURED RESPONSE + RAG TRACE]")
                console.log(json_text(row["result"]))
                console.log("\n[HUMAN/DB GROUND TRUTH]")
                console.log(json_text(row["ground_truth"]))
                console.log("\n[DETERMINISTIC JUDGEMENT]")
                console.log(json_text(row["score"]))
                console.log(f"\n[CASE ELAPSED] {format_elapsed(row['elapsed_seconds'])}")
    finally:
        console.stop()

    elapsed = time.perf_counter() - started
    rows.sort(key=lambda row: row["case_id"])
    per_category: dict[str, dict[str, Any]] = {}
    for category in sorted({row["category"] for row in rows}):
        selected = [row for row in rows if row["category"] == category]
        category_correct = sum(row["score"]["correct"] for row in selected)
        per_category[category] = {
            "correct": category_correct,
            "total": len(selected),
            "accuracy": category_correct / len(selected),
            "mean_component_accuracy": statistics.fmean(
                row["score"]["component_accuracy"] for row in selected),
        }
    credential = os.environ.get("OPENAI_API_KEY", "")
    summary = {
        "correct": correct,
        "total": completed,
        "exact_case_accuracy": correct / completed if completed else 0.0,
        "mean_component_accuracy": statistics.fmean(
            row["score"]["component_accuracy"] for row in rows) if rows else 0.0,
        "elapsed_seconds": elapsed,
        "configured_workers": workers,
        "observed_peak_concurrency": _observed_peak_concurrency(rows),
        "mean_case_seconds": statistics.fmean(
            row["elapsed_seconds"] for row in rows) if rows else 0.0,
        "per_category": per_category,
    }
    report = {
        "metadata": {
            "experiment": "advisor_hallucination_comparison",
            "mode": mode,
            "started_at": datetime.now().astimezone().isoformat(),
            "query_count": len(cases),
            "category_distribution": selected_distribution,
            "llm_provider": getattr(llm, "provider", "local"),
            "llm_model": getattr(llm, "model", "rule"),
            "api_credential_fingerprint": (
                hashlib.sha256(credential.encode("utf-8")).hexdigest()[:12]
                if credential and delegate_live else None),
            "planner_prompt_fingerprint": (
                _prompt_fingerprint() if mode == "optimized"
                else naive_prompt_fingerprint()
            ),
            "pipeline_policy": {
                "atomic_processing": mode == "optimized",
                "parallel_processing": mode == "optimized",
                "dependency_scheduling": mode == "optimized",
                "whole_prompt_single_pass": mode == "naive",
                "llm_planning_calls_per_query": 1,
                "llm_text2sql_stage_enabled": mode == "optimized",
                "llm_final_answer_synthesis_enabled": require_live_synthesis,
            },
            "database_path": str(config.DB_PATH.resolve()),
            "database_fingerprint": database_fingerprint(),
            "evaluation_policy": {
                "llm_as_judge": False,
                "exact_case_requires_all_checks": True,
                "oracle_sources": [
                    "human_authored_intent_and_tool_contract",
                    "read_only_database_verification_for_recommended_properties",
                    "deterministic_financial_and_forecast_arithmetic",
                    "final_answer_numeric_claim_grounding",
                ],
            },
        },
        "summary": summary,
        "cases": rows,
    }
    path = _save_report(args.output_dir, "hallucination", mode, report)
    print("\n" + "=" * 78)
    print(f"최종 정답률: {correct:02d}/{completed:02d} "
          f"({summary['exact_case_accuracy'] * 100:.1f}%)")
    print(f"평균 세부항목 정확도: {summary['mean_component_accuracy'] * 100:.1f}%")
    print(f"workers {workers} | 관측 최대 동시성 {summary['observed_peak_concurrency']}")
    for category, value in per_category.items():
        print(f"  {category}: {value['correct']}/{value['total']} "
              f"({value['accuracy'] * 100:.1f}%)")
    print(f"결과 JSON: {path.resolve()}")
    if not args.no_wait:
        wait_for_key()
    return 0 if completed == len(cases) else 1


def main_hallucination() -> None:
    try:
        raise SystemExit(run_hallucination())
    except KeyboardInterrupt:
        print("\n사용자가 실험을 중단했습니다.")
        raise SystemExit(130)
