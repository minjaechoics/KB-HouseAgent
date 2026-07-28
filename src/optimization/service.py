"""유효한 조합을 이산화하고 scipy MILP로 대표 파레토 후보를 선택한다."""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from src.preferences import normalize_preferences
from src.report.budget import (
    _affordable_loan_cap, _annuity_payment, _capital, _compatible_loan,
    _effective_rate_pct, _loan_term_years, _loan_to_capital_ratio,
    _repayment_style,
)


PROFILE_WEIGHTS = {
    "personalized": None,
    "asset_growth": dict(asset_growth=.58, monthly_burden=.08, safety=.10,
                         commute=.06, liquidity=.10, debt_aversion=.08),
    "monthly_burden": dict(asset_growth=.12, monthly_burden=.52, safety=.10,
                           commute=.08, liquidity=.10, debt_aversion=.08),
    "safety": dict(asset_growth=.10, monthly_burden=.10, safety=.58,
                   commute=.08, liquidity=.08, debt_aversion=.06),
    "commute": dict(asset_growth=.10, monthly_burden=.10, safety=.10,
                    commute=.58, liquidity=.06, debt_aversion=.06),
}


def _loan_variants(minimum: float, maximum: float, requested: float | None = None) -> list[float]:
    if maximum + 0.01 < minimum:
        return []
    values = {minimum, maximum, minimum + (maximum - minimum) * .5}
    if requested is not None and minimum <= requested <= maximum:
        values.add(requested)
    rounded = {round(value / 500.0) * 500.0 for value in values}
    rounded |= {minimum, maximum}
    return sorted(max(minimum, min(value, maximum)) for value in rounded)


def _project_option(user: dict, prop: dict, program: dict | None, loan: float,
                    years: int, assumptions: dict) -> dict:
    transaction = str(prop.get("transaction_type") or "")
    capital = _capital(prop)
    asset = float(user.get("total_asset_manwon") or 0)
    income = float(user.get("monthly_income_manwon") or 0)
    living = float(user.get("monthly_living_cost_manwon") or 0)
    rent = float(prop.get("monthly_rent_manwon") or 0) if transaction == "월세" else 0.0
    maintenance = float(prop.get("maintenance_fee_manwon") or 0)
    transaction_cost = capital * (0.015 if transaction == "매매" else 0.003)
    emergency = living * 6
    cash_needed = max(capital + transaction_cost - loan, 0.0)
    liquid = asset - cash_needed
    rate_pct, _ = _effective_rate_pct(program, float(
        assumptions.get("fallback_financing_rate", .045)))
    rate = rate_pct / 100.0
    term = _loan_term_years(program or {}, transaction)
    style = _repayment_style(program or {}, transaction)
    monthly_payment = (_annuity_payment(loan, rate, term)
                       if style == "annuity" else loan * rate / 12.0)
    income_growth = float(assumptions.get("income_growth_rate", .03))
    inflation = float(assumptions.get("inflation_rate", .02))
    asset_return = float(assumptions.get("liquid_asset_return_rate", .03))
    house_growth = float(assumptions.get("house_growth_rate", .02))
    debt = loan
    property_value = capital if transaction == "매매" else 0.0
    deposit = capital if transaction != "매매" else 0.0
    current_income, current_living = income, living
    for _ in range(years):
        annual_interest = debt * rate
        principal = 0.0
        if style == "annuity":
            principal = min(max(monthly_payment * 12 - annual_interest, 0.0), debt)
            debt -= principal
        annual_surplus = ((current_income - current_living - rent - maintenance
                           - monthly_payment) * 12)
        liquid = liquid * (1 + asset_return) + annual_surplus
        property_value *= 1 + house_growth
        current_income *= 1 + income_growth
        current_living *= 1 + inflation
    if transaction != "매매":
        repayment = min(deposit, debt)
        debt -= repayment
        liquid += deposit - repayment
        deposit = 0.0
    net_worth = liquid + property_value + deposit - debt
    fraud = prop.get("fraud_score")
    risk = float(fraud) if fraud is not None else 0.25
    safety = max(0.0, min(1.0, 1.0 - risk))
    commute_minutes = prop.get("commute_minutes")
    distance = prop.get("distance_km")
    if commute_minutes is not None:
        commute = max(0.0, 1.0 - float(commute_minutes) / 90.0)
        commute_evidence = "route_minutes"
    elif distance is not None:
        commute = max(0.0, 1.0 - float(distance) / 30.0)
        commute_evidence = "distance_proxy"
    else:
        commute = 0.5
        commute_evidence = "neutral_missing_destination"
    return {
        "terminal_net_worth_manwon": round(net_worth, 1),
        "monthly_housing_cost_manwon": round(rent + maintenance + monthly_payment, 1),
        "terminal_liquid_assets_manwon": round(liquid, 1),
        "safety_score": round(safety, 6), "contract_risk": round(risk, 6),
        "commute_score": round(commute, 6), "commute_evidence": commute_evidence,
        "debt_amount_manwon": round(loan, 1), "final_debt_manwon": round(debt, 1),
        "monthly_payment_manwon": round(monthly_payment, 1),
        "cash_after_contract_manwon": round(asset - cash_needed, 1),
        "emergency_fund_manwon": round(emergency, 1),
    }


def _build_candidates(properties: list[dict], programs: list[dict], user: dict,
                      assumptions: dict) -> tuple[list[dict], dict]:
    years = max(1, min(int(assumptions.get("horizon_years", 10)), 30))
    living = float(user.get("monthly_living_cost_manwon") or 0)
    asset = float(user.get("total_asset_manwon") or 0)
    available = max(asset - living * 6, 0.0)
    requested = assumptions.get("requested_loan_amount_manwon")
    candidates: list[dict] = []
    rejected = {"funding": 0, "eligibility": 0, "repayment": 0}
    eligible = [program for program in programs if program.get(
        "eligibility_status") in (None, "", "preliminarily_eligible")]
    for prop in properties[:60]:
        capital = _capital(prop)
        transaction = str(prop.get("transaction_type") or "")
        transaction_cost = capital * (0.015 if transaction == "매매" else 0.003)
        minimum_loan = max(capital + transaction_cost - available, 0.0)
        option_programs: list[dict | None] = [None] if minimum_loan <= .01 else []
        compatible = [p for p in eligible if _compatible_loan(p, transaction, prop)]
        option_programs.extend(compatible[:8])
        if not option_programs:
            rejected["eligibility"] += 1
        for program in option_programs:
            if program is None:
                loan_values = [0.0]
            else:
                rate_pct, _ = _effective_rate_pct(program, float(
                    assumptions.get("fallback_financing_rate", .045)))
                term = _loan_term_years(program, transaction)
                style = _repayment_style(program, transaction)
                ratio_limit = capital * _loan_to_capital_ratio(program, transaction)
                published = float(program.get("max_amount_manwon") or ratio_limit)
                affordable, _ = _affordable_loan_cap(
                    float(user.get("monthly_income_manwon") or 0), living,
                    float(prop.get("monthly_rent_manwon") or 0),
                    float(prop.get("maintenance_fee_manwon") or 0),
                    rate_pct / 100.0, term, style)
                maximum = min(ratio_limit, published, affordable)
                loan_values = _loan_variants(minimum_loan, maximum, requested)
                if not loan_values:
                    rejected["funding"] += 1
            for loan in loan_values:
                metrics = _project_option(user, prop, program, loan, years, assumptions)
                disposable = max(float(user.get("monthly_income_manwon") or 0) - living, 0)
                if metrics["monthly_payment_manwon"] > disposable * .35 + .01:
                    rejected["repayment"] += 1
                    continue
                candidates.append({
                    "option_id": f"{prop.get('property_id')}::{(program or {}).get('program_id') or 'cash'}::{loan:.1f}",
                    "property_id": prop.get("property_id"),
                    "property": {key: prop.get(key) for key in (
                        "house_type", "transaction_type", "road_address", "dong",
                        "sale_price_manwon", "deposit_manwon", "monthly_rent_manwon",
                        "fraud_score", "lat", "lng")},
                    "finance_program_id": (program or {}).get("program_id"),
                    "finance_program_name": (program or {}).get("name") or "금융상품 미적용",
                    "loan_amount_manwon": round(float(loan), 1),
                    "hard_constraints": {
                        "funding": True, "preliminary_eligibility": True,
                        "monthly_repayment_within_35pct_disposable": True,
                    },
                    **metrics,
                })
    return candidates, rejected


def _normalize(candidates: list[dict]) -> None:
    dimensions = {
        "asset_growth": ("terminal_net_worth_manwon", True),
        "monthly_burden": ("monthly_housing_cost_manwon", False),
        "safety": ("safety_score", True), "commute": ("commute_score", True),
        "liquidity": ("terminal_liquid_assets_manwon", True),
        "debt_aversion": ("debt_amount_manwon", False),
    }
    for target, (source, positive) in dimensions.items():
        values = np.array([float(row[source]) for row in candidates], dtype=float)
        low, high = float(values.min()), float(values.max())
        for row, value in zip(candidates, values):
            score = .5 if high <= low else (float(value) - low) / (high - low)
            row.setdefault("normalized_scores", {})[target] = round(
                score if positive else 1.0 - score, 6)


def _utility(row: dict, weights: dict) -> float:
    return sum(float(weights.get(key, 0)) * float(value)
               for key, value in row["normalized_scores"].items())


def _milp_pick(candidates: list[dict], weights: dict) -> tuple[int, dict]:
    utilities = np.array([_utility(row, weights) for row in candidates])
    trace: dict[str, Any] = {
        "solver": "scipy.optimize.milp_highs", "binary_variables": len(candidates),
        "constraint": "exactly_one_candidate", "objective": weights,
    }
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        result = milp(
            c=-utilities, integrality=np.ones(len(candidates), dtype=int),
            bounds=Bounds(np.zeros(len(candidates)), np.ones(len(candidates))),
            constraints=LinearConstraint(np.ones((1, len(candidates))), [1.0], [1.0]),
            options={"time_limit": 5.0},
        )
        if not result.success or result.x is None:
            raise RuntimeError(str(result.message))
        index = int(np.argmax(result.x))
        trace.update(status="optimal" if result.status == 0 else "feasible",
                     message=str(result.message), objective_value=round(float(-result.fun), 6))
        return index, trace
    except Exception as exc:
        index = int(np.argmax(utilities))
        trace.update(status="deterministic_fallback", error=type(exc).__name__,
                     message=str(exc)[:300], objective_value=round(float(utilities[index]), 6))
        return index, trace


def _pareto_front(candidates: list[dict]) -> list[int]:
    values = np.array([[row["normalized_scores"][key] for key in (
        "asset_growth", "monthly_burden", "safety", "commute", "liquidity",
        "debt_aversion")] for row in candidates])
    keep = np.ones(len(values), dtype=bool)
    for index, value in enumerate(values):
        if not keep[index]:
            continue
        dominated = np.all(values >= value, axis=1) & np.any(values > value, axis=1)
        if dominated.any():
            keep[index] = False
    return np.flatnonzero(keep).tolist()


def optimize_housing_choices(properties: list[dict], programs: list[dict], user: dict,
                             preferences: dict | None = None,
                             assumptions: dict | None = None) -> dict:
    started = time.perf_counter()
    assumptions = dict(assumptions or {})
    preference = normalize_preferences(preferences)
    candidates, rejected = _build_candidates(properties, programs, user, assumptions)
    if not candidates:
        return {
            "status": "infeasible", "candidates_evaluated": 0,
            "message": "자금·자격·월 상환 제약을 모두 만족하는 조합이 없습니다.",
            "rejected": rejected, "preference_profile": preference,
        }
    _normalize(candidates)
    pareto_indices = _pareto_front(candidates)
    pool = [candidates[index] for index in pareto_indices]
    representatives = []
    traces = []
    seen = set()
    for name, preset in PROFILE_WEIGHTS.items():
        weights = preference["weights"] if preset is None else preset
        index, trace = _milp_pick(pool, weights)
        selected = dict(pool[index])
        selected["profile"] = name
        selected["utility_score"] = round(_utility(selected, weights) * 100, 1)
        selected["selection_reason"] = {
            "personalized": "승인된 사용자 성향",
            "asset_growth": "10년 후 순자산 우선",
            "monthly_burden": "월 주거비 우선",
            "safety": "계약·보증사고 안전 우선",
            "commute": "통근 만족도 우선",
        }[name]
        trace["profile"] = name
        traces.append(trace)
        if selected["option_id"] not in seen:
            representatives.append(selected)
            seen.add(selected["option_id"])
    return {
        "status": "ok", "solver": "mixed_integer_linear_programming",
        "candidate_generation": "property_finance_discrete_loan_grid",
        "candidates_evaluated": len(candidates), "pareto_candidate_count": len(pool),
        "representatives": representatives, "preference_profile": preference,
        "rejected": rejected, "solver_traces": traces,
        "risk_policy": "fraud risk is a soft objective, never a hard search filter",
        "commute_policy": "live route evidence when present; missing destination uses neutral score",
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
