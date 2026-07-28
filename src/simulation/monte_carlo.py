"""NumPy 기반 월 단위 확률적 자산 시뮬레이션.

LLM은 이 모듈에 관여하지 않는다. 동일 입력과 seed는 동일 결과를 반환한다.
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from .distributions import (
    annual_interval_sigma, correlated_monthly_shocks, monthly_log_mean,
)
from .metrics import lower_tail_mean, percentile_snapshot, probability


DEFAULT_PATHS = 10_000


def _selected_projection(budget: dict) -> dict:
    comparison = budget.get("finance_comparison") or {}
    selected = str(comparison.get("selected_program_id") or "")
    if selected:
        for option in comparison.get("options") or []:
            if str(option.get("program_id") or "") == selected:
                return option
    return comparison.get("baseline") or {}


def _annual_schedule(base_path: list[dict], key: str, years: int) -> np.ndarray:
    values = np.zeros(years * 12 + 1, dtype=float)
    by_year = {int(row.get("year") or 0): float(row.get(key) or 0)
               for row in base_path}
    for month in range(1, years * 12 + 1):
        values[month] = by_year.get(math.ceil(month / 12), 0.0) / 12.0
    return values


def _simulate(
    user: dict,
    prop: dict,
    forecast: dict,
    budget: dict,
    assumptions: dict,
    *,
    paths: int,
    seed: int,
    rate_stress: float,
) -> dict:
    rng = np.random.default_rng(seed)
    base_path = (budget.get("scenarios") or {}).get("base") or []
    first = base_path[0] if base_path else {}
    years = int((budget.get("assumptions") or {}).get("horizon_years") or 10)
    months = years * 12
    transaction = str(prop.get("transaction_type") or "")
    projection = _selected_projection(budget)

    liquid = np.full(paths, float(first.get("liquid_assets") or 0), dtype=float)
    property_value = np.full(paths, float(first.get("property_value") or 0), dtype=float)
    deposit_asset = np.full(paths, float(first.get("deposit_asset") or 0), dtype=float)
    debt = np.full(paths, float(first.get("debt") or 0), dtype=float)
    initial_net = liquid + property_value + deposit_asset - debt

    income = np.full(paths, float(user.get("monthly_income_manwon") or 0), dtype=float)
    living = np.full(paths, float((budget.get("cashflow") or {}).get(
        "monthly_living_total_manwon") or user.get("monthly_living_cost_manwon") or 0), dtype=float)
    rent = float((budget.get("cashflow") or {}).get("monthly_rent_manwon") or 0)
    maintenance = float((budget.get("cashflow") or {}).get("monthly_maintenance_manwon") or 0)
    scheduled_payment = float(projection.get("initial_monthly_payment_manwon") or 0)
    base_rate = float(projection.get("rate_pct") or 0) / 100.0 + rate_stress
    rate = np.full(paths, max(base_rate, 0), dtype=float)
    repayment_style = str(projection.get("repayment_style") or "")
    if not repayment_style:
        repayment_style = "annuity" if transaction == "매매" else "bullet"

    income_growth = float(assumptions.get("income_growth_rate", 0.03))
    inflation = float(assumptions.get("inflation_rate", 0.02))
    asset_return = float(assumptions.get("liquid_asset_return_rate", 0.03))
    house_growth = float(forecast.get("annual_growth_rate") or 0)
    house_sigma = annual_interval_sigma(
        float(forecast.get("annual_low", house_growth - 0.04)),
        float(forecast.get("annual_high", house_growth + 0.04)),
    )
    vol = np.array([0.025, 0.012, 0.12, house_sigma]) / math.sqrt(12)
    means = np.array([
        monthly_log_mean(income_growth), monthly_log_mean(inflation),
        monthly_log_mean(asset_return), monthly_log_mean(house_growth),
    ])

    child_cost = _annual_schedule(base_path, "annual_child_cost", years)
    inheritance = _annual_schedule(base_path, "inheritance_inflow", years)
    enable_job_loss = bool(assumptions.get("enable_job_loss", False))
    annual_job_loss_probability = max(0.0, min(float(
        assumptions.get("annual_job_loss_probability", 0.05)), 0.5))
    job_loss_income_ratio = max(0.0, min(float(
        assumptions.get("job_loss_income_ratio", 0.2)), 1.0))
    unemployed_months = np.zeros(paths, dtype=int)

    depleted = np.zeros(paths, dtype=bool)
    distress = np.zeros(paths, dtype=bool)
    distress_streak = np.zeros(paths, dtype=np.int16)
    yearly: list[dict[str, Any]] = []

    def capture(year: int) -> None:
        net = liquid + property_value + deposit_asset - debt
        yearly.append({
            "year": year,
            "age": int(user.get("age") or 0) + year if user.get("age") else None,
            "liquid_assets": percentile_snapshot(liquid),
            "property_value": percentile_snapshot(property_value),
            "debt": percentile_snapshot(debt),
            "net_worth": percentile_snapshot(net),
        })

    capture(0)
    for month in range(1, months + 1):
        shocks = correlated_monthly_shocks(rng, paths)
        rates = np.exp(means + shocks * vol)

        if enable_job_loss:
            entering = ((unemployed_months == 0) &
                        (rng.random(paths) < annual_job_loss_probability / 12.0))
            unemployed_months[entering] = rng.integers(3, 10, size=int(entering.sum()))
            income_multiplier = np.where(unemployed_months > 0, job_loss_income_ratio, 1.0)
        else:
            income_multiplier = 1.0

        effective_income = income * income_multiplier
        interest = debt * rate / 12.0
        if repayment_style == "annuity":
            payment = np.minimum(np.full(paths, scheduled_payment), debt + interest)
            principal = np.maximum(payment - interest, 0.0)
            debt = np.maximum(debt - principal, 0.0)
        else:
            payment = np.where(debt > 0, interest, 0.0)

        disposable_before_debt = effective_income - living - rent - maintenance - child_cost[month]
        distress_streak = np.where(payment > np.maximum(disposable_before_debt, 0),
                                   distress_streak + 1, 0)
        distress |= distress_streak >= 3
        cashflow = disposable_before_debt - payment + inheritance[month]
        liquid = liquid * rates[:, 2] + cashflow
        depleted |= liquid < 0
        if transaction == "매매":
            property_value *= rates[:, 3]
        income *= rates[:, 0]
        living *= rates[:, 1]

        if enable_job_loss:
            unemployed_months = np.maximum(unemployed_months - 1, 0)
        if month % 12 == 0:
            capture(month // 12)

    if transaction != "매매":
        repayment = np.minimum(deposit_asset, debt)
        debt -= repayment
        liquid += deposit_asset - repayment
        deposit_asset[:] = 0
        yearly[-1]["liquid_assets"] = percentile_snapshot(liquid)
        yearly[-1]["debt"] = percentile_snapshot(debt)
        yearly[-1]["net_worth"] = percentile_snapshot(
            liquid + property_value + deposit_asset - debt)

    terminal = liquid + property_value + deposit_asset - debt
    delta = terminal - initial_net
    ten_year_index = min(10, years)
    ten_year = yearly[ten_year_index]["net_worth"]
    return {
        "yearly_percentiles": yearly,
        "terminal_net_worth": percentile_snapshot(terminal),
        "ten_year_net_worth": ten_year,
        "cash_depletion_probability": probability(depleted),
        "repayment_distress_probability": probability(distress),
        "cvar_5_terminal_change_manwon": round(lower_tail_mean(delta, 0.05), 1),
        "expected_terminal_net_worth_manwon": round(float(terminal.mean()), 1),
    }


def simulate_probabilistic(
    user: dict,
    prop: dict,
    forecast: dict,
    deterministic_budget: dict,
    assumptions: dict | None = None,
    *,
    paths: int = DEFAULT_PATHS,
    seed: int = 42,
) -> dict:
    """기준·금리 2%p 스트레스 경로를 같은 난수로 계산한다."""
    started = time.perf_counter()
    assumptions = dict(assumptions or {})
    paths = max(500, min(int(paths), 20_000))
    base = _simulate(user, prop, forecast, deterministic_budget, assumptions,
                     paths=paths, seed=int(seed), rate_stress=0.0)
    stress = _simulate(user, prop, forecast, deterministic_budget, assumptions,
                       paths=paths, seed=int(seed), rate_stress=0.02)
    return {
        "model": "vectorized_monthly_monte_carlo_v1",
        "path_count": paths,
        "seed": int(seed),
        "horizon_years": int((deterministic_budget.get("assumptions") or {}).get(
            "horizon_years") or 10),
        "base": base,
        "rate_plus_2pp": stress,
        "stress_delta": {
            "cash_depletion_probability": round(
                stress["cash_depletion_probability"] - base["cash_depletion_probability"], 6),
            "repayment_distress_probability": round(
                stress["repayment_distress_probability"] - base["repayment_distress_probability"], 6),
            "terminal_net_worth_p50_manwon": round(
                stress["terminal_net_worth"]["p50"] - base["terminal_net_worth"]["p50"], 1),
        },
        "distribution_method": (
            "상관 월 충격과 집값 예측구간을 사용하며, 금융상품 비교에는 동일 난수 경로를 적용"
        ),
        "event_policy": {
            "job_loss_enabled": bool(assumptions.get("enable_job_loss", False)),
            "marriage_birth": "사용자가 입력한 계획만 반영",
            "inheritance": "사용자가 입력한 나이·금액만 반영",
        },
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "disclaimer": (
            "확률은 입력 가정과 과거 변동성에 따른 모형상 결과이며 실제 연체확률이나 투자수익을 보장하지 않습니다."
        ),
    }
