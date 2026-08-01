"""선택 매물에서 거주할 때의 조건부 자산·현금흐름 시뮬레이션."""
from __future__ import annotations

import math
import re
from datetime import date


DEFAULTS = {
    "horizon_years": 10, "income_growth_rate": 0.03,
    "inflation_rate": 0.02, "liquid_asset_return_rate": 0.03,
    "fallback_financing_rate": 0.045,
}


def _child_annual_cost_manwon(child_age: int) -> float:
    """Scenario cost envelope per child (만원/년), not a household prediction."""
    if child_age < 0:
        return 0.0
    anchors = ((0, 600.0), (5, 850.0), (12, 1150.0), (18, 1700.0),
               (22, 2200.0), (25, 2600.0), (26, 650.0), (28, 180.0),
               (30, 0.0))
    if child_age >= anchors[-1][0]:
        return 0.0
    for (left_age, left_cost), (right_age, right_cost) in zip(anchors, anchors[1:]):
        if left_age <= child_age <= right_age:
            ratio = (child_age - left_age) / max(right_age - left_age, 1)
            return left_cost + (right_cost - left_cost) * ratio
    return 0.0


def _family_event(user: dict, current_age: int, year: int) -> tuple[float, float, list[str]]:
    calendar_year = date.today().year + year
    child_cost = 0.0
    labels: list[str] = []
    for index, plan in enumerate(user.get("children_plans") or [], start=1):
        birth_year = int((plan or {}).get("birth_year") or 0)
        if not birth_year:
            continue
        child_age = calendar_year - birth_year
        child_cost += _child_annual_cost_manwon(child_age)
        if child_age in {0, 7, 20, 23, 26, 30}:
            labels.append(f"자녀 {index} · {child_age}세")
    inheritance = 0.0
    inheritance_age = user.get("expected_inheritance_age")
    if inheritance_age is not None and current_age + year == int(inheritance_age):
        inheritance = float(user.get("expected_inheritance_manwon") or 0)
        if inheritance > 0:
            labels.append("예상 증여·상속 유입")
    return child_cost, inheritance, labels


def _annuity_payment(principal: float, annual_rate: float, years: int) -> float:
    if principal <= 0:
        return 0.0
    monthly_rate = annual_rate / 12.0
    months = years * 12
    if monthly_rate <= 0:
        return principal / months
    return principal * monthly_rate / (1 - (1 + monthly_rate) ** -months)


def _capital(prop: dict) -> float:
    transaction = prop.get("transaction_type")
    if transaction == "매매":
        return float(prop.get("sale_price_manwon") or prop.get("asking_price_manwon") or 0)
    return float(prop.get("deposit_manwon") or 0)


def _compatible_loan(
    program: dict, transaction: str, prop: dict | None = None
) -> bool:
    text = " ".join(str(program.get(key) or "") for key in
                    ("name", "category", "product_kind", "support_content",
                     "eligibility_text", "loan_limit_text"))
    if "대출" not in text:
        return False
    if transaction == "매매":
        if "청약통장" in text:
            return False
        if not any(term in text for term in (
            "주택담보", "아파트담보", "주택자금", "주택구입",
            "주택 구입", "주택 매매", "구입대출", "경매주택", "부동산 구입",
        )):
            return False
        if any(term in text for term in ("전세자금", "임차보증금", "자동차대출")):
            return False
        if "아파트담보" in text and prop:
            house_text = " ".join(str(prop.get(key) or "") for key in
                                  ("house_type", "property_type"))
            if "아파트" not in house_text:
                return False
        return True
    return any(term in text for term in (
        "전세", "임차", "보증금", "월세자금",
    ))


def _effective_rate_pct(program: dict, fallback_rate: float) -> tuple[float, str]:
    """Return a conservative usable rate and its evidence level."""
    if not program:
        return 0.0, "none"
    for key, source in (
        ("rate_max_pct", "published_range_max"),
        ("rate_pct", "published_representative"),
        ("rate_min_pct", "published_range_min"),
    ):
        value = program.get(key)
        if value is not None and float(value) >= 0:
            return float(value), source
    return float(fallback_rate) * 100.0, "fallback_assumption"


def _loan_term_years(program: dict, transaction: str) -> int:
    if transaction != "매매":
        return 2
    text = " ".join(str((program or {}).get(key) or "") for key in
                    ("loan_period_text", "repayment_method"))
    values = [int(value) for value in re.findall(r"(\d+)\s*년", text)]
    return max(1, min(max(values) if values else 30, 50))


def _loan_to_capital_ratio(program: dict, transaction: str = "") -> float:
    text = str((program or {}).get("loan_limit_text") or "")
    values = [float(value) / 100.0 for value in re.findall(r"(\d+(?:\.\d+)?)\s*%", text)]
    if values:
        return max(0.0, min(max(values), 1.0))
    product_text = " ".join(str((program or {}).get(key) or "") for key in
                            ("name", "category", "eligibility_text"))
    if transaction == "매매" and any(
            term in product_text for term in ("담보대출", "주택담보", "아파트담보")):
        return 0.70
    return 0.80 if transaction != "매매" else 0.70


def _repayment_style(program: dict, transaction: str) -> str:
    text = str((program or {}).get("repayment_method") or "")
    if transaction != "매매" and "일시상환" in text:
        return "bullet"
    if "원리금균등" in text or "분할상환" in text or transaction == "매매":
        return "annuity"
    return "bullet" if transaction != "매매" else "annuity"


def _affordable_loan_cap(income: float, living: float, monthly_rent: float,
                         maintenance: float, annual_rate: float, years: int,
                         style: str) -> tuple[float, float]:
    """Internal cash-flow guardrail, not a bank DSR approval calculation."""
    payment_cap = max(min(income * 0.30, income - living - monthly_rent - maintenance), 0.0)
    if payment_cap <= 0:
        return 0.0, payment_cap
    if style == "bullet":
        principal = payment_cap * 12.0 / max(annual_rate, 0.0001)
    else:
        monthly_rate = annual_rate / 12.0
        months = max(years * 12, 1)
        principal = (payment_cap * months if monthly_rate <= 0 else
                     payment_cap * (1 - (1 + monthly_rate) ** -months) / monthly_rate)
    return max(principal, 0.0), payment_cap


def _finance_comparison_projection(
    user: dict,
    prop: dict,
    forecast: dict,
    program: dict,
    assumptions: dict,
    lifestyle: dict,
    years: int,
    current_age: int,
) -> dict:
    """Project one financing choice, including principal repayment."""
    transaction = str(prop.get("transaction_type") or "")
    capital = _capital(prop)
    asset = float(user.get("total_asset_manwon") or 0)
    income = float(user.get("monthly_income_manwon") or 0)
    living = float(
        lifestyle.get("effective_monthly_living_cost_manwon")
        if lifestyle.get("effective_monthly_living_cost_manwon") is not None
        else user.get("monthly_living_cost_manwon") or 0
    )
    maintenance = float(prop.get("maintenance_fee_manwon") or 0)
    monthly_rent = (float(prop.get("monthly_rent_manwon") or 0)
                    if transaction == "월세" else 0.0)
    emergency_fund = living * 6
    available_cash = max(asset - emergency_fund, 0.0)
    cash_used = min(capital, available_cash)
    gap = max(capital - cash_used, 0.0)
    rate_pct, rate_source = _effective_rate_pct(
        program, float(assumptions["fallback_financing_rate"]))
    annual_rate = rate_pct / 100.0
    term_years = _loan_term_years(program, transaction)
    repayment_style = _repayment_style(program, transaction)
    loan_ratio = _loan_to_capital_ratio(program, transaction)
    ratio_limit = capital * loan_ratio
    published_limit = float((program or {}).get("max_amount_manwon") or 0)
    effective_limit = published_limit if published_limit > 0 else ratio_limit
    affordable_limit, payment_cap = _affordable_loan_cap(
        income, living, monthly_rent, maintenance, annual_rate,
        term_years, repayment_style)
    recommended_loan = min(
        gap, max(effective_limit, 0.0), ratio_limit, affordable_limit
    ) if program else 0.0
    requested = assumptions.get("requested_loan_amount_manwon")
    loan = (min(float(requested), gap, max(effective_limit, 0.0), ratio_limit)
            if program and requested is not None else recommended_loan)
    shortfall = max(gap - loan, 0.0)
    scheduled_monthly = (
        _annuity_payment(loan, annual_rate, term_years)
        if repayment_style == "annuity" else loan * annual_rate / 12.0
    )

    transaction_cost = capital * (0.015 if transaction == "매매" else 0.003)
    liquid = max(asset - cash_used - transaction_cost, 0.0)
    property_value = capital if transaction == "매매" else 0.0
    deposit_asset = capital if transaction != "매매" else 0.0
    debt = loan
    income_growth = float(assumptions["income_growth_rate"])
    inflation = float(assumptions["inflation_rate"])
    asset_return = float(assumptions["liquid_asset_return_rate"])
    house_growth = float(forecast.get("annual_growth_rate") or 0)
    cumulative_interest = 0.0
    cumulative_principal = 0.0
    payoff_year = None

    def snapshot(year: int, monthly_payment: float = 0.0,
                 monthly_income: float | None = None,
                 monthly_living: float | None = None,
                 annual_surplus: float = 0.0,
                 annual_asset_return: float = 0.0,
                 annual_interest: float = 0.0,
                 annual_principal: float = 0.0,
                 child_cost: float = 0.0,
                 inheritance_inflow: float = 0.0,
                 event_labels: list[str] | None = None) -> dict:
        return {
            "year": year,
            "age": current_age + year if current_age else None,
            "liquid_assets": round(liquid, 1),
            "property_value": round(property_value, 1),
            "deposit_asset": round(deposit_asset, 1),
            "debt": round(debt, 1),
            "unfunded_gap": round(shortfall, 1),
            "monthly_loan_payment": round(monthly_payment, 1),
            "monthly_income": round(income if monthly_income is None else monthly_income, 1),
            "monthly_living_cost": round(living if monthly_living is None else monthly_living, 1),
            "annual_surplus": round(annual_surplus, 1),
            "annual_asset_return": round(annual_asset_return, 1),
            "annual_interest": round(annual_interest, 1),
            "annual_principal": round(annual_principal, 1),
            "annual_child_cost": round(child_cost, 1),
            "inheritance_inflow": round(inheritance_inflow, 1),
            "event_labels": event_labels or [],
            "cumulative_interest": round(cumulative_interest, 1),
            "cumulative_principal": round(cumulative_principal, 1),
            "net_worth": round(
                liquid + property_value + deposit_asset - debt - shortfall, 1),
        }

    path = [snapshot(0, scheduled_monthly)]
    current_income, current_living = income, living
    for year in range(1, years + 1):
        interest_paid = 0.0
        principal_paid = 0.0
        loan_paid = 0.0
        if repayment_style == "annuity":
            for _ in range(12):
                if debt <= 0.00001:
                    debt = 0.0
                    break
                interest = debt * annual_rate / 12.0
                payment = min(scheduled_monthly, debt + interest)
                principal = max(payment - interest, 0.0)
                debt = max(debt - principal, 0.0)
                interest_paid += interest
                principal_paid += principal
                loan_paid += payment
            if loan > 0 and debt <= 0.00001 and payoff_year is None:
                payoff_year = year
        else:
            interest_paid = debt * annual_rate
            loan_paid = interest_paid

        child_cost, inheritance_inflow, event_labels = _family_event(
            user, current_age, year)
        annual_housing = (monthly_rent + maintenance) * 12 + loan_paid
        annual_surplus = ((current_income - current_living) * 12
                          - annual_housing - child_cost)
        annual_asset_return = liquid * asset_return
        liquid = max(0.0, liquid + annual_asset_return + annual_surplus
                     + inheritance_inflow)
        property_value *= 1 + house_growth
        cumulative_interest += interest_paid
        cumulative_principal += principal_paid

        if transaction != "매매" and year == years:
            repayment = min(deposit_asset, debt)
            debt = max(debt - repayment, 0.0)
            cumulative_principal += repayment
            liquid += max(deposit_asset - repayment, 0.0)
            deposit_asset = 0.0
            payoff_year = year if loan > 0 and debt <= 0.00001 else None

        current_income *= 1 + income_growth
        current_living *= 1 + inflation
        path.append(snapshot(
            year, loan_paid / 12.0, current_income, current_living,
            annual_surplus, annual_asset_return,
            interest_paid, principal_paid, child_cost,
            inheritance_inflow, event_labels))

    repayment_method = (
        f"{term_years}년 원리금균등상환 가정"
        if repayment_style == "annuity" else
        "상품 지침상 거주 중 이자 납부·만기 시 반환보증금으로 원금 상환"
    )
    return {
        "program_id": (program or {}).get("program_id"),
        "name": (program or {}).get("name") or "금융상품 미적용",
        "provider": (program or {}).get("provider"),
        "eligibility_status": (program or {}).get("eligibility_status"),
        "eligibility_checks": (program or {}).get("eligibility_checks") or [],
        "eligibility_reviews": (program or {}).get("eligibility_reviews") or [],
        "eligibility_failures": (program or {}).get("eligibility_failures") or [],
        "loan_limit_text": (program or {}).get("loan_limit_text"),
        "source_url": (program or {}).get("source_url"),
        "feasible": (
            shortfall <= 0.01
            and scheduled_monthly <= payment_cap + 0.01
        ),
        "loan_amount_manwon": round(loan, 1),
        "recommended_loan_amount_manwon": round(recommended_loan, 1),
        "requested_loan_amount_manwon": (
            round(float(requested), 1) if requested is not None else None),
        "loan_limit_ratio_pct": round(loan_ratio * 100, 1),
        "published_loan_limit_manwon": (
            round(published_limit, 1) if published_limit > 0 else None),
        "loan_limit_source": (
            "published_amount_and_ratio"
            if published_limit > 0 else "provisional_property_ratio"),
        "monthly_payment_guardrail_manwon": round(payment_cap, 1),
        "monthly_payment_to_income_ratio": (
            round(scheduled_monthly / income, 3) if income else None),
        "affordable_under_guardrail": scheduled_monthly <= payment_cap + 0.01,
        "unfunded_gap_manwon": round(shortfall, 1),
        "rate_pct": round(rate_pct, 3),
        "rate_source": rate_source,
        "rate_is_assumption": rate_source == "fallback_assumption",
        "loan_term_years": term_years,
        "repayment_style": repayment_style,
        "initial_monthly_payment_manwon": round(scheduled_monthly, 1),
        "total_interest_manwon": round(cumulative_interest, 1),
        "total_principal_repaid_manwon": round(cumulative_principal, 1),
        "final_debt_manwon": round(debt, 1),
        "final_net_worth_manwon": path[-1]["net_worth"],
        "payoff_year": payoff_year,
        "payoff_age": current_age + payoff_year
                      if current_age and payoff_year is not None else None,
        "repayment_method": repayment_method,
        "path": path,
    }


def simulate(user: dict, prop: dict, forecast: dict, programs: list[dict],
             assumptions: dict | None = None) -> dict:
    assumptions = {**DEFAULTS, **(assumptions or {})}
    current_age = max(0, int(user.get("age") or 0))
    end_age = assumptions.get("simulation_end_age")
    if end_age is not None and current_age:
        years = int(end_age) - current_age
    else:
        years = int(assumptions["horizon_years"])
    years = max(1, min(years, 50))
    end_age = current_age + years if current_age else None
    income_growth = max(-0.2, min(float(assumptions["income_growth_rate"]), 0.3))
    inflation = max(-0.05, min(float(assumptions["inflation_rate"]), 0.3))
    asset_return = max(-0.3, min(float(assumptions["liquid_asset_return_rate"]), 0.5))
    transaction = str(prop.get("transaction_type") or "")
    capital = _capital(prop)
    asset = float(user.get("total_asset_manwon") or 0)
    income = float(user.get("monthly_income_manwon") or 0)
    lifestyle = dict(assumptions.get("lifestyle") or {})
    living = float(
        lifestyle.get("effective_monthly_living_cost_manwon")
        if lifestyle.get("effective_monthly_living_cost_manwon") is not None
        else user.get("monthly_living_cost_manwon") or 0
    )
    maintenance = float(prop.get("maintenance_fee_manwon") or 0)
    monthly_rent = float(prop.get("monthly_rent_manwon") or 0) if transaction == "월세" else 0.0
    emergency_fund = living * 6
    available_cash = max(asset - emergency_fund, 0.0)
    cash_used = min(capital, available_cash)
    gap = max(capital - cash_used, 0.0)

    compatible = [p for p in programs if _compatible_loan(p, transaction, prop)]
    compatible.sort(key=lambda p: (
        1 if all(p.get(key) is None for key in
                 ("rate_pct", "rate_max_pct", "rate_min_pct")) else 0,
        _effective_rate_pct(p, float(assumptions["fallback_financing_rate"]))[0],
        -float(p.get("max_amount_manwon") or 0),
    ))
    # "구매가능" 필터(harness.py의 _classify_transaction_finance)는 자격요건이
    # 미입력이라 "needs_review"인 상품도 "가입한다고 가정하면 조달 가능"으로
    # 취급해 후보에 포함한다. 여기서 "preliminarily_eligible"만 통과시키면
    # 같은 매물이 필터에서는 구매가능으로 나오고 상세페이지에서는 조달불가로
    # 나오는 모순이 생기므로, 명백히 불충족(not_eligible)인 상품만 제외한다.
    eligible_compatible = [
        p for p in compatible
        if p.get("eligibility_status") != "not_eligible"
    ]
    selected_program_id = str(assumptions.get("selected_finance_program_id") or "")
    candidate_projections = [
        (program, _finance_comparison_projection(
            user, prop, forecast, program, assumptions, lifestyle, years, current_age))
        for program in eligible_compatible
    ]
    chosen = None
    selected_projection = None
    if selected_program_id and selected_program_id != "__none__":
        pair = next((
            pair for pair in candidate_projections
            if str(pair[0].get("program_id") or "") == selected_program_id
        ), None)
        if pair:
            chosen, selected_projection = pair
    elif not selected_program_id and candidate_projections and gap > 0.01:
        chosen, selected_projection = min(
            candidate_projections,
            key=lambda pair: (
                0 if pair[1]["feasible"] else 1,
                float(pair[1]["unfunded_gap_manwon"]),
                float(pair[1]["initial_monthly_payment_manwon"]),
                float(pair[1]["rate_pct"]),
            ),
        )
    if selected_projection is None:
        selected_projection = _finance_comparison_projection(
            user, prop, forecast, None, assumptions, lifestyle, years, current_age)
    loan = float(selected_projection["loan_amount_manwon"])
    shortfall = float(selected_projection["unfunded_gap_manwon"])
    chosen_rate_pct = float(selected_projection["rate_pct"])
    chosen_rate_source = str(selected_projection["rate_source"])
    rate = chosen_rate_pct / 100.0
    monthly_loan = float(selected_projection["initial_monthly_payment_manwon"])

    transaction_cost = capital * (0.015 if transaction == "매매" else 0.003)
    liquid_start = max(asset - cash_used - transaction_cost, 0.0)
    property_start = capital if transaction == "매매" else 0.0
    deposit_asset = capital if transaction != "매매" else 0.0
    base_growth = float(forecast.get("annual_growth_rate") or 0)
    growths = {
        "conservative": float(forecast.get("annual_low", base_growth - 0.04)),
        "base": base_growth,
        "optimistic": float(forecast.get("annual_high", base_growth + 0.04)),
    }

    scenarios = {}
    for scenario, house_growth in growths.items():
        liquid, property_value, debt = liquid_start, property_start, loan
        path = [{"year": 0, "age": current_age or None,
                 "liquid_assets": round(liquid, 1),
                 "property_value": round(property_value, 1),
                 "house_price_index": 100.0,
                 "deposit_asset": round(deposit_asset, 1), "debt": round(debt, 1),
                 "unfunded_gap": round(shortfall, 1),
                 "monthly_income": round(income, 1),
                 "monthly_living_cost": round(living, 1),
                 "annual_surplus": 0.0, "annual_asset_return": 0.0,
                 "annual_interest": 0.0, "annual_principal": 0.0,
                 "annual_child_cost": 0.0, "inheritance_inflow": 0.0,
                 "event_labels": [],
                 "net_worth": round(
                     liquid + property_value + deposit_asset - debt - shortfall, 1)}]
        current_income, current_living = income, living
        for year in range(1, years + 1):
            child_cost, inheritance_inflow, event_labels = _family_event(
                user, current_age, year)
            annual_housing = (monthly_rent + maintenance + monthly_loan) * 12
            annual_surplus = ((current_income - current_living) * 12
                              - annual_housing - child_cost)
            annual_asset_gain = liquid * asset_return
            liquid = max(0.0, liquid + annual_asset_gain + annual_surplus
                         + inheritance_inflow)
            property_value *= 1 + house_growth
            annual_interest = debt * rate
            annual_principal = 0.0
            if transaction == "매매" and debt > 0:
                annual_principal = max(monthly_loan * 12 - debt * rate, 0.0)
                debt = max(0.0, debt - annual_principal)
            current_income *= 1 + income_growth
            current_living *= 1 + inflation
            path.append({"year": year,
                         "age": (current_age + year) if current_age else None,
                         "liquid_assets": round(liquid, 1),
                         "property_value": round(property_value, 1),
                         "house_price_index": round(100 * (1 + house_growth) ** year, 2),
                         "deposit_asset": round(deposit_asset, 1), "debt": round(debt, 1),
                         "unfunded_gap": round(shortfall, 1),
                         "net_worth": round(
                             liquid + property_value + deposit_asset - debt - shortfall, 1),
                         "monthly_income": round(current_income, 1),
                         "monthly_living_cost": round(current_living, 1),
                         "annual_surplus": round(annual_surplus, 1),
                         "annual_asset_return": round(annual_asset_gain, 1),
                         "annual_interest": round(annual_interest, 1),
                         "annual_principal": round(annual_principal, 1),
                         "annual_child_cost": round(child_cost, 1),
                         "inheritance_inflow": round(inheritance_inflow, 1),
                         "event_labels": event_labels})
        scenarios[scenario] = path

    monthly_housing = monthly_rent + maintenance + monthly_loan
    price_affects_user_asset = transaction == "매매"
    base_path = scenarios["base"]
    start, end = base_path[0], base_path[-1]
    drivers = {
        "net_worth_change_manwon": round(end["net_worth"] - start["net_worth"], 1),
        "liquid_asset_change_manwon": round(
            end["liquid_assets"] - start["liquid_assets"], 1),
        "property_value_change_manwon": round(
            end["property_value"] - start["property_value"], 1),
        "debt_reduction_manwon": round(start["debt"] - end["debt"], 1),
    }
    if not price_affects_user_asset:
        explanation = (
            "전·월세 매물이므로 집값 전망은 임대인 자산 참고값일 뿐 사용자의 순자산에 "
            "반영하지 않습니다. 순자산 변화는 저축·금융자산·보증금·대출로 계산합니다."
        )
    elif base_growth < 0 and drivers["net_worth_change_manwon"] > 0:
        explanation = (
            "집값 기준 전망은 하락이지만 예상 저축과 대출 원금 감소가 주택가치 감소보다 "
            "커서 총순자산은 증가합니다. 아래 구성요소를 함께 확인하세요."
        )
    else:
        explanation = (
            "총순자산은 주택가치만이 아니라 금융자산 저축과 대출 원금 감소를 함께 반영합니다."
        )
    comparison_programs = eligible_compatible
    finance_baseline = _finance_comparison_projection(
        user, prop, forecast, None, assumptions, lifestyle, years, current_age)
    finance_options = [
        _finance_comparison_projection(
            user, prop, forecast, program, assumptions, lifestyle, years, current_age)
        for program in comparison_programs
    ]
    funding_possible = bool(selected_projection["feasible"])
    contract_label = "구매" if transaction == "매매" else "계약"
    if gap <= 0.01:
        verdict_code = "cash_possible"
        verdict_title = f"대출 없이 {contract_label} 가능"
        verdict_message = (
            f"계약자금 {capital:,.0f}만원을 비상예비자금 {emergency_fund:,.0f}만원을 "
            "남긴 자기자금만으로 충족하므로 대출을 적용하지 않아도 됩니다."
        )
    elif not eligible_compatible:
        verdict_code = "no_eligible_finance"
        verdict_title = f"현재 입력 기준 {contract_label} 불가"
        verdict_message = (
            "예비 적합으로 판정된 주거대출이 없습니다. 추가 확인 상품의 자격정보를 "
            "보완하거나 자기자금을 늘려야 합니다."
        )
    elif shortfall > 0.01:
        verdict_code = "funding_shortfall"
        verdict_title = f"현재 조달안으로 {contract_label} 불가"
        verdict_message = (
            f"가장 유리한 예비 적합 상품을 적용해도 {shortfall:,.0f}만원이 부족합니다."
        )
    elif not selected_projection["affordable_under_guardrail"]:
        verdict_code = "repayment_overload"
        verdict_title = f"월 상환여력 초과로 {contract_label} 보류"
        verdict_message = "선택 대출의 월 상환액이 현재 소득 기반 내부 안전가이드를 넘습니다."
    else:
        verdict_code = "financeable"
        verdict_title = f"예비 금융조건으로 {contract_label} 가능"
        verdict_message = (
            f"자기자금 {cash_used:,.0f}만원과 {chosen.get('name') if chosen else '예비 적합 대출'} "
            f"{loan:,.0f}만원으로 계약자금 {capital:,.0f}만원을 충족합니다. "
            "공시 자격은 예비 통과했지만 담보평가·DSR·보증서 발급은 은행 최종 심사가 필요합니다."
        )
    return {
        "assumptions": {**assumptions, "horizon_years": years,
                        "simulation_end_age": end_age,
                        "house_price_growth_base": round(base_growth, 4)},
        "funding": {
            "required_capital_manwon": round(capital, 1),
            "cash_used_manwon": round(cash_used, 1), "emergency_fund_manwon": round(emergency_fund, 1),
            "known_product_loan_manwon": round(loan, 1), "funding_gap_manwon": round(shortfall, 1),
            "initial_cash_shortfall_manwon": round(gap, 1),
            "monthly_budget_shortfall_manwon": round(
                max(monthly_loan - float(selected_projection.get(
                    "monthly_payment_guardrail_manwon") or 0), 0.0), 1),
            "recommended_loan_amount_manwon": selected_projection.get("recommended_loan_amount_manwon"),
            "requested_loan_amount_manwon": selected_projection.get("requested_loan_amount_manwon"),
            "monthly_payment_guardrail_manwon": selected_projection.get("monthly_payment_guardrail_manwon"),
            "monthly_payment_to_income_ratio": selected_projection.get("monthly_payment_to_income_ratio"),
            "affordable_under_guardrail": selected_projection.get("affordable_under_guardrail"),
            "repayment_method": selected_projection.get("repayment_method"),
            "feasible_with_known_products": funding_possible,
            "feasible_under_preliminary_product_limits": funding_possible,
            "simulation_valid": funding_possible,
            "verdict_code": verdict_code,
            "verdict_title": verdict_title,
            "verdict_message": verdict_message,
            "eligible_product_count": len(eligible_compatible),
            "review_product_count": sum(
                program.get("eligibility_status") == "needs_review"
                for program in compatible),
            "selection_reason": (
                "필요자금 충족 여부 → 부족액 → 월 상환액 → 공시금리 순 자동선택"
                if not selected_program_id else "사용자가 선택한 금융상품"
            ),
            "chosen_program_id": chosen.get("program_id") if chosen else None,
            "chosen_program_name": chosen.get("name") if chosen else None,
            "assumed_rate_pct": round(rate * 100, 3),
            "rate_source": chosen_rate_source,
            "rate_is_assumption": chosen_rate_source == "fallback_assumption",
        },
        "cashflow": {
            "monthly_rent_manwon": round(monthly_rent, 1),
            "monthly_maintenance_manwon": round(maintenance, 1),
            "monthly_loan_payment_manwon": round(monthly_loan, 1),
            "monthly_housing_total_manwon": round(monthly_housing, 1),
            "monthly_living_total_manwon": round(living, 1),
            "monthly_total_outflow_manwon": round(monthly_housing + living, 1),
            "housing_to_income_ratio": round(monthly_housing / income, 3) if income else None,
        },
        "lifestyle": lifestyle,
        "compatible_finance_programs": [{
            key: program.get(key) for key in (
                "program_id", "name", "provider", "category", "rate_pct",
                "rate_min_pct", "rate_max_pct", "max_amount_manwon", "source_url",
                "eligibility_status", "eligibility_reviews", "repayment_method",
                "eligibility_failures", "eligibility_checks", "loan_period_text",
                "loan_limit_text",
            )
        } for program in compatible],
        "finance_comparison": {
            "selected_program_id": chosen.get("program_id") if chosen else None,
            "baseline": finance_baseline,
            "options": finance_options,
            "basis": "동일한 집값 기준 경로·소득·생활비에서 금융조건만 바꾼 비교",
            "repayment_notice": (
                "상품 공시의 상환방식을 우선 적용합니다. 매매 분할상환은 월 원리금, "
                "전·월세 일시상환은 거주 중 이자와 만기 반환보증금 상환을 반영합니다. "
                "월 상환여력 30%는 내부 안전가이드이며 은행 DSR 심사를 대체하지 않습니다."
            ),
        },
        "scenario_metadata": {
            "price_affects_user_asset": price_affects_user_asset,
            "display_keys": ["conservative", "base", "optimistic"]
                            if price_affects_user_asset else ["base"],
            "labels": {
                "conservative": "집값 하방 경로",
                "base": "집값 기준 경로" if price_affects_user_asset else "거주 기준 경로",
                "optimistic": "집값 상방 경로",
            },
            "growth_rates": {key: round(value, 4) for key, value in growths.items()},
            "explanation": explanation,
            "base_path_drivers": drivers,
            "family_cost_method": (
                "자녀별 연간 비용은 0~25세까지 단계적으로 증가하고 23~25세에 "
                "정점, 26세 이후 급감하는 편집 가능한 시나리오 곡선입니다."
            ),
        },
        "scenarios": scenarios,
        "disclaimer": (
            "세금·대출심사·중도상환·공실 등은 단순화한 조건부 시뮬레이션입니다. "
            "자산 전망이나 금융상품 가입을 보장하지 않습니다."
        ),
    }
