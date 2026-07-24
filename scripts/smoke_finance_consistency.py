"""Production smoke test for purchase-finance consistency."""
from __future__ import annotations

import json
import os

import requests


def main() -> None:
    base = os.environ.get("SMOKE_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    session = requests.post(f"{base}/session", json={
        "age": 29, "monthly_income_manwon": 350, "total_asset_manwon": 5_000,
        "monthly_living_cost_manwon": 120, "transaction_types": ["매매"],
        "employment_type": "employee", "employment_months": 36,
        "household_role": "prospective", "home_ownership_count": 0,
        "marital_status": "single", "spouse_annual_income_manwon": None,
        "minor_children_count": 0, "is_korean_national": True,
        "has_income_proof": True, "contract_deposit_paid_5pct": True,
    }, timeout=60)
    session.raise_for_status()
    session_id = session.json()["session_id"]

    search = requests.post(f"{base}/api/properties/search", json={
        "session_id": session_id, "limit": 1, "sort_by": "price_desc",
    }, timeout=120)
    search.raise_for_status()
    rows = search.json().get("properties") or search.json().get("rows") or []
    if not rows:
        raise RuntimeError("No purchase listing was returned.")

    report = requests.post(f"{base}/api/properties/report", json={
        "session_id": session_id, "property_id": rows[0]["property_id"],
        "horizon_years": 5,
    }, timeout=240)
    report.raise_for_status()
    budget = report.json().get("budget") or {}
    funding = budget.get("funding") or {}
    products = budget.get("compatible_finance_programs") or []

    names = [str(p.get("name") or "") for p in products]
    if any("매직카" in name or "자동차" in name for name in names):
        raise RuntimeError(f"Car loan leaked into housing candidates: {names}")
    if any("청약통장" in name for name in names):
        raise RuntimeError(f"Savings account leaked into immediate purchase loans: {names}")
    if products and not any(p.get("eligibility_checks") for p in products):
        raise RuntimeError("No structured eligibility evidence was returned.")

    valid = bool(funding.get("simulation_valid"))
    gap = float(funding.get("funding_gap_manwon") or 0)
    affordable = bool(funding.get("affordable_under_guardrail"))
    verdict = str(funding.get("verdict_code") or "")
    if valid != (gap <= 0.01 and affordable):
        raise RuntimeError("Simulation validity contradicts funding or affordability.")
    if not valid and verdict not in {"no_eligible_finance", "funding_shortfall", "repayment_overload"}:
        raise RuntimeError(f"Unexpected purchase-block reason: {verdict!r}")

    print(json.dumps({
        "property_id": rows[0]["property_id"], "compatible_products": names,
        "chosen_program": funding.get("chosen_program_name"),
        "loan_amount_manwon": funding.get("known_product_loan_manwon"),
        "funding_gap_manwon": gap, "monthly_payment_affordable": affordable,
        "simulation_valid": valid, "verdict_code": verdict,
        "verdict_title": funding.get("verdict_title"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
