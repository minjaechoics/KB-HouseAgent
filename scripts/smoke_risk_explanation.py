"""Production smoke test for grounded LLM jeonse-risk explanations."""
from __future__ import annotations

import json
import os

import requests


def main() -> None:
    base = os.environ.get("SMOKE_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    session = requests.post(f"{base}/session", json={
        "age": 29, "monthly_income_manwon": 350, "total_asset_manwon": 8_000,
        "monthly_living_cost_manwon": 120, "transaction_types": ["전세"],
        "employment_type": "employee", "employment_months": 36,
        "household_role": "prospective", "home_ownership_count": 0,
        "marital_status": "single", "minor_children_count": 0,
        "is_korean_national": True, "has_income_proof": True,
        "contract_deposit_paid_5pct": True,
    }, timeout=60)
    session.raise_for_status()
    session_id = session.json()["session_id"]

    search = requests.post(f"{base}/api/properties/search", json={
        "session_id": session_id, "limit": 1, "sort_by": "risk_desc",
    }, timeout=120)
    search.raise_for_status()
    rows = search.json().get("properties") or search.json().get("rows") or []
    if not rows:
        raise RuntimeError("No jeonse listing was returned.")

    report = requests.post(f"{base}/api/properties/report", json={
        "session_id": session_id, "property_id": rows[0]["property_id"],
        "horizon_years": 2, "selected_finance_program_id": "__none__",
    }, timeout=300)
    report.raise_for_status()
    explanation = ((report.json().get("contract_safety") or {})
                   .get("risk_explanation") or {})
    if explanation.get("strategy") != "llm_structured":
        raise RuntimeError(f"Risk explanation did not use the live LLM: {explanation}")
    if not (explanation.get("model_evidence") or {}).get("model_drivers"):
        raise RuntimeError("Risk explanation is missing model-grounded drivers.")
    if not explanation.get("factors") or not explanation.get("next_checks"):
        raise RuntimeError("Risk explanation is missing factors or next checks.")
    combined = json.dumps(explanation, ensure_ascii=False)
    for forbidden in ("사기범일 확률", "사기 범죄자일 확률", "손실이 확정됩니다"):
        if forbidden in combined:
            raise RuntimeError(f"Forbidden overclaim in explanation: {forbidden}")

    print(json.dumps({
        "property_id": rows[0]["property_id"],
        "fraud_score": rows[0].get("fraud_score"),
        "strategy": explanation.get("strategy"),
        "headline": explanation.get("headline"),
        "factor_count": len(explanation.get("factors") or []),
        "next_check_count": len(explanation.get("next_checks") or []),
        "model_method": (explanation.get("model_evidence") or {}).get("method"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
