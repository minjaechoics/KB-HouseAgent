"""End-to-end smoke test for the deployed final product.

Run this inside the app container so Basic Auth and transient public-network
conditions do not affect application verification.
"""
from __future__ import annotations

import json
import os

import requests


def main() -> None:
    base = os.environ.get("SMOKE_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    session = requests.post(
        f"{base}/session",
        json={
            "age": 29,
            "monthly_income_manwon": 350,
            "total_asset_manwon": 12_000,
            "monthly_living_cost_manwon": 120,
            "transaction_types": ["전세"],
            "employment_type": "employee",
            "employment_months": 36,
            "household_role": "prospective",
            "home_ownership_count": 0,
            "marital_status": "single",
            "spouse_annual_income_manwon": None,
            "minor_children_count": 0,
            "children_plans": [],
            "expected_inheritance_manwon": 2000,
            "expected_inheritance_age": 40,
            "workplace_or_school": "아주대학교",
            "is_korean_national": True,
            "has_income_proof": True,
            "contract_deposit_paid_5pct": True,
        },
        timeout=60,
    )
    session.raise_for_status()
    session_id = session.json()["session_id"]

    chat = requests.post(
        f"{base}/chat",
        json={
            "session_id": session_id,
            "text": "내 조건에서 이용할 수 있는 전세 금융상품을 알려줘",
        },
        timeout=180,
    )
    chat.raise_for_status()
    chat_payload = chat.json()
    chat_trace = json.dumps(
        chat_payload.get("agent_trace") or {}, ensure_ascii=False)

    search = requests.post(
        f"{base}/api/properties/search",
        json={"session_id": session_id, "limit": 1, "sort_by": "recommended"},
        timeout=120,
    )
    search.raise_for_status()
    search_payload = search.json()
    rows = search_payload.get("properties") or search_payload.get("rows") or []
    if not rows:
        raise RuntimeError("property search returned no rows")

    report = requests.post(
        f"{base}/api/properties/report",
        json={
            "session_id": session_id,
            "property_id": rows[0]["property_id"],
            "horizon_years": 2,
            "selected_finance_program_id": "__none__",
            "requested_loan_amount_manwon": 0,
        },
        timeout=240,
    )
    report.raise_for_status()
    report_payload = report.json()
    budget = report_payload.get("budget") or {}
    funding = budget.get("funding") or {}
    scenario_metadata = budget.get("scenario_metadata") or {}
    final_assessment = report_payload.get("final_assessment") or {}

    summary = {
        "chat_answer_present": bool(chat_payload.get("answer")),
        "dadungi_in_trace": "다둥이" in chat_trace,
        "property_count": len(rows),
        "transaction_type": rows[0].get("transaction_type"),
        "finance_applied": bool(funding.get("chosen_program_id")),
        "loan_amount_manwon": funding.get("known_product_loan_manwon"),
        "price_affects_user_asset": scenario_metadata.get(
            "price_affects_user_asset"),
        "emphasis_count": len(
            (report_payload.get("ai_emphasis") or {}).get("items") or []
        ),
        "official_public_csv_present": (
            "official_public_csv"
            in json.dumps(report_payload, ensure_ascii=False)
        ),
        "final_assessment_strategy": final_assessment.get("strategy"),
        "final_assessment_recommendation": final_assessment.get("recommendation"),
        "final_assessment_score": final_assessment.get("score"),
        "workplace_route_rows": len((budget.get("lifestyle") or {}).get("destinations") or []),
        "guarantee_candidates": len(
            (report_payload.get("contract_safety") or {}).get("guarantee_candidates") or []),
    }
    expected = {
        "chat_answer_present": True,
        "dadungi_in_trace": False,
        "transaction_type": "전세",
        "finance_applied": False,
        "loan_amount_manwon": 0.0,
        "price_affects_user_asset": False,
    }
    for key, expected_value in expected.items():
        if summary[key] != expected_value:
            raise RuntimeError(
                f"{key}: expected {expected_value!r}, got {summary[key]!r}")
    if summary["final_assessment_strategy"] not in {"llm_structured", "deterministic_fallback"}:
        raise RuntimeError("final assessment was not generated")
    if summary["guarantee_candidates"] < 1:
        raise RuntimeError("lease guarantee candidates were not generated")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
