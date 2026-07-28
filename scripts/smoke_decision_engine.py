"""End-to-end smoke test for the probabilistic decision engine.

Run inside the application container (or against a local server):

    python scripts/smoke_decision_engine.py

The script creates disposable in-memory session state and a persisted audit
record.  It never mutates property or finance source data.
"""
from __future__ import annotations

import json
import os

import requests


def _post(base: str, path: str, payload: dict, timeout: int = 180) -> dict:
    response = requests.post(f"{base}{path}", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def main() -> None:
    base = os.environ.get("SMOKE_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    session = _post(base, "/session", {
        "age": 29,
        "monthly_income_manwon": 420,
        "total_asset_manwon": 12_000,
        "monthly_living_cost_manwon": 150,
        "transaction_types": [],
        "house_types": [],
        "employment_type": "employee",
        "employment_months": 48,
        "home_ownership_count": 0,
        "marital_status": "single",
        "minor_children_count": 0,
        "is_korean_national": True,
        "has_income_proof": True,
        "preferences": {
            "mode": "balanced",
            "risk_tolerance": "balanced",
            "approved": True,
        },
    })
    session_id = session["session_id"]

    search = _post(base, "/api/properties/search", {
        "session_id": session_id,
        "limit": 8,
        "sort_by": "recommended",
    })
    properties = search.get("properties") or []
    if not properties:
        raise RuntimeError("property search returned no rows")
    search_ids = {row["property_id"] for row in properties}

    optimized = _post(base, "/api/optimization/pareto", {
        "session_id": session_id,
        "property_ids": list(search_ids),
        "horizon_years": 10,
    })
    representatives = optimized.get("representatives") or []
    if optimized.get("status") != "ok" or not representatives:
        raise RuntimeError(f"optimizer failed: {optimized.get('status')!r}")
    if any(row.get("property_id") not in search_ids for row in representatives):
        raise RuntimeError("optimizer escaped the current search intersection")

    optimization_run_id = optimized["decision_run_id"]
    optimization_audit = requests.get(
        f"{base}/api/decisions/{optimization_run_id}", timeout=30,
    )
    optimization_audit.raise_for_status()
    optimization_audit_payload = optimization_audit.json()
    if optimization_audit_payload.get("status") != "completed":
        raise RuntimeError("optimization audit is incomplete")

    report = _post(base, "/api/properties/report", {
        "session_id": session_id,
        "property_id": properties[0]["property_id"],
        "horizon_years": 3,
        "monte_carlo_paths": 10_000,
        "simulation_seed": 20260727,
        "enable_job_loss": True,
    }, timeout=300)
    simulation = report.get("probabilistic_simulation") or {}
    if simulation.get("path_count") != 10_000:
        raise RuntimeError("report did not execute 10,000 Monte Carlo paths")
    if simulation.get("seed") != 20260727:
        raise RuntimeError("simulation seed was not preserved")

    report_run_id = report["decision_run_id"]
    report_audit = requests.get(
        f"{base}/api/decisions/{report_run_id}", timeout=30,
    )
    report_audit.raise_for_status()
    report_audit_payload = report_audit.json()
    if report_audit_payload.get("status") != "completed":
        raise RuntimeError("property report audit is incomplete")

    forecast = report.get("forecast") or {}
    summary = {
        "search_total": search.get("total"),
        "search_returned": len(properties),
        "optimizer_status": optimized.get("status"),
        "optimizer_candidates": optimized.get("candidates_evaluated"),
        "pareto_candidates": optimized.get("pareto_candidate_count"),
        "representatives": len(representatives),
        "optimization_audit_steps": len(optimization_audit_payload.get("steps") or []),
        "simulation_paths": simulation.get("path_count"),
        "simulation_seed": simulation.get("seed"),
        "cash_depletion_probability": (
            simulation.get("base") or {}).get("cash_depletion_probability"),
        "repayment_distress_probability": (
            simulation.get("base") or {}).get("repayment_distress_probability"),
        "forecast_model_version": forecast.get("model_version"),
        "report_audit_steps": len(report_audit_payload.get("steps") or []),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
