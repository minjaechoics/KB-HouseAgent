"""End-to-end smoke check for a deployed Jeonse Helper API."""
from __future__ import annotations

import json
import os

import requests


def main() -> None:
    base = os.environ.get("SMOKE_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    health = requests.get(f"{base}/health", timeout=10).json()
    sources = requests.get(f"{base}/api/data-sources/status", timeout=10).json()
    session = requests.post(
        f"{base}/session",
        json={
            "age": 29,
            "monthly_income_manwon": 350,
            "total_asset_manwon": 12000,
            "preferred_sido": "대전",
            "preferred_gugun": "유성구",
            "transaction_types": ["전세", "월세", "매매"],
        },
        timeout=10,
    ).json()
    chat_response = requests.post(
        f"{base}/chat",
        json={
            "session_id": session["session_id"],
            "text": "금리 3% 이하 청년 전세대출을 알려줘",
        },
        timeout=120,
    )
    chat_response.raise_for_status()
    chat = chat_response.json()
    if health.get("llm") == "APILLM" and not chat.get("answer"):
        raise RuntimeError("live LLM did not synthesize an answer")
    search = requests.post(
        f"{base}/api/properties/search",
        json={"session_id": session["session_id"], "limit": 3},
        timeout=30,
    ).json()
    rows = search.get("properties") or search.get("rows") or []
    if not rows:
        raise RuntimeError("property search returned no rows")
    report = requests.post(
        f"{base}/api/properties/report",
        json={
            "session_id": session["session_id"],
            "property_id": rows[0]["property_id"],
            "horizon_years": 2,
        },
        timeout=120,
    )
    report.raise_for_status()
    payload = report.json()
    finance_comparison = (payload.get("budget") or {}).get(
        "finance_comparison") or {}
    finance_options = finance_comparison.get("options") or []
    if not finance_options:
        raise RuntimeError("finance comparison returned no options")
    first_option = finance_options[0]
    if not first_option.get("path"):
        raise RuntimeError("finance comparison returned no asset path")
    hero = requests.get(
        f"{base}/assets/youth-home-hero-v1.png", timeout=20)
    hero.raise_for_status()
    if len(hero.content) < 10000:
        raise RuntimeError("landing hero asset is unexpectedly small")
    print(json.dumps({
        "health": health.get("status"),
        "llm": health.get("llm"),
        "model": health.get("model"),
        "chat_answer": bool(chat.get("answer")),
        "chat_synthesis": (
            (chat.get("agent_trace") or {}).get("synthesis") or {}
        ).get("strategy"),
        "property_count": len(rows),
        "property_id": rows[0]["property_id"],
        "regional_market_available": bool(
            (payload.get("regional_market") or {}).get("available")),
        "rone_observations": (sources.get("rone") or {}).get("observation_count"),
        "rtms_observations": (sources.get("price_observations") or {}).get("count"),
        "synthetic_live_policy": (sources.get("policy") or {}).get(
            "synthetic_is_never_presented_as_live"),
        "finance_comparison_options": len(finance_options),
        "finance_repayment_path_years": len(first_option["path"]),
        "landing_hero_bytes": len(hero.content),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
