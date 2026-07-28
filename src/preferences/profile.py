"""사용자 성향 정규화와 보수적인 자연어 초안 추출."""
from __future__ import annotations

import re


DIMENSIONS = (
    "asset_growth", "monthly_burden", "safety", "commute",
    "liquidity", "debt_aversion",
)

PRESETS = {
    "balanced": dict(asset_growth=0.24, monthly_burden=0.20, safety=0.20,
                     commute=0.14, liquidity=0.12, debt_aversion=0.10),
    "stable": dict(asset_growth=0.12, monthly_burden=0.21, safety=0.27,
                   commute=0.10, liquidity=0.16, debt_aversion=0.14),
    "growth": dict(asset_growth=0.39, monthly_burden=0.13, safety=0.13,
                   commute=0.10, liquidity=0.12, debt_aversion=0.13),
}


def normalize_preferences(raw: dict | None = None) -> dict:
    raw = dict(raw or {})
    mode = str(raw.get("mode") or "balanced")
    values = dict(PRESETS.get(mode, PRESETS["balanced"]))
    supplied = raw.get("weights") or raw
    for dimension in DIMENSIONS:
        if supplied.get(dimension) is not None:
            values[dimension] = max(0.0, min(float(supplied[dimension]), 1.0))
    total = sum(values.values()) or 1.0
    values = {key: round(value / total, 6) for key, value in values.items()}
    risk_tolerance = str(raw.get("risk_tolerance") or mode)
    if risk_tolerance not in {"stable", "balanced", "growth"}:
        risk_tolerance = "balanced"
    return {
        "version": "preference_profile_v1", "mode": mode,
        "risk_tolerance": risk_tolerance, "weights": values,
        "approved": bool(raw.get("approved", False)),
        "source": str(raw.get("source") or "setup_ui"),
    }


def preferences_from_text(text: str) -> dict:
    """LLM 승인 전 사용할 가중치 초안. 명시된 표현만 반영한다."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    signals: dict[str, float] = {}
    rules = {
        "asset_growth": ("자산", "수익", "오를", "성장"),
        "monthly_burden": ("월 부담", "생활비", "주거비", "저렴", "월세"),
        "safety": ("안전", "치안", "전세사기", "보증사고", "보증"),
        "commute": ("통근", "출퇴근", "직장", "학교", "가까"),
        "liquidity": ("현금", "유동성", "비상금"),
        "debt_aversion": ("대출", "빚", "부채", "상환 부담"),
    }
    for key, words in rules.items():
        hits = sum(word in text for word in words)
        if hits:
            signals[key] = min(0.25 + hits * 0.15, 0.7)
    draft = normalize_preferences({"weights": signals, "source": "natural_language_draft"})
    draft["approved"] = False
    draft["detected_dimensions"] = sorted(signals)
    draft["requires_confirmation"] = bool(signals)
    return draft
