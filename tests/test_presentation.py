"""LLM 주 답변과 결정론적 조회 근거가 중복 없이 분리되는지 검증한다."""
from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import os
from pathlib import Path
from unittest.mock import patch

from src.agent.cli import render


def _render_to_text(response: dict) -> str:
    output = StringIO()
    with patch.dict(os.environ, {"JEONSE_SHOW_RAG_TRACE": "0"}), redirect_stdout(output):
        render(response)
    return output.getvalue()


def test_finance_llm_answer_is_primary_and_db_rows_are_evidence():
    narrative = "현재 조건에서는 A 대출이 금리 제한을 충족합니다. 최종 자격은 심사가 필요합니다."
    response = {
        "status": "qa",
        "qa_type": "finance",
        "answer": narrative,
        "message": "기존 고정 금융 안내 문장",
        "programs": [{
            "name": "A 대출",
            "category": "전세대출",
            "region_scope": "전국",
            "application_status": "상시",
            "rate_pct": 1.5,
            "max_amount_manwon": 10000,
            "source_url": "https://example.com/a",
        }],
    }

    text = _render_to_text(response)

    assert "AI 상담 답변" in text
    assert text.count(narrative) == 1
    assert "조회 근거 · 금융서비스 DB 1건" in text
    assert "A 대출" in text
    assert "기존 고정 금융 안내 문장" not in text


def test_recommendation_llm_answer_is_separate_from_property_evidence():
    narrative = "요청 지역과 예산을 모두 만족하는 전세 후보가 있습니다."
    response = {
        "status": "recommendation",
        "answer": narrative,
        "groups": {0: [{
            "sido": "대전",
            "gugun": "유성구",
            "lease_type": "전세",
            "deposit_manwon": 8000,
            "monthly_rent_manwon": 0,
            "fraud_score": 0.2,
            "missing_conditions": [],
        }]},
    }

    text = _render_to_text(response)

    assert text.count(narrative) == 1
    assert "조회 근거 · 부동산 DB 1건" in text
    assert "대전 유성구 전세" in text


def test_gui_defines_separate_answer_and_evidence_blocks():
    gui = (Path(__file__).parents[1] / "src" / "server" / "gui.html").read_text(
        encoding="utf-8"
    )
    assert 'class="answer-block"' in gui
    assert 'class="evidence-block"' in gui
    assert "조회 근거 · 금융서비스 DB" in gui
    assert "조회 근거 · 부동산 DB" in gui
    assert "목표 달성 계산 · 전세 조달 계획" in gui


def test_financed_goal_cli_shows_budget_finance_and_property_evidence():
    response = {
        "status": "recommendation",
        "recommendation_mode": "financed_jeonse_goal",
        "message": "금융과 매물을 연결한 추천입니다.",
        "financing_plan": {
            "base_jeonse_budget_manwon": 5000,
            "direct_loan_limit_manwon": 10000,
            "estimated_max_deposit_manwon": 15000,
            "selected_program_name": "청년 전세대출",
            "limitation": None,
        },
        "finance_programs": [{
            "name": "청년 전세대출", "category": "전세대출", "region_scope": "전국",
            "goal_role": "전세보증금 증액 후보", "rate_pct": 2.0,
            "max_amount_manwon": 10000, "application_status": "상시",
            "source_url": "https://example.com/loan",
        }],
        "groups": {0: []},
    }

    text = _render_to_text(response)

    assert "목표 달성 계산 · 전세 조달 계획" in text
    assert "추정 최대 보증금 15,000만원" in text
    assert "목표 관련 금융서비스 DB 1건" in text
    assert "부동산 DB 0건" in text


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ✓ {name}")
    print("OK: presentation tests passed")
