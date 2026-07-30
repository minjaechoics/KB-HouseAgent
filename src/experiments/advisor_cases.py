"""Human-authored 50-case benchmark for the production advisor channel."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.experiments.cases import CASES as SEARCH_CASES


@dataclass(frozen=True)
class AdvisorCase:
    case_id: str
    category: str
    query: str
    expected_intent: str
    expected_status: str
    required_tools: tuple[str, ...]
    expected_qa_type: str | None = None
    expected_mode: str | None = None
    expected_slots: dict[str, Any] | None = None
    context_kind: str = "none"
    rationale: str = ""


BEST_AFFORDABLE_QUERIES = (
    "수원에서 내 예산과 대출로 제일 좋은 집이 뭐야?",
    "수원에서 가진 예산과 대출을 포함해 감당 가능한 가장 좋은 아파트를 추천해줘.",
    "내 보유자산과 대출 한도 안에서 살 수 있는 최적 주택을 골라줘.",
    "수원시에서 자금 조달까지 고려한 베스트 매물을 찾아줘.",
    "내 예산으로 살 수 있는 집 중 금융상품까지 적용해서 가장 좋은 곳을 추천해줘.",
    "대출 포함해서 감당 가능한 최적의 수원 매물을 골라줘.",
    "내 자산과 월 상환능력으로 살 수 있는 제일 좋은 주택은 어디야?",
    "수원에서 보유 예산에 금융상품을 더해 살 수 있는 가장 좋은 집을 찾아줘.",
    "내 형편에서 대출을 받아 구매 가능한 최적 아파트를 추천해줘.",
    "자기자금과 대출 조건을 모두 지키면서 살 수 있는 베스트 집을 보여줘.",
)

LEASE_COMPARE_QUERIES = (
    "전세가 좋을까 월세가 좋을까?",
    "내 상황에서는 전세랑 월세 중 어느 쪽이 더 유리해?",
    "전세와 월세를 장기 자산 기준으로 비교해줘.",
    "월세보다 전세가 나은지 확률적으로 계산해줘.",
    "전세로 사는 것과 월세로 사는 것 중 뭐가 좋아?",
    "금리 변동까지 생각하면 전세와 월세 중 무엇이 유리할까?",
    "내 소득과 자산 기준으로 전월세 선택을 비교해줘.",
    "전세랑 월세의 10년 뒤 자산 결과를 비교해줘.",
)

MARKET_QUERIES = (
    "이 동네 집값 앞으로 오를까 내릴까?",
    "선택한 집이 있는 동네의 가격 전망이 상승이야 하락이야?",
    "이 지역 주택 가격은 앞으로 상승할지 하락할지 전망해줘.",
    "선택한 매물 주변 집값 전망을 실거래 근거로 알려줘.",
    "인계동 집값이 앞으로 오를까 내릴까?",
    "이 집이 있는 지역의 향후 시세 방향을 분석해줘.",
    "최근 실거래와 뉴스를 기준으로 이 동네 가격 전망을 알려줘.",
)

BUY_WAIT_QUERIES = (
    "지금 사는 게 나을까, 1~2년 기다리는 게 나을까?",
    "이 집을 지금 매수할까 2년 뒤까지 기다릴까?",
    "지금 구매하는 것과 1년 기다리는 것 중 무엇이 나아?",
    "선택한 집은 바로 사는 게 좋아, 아니면 1~2년 후가 좋아?",
    "집값과 대기 주거비를 생각하면 지금 매수해야 할까 기다려야 할까?",
)

ALTERNATIVE_QUERIES = (
    "여기 말고 예산 맞는 다른 동네도 있을까?",
    "현재 동네 말고 내 예산으로 살 수 있는 다른 지역을 추천해줘.",
    "이곳 대신 대출 포함 감당 가능한 대안 동네의 집을 골라줘.",
    "여기 말고 같은 가격으로 살 수 있는 옆 동네 매물도 찾아줘.",
    "현재 선택지를 제외하고 내 예산에 맞는 다른 동네의 최적 집을 추천해줘.",
)


def build_advisor_cases() -> tuple[AdvisorCase, ...]:
    cases: list[AdvisorCase] = []
    number = 1

    for source in SEARCH_CASES[:15]:
        cases.append(AdvisorCase(
            case_id=f"H{number:02d}", category="condition_dialogue",
            query=source.query, expected_intent="condition_dialogue",
            expected_status="ask_confirmation",
            required_tools=(),
            expected_slots=dict(source.expected_slots),
            rationale=(
                "조건 추가 채널은 참인 분기의 슬롯을 제안하되 UI 버튼 승인 전에는 "
                "검색·SQL 도구를 실행하지 않아야 한다."
            ),
        ))
        number += 1

    for query in BEST_AFFORDABLE_QUERIES:
        cases.append(AdvisorCase(
            case_id=f"H{number:02d}", category="best_affordable",
            query=query, expected_intent="goal_best_affordable",
            expected_status="recommendation",
            expected_mode="best_affordable_pareto",
            required_tools=(
                "property_text2sql", "finance_search", "pareto_milp_optimizer",
            ),
            rationale="자기자금·금융자격·월 상환 제약을 적용한 Pareto 대표 매물을 반환해야 한다.",
        ))
        number += 1

    for query in LEASE_COMPARE_QUERIES:
        cases.append(AdvisorCase(
            case_id=f"H{number:02d}", category="lease_compare",
            query=query, expected_intent="qa_lease_compare",
            expected_status="qa", expected_qa_type="lease_compare",
            required_tools=("lease_monte_carlo",),
            rationale="전세와 월세 각각 3,000개 경로의 P10·P50·P90과 스트레스 결과를 비교해야 한다.",
        ))
        number += 1

    for query in MARKET_QUERIES:
        cases.append(AdvisorCase(
            case_id=f"H{number:02d}", category="market_outlook",
            query=query, expected_intent="qa_market",
            expected_status="qa", expected_qa_type="market",
            required_tools=("selected_property_market_forecast",),
            context_kind="selected_sale",
            rationale="선택 매물에 고정된 실거래 시계열·뉴스 전망 수치만 전달해야 한다.",
        ))
        number += 1

    for query in BUY_WAIT_QUERIES:
        cases.append(AdvisorCase(
            case_id=f"H{number:02d}", category="buy_or_wait",
            query=query, expected_intent="qa_buy_or_wait",
            expected_status="qa", expected_qa_type="buy_or_wait",
            required_tools=("buy_now_vs_wait",),
            context_kind="selected_sale",
            rationale="선택 매물 가격 전망과 1·2년 대기 주거비의 산술 결과가 일치해야 한다.",
        ))
        number += 1

    for query in ALTERNATIVE_QUERIES:
        cases.append(AdvisorCase(
            case_id=f"H{number:02d}", category="alternative_areas",
            query=query, expected_intent="goal_alternative_areas",
            expected_status="recommendation",
            expected_mode="alternative_areas_pareto",
            required_tools=(
                "property_text2sql", "finance_search", "pareto_milp_optimizer",
            ),
            context_kind="selected_area",
            rationale="현재 선택 동을 제외하고 동일 자금·상환 제약의 Pareto 대안을 반환해야 한다.",
        ))
        number += 1

    assert len(cases) == 50
    assert len({case.query for case in cases}) == 50
    return tuple(cases)


ADVISOR_CASES = build_advisor_cases()


def category_counts() -> dict[str, int]:
    result: dict[str, int] = {}
    for case in ADVISOR_CASES:
        result[case.category] = result.get(case.category, 0) + 1
    return result
