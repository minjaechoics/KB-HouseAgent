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
    # 각 문장에 투자 여력(투자처 유무)과 거주 예정 기간을 함께 밝혀서,
    # AI가 "자산 운용형/비용 절감형/안전 최우선형" 상담 질문을 되묻지
    # 않고 한 번에 계산·상담까지 끝내도록 한다(harness.py의
    # qa_lease_compare가 investment_edge/planned_stay_years 둘 다 없으면
    # 먼저 되묻기 때문에, 벤치마크는 매번 답변을 기다릴 수 없다).
    "특별한 투자처는 없고 이 집에서 3년 정도 살 계획인데, 전세가 좋을까 월세가 좋을까?",
    "투자로 대출 금리보다 나은 수익을 낼 자신은 없고 2년은 살 건데, "
    "내 상황에서는 전세랑 월세 중 어느 쪽이 더 유리해?",
    "특별히 투자할 곳도 없고 여기서 4년 정도 지낼 생각인데, "
    "전세와 월세를 장기 자산 기준으로 비교해줘.",
    "투자처는 마땅히 없고 3년 정도 거주할 예정인데, "
    "월세보다 전세가 나은지 확률적으로 계산해줘.",
    "목돈을 주식이나 사업에 굴려서 대출 금리보다 높은 수익을 낼 자신이 있는데, "
    "전세로 사는 것과 월세로 사는 것 중 뭐가 좋아?",
    "따로 투자할 곳은 없고 2년 이상 살 건데, "
    "금리 변동까지 생각하면 전세와 월세 중 무엇이 유리할까?",
    "투자 기회는 없고 3년 살 계획인데, 내 소득과 자산 기준으로 전월세 선택을 비교해줘.",
    "특별한 투자처 없이 10년 정도 살 건데, 전세랑 월세의 10년 뒤 자산 결과를 비교해줘.",
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

# 아래 5개 카테고리는 이번 세션에 새로 만든 기능(조건 카테고리화의 신규
# atom, KB 대출 데이터 새로고침 + 신용등급·차량·개인사업자 자격 확인,
# NAVER 기반 치안·편의 조회, 적정예산 계산)을 검증한다. 특히
# CONDITION_NEW_ATOM_CASES의 슬롯(region_dong, max_subway_walk_min,
# max_facility_walk_min, max_police_distance_min, min_room_count,
# elevator_required, pet_allowed_required)은 CONDITION_DECISION_JSON_SCHEMA
# 에만 있고 메인 플래너의 PLAN_JSON_SCHEMA에는 없다 — NAIVE의
# plan_condition_dialogue()는 PLAN_JSON_SCHEMA로 단일 추출하므로 이 슬롯을
# 구조적으로 아예 만들어낼 수 없다(스키마에 없는 필드).
CONDITION_NEW_ATOM_CASES: tuple[tuple[str, dict[str, Any]], ...] = (
    ("우만동에서 지하철역까지 도보 10분 이내인 전세 찾아줘.",
     {"region_dong": ["우만동"], "max_subway_walk_min": 10}),
    ("화서동에서 경찰서까지 도보 15분 이내이고 방 2개 이상인 집을 찾아줘.",
     {"region_dong": ["화서동"], "max_police_distance_min": 15, "min_room_count": 2}),
    ("인계동에서 엘리베이터 있는 집으로 찾아줘.",
     {"region_dong": ["인계동"], "elevator_required": True}),
    ("지동에서 반려동물 키울 수 있는 전세를 찾아줘.",
     {"region_dong": ["지동"], "pet_allowed_required": True, "transaction_type": "전세"}),
    ("매교동에서 마트까지 도보 5분 이내인 곳을 찾아줘.",
     {"region_dong": ["매교동"], "max_facility_walk_min": 5}),
    ("우만동에서 방 3개 이상이고 엘리베이터 있는 집을 찾아줘.",
     {"region_dong": ["우만동"], "min_room_count": 3, "elevator_required": True}),
    ("화서동에서 반려동물 가능하고 경찰서까지 도보 10분 이내인 곳을 찾아줘.",
     {"region_dong": ["화서동"], "pet_allowed_required": True, "max_police_distance_min": 10}),
    ("인계동에서 지하철까지 도보 15분 이내, 마트까지 도보 10분 이내인 매물을 찾아줘.",
     {"region_dong": ["인계동"], "max_subway_walk_min": 15, "max_facility_walk_min": 10}),
    ("지동에서 방 2개 이상이고 반려동물 가능한 월세를 찾아줘.",
     {"region_dong": ["지동"], "min_room_count": 2, "pet_allowed_required": True,
      "transaction_type": "월세"}),
    ("매교동에서 엘리베이터 있고 경찰서까지 도보 20분 이내인 곳을 찾아줘.",
     {"region_dong": ["매교동"], "elevator_required": True, "max_police_distance_min": 20}),
    ("우만동에서 마트까지 도보 8분 이내이고 반려동물 가능한 곳을 찾아줘.",
     {"region_dong": ["우만동"], "max_facility_walk_min": 8, "pet_allowed_required": True}),
    ("화서동에서 방 2개 이상이고 지하철까지 도보 12분 이내인 집을 찾아줘.",
     {"region_dong": ["화서동"], "min_room_count": 2, "max_subway_walk_min": 12}),
    ("인계동에서 엘리베이터 있고 반려동물도 가능한 전세를 찾아줘.",
     {"region_dong": ["인계동"], "elevator_required": True, "pet_allowed_required": True,
      "transaction_type": "전세"}),
    ("지동에서 경찰서까지 도보 10분 이내이고 지하철역도 도보 10분 이내인 곳을 찾아줘.",
     {"region_dong": ["지동"], "max_police_distance_min": 10, "max_subway_walk_min": 10}),
    ("매교동에서 방 3개 이상이고 마트까지 도보 15분 이내인 집을 찾아줘.",
     {"region_dong": ["매교동"], "min_room_count": 3, "max_facility_walk_min": 15}),
    ("우만동에서 엘리베이터 있고 지하철역까지 도보 10분 이내, 방 2개 이상인 곳을 찾아줘.",
     {"region_dong": ["우만동"], "elevator_required": True, "max_subway_walk_min": 10,
      "min_room_count": 2}),
)

QA_FINANCE_QUERIES = (
    "지금 내 신용등급으로 받을 수 있는 신용대출 상품 뭐가 있어?",
    "중고차 8천만원짜리 사려는데 자동차 대출 조건 알려줘.",
    "저는 개인사업자인데 사업자 대상 대출 상품 있어?",
    "신차 구매하려는데 금리 낮은 자동차대출 추천해줘.",
    "전세자금대출 중에 금리 제일 낮은 상품 알려줘.",
    "담보대출 받을 수 있는 상품 중에 한도 큰 거 알려줘.",
    "개인사업자 신용대출 자격 조건이 뭐야?",
    "지금 내 조건으로 받을 수 있는 대출상품 다 알려줘.",
    "주택담보대출 금리랑 한도 알려줘.",
    "신혼부부 전세자금대출 있어?",
    "청년 전세자금대출 조건 알려줘.",
    "자동차 대출 받으려는데 신차랑 중고차 중에 뭐가 유리해?",
)

QA_SAFETY_QUERIES = (
    # "파출소/비상벨/소방서 있어?" 류의 특정 시설 유무 질문은 qa_poi(장소검색)
    # 와도 겹쳐 의도가 모호해진다. qa_safety 고유 영역인 "종합 치안 수준"
    # 프레이밍으로 물어 의도 경계를 명확히 한다.
    "이 동네 치안 어때?",
    "여기 밤에 다녀도 안전해?",
    "이 동네 방범 인프라가 전체적으로 잘 갖춰져 있어?",
    "이 지역 치안이 전반적으로 얼마나 안전한 편이야?",
    "동네 안전 점수 알려줘.",
    "이 동네 안전 관련 시설이 종합적으로 충분한 편이야?",
    "여기 밤늦게 다녀도 될 만큼 안전한 동네야?",
    "이 동네 치안 수준이 어느 정도야?",
)

QA_CONVENIENCE_QUERIES = (
    # "편의점 있어?" 류의 특정 시설 유무 질문은 qa_poi(장소검색)와도 겹쳐서
    # 의도가 모호해진다. qa_convenience 고유 영역인 "종합 편의성 점수"
    # 프레이밍으로 질문해 의도 경계가 명확하도록 한다.
    "이 동네 생활 편의성 점수가 어떻게 돼?",
    "여기 생활하기 얼마나 편리한 동네야?",
    "주변 편의시설이 전체적으로 얼마나 잘 갖춰져 있어?",
    "이 동네 생활 편의점수 알려줘.",
    "이 동네 편의시설 종합 점수 알려줘.",
    "여기 생활 인프라가 잘 갖춰진 동네야?",
)

QA_AFFORDABILITY_QUERIES = (
    "내 소득이랑 자산으로 적정 예산이 얼마야?",
    "내가 감당할 수 있는 월세는 얼마 정도야?",
    "내 상황에서 적정 전세보증금이 얼마나 돼?",
    "내 소득 대비 적정 주거비가 얼마야?",
    "지금 자산으로 적정한 보증금 규모를 알려줘.",
    "내 형편에 맞는 월 주거비 상한은 얼마야?",
    "적정 주거예산을 소득과 자산 기준으로 계산해줘.",
    "내가 무리없이 낼 수 있는 월세 상한이 얼마야?",
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

    for query, slots in CONDITION_NEW_ATOM_CASES:
        cases.append(AdvisorCase(
            case_id=f"H{number:02d}", category="condition_new_atoms",
            query=query, expected_intent="condition_dialogue",
            expected_status="ask_confirmation",
            required_tools=(),
            expected_slots=dict(slots),
            rationale=(
                "region_dong/max_subway_walk_min/max_facility_walk_min/"
                "max_police_distance_min/min_room_count/elevator_required/"
                "pet_allowed_required는 CONDITION_DECISION_JSON_SCHEMA에만 있고 "
                "메인 플래너 스키마에는 없어 NAIVE는 구조적으로 추출할 수 없다."
            ),
        ))
        number += 1

    for query in QA_FINANCE_QUERIES:
        cases.append(AdvisorCase(
            case_id=f"H{number:02d}", category="qa_finance",
            query=query, expected_intent="qa_finance",
            expected_status="qa", expected_qa_type="finance",
            required_tools=("finance_text2sql",),
            rationale="실제 KB 대출 DB(신용등급/차량/개인사업자 자격 포함)를 조회해 근거 있는 금리·한도만 말해야 한다.",
        ))
        number += 1

    for query in QA_SAFETY_QUERIES:
        cases.append(AdvisorCase(
            case_id=f"H{number:02d}", category="qa_safety",
            query=query, expected_intent="qa_safety",
            expected_status="qa", expected_qa_type="safety",
            required_tools=("safety_assess",),
            rationale="SafetyTool의 실제 CCTV·비상벨·치안센터·소방서 집계값만 말해야 한다.",
        ))
        number += 1

    for query in QA_CONVENIENCE_QUERIES:
        cases.append(AdvisorCase(
            case_id=f"H{number:02d}", category="qa_convenience",
            query=query, expected_intent="qa_convenience",
            expected_status="qa", expected_qa_type="convenience",
            required_tools=("convenience_assess",),
            rationale="ConvenienceTool의 실제 편의시설 집계값만 말해야 한다.",
        ))
        number += 1

    for query in QA_AFFORDABILITY_QUERIES:
        cases.append(AdvisorCase(
            case_id=f"H{number:02d}", category="qa_affordability",
            query=query, expected_intent="qa_affordability",
            expected_status="qa", expected_qa_type="affordability",
            required_tools=(),
            rationale="compute_affordability()의 결정론적 계산값과 정확히 일치해야 한다.",
        ))
        number += 1

    assert len(cases) == 100
    assert len({case.query for case in cases}) == 100
    return tuple(cases)


ADVISOR_CASES = build_advisor_cases()


def category_counts() -> dict[str, int]:
    result: dict[str, int] = {}
    for case in ADVISOR_CASES:
        result[case.category] = result.get(case.category, 0) + 1
    return result
