"""제품의 핵심 자연어 의도를 고르게 포함하는 150개 회귀 질문."""
from __future__ import annotations

from typing import Any


_GROUPS: list[dict[str, Any]] = [
    {"intent": "qa_finance", "action": "proceed", "texts": [
        "금리 {n}% 미만 대출 상품 알려줘", "내가 받을 수 있는 주거 금융상품이 뭐야",
        "청년 전세대출 자격이 되는지 확인해줘"]},
    {"intent": "qa_contract", "action": "proceed", "texts": [
        "전세 계약할 때 체크리스트 알려줘", "확정일자와 전입신고는 어떻게 해",
        "임대차 계약서 특약을 알려줘"]},
    {"intent": "qa_cost", "action": "proceed", "texts": [
        "이 집은 한 달에 얼마가 들어", "월 실부담을 계산해줘", "총 주거비가 얼마나 들어"]},
    {"intent": "qa_poi", "action": "proceed", "texts": [
        "주변 편의점은 어디 있어", "근처 병원을 알려줘", "가까운 지하철역 있어"]},
    {"intent": "qa_market", "action": "proceed", "texts": [
        "이 동네 실거래 시세 알려줘", "이 가격이 적정한지 봐줘", "바가지 가격 아닌가"]},
    {"intent": "qa_registry", "action": "proceed", "texts": [
        "등기부는 어떻게 확인해", "근저당 확인 방법 알려줘", "신탁등기 위험을 설명해줘"]},
    {"intent": "qa_safety", "action": "proceed", "texts": [
        "이 동네 치안은 어때", "주변 CCTV가 충분한가", "밤에 안전한 지역인지 알려줘"]},
    {"intent": "qa_convenience", "action": "proceed", "texts": [
        "주변 생활 인프라가 어때", "이 동네 생활 편의시설 알려줘", "살기 편한 동네인지 알려줘"]},
    {"intent": "qa_affordability", "action": "proceed", "texts": [
        "내 소득으로 얼마짜리 집까지 가능해", "내 형편에 적정한 보증금은 얼마야",
        "월 주거비를 얼마까지 감당할 수 있어"]},
    {"intent": "recommend", "action": "confirm", "texts": [
        "월세 {n}만원 이하 집 찾아줘", "전세 {n}억 이하 아파트 찾아줘",
        "안전한 오피스텔 매물을 추천해줘"]},
]


def build_golden_cases() -> list[dict[str, Any]]:
    """10개 의도 × 3개 표현 × 5개 숫자 변형 = 정확히 150개."""
    cases: list[dict[str, Any]] = []
    values = [1, 2, 3, 4, 5]
    for group_index, group in enumerate(_GROUPS):
        for text_index, template in enumerate(group["texts"]):
            for variant, value in enumerate(values):
                text = template.format(n=value if "금리" in template else value * 20)
                case = {
                    "case_id": f"g{group_index:02d}-{text_index}-{variant}",
                    "text": text,
                    "expected_intent": group["intent"],
                    "expected_action": group["action"],
                    "required_slots": {},
                    "tags": ["korean", "regression", group["intent"]],
                }
                if group["intent"] == "recommend" and template.startswith("월세"):
                    case["required_slots"] = {"transaction_type": "월세"}
                elif group["intent"] == "recommend" and template.startswith("전세"):
                    case["required_slots"] = {"transaction_type": "전세"}
                cases.append(case)
    assert len(cases) == 150
    return cases
