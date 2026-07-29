"""청년 주거 Agent의 자연어 의도를 고정하는 한국어 골든셋."""
from __future__ import annotations

from typing import Any


_GROUPS: list[dict[str, Any]] = [
    {"intent": "qa_finance", "action": "proceed", "texts": [
        "금리 {n}% 미만 대출 상품 알려줘",
        "내가 받을 수 있는 주거 금융상품은 뭐야",
        "청년 전세대출 자격이 되는지 확인해줘",
    ]},
    {"intent": "qa_contract", "action": "proceed", "texts": [
        "전세 계약할 때 체크리스트 알려줘",
        "확정일자와 전입신고는 어떻게 해",
        "이 집 계약해도 안전한지 알려줘",
    ]},
    {"intent": "qa_cost", "action": "proceed", "texts": [
        "이 집은 한 달에 얼마가 들어",
        "총 주거비를 계산해줘",
        "실제 부담액이 얼마나 돼",
    ]},
    {"intent": "qa_poi", "action": "proceed", "texts": [
        "주변 편의점이 어디 있어",
        "근처 병원을 알려줘",
        "가까운 지하철역이 있어",
    ]},
    {"intent": "qa_market", "action": "proceed", "texts": [
        "이 동네 실거래 시세 알려줘",
        "이 가격이 적정한지 봐줘",
        "바가지 가격 아닌가",
    ]},
    {"intent": "qa_registry", "action": "proceed", "texts": [
        "등기부는 어떻게 확인해",
        "근저당 확인 방법 알려줘",
        "신탁등기 위험을 설명해줘",
    ]},
    {"intent": "qa_safety", "action": "proceed", "texts": [
        "이 동네 치안은 어때",
        "주변 CCTV가 충분해",
        "밤에 안전한 지역인지 알려줘",
    ]},
    {"intent": "qa_convenience", "action": "proceed", "texts": [
        "주변 생활 인프라는 어때",
        "이 동네 생활 편의시설 알려줘",
        "여기 살기 편한 동네인지 알려줘",
    ]},
    {"intent": "qa_affordability", "action": "proceed", "texts": [
        "내 소득으로 얼마짜리 집까지 가능해",
        "내 형편에 적정한 보증금은 얼마야",
        "월 주거비를 얼마까지 감당할 수 있어",
    ]},
    {"intent": "recommend", "action": "confirm", "texts": [
        "월세 {n}만원 이하 집 찾아줘",
        "전세 {n}억 이하 아파트 찾아줘",
        "안전한 오피스텔 매물을 추천해줘",
    ]},
]


_DECISION_CASES = [
    ("lease-compare-0", "전세가 좋을까 월세가 좋을까?", "qa_lease_compare"),
    ("lease-compare-1", "전세랑 월세 중 내 자산에 뭐가 유리해?", "qa_lease_compare"),
    ("lease-compare-2", "월세가 나아 아니면 전세가 나아?", "qa_lease_compare"),
    ("best-affordable-0", "수원에서 내 예산과 대출로 제일 좋은 집은?", "goal_best_affordable"),
    ("best-affordable-1", "내가 가진 돈과 대출을 포함해 감당 가능한 최적 매물을 추천해줘", "goal_best_affordable"),
    ("best-affordable-2", "보유 자산으로 살 수 있는 가장 좋은 주택을 골라줘", "goal_best_affordable"),
    ("market-outlook-0", "이 동네 집값 앞으로 오를까 내릴까?", "qa_market"),
    ("market-outlook-1", "여기 집값 전망이 상승이야 하락이야?", "qa_market"),
    ("market-outlook-2", "선택한 집 주변 가격이 앞으로 내릴까?", "qa_market"),
    ("buy-wait-0", "지금 사는 게 나을까 1년 기다리는 게 나을까?", "qa_buy_or_wait"),
    ("buy-wait-1", "이 집 지금 매수할까 2년 뒤 살까?", "qa_buy_or_wait"),
    ("buy-wait-2", "구매를 1~2년 기다리는 편이 유리해?", "qa_buy_or_wait"),
    ("alternative-0", "여기 말고 예산 맞는 다른 동네도 있을까?", "goal_alternative_areas"),
    ("alternative-1", "같은 가격으로 살 수 있는 대안 지역을 추천해줘", "goal_alternative_areas"),
    ("alternative-2", "현재 동네 말고 대출 포함 감당 가능한 다른 지역은?", "goal_alternative_areas"),
]


def build_golden_cases() -> list[dict[str, Any]]:
    """기존 150개 회귀셋에 의사결정 질문 15개를 추가한다."""
    cases: list[dict[str, Any]] = []
    values = [1, 2, 3, 4, 5]
    for group_index, group in enumerate(_GROUPS):
        for text_index, template in enumerate(group["texts"]):
            for variant, value in enumerate(values):
                rendered = (
                    value if "금리" in template else
                    value * 20 if "월세" in template else value
                )
                text = template.format(n=rendered)
                required_slots: dict[str, Any] = {}
                if group["intent"] == "recommend" and text.startswith("월세"):
                    required_slots = {"transaction_type": "월세"}
                elif group["intent"] == "recommend" and text.startswith("전세"):
                    required_slots = {"transaction_type": "전세"}
                cases.append({
                    "case_id": f"g{group_index:02d}-{text_index}-{variant}",
                    "text": text,
                    "expected_intent": group["intent"],
                    "expected_action": group["action"],
                    "required_slots": required_slots,
                    "tags": ["korean", "regression", group["intent"]],
                })
    for case_id, text, intent in _DECISION_CASES:
        cases.append({
            "case_id": case_id,
            "text": text,
            "expected_intent": intent,
            "expected_action": "proceed",
            "required_slots": {},
            "tags": ["korean", "decision_support", intent],
        })
    assert len(cases) == 165
    return cases
