"""The shared, deterministic 50-query benchmark set.

Every query contains a conditional branch, a distractor branch, or a negated
preference.  ``expected_slots`` is the human-authored answer key; it is never
shown to the planner.  Both experiment modes consume this exact tuple.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExperimentCase:
    case_id: str
    query: str
    expected_slots: dict[str, Any]
    rationale: str


_TARGETS = (
    ("월세", "원룸", {"max_deposit_manwon": 10000, "max_monthly_rent_manwon": 70,
                       "min_area_m2": 12, "sort_by": "price_asc"}),
    ("월세", "오피스텔", {"max_deposit_manwon": 8500, "max_monthly_rent_manwon": 115,
                          "min_area_m2": 18, "sort_by": "price_asc"}),
    ("월세", "다가구주택", {"max_deposit_manwon": 10000, "max_monthly_rent_manwon": 80,
                            "min_area_m2": 19, "sort_by": "price_asc"}),
    ("월세", "다세대주택", {"max_deposit_manwon": 13000, "max_monthly_rent_manwon": 120,
                            "min_area_m2": 13, "sort_by": "price_asc"}),
    ("월세", "단독주택", {"max_deposit_manwon": 7000, "max_monthly_rent_manwon": 82,
                           "min_area_m2": 17, "sort_by": "price_asc"}),
    ("전세", "아파트", {"max_deposit_manwon": 68000, "min_area_m2": 12,
                         "sort_by": "price_asc"}),
    ("전세", "원룸", {"max_deposit_manwon": 13800, "min_area_m2": 14,
                        "sort_by": "price_asc"}),
    ("전세", "다가구주택", {"max_deposit_manwon": 23000, "min_area_m2": 20,
                            "sort_by": "risk_asc"}),
    ("전세", "단독주택", {"max_deposit_manwon": 15000, "min_area_m2": 26,
                           "sort_by": "risk_asc"}),
    ("매매", "아파트", {"max_sale_price_manwon": 125000, "min_area_m2": 14,
                         "sort_by": "price_asc"}),
)


def _money_phrase(slots: dict[str, Any]) -> str:
    parts: list[str] = []
    if "max_sale_price_manwon" in slots:
        parts.append(f"매매가 {slots['max_sale_price_manwon']:,}만원 이하")
    if "max_deposit_manwon" in slots:
        parts.append(f"보증금 {slots['max_deposit_manwon']:,}만원 이하")
    if "max_monthly_rent_manwon" in slots:
        parts.append(f"월세 {slots['max_monthly_rent_manwon']:,}만원 이하")
    parts.append(f"전용면적 {slots['min_area_m2']:,}㎡ 이상")
    parts.append("가격 오름차순" if slots["sort_by"] == "price_asc" else "전세사기 추정 위험도 낮은 순")
    return ", ".join(parts)


def _query(style: int, index: int, transaction: str, house: str,
           slots: dict[str, Any]) -> str:
    wanted = _money_phrase(slots)
    alternatives = (
        "서울 강남구의 10억원 넘는 신축 아파트",
        "보증금 제한 없는 바다 전망 빌라",
        "경기도 밖의 공원 인접 오피스텔",
        "위험도가 높은 대신 넓은 상가주택",
        "월 200만원을 넘는 초고가 월세",
    )
    alt = alternatives[index % len(alternatives)]
    if style == 0:
        return (
            f"조건부로 판단해 줘. 내 가용자금 지표는 {40 + index}이고 기준값 {60 + index} 이하이므로 "
            f"첫 번째 분기를 적용하고, 초과할 때만 {alt}를 찾는다. 첫 분기는 수원시 팔달구 "
            f"{transaction} {house}, {wanted}다. 과거 희망사항과 두 번째 분기는 무시하고 "
            "현재 성립한 분기의 집을 최대 5개 추천해 줘."
        )
    if style == 1:
        return (
            f"다음 규칙의 참인 결론만 SQL 조건으로 써 줘: (재직 중 AND 이사시점 확정)이면 "
            f"수원시 팔달구 {transaction} {house} 중 {wanted}; 둘 중 하나라도 거짓이면 {alt}. "
            "나는 현재 재직 중이고 이사시점도 확정했으므로 전건이 참이다. '제한 없음'으로 "
            "완화하지 말고 참인 AND 교집합에서만 5채를 골라 줘."
        )
    if style == 2:
        return (
            f"중첩 규칙이다. A=수원 거주 유지(true), B=팔달구 선호(true), C=타지역 가능(false). "
            f"IF A THEN (IF B THEN 수원시 팔달구 {transaction} {house} + {wanted} ELSE {alt}) "
            f"ELSE C 후보를 써라. A와 B가 모두 참이니 가장 안쪽 THEN만 적용하고, OR처럼 넓히지 "
            "말고 모든 수치 경계를 포함해 추천해 줘."
        )
    if style == 3:
        return (
            f"부정문을 주의해. 나는 {alt}를 원하는 것이 아니고, 매매·전세·월세를 전부 섞어 "
            f"달라는 것도 아니다. 실제 요청은 오직 수원시 팔달구의 {transaction} {house}이며 "
            f"{wanted}다. 단, 값이 정확히 상한과 같은 매물은 포함하고 각 조건의 합집합이 아닌 "
            "교집합만 추천해 줘."
        )
    return (
        f"논리식 ((P∧Q)∨R)∧¬S를 평가해 줘. P='수원시 팔달구 유지'=참, "
        f"Q='{transaction} {house} 선택'=참, R='{alt}'=거짓, S='예산 상한 제거'=거짓이다. "
        f"따라서 유효 목표는 수원시 팔달구 {transaction} {house}, {wanted}이고, 거짓 분기의 "
        "단어를 조건으로 오인하지 말고 그 교집합에서 최대 5개를 추천해 줘."
    )


def build_cases() -> tuple[ExperimentCase, ...]:
    cases: list[ExperimentCase] = []
    number = 1
    for style in range(5):
        for target_index, (transaction, house, values) in enumerate(_TARGETS):
            expected = {
                "transaction_type": transaction,
                "property_type": house,
                "region_sido": "경기",
                "region_gugun": ["수원시 팔달구"],
                **values,
            }
            cases.append(ExperimentCase(
                case_id=f"Q{number:02d}",
                query=_query(style, target_index, transaction, house, values),
                expected_slots=expected,
                rationale=(
                    f"명시된 논리식의 참 분기는 경기 수원시 팔달구 {transaction} {house}; "
                    f"경계값은 포함되며 {_money_phrase(values)}를 모두 AND로 적용한다."
                ),
            ))
            number += 1
    assert len(cases) == 50
    assert len({case.query for case in cases}) == 50
    return tuple(cases)


CASES = build_cases()
