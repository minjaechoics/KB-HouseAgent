"""
(2)-1 적정 주거비 산정 — 고정 정적 공식(static formula).

사용자 재무정보로 감당 가능한 월 주거비 상한과, 그에 대응하는
적정 전세보증금 / 월세보증금 / 월세를 계산한다.

── 공식 근거(반드시 링크로 명시) ───────────────────────────────────────
1) RIR(Rent to Income Ratio) 30% 룰 (UN-Habitat / OECD):
   가처분소득의 30% 초과 시 '주거비 과부담'.
   - https://calculkorea.com/rent-burden-calculator
   - 연세대 IPAID, 저소득 임차가구 주거비부담 분석(RIR 30% 룰):
     https://ycms.yonsei.ac.kr/ipaid/Publication_a.do (jhuf-4-2-159)
2) 한국 지역사회보장지표: 월소득 대비 임대료 25% 초과를 과부담으로 정의.
   - https://www.ejhuf.org/archive/view_article?pid=jhuf-4-2-159
3) 50/30/20 예산 법칙(필수지출 50% 중 주거는 절반 이하 → 소득의 ~25%):
   - https://calculkorea.com/rent-burden-calculator
4) 전월세전환율로 보증금↔월세 환산:
   월세환산액 = 보증금 * 전환율(연) / 12

계산 로직:
  max_monthly_housing = min(
        income * RIR_TARGET,                      # 소득 기준(권장 25%)
        income - living_cost - 최소저축            # 현금흐름 기준
  )
  - 관리비 포함 여부는 config.INCLUDE_MAINTENANCE_IN_RIR.
  - 전세 적정 보증금은 "가용자산 - 비상금" 과 "월주거비를 전세로 환산" 중 보수적 값.
"""
from __future__ import annotations
from typing import Union

from src import config
from src.schemas import UserProfile, AffordabilityResult


def compute_affordability(user: Union[dict, UserProfile]) -> AffordabilityResult:
    if isinstance(user, dict):
        user = UserProfile(**user)

    income = user.monthly_income_manwon
    living = user.monthly_living_cost_manwon
    asset = user.total_asset_manwon

    notes: list[str] = []

    # 1) 소득 기준 상한 (RIR 권장 25%)
    cap_income = income * config.RIR_TARGET

    # 2) 현금흐름 기준 상한 (소득 - 생활비 - 최소저축여력)
    #    최소저축: 소득의 10%는 남기도록(재무건전성).
    min_saving = income * 0.10
    cap_cashflow = max(income - living - min_saving, 0)

    max_monthly = min(cap_income, cap_cashflow)
    if cap_cashflow < cap_income:
        notes.append("현금흐름(생활비) 제약이 소득기준보다 빡빡하여 상한이 낮아짐")

    # 3) 전세 적정 보증금
    #    (a) 자산 기준: 가용자산 - 비상금(생활비 N개월)
    emergency = living * config.EMERGENCY_FUND_MONTHS
    asset_based_deposit = max(asset - emergency, 0)
    #    (b) 월주거비를 전세로 환산: 월세로 낼 돈을 보증금 기회비용으로 환산
    #        deposit_equiv = max_monthly * 12 / 전환율
    rate = config.JEONSE_TO_MONTHLY_RATE
    flow_based_deposit = (max_monthly * 12) / rate if rate > 0 else 0
    rec_jeonse = min(asset_based_deposit, flow_based_deposit)
    if asset_based_deposit < flow_based_deposit:
        notes.append("보유자산이 부족하여 전세보증금 상한이 자산에 의해 제한됨 → 대출 검토 권장")

    # 4) 월세: 보증금은 소액(자산의 일부), 나머지를 월세로
    rec_monthly_deposit = min(asset * 0.3, asset_based_deposit)
    deposit_offset = rec_monthly_deposit * rate / 12   # 보증금이 상쇄하는 월세분
    rec_monthly_rent = max(max_monthly - deposit_offset, 0)

    # 관리비 고려: 월주거비 상한에 관리비가 포함된다면 순월세는 더 낮아짐(안내만)
    if config.INCLUDE_MAINTENANCE_IN_RIR:
        notes.append("월주거비 상한은 관리비 포함 기준. 실제 순월세는 관리비만큼 낮춰 계약할 것")

    rir = (max_monthly / income) if income > 0 else 0.0

    return AffordabilityResult(
        max_monthly_housing_manwon=round(max_monthly, 1),
        recommended_jeonse_deposit_manwon=round(rec_jeonse, 1),
        recommended_monthly_deposit_manwon=round(rec_monthly_deposit, 1),
        recommended_monthly_rent_manwon=round(rec_monthly_rent, 1),
        rir_at_recommended=round(rir, 3),
        notes=notes,
    )


if __name__ == "__main__":
    demo = dict(
        user_id="U_DEMO", age=29,
        monthly_income_manwon=280, total_asset_manwon=4000,
        monthly_living_cost_manwon=110, income_decile=5,
    )
    res = compute_affordability(demo)
    print("[적정 주거비] 입력:", demo)
    print(res.model_dump_json(indent=2))
