"""
자문/계산 도구 (Agent Tools) — 외부 API 없이 규칙·공식으로 동작.

  - contract_checklist : 전세/월세 계약 특약·확인 체크리스트 (전세사기 예방).
  - lease_compare      : 전세 vs 월세 유불리 비교(전월세전환율·기회비용 기반).
  - cost_breakdown     : 월 실부담금 상세 분해(월세+관리비+보증금 기회비용).

근거(체크리스트/공식):
  - 전세사기 예방 체크리스트(국토부/정책브리핑, 2026 갱신):
    https://www.korea.kr/multi/visualNewsView.do?newsId=148905628
    https://www.innno.co.kr/2026/03/jeonse-fraud-checklist-10.html
    https://www.tossbank.com/articles/rental-scam
  - 안전식: (보증금+선순위채권) ≤ 시세 70~80%
  - 전월세전환율로 전세↔월세 환산(월세=보증금*전환율/12)
"""
from __future__ import annotations
from src import config


# ----------------------------------------------------------------------
# 1) 계약 체크리스트 / 특약
# ----------------------------------------------------------------------
def contract_checklist(lease_type: str = "전세",
                       is_multi_family: bool = True) -> dict:
    """
    계약 단계별 체크리스트 + 권장 특약 문구 반환.
    다가구주택이면 선순위 보증금 확인 항목을 강조.
    """
    before = [
        "주변 매매가·전세가 시세 확인 (국토부 실거래가, 부동산테크)",
        "등기부등본(등기사항증명서) 갑구·을구 확인 — 압류·근저당·신탁등기 여부",
        "임대인 = 소유자 일치 확인 (신분증 진위: 정부24/1382)",
        "임대인 국세·지방세 체납 여부 확인 (홈택스/위택스)",
        "전세보증금 반환보증(HUG/SGI) 가입 가능 여부를 계약금 송금 전 확인",
        "건축물대장으로 주소·동호수·용도 일치 확인",
    ]
    if is_multi_family:
        before.insert(2,
            "★다가구 필수: 선순위 보증금 총액 확인 (임대인 동의 후 확정일자 부여현황 열람) "
            "— (내 보증금 + 선순위 보증금 + 근저당) ≤ 시세 70~80%인지 계산")

    after = [
        "임대차 신고 (계약일로부터 30일 이내, 보증금 6천만원 또는 월세 30만원 초과 시 의무)",
        "이사 당일 전입신고 (대항력 확보 — 익일 0시 발효)",
        "확정일자 취득 (우선변제권 확보 — 주민센터/인터넷등기소, 600원)",
        "전세보증금 반환보증 가입 (계약기간+30일 이내)",
    ]
    special_terms = [
        "「확정일자·전입신고 익일까지 임대인은 근저당 등 담보권을 설정하지 않는다. "
        "위반 시 계약 즉시 해지 및 손해배상」",
        "「잔금일 전까지 근저당 설정액이 현 상태를 초과하지 않는다」",
        "「전세보증금 반환보증 가입 불가(거절) 시 계약을 무효로 하고 계약금 전액 반환한다」",
        "「임대인의 세금 체납 사실이 발견되면 임차인은 계약을 해제할 수 있다」",
        "「임차인의 귀책 없는 전세자금대출 미승인 시 계약금을 반환한다」",
    ]
    if is_multi_family:
        special_terms.insert(0,
            "「계약 시점의 선순위 보증금·근저당 총액을 임대인이 고지하며, "
            "고지 내용과 다를 경우 계약을 해제하고 손해배상한다」")

    return {
        "lease_type": lease_type,
        "is_multi_family": is_multi_family,
        "before_contract": before,
        "after_contract": after,
        "recommended_special_terms": special_terms,
        "sources": [
            "https://www.korea.kr/multi/visualNewsView.do?newsId=148905628",
            "https://www.tossbank.com/articles/rental-scam",
        ],
    }


# ----------------------------------------------------------------------
# 2) 전세 vs 월세 비교
# ----------------------------------------------------------------------
def lease_compare(jeonse_deposit_manwon: float,
                  monthly_deposit_manwon: float,
                  monthly_rent_manwon: float,
                  loan_rate_pct: float = 4.0,
                  deposit_opportunity_rate_pct: float | None = None) -> dict:
    """
    같은 집을 전세로 vs 월세로 살 때 '월 실부담'을 비교.

    - 전세: 보증금을 마련하는 데 드는 비용 = (전세보증금 × 대출금리 또는 기회비용) / 12.
            (대출로 조달하면 대출이자, 자기자본이면 예금 기회비용)
    - 월세: 월세 + (월세보증금 × 기회비용)/12.
    기회비용률 미지정 시 전월세전환율(config) 사용.
    """
    opp = (deposit_opportunity_rate_pct or config.JEONSE_TO_MONTHLY_RATE * 100) / 100.0
    loan = loan_rate_pct / 100.0

    # 전세 월환산 부담(전액 대출 가정 상한 + 전액 자기자본 하한)
    jeonse_cost_loan = jeonse_deposit_manwon * loan / 12
    jeonse_cost_own = jeonse_deposit_manwon * opp / 12

    # 월세 월환산 부담
    wolse_deposit_cost = monthly_deposit_manwon * opp / 12
    wolse_total = monthly_rent_manwon + wolse_deposit_cost

    cheaper = "전세" if jeonse_cost_loan < wolse_total else "월세"
    return {
        "jeonse_monthly_if_loan": round(jeonse_cost_loan, 1),
        "jeonse_monthly_if_own_capital": round(jeonse_cost_own, 1),
        "wolse_monthly_total": round(wolse_total, 1),
        "wolse_rent": round(monthly_rent_manwon, 1),
        "wolse_deposit_opportunity_cost": round(wolse_deposit_cost, 1),
        "cheaper_option_if_loan": cheaper,
        "assumptions": {
            "loan_rate_pct": loan_rate_pct,
            "opportunity_rate_pct": round(opp * 100, 2),
            "note": "전세를 대출로 조달할 때와 월세 총부담을 비교. "
                    "자기자본이면 전세 부담은 예금 기회비용(하한)에 가까움.",
        },
    }


# ----------------------------------------------------------------------
# 3) 월 실부담금 상세 분해
# ----------------------------------------------------------------------
def cost_breakdown(deposit_manwon: float,
                   monthly_rent_manwon: float,
                   maintenance_fee_manwon: float,
                   onetime_fee_manwon: float = 0.0,
                   deposit_opportunity_rate_pct: float | None = None,
                   lease_months: int = 24) -> dict:
    """
    월 실부담금을 구성요소로 분해.
      월 실부담 = 월세 + 관리비 + 보증금 기회비용/월 + 일회성비용 월분할
    """
    opp = (deposit_opportunity_rate_pct or config.JEONSE_TO_MONTHLY_RATE * 100) / 100.0
    deposit_monthly = deposit_manwon * opp / 12
    onetime_monthly = onetime_fee_manwon / max(lease_months, 1)
    total = monthly_rent_manwon + maintenance_fee_manwon + deposit_monthly + onetime_monthly
    return {
        "monthly_rent": round(monthly_rent_manwon, 1),
        "maintenance": round(maintenance_fee_manwon, 1),
        "deposit_opportunity_cost_monthly": round(deposit_monthly, 1),
        "onetime_amortized_monthly": round(onetime_monthly, 1),
        "total_monthly_real_cost": round(total, 1),
        "assumptions": {
            "opportunity_rate_pct": round(opp * 100, 2),
            "lease_months": lease_months,
        },
    }


if __name__ == "__main__":
    import json
    print("[contract_checklist] 다가구 전세:")
    cc = contract_checklist("전세", is_multi_family=True)
    print("  계약 전 항목 수:", len(cc["before_contract"]))
    print("  권장 특약 수:", len(cc["recommended_special_terms"]))
    print("  대표 특약:", cc["recommended_special_terms"][0][:50], "...")

    print("\n[lease_compare] 전세 2억 vs (월세보증금 2천/월세 70):")
    print(json.dumps(lease_compare(20000, 2000, 70), ensure_ascii=False, indent=2))

    print("\n[cost_breakdown] 보증금 2천/월세 70/관리비 8/중개 60:")
    print(json.dumps(cost_breakdown(2000, 70, 8, 60), ensure_ascii=False, indent=2))
