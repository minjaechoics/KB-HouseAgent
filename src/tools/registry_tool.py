"""
등기부등본 확인 도구 (Agent Tool) — 스텁.

인터넷등기소(iros.go.kr)는 공식 오픈 API가 없고 건당 열람 수수료가 있으며
자동화에 법적 제약이 있다. 따라서 자동 조회 대신 '수동 확인 가이드'를 제공한다.

향후: 정부 '안심전세App' 통합 정보 제공이 API로 개방되면 연동 가능
      (2026 전세사기 방지대책에서 위험정보 통합 제공 방향).
"""
from __future__ import annotations


def registry_check_guide(address: str = "") -> dict:
    """등기부등본에서 무엇을, 어떻게 확인할지 안내."""
    return {
        "status": "manual_required",
        "address": address,
        "message": "등기부등본은 자동 조회를 제공하지 않아요. 아래 절차로 직접 확인하세요.",
        "how_to": [
            "인터넷등기소(iros.go.kr) 또는 주민센터에서 '등기사항전부증명서' 열람(건당 700원)",
            "갑구: 소유권·압류·가압류 확인 (임대인=소유자 일치?)",
            "을구: 근저당권·전세권·신탁등기 확인 (채권최고액 합계 계산)",
            "다가구는 건물 전체 등기부 + 확정일자 부여현황(선순위 보증금)까지 확인",
        ],
        "danger_signals": [
            "신탁등기 존재 → 소유·처분 권한이 신탁사에 있음, 특히 위험",
            "(근저당 채권최고액 + 선순위 보증금 + 내 보증금) > 시세 70~80%",
            "계약일 직전 소유권 이전 또는 근저당 설정",
        ],
        "related_tools": ["market_price(시세 대비 판단)", "fraud_risk(위험도 점수)"],
        "sources": [
            "https://www.tossbank.com/articles/rental-scam",
            "https://www.innno.co.kr/2026/03/jeonse-fraud-checklist-10.html",
        ],
    }


if __name__ == "__main__":
    import json
    print(json.dumps(registry_check_guide("서울 관악구 …"), ensure_ascii=False, indent=2))
