"""
외부 데이터 도구 (Agent Tools) — 실 API + mock 폴백.

  - MarketPriceTool    : 지역 실거래가 시세 조회로 매물가 적정성 판단.
                         실 API: 국토부 실거래가 공개시스템. 키 없으면 mock.

매물 주변 편의시설(지하철/편의점/병원 등) 검색은 카카오 로컬 검색 대신
`src.tools.naver_local_tool.NaverLocalSearchTool`(NAVER API HUB 지역검색)을
쓴다 — `ConvenienceTool.local_search`로 이미 공유 인스턴스가 있으므로 별도
클라이언트를 새로 만들지 않는다. `CATEGORY_KO`는 그 검색어(한글 키워드)를
만들 때 재사용하는 카테고리 코드 매핑이다.

키(환경변수, 코드에 넣지 말 것):
  MOLIT_REALPRICE_API_KEY   (국토부 실거래가, data.go.kr 서비스키)

MarketPriceTool은 키가 없으면 결정론적 mock으로 동작해 오프라인 테스트가 가능하다.
"""
from __future__ import annotations
import os


CATEGORY_KO = {
    "subway": "지하철역", "convenience": "편의점", "mart": "마트",
    "hospital": "병원", "cafe": "카페", "gym": "헬스장",
}


# ----------------------------------------------------------------------
# 실거래가 시세
# ----------------------------------------------------------------------
class MarketPriceTool:
    def __init__(self):
        self.key = os.environ.get("MOLIT_REALPRICE_API_KEY")
        self.online = bool(self.key)

    def appraise(self, sido: str, gugun: str, area_m2: float,
                 asking_deposit_manwon: float,
                 market_price_manwon: float | None = None) -> dict:
        """
        해당 지역/면적의 시세 대비 매물가 적정성 판단.
        online이면 국토부 실거래가로 지역 평균 산출, 아니면 market_price_manwon(합성값) 사용.
        """
        if self.online:
            try:
                est = self._molit_estimate(sido, gugun, area_m2)
                src = "molit_realprice"
            except Exception:
                est = market_price_manwon
                src = "mock(fallback)"
        else:
            est = market_price_manwon
            src = "given/mock"

        if not est or est <= 0:
            # 시세 정보가 없으면 판단 보류
            return {"estimated_market_manwon": None, "verdict": "unknown",
                    "ratio": None, "source": src,
                    "message": "시세 정보를 확인할 수 없어 적정성 판단 보류. 실거래가 직접 확인 권장."}

        ratio = asking_deposit_manwon / est
        # 안전식 기반 판정: 전세가율(보증금/시세)
        if ratio <= 0.7:
            verdict = "안전"
        elif ratio <= 0.8:
            verdict = "주의"
        else:
            verdict = "위험(깡통전세 가능)"
        return {"estimated_market_manwon": round(est, 1),
                "asking_deposit_manwon": round(asking_deposit_manwon, 1),
                "ratio": round(ratio, 3), "verdict": verdict, "source": src,
                "message": f"전세가율 {ratio:.0%} → {verdict}"}

    def _molit_estimate(self, sido, gugun, area_m2) -> float:
        """
        국토부 실거래가 API 호출 스텁.
        실제 구현: getRTMSDataSvc... 엔드포인트에서 지역코드+계약월 조회 →
                   유사 면적 거래 평균가 산출. (사용자 환경 + 서비스키 필요)
        """
        raise NotImplementedError("MOLIT 실거래가 실 연동은 서비스키 필요")


if __name__ == "__main__":
    from src.tools.naver_local_tool import NaverLocalSearchTool
    poi = NaverLocalSearchTool()
    print(f"[POI] configured={poi.configured}")
    for cat in ("subway", "convenience", "hospital"):
        r = poi.search(37.5665, 126.9780, "", CATEGORY_KO[cat], 1000)
        nearest = r["places"][0]["distance_m"] if r["places"] else None
        print(f"  {CATEGORY_KO[cat]}: {r['count']}개, 최근접 {nearest}m ({r['source']})")

    mp = MarketPriceTool()
    print(f"\n[MarketPrice] online={mp.online}")
    print(" ", mp.appraise("서울", "관악구", 33, asking_deposit_manwon=2400,
                           market_price_manwon=3000))
    print(" ", mp.appraise("서울", "강서구", 30, asking_deposit_manwon=2800,
                           market_price_manwon=3000))
