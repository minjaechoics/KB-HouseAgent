"""
외부 데이터 도구 (Agent Tools) — 실 API + mock 폴백.

  - POISearchTool      : 매물 주변 편의시설(지하철/편의점/병원 등) 검색.
                         실 API: 카카오 로컬(Local) Search. 키 없으면 mock.
  - MarketPriceTool    : 지역 실거래가 시세 조회로 매물가 적정성 판단.
                         실 API: 국토부 실거래가 공개시스템. 키 없으면 mock.

키(환경변수, 코드에 넣지 말 것):
  KAKAO_REST_API_KEY        (카카오 로컬)
  MOLIT_REALPRICE_API_KEY   (국토부 실거래가, data.go.kr 서비스키)

두 도구 모두 키가 없으면 결정론적 mock으로 동작해 오프라인 테스트가 가능하다.
"""
from __future__ import annotations
import hashlib
import os


def _seed_from(*parts) -> int:
    h = hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


# ----------------------------------------------------------------------
# POI (주변 편의시설)
# ----------------------------------------------------------------------
CATEGORY_KO = {
    "subway": "지하철역", "convenience": "편의점", "mart": "마트",
    "hospital": "병원", "cafe": "카페", "gym": "헬스장",
}


class POISearchTool:
    def __init__(self):
        self.key = os.environ.get("KAKAO_REST_API_KEY")
        self.online = bool(self.key)

    def search(self, lat: float, lng: float, category: str = "subway",
               radius_m: int = 1000) -> dict:
        if self.online:
            try:
                return self._kakao(lat, lng, category, radius_m)
            except Exception as e:
                r = self._mock(lat, lng, category, radius_m)
                r["source"] = f"mock(fallback:{type(e).__name__})"
                return r
        return self._mock(lat, lng, category, radius_m)

    def _kakao(self, lat, lng, category, radius_m) -> dict:
        """카카오 로컬 키워드 검색 API."""
        import requests
        url = "https://dapi.kakao.com/v2/local/search/keyword.json"
        headers = {"Authorization": f"KakaoAK {self.key}"}
        params = {"query": CATEGORY_KO.get(category, category),
                  "x": lng, "y": lat, "radius": radius_m, "sort": "distance"}
        r = requests.get(url, headers=headers, params=params, timeout=5)
        r.raise_for_status()
        docs = r.json().get("documents", [])
        places = [{"name": d["place_name"],
                   "distance_m": int(d.get("distance", 0)),
                   "category": d.get("category_name", "")} for d in docs[:5]]
        return {"category": category, "count": len(places),
                "nearest_m": places[0]["distance_m"] if places else None,
                "places": places, "source": "kakao_local"}

    def _mock(self, lat, lng, category, radius_m) -> dict:
        """좌표+카테고리 기반 결정론적 mock."""
        rng = _seed_from(round(lat, 4), round(lng, 4), category)
        n = 1 + (rng % 5)                       # 1~5개
        nearest = 80 + (rng % (radius_m - 80))   # 반경 이내 최근접 거리
        places = []
        for i in range(min(n, 5)):
            d = nearest + i * (120 + rng % 200)
            if d > radius_m:
                break
            places.append({"name": f"{CATEGORY_KO.get(category, category)} {i+1}",
                           "distance_m": int(d), "category": CATEGORY_KO.get(category, category)})
        return {"category": category, "count": len(places),
                "nearest_m": places[0]["distance_m"] if places else None,
                "places": places, "source": "mock"}


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
    poi = POISearchTool()
    print(f"[POI] online={poi.online}")
    for cat in ("subway", "convenience", "hospital"):
        r = poi.search(37.5665, 126.9780, cat, 1000)
        print(f"  {CATEGORY_KO[cat]}: {r['count']}개, 최근접 {r['nearest_m']}m ({r['source']})")

    mp = MarketPriceTool()
    print(f"\n[MarketPrice] online={mp.online}")
    print(" ", mp.appraise("서울", "관악구", 33, asking_deposit_manwon=2400,
                           market_price_manwon=3000))
    print(" ", mp.appraise("서울", "강서구", 30, asking_deposit_manwon=2800,
                           market_price_manwon=3000))
