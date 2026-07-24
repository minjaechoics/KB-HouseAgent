"""
생활/편의 판단 모듈 (Agent Tool).

편의점은 생활안전지도 편의점 정보조회 REST API(IF_0039)를 사용한다. 이 API는
위치/bbox 검색을 지원하지 않으므로 전국 데이터를 페이지 단위로 한 번 내려받아
``data/generated/safemap_convenience.csv``에 캐시한 뒤 매물 주변을 반경 집계한다.
API의 x/y는 Web Mercator(EPSG:3857)이므로 WGS84 위경도로 변환해 저장한다.

그 밖의 시설은 NAVER API HUB 지역검색(또는 설정된 카카오 카테고리 검색)을 사용한다.
키 또는 캐시가 없으면 숫자를 만들어내지 않고 ``None/unavailable``로 반환한다.

주의: 생활안전지도 데이터는 '영업 중 편의점'이며 24시간 영업 여부를 제공하지
않는다. 기존 인터페이스 호환을 위해 키 이름만 convenience_24h로 유지한다.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.tools.naver_local_tool import NaverLocalSearchTool
from src.tools.public_facility_cache import PublicFacilityCache

RADIUS_M = 500
SAFEMAP_CACHE = config.DATA_GEN / "safemap_convenience.csv"

KAKAO_CATEGORY = {
    "hospital": "HP8",
    "pharmacy": "PM9",
    "mart": "MT1",
    "academy": "AC5",
    "restaurant": "FD6",
    "cafe": "CE7",
}
CATEGORY_KO = {
    "convenience_24h": "편의점(24시간 여부 미확인)",
    "mart": "대형마트",
    "academy": "학원",
    "restaurant": "음식점",
    "cafe": "카페",
    "hospital": "병원",
    "pharmacy": "약국",
}
CONV_WEIGHTS = {
    "convenience_24h": 3.0,
    "mart": 3.0,
    "academy": 1.5,
    "restaurant": 1.0,
    "cafe": 1.0,
    "hospital": 2.5,
    "pharmacy": 2.0,
}


def _web_mercator_to_wgs84(x: float, y: float) -> tuple[float, float]:
    """EPSG:3857 x/y(m)를 (lat, lng)로 변환한다."""
    radius = 6378137.0
    lng = math.degrees(float(x) / radius)
    lat = math.degrees(2.0 * math.atan(math.exp(float(y) / radius)) - math.pi / 2.0)
    return lat, lng


def _haversine_mask(df: pd.DataFrame, lat: float, lng: float,
                    radius_m: int) -> np.ndarray:
    """DataFrame의 lat/lng 전체에 대한 반경 포함 마스크."""
    lat2 = np.radians(df["lat"].to_numpy(dtype=float))
    lng2 = np.radians(df["lng"].to_numpy(dtype=float))
    lat1, lng1 = math.radians(lat), math.radians(lng)
    dlat, dlng = lat2 - lat1, lng2 - lng1
    a = np.sin(dlat / 2.0) ** 2 + math.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2.0) ** 2
    distance = 6371000.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return distance <= radius_m


class SafeMapConvenienceClient:
    """생활안전지도 IF_0039 전용 클라이언트와 로컬 좌표 캐시."""

    def __init__(self, service_key: str | None = None,
                 cache_path: Path = SAFEMAP_CACHE, session=None):
        self.service_key = config.SAFEMAP_SERVICE_KEY if service_key is None else service_key
        self.cache_path = Path(cache_path)
        self.session = session
        self._data: pd.DataFrame | None = None

    @property
    def available(self) -> bool:
        return bool(self.service_key) or self.cache_path.exists()

    def _get(self, params: dict) -> dict:
        if self.session is None:
            import requests
            response = requests.get(
                config.SAFEMAP_CONVENIENCE_URL,
                params=params,
                timeout=30,
            )
        else:
            response = self.session.get(
                config.SAFEMAP_CONVENIENCE_URL,
                params=params,
                timeout=30,
            )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _unwrap(payload: dict) -> tuple[list[dict], int]:
        """IF_0039 JSON의 body/items 모양 차이를 흡수한다."""
        if "response" in payload:
            payload = payload["response"]
        header = payload.get("header", {})
        body = payload.get("body", payload)
        result_code = str(header.get("resultCode", body.get("resultCode", "0")))
        if result_code not in ("0", "00", "0000", "None", ""):
            message = header.get("resultMsg") or header.get("errorMsg") or "unknown error"
            raise RuntimeError(f"Safemap API {result_code}: {message}")

        items: Any = body.get("items", [])
        if isinstance(items, dict):
            items = items.get("item", items.get("items", []))
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            items = []
        total = int(body.get("totalCount", len(items)) or len(items))
        return items, total

    def _fetch_page(self, page_no: int, page_size: int) -> tuple[list[dict], int]:
        if not self.service_key:
            raise RuntimeError("SAFEMAP_SERVICE_KEY가 비어 있습니다.")
        payload = self._get({
            "serviceKey": self.service_key,
            "pageNo": page_no,
            "numOfRows": page_size,
            "returnType": "json",
        })
        return self._unwrap(payload)

    def refresh(self, page_size: int = 1000) -> pd.DataFrame:
        """전국 편의점 페이지를 모두 받아 원자적 CSV 캐시로 저장한다."""
        rows: list[dict] = []
        page_no, total = 1, None
        while total is None or len(rows) < total:
            page, total = self._fetch_page(page_no, page_size)
            if not page:
                break
            rows.extend(page)
            page_no += 1

        normalized = []
        for row in rows:
            try:
                lat, lng = _web_mercator_to_wgs84(row["x"], row["y"])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            normalized.append({
                "objt_id": row.get("objt_id"),
                "name": row.get("fclty_nm", ""),
                "address": row.get("rn_adres") or row.get("adres", ""),
                "lat": lat,
                "lng": lng,
                "data_year": row.get("data_yr", ""),
            })

        df = pd.DataFrame(normalized).drop_duplicates(subset=["objt_id", "lat", "lng"])
        if df.empty:
            raise RuntimeError("생활안전지도 편의점 응답에서 유효한 좌표를 찾지 못했습니다.")
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        df.to_csv(temp_path, index=False, encoding="utf-8-sig")
        temp_path.replace(self.cache_path)
        self._data = df
        return df

    def load(self) -> pd.DataFrame | None:
        if self._data is not None:
            return self._data
        if self.cache_path.exists():
            df = pd.read_csv(self.cache_path, encoding="utf-8-sig")
            df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
            df["lng"] = pd.to_numeric(df["lng"], errors="coerce")
            self._data = df.dropna(subset=["lat", "lng"])
            return self._data
        if self.service_key:
            return self.refresh()
        return None

    def count_nearby(self, lat: float, lng: float, radius_m: int) -> int | None:
        df = self.load()
        if df is None:
            return None
        dlat = radius_m / 111000.0
        dlng = radius_m / (111000.0 * math.cos(math.radians(lat)) + 1e-9)
        nearby = df[
            df["lat"].between(lat - dlat, lat + dlat)
            & df["lng"].between(lng - dlng, lng + dlng)
        ]
        if nearby.empty:
            return 0
        return int(_haversine_mask(nearby, lat, lng, radius_m).sum())


class ConvenienceTool:
    def __init__(self, safemap: SafeMapConvenienceClient | None = None,
                 local_search: NaverLocalSearchTool | None = None,
                 public_facilities: PublicFacilityCache | None = None):
        import os
        self.kakao_key = os.environ.get("KAKAO_REST_API_KEY")
        self.safemap = safemap or SafeMapConvenienceClient()
        self.local_search = local_search or NaverLocalSearchTool()
        self.public_facilities = public_facilities or PublicFacilityCache()
        self.online = bool(self.kakao_key) or self.safemap.available \
            or self.local_search.configured or self.public_facilities.available

    def refresh_safemap(self) -> dict:
        df = self.safemap.refresh()
        return {"rows": len(df), "cache": str(self.safemap.cache_path),
                "source": "safemap"}

    def _count_kakao(self, lat, lng, code, radius_m) -> int:
        import requests
        url = "https://dapi.kakao.com/v2/local/search/category.json"
        headers = {"Authorization": f"KakaoAK {self.kakao_key}"}
        params = {"category_group_code": code, "x": lng, "y": lat,
                  "radius": radius_m, "size": 15}
        response = requests.get(url, headers=headers, params=params, timeout=5)
        response.raise_for_status()
        return int(response.json().get("meta", {}).get("total_count", 0))

    def count_convenience_stores(self, lat: float, lng: float,
                                 radius_m: int = RADIUS_M, context: str = "",
                                 allow_local: bool = True) -> tuple[int | None, str, list]:
        try:
            count = self.safemap.count_nearby(lat, lng, radius_m)
        except Exception as exc:
            count = None
            safemap_reason = type(exc).__name__
        else:
            safemap_reason = "cache_missing"
        if count is not None:
            return count, "safemap", []
        if allow_local:
            result = self.local_search.search(
                lat, lng, context, "편의점", radius_m)
            return result["count"], result["source"], result["places"]
        return None, f"unavailable(safemap_{safemap_reason})", []

    def assess(self, lat: float, lng: float, radius_m: int = RADIUS_M,
               context: str = "") -> dict:
        counts, sources, places = {}, {}, {}
        (counts["convenience_24h"], sources["convenience_24h"],
         places["convenience_24h"]) = self.count_convenience_stores(
            lat, lng, radius_m, context=context)

        for key, code in KAKAO_CATEGORY.items():
            # LOCALDATA cache is the first source for every supported category.
            # Online map search is only a fallback when the public-data category
            # has not yet been downloaded/geocoded.
            if key in {"hospital", "pharmacy", "mart", "academy", "restaurant", "cafe"}:
                cached = self.public_facilities.nearby(
                    key, lat, lng, radius_m)
                if cached is not None:
                    counts[key] = cached["count"]
                    sources[key] = cached["source"]
                    places[key] = cached["places"]
                    continue
            if self.kakao_key:
                try:
                    counts[key] = self._count_kakao(lat, lng, code, radius_m)
                    sources[key] = "kakao"
                    places[key] = []
                    continue
                except Exception as exc:
                    sources[key] = f"kakao_failed({type(exc).__name__})"
            result = self.local_search.search(
                lat, lng, context, CATEGORY_KO[key].split("(")[0], radius_m)
            counts[key], sources[key], places[key] = (
                result["count"], result["source"], result["places"])

        available = [key for key, value in counts.items() if value is not None]
        source = "real" if len(available) == len(counts) else (
            "partial" if available else "unavailable")

        raw = 0.0
        for key in available:
            count = counts[key]
            present = 1.0 if count > 0 else 0.0
            raw += CONV_WEIGHTS[key] * (present + min(math.log1p(count) / 3.0, 1.0))
        cap = sum(CONV_WEIGHTS[key] for key in available) * 2.0
        score = round(min(100.0, raw / cap * 100), 1) if cap else None
        grade = ("조회 불가" if score is None else
                 ("우수" if score >= 65 else ("보통" if score >= 35 else "부족")))
        return {
            "radius_m": radius_m,
            "counts": counts,
            "convenience_score": score,
            "grade": grade,
            "source": source,
            "sources": sources,
            "places": places,
            "has": {key: (None if count is None else count > 0)
                    for key, count in counts.items()},
            "coverage": {"available": len(available), "total": len(counts)},
            "detail_ko": CATEGORY_KO,
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true",
                        help="생활안전지도 전국 편의점 캐시 갱신")
    args = parser.parse_args()
    tool = ConvenienceTool()
    if args.refresh:
        print(tool.refresh_safemap())
    for name, (lat, lng) in {
        "관악구": (37.4784, 126.9516), "외곽": (36.5, 127.5),
    }.items():
        result = tool.assess(lat, lng)
        print(name, result)
