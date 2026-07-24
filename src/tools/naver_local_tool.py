"""NAVER API HUB 지역검색을 이용한 선택 매물 주변 장소 확인.

NAVER Maps의 Dynamic Map/Geocoding 키는 장소 검색 권한이 없다. 이 도구는
별도 NAVER API HUB 키를 사용하며, API 결과의 WGS84 좌표를 매물 좌표와 다시
거리 계산하여 요청 반경 밖의 결과를 제거한다. 지역검색 자체가 질의당 최대 5건만
반환하므로 결과는 '전체 시설 수'가 아니라 '검색으로 확인된 최대 5건'이다.
"""
from __future__ import annotations

import html
import math
import os
import re
import threading
from typing import Any

import requests

LOCAL_URL = os.environ.get(
    "NAVER_LOCAL_SEARCH_URL",
    "https://naverapihub.apigw.ntruss.com/search/v1/local",
).strip()


def _plain(value: str) -> str:
    return re.sub(r"<[^>]+>", "", html.unescape(str(value or ""))).strip()


def _coordinate(value: Any) -> float:
    number = float(value)
    # 구 Developers API의 일부 응답은 WGS84 좌표를 1e7 배 정수로 반환했다.
    return number / 10_000_000.0 if abs(number) > 1000 else number


def _distance_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat, dlng = p2 - p1, math.radians(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlng / 2) ** 2
    return 6371000.0 * 2 * math.asin(math.sqrt(min(1.0, a)))


class NaverLocalSearchTool:
    """Per-report call-budgeted NAVER local search client."""

    def __init__(self, timeout: float = 5.0):
        self.client_id = os.environ.get("NAVER_API_HUB_CLIENT_ID", "").strip()
        self.client_secret = os.environ.get("NAVER_API_HUB_CLIENT_SECRET", "").strip()
        self.timeout = timeout
        self._state = threading.local()
        self._state.max_calls = 5
        self._state.calls = 0
        self._cache: dict[tuple, dict] = {}

    @property
    def configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    @property
    def max_calls(self) -> int:
        return int(getattr(self._state, "max_calls", 5))

    @property
    def calls(self) -> int:
        return int(getattr(self._state, "calls", 0))

    def begin_request(self, max_calls: int = 5) -> None:
        self._state.max_calls = max(1, min(int(max_calls), 20))
        self._state.calls = 0

    def search(self, lat: float, lng: float, context: str, keyword: str,
               radius_m: int) -> dict:
        query = " ".join(part for part in (context.strip(), keyword.strip()) if part)
        key = (round(lat, 5), round(lng, 5), query, int(radius_m))
        if key in self._cache:
            return {**self._cache[key], "cached": True}
        if not self.configured:
            return self._unavailable(query, "NAVER_API_HUB_CLIENT_ID/SECRET 미설정")
        if self.calls >= self.max_calls:
            return self._unavailable(query, f"요청당 {self.max_calls}회 호출 한도 도달")
        self._state.calls = self.calls + 1
        try:
            response = requests.get(
                LOCAL_URL,
                headers={
                    "X-NCP-APIGW-API-KEY-ID": self.client_id,
                    "X-NCP-APIGW-API-KEY": self.client_secret,
                },
                params={"query": query, "display": 5, "start": 1,
                        "sort": "random", "format": "json"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            places = []
            for item in response.json().get("items", [])[:5]:
                try:
                    place_lng = _coordinate(item.get("mapx"))
                    place_lat = _coordinate(item.get("mapy"))
                    distance = _distance_m(lat, lng, place_lat, place_lng)
                except (TypeError, ValueError):
                    continue
                if distance <= radius_m:
                    places.append({
                        "name": _plain(item.get("title")),
                        "category": _plain(item.get("category")),
                        "address": item.get("roadAddress") or item.get("address") or "",
                        "lat": round(place_lat, 7), "lng": round(place_lng, 7),
                        "distance_m": round(distance), "link": item.get("link") or "",
                    })
            places.sort(key=lambda item: item["distance_m"])
            result = {
                "available": True, "source": "naver_api_hub_local",
                "query": query, "radius_m": radius_m, "count": len(places),
                "places": places, "result_limit": 5,
                "count_is_lower_bound": len(places) == 5,
                "warning": "NAVER 지역검색 상위 5건을 좌표 거리로 재검증한 결과입니다.",
                "cached": False,
            }
            self._cache[key] = result
            return result
        except Exception as exc:
            return self._unavailable(query, f"{type(exc).__name__}로 지역검색 실패")

    @staticmethod
    def _unavailable(query: str, reason: str) -> dict:
        return {
            "available": False, "source": "unavailable", "query": query,
            "count": None, "places": [], "result_limit": 5,
            "count_is_lower_bound": False, "reason": reason,
        }
