"""Route and map tools used by the agent and the web map UI.

Provider ownership is explicit:

* driving: NAVER Directions 5
* public transport: TMAP Transit summary routes
* walking/bicycling: clearly labelled local estimates (until a dedicated API
  is configured)

Provider failures never make up an "exact" route.  They fall back to a result
whose ``estimated`` flag and ``fallback_reason`` are visible in the RAG trace.
"""
from __future__ import annotations

import math
import os
import threading
import time
from pathlib import Path
from typing import Optional

import requests

from src import config


GEOCODE_URL = "https://maps.apigw.ntruss.com/map-geocode/v2/geocode"
REVERSE_GEOCODE_URL = "https://maps.apigw.ntruss.com/map-reversegeocode/v2/gc"
DIRECTIONS_URL = "https://maps.apigw.ntruss.com/map-direction/v1/driving"
STATIC_MAP_URL = "https://maps.apigw.ntruss.com/map-static/v2/raster"
TMAP_TRANSIT_URL = "https://apis.openapi.sk.com/transit/routes"
TMAP_TRANSIT_SUMMARY_URL = "https://apis.openapi.sk.com/transit/routes/sub"

# Geocoding searches postal addresses rather than POI names.  A small audited
# catalogue covers common demo landmarks; unknown place names are returned as
# unresolved so the UI can ask for an address instead of inventing coordinates.
KNOWN_LANDMARKS = {
    "아주대": {
        "lat": 37.282943, "lng": 127.043824,
        "address": "경기도 수원시 영통구 월드컵로 206",
    },
    "아주대학교": {
        "lat": 37.282943, "lng": 127.043824,
        "address": "경기도 수원시 영통구 월드컵로 206",
    },
    "카이스트": {
        "lat": 36.372118, "lng": 127.360703,
        "address": "대전광역시 유성구 대학로 291",
    },
    "kaist": {
        "lat": 36.372118, "lng": 127.360703,
        "address": "대전광역시 유성구 대학로 291",
    },
    "ifc몰": {
        "lat": 37.525164, "lng": 126.925549,
        "address": "서울특별시 영등포구 국제금융로 10",
    },
}

SIDO_GUGUN_CENTROIDS = {
    ("서울", "종로구"): (37.5735, 126.9790),
    ("서울", "중구"): (37.5636, 126.9976),
    ("서울", "용산구"): (37.5326, 126.9906),
    ("서울", "성동구"): (37.5633, 127.0371),
    ("서울", "광진구"): (37.5385, 127.0824),
    ("서울", "동대문구"): (37.5744, 127.0396),
    ("서울", "마포구"): (37.5663, 126.9019),
    ("서울", "영등포구"): (37.5264, 126.8963),
    ("서울", "강서구"): (37.5509, 126.8495),
    ("서울", "양천구"): (37.5170, 126.8664),
    ("서울", "구로구"): (37.4954, 126.8874),
    ("서울", "관악구"): (37.4784, 126.9516),
    ("서울", "동작구"): (37.5124, 126.9393),
    ("서울", "서초구"): (37.4836, 127.0327),
    ("서울", "강남구"): (37.5172, 127.0473),
    ("서울", "송파구"): (37.5145, 127.1060),
    ("서울", "노원구"): (37.6542, 127.0568),
    ("서울", "은평구"): (37.6027, 126.9291),
}

_AVG_SPEED_KMH = {
    "driving": 22.0, "transit": 18.0, "walking": 4.5, "bicycling": 12.0,
}


class RouteNotFoundError(RuntimeError):
    """The provider answered successfully but returned no route."""


def _private_credentials() -> tuple[str, str]:
    """Read credentials from environment, with a local-development file fallback."""
    client_id = os.environ.get("NAVER_MAP_CLIENT_ID", "").strip()
    client_secret = os.environ.get("NAVER_MAP_CLIENT_SECRET", "").strip()
    if client_id and client_secret:
        return client_id, client_secret

    private_file = config.ROOT / "deploy" / "NAVER_KEYS.private.env"
    if private_file.exists():
        values: dict[str, str] = {}
        for raw in private_file.read_text(encoding="utf-8-sig").splitlines():
            raw = raw.strip()
            if raw and not raw.startswith("#") and "=" in raw:
                key, value = raw.split("=", 1)
                values[key.strip()] = value.strip()
        client_id = values.get("NAVER_MAP_CLIENT_ID", "")
        client_secret = values.get("NAVER_MAP_CLIENT_SECRET", "")
    return client_id, client_secret


def _private_tmap_app_key() -> str:
    """Read the server-side TMAP appKey without exposing it to the browser."""
    app_key = os.environ.get("TMAP_APP_KEY", "").strip()
    if app_key:
        return app_key

    private_file = config.ROOT / "deploy" / "TMAP_KEYS.private.env"
    if private_file.exists():
        for raw in private_file.read_text(encoding="utf-8-sig").splitlines():
            raw = raw.strip()
            if raw and not raw.startswith("#") and "=" in raw:
                key, value = raw.split("=", 1)
                if key.strip() == "TMAP_APP_KEY":
                    return value.strip()
    return ""


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    radius = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = (math.sin(dlat / 2) ** 2
             + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return 2 * radius * math.asin(math.sqrt(value))


class MapTool:
    def __init__(self, timeout_seconds: float = 8.0):
        self.client_id, self.client_secret = _private_credentials()
        live_disabled = os.environ.get("MAP_LIVE_DISABLED", "").strip().lower() \
            in {"1", "true", "yes"}
        self.online = bool(self.client_id and self.client_secret) and not live_disabled
        self.tmap_app_key = _private_tmap_app_key()
        self.tmap_online = bool(self.tmap_app_key) and not live_disabled
        self.timeout_seconds = timeout_seconds
        self.route_cache_ttl_seconds = max(
            0, int(os.environ.get("MAP_ROUTE_CACHE_TTL_SECONDS", "900")))
        self._route_cache: dict[tuple, tuple[float, dict]] = {}
        self._route_cache_lock = threading.Lock()

    @property
    def headers(self) -> dict[str, str]:
        return {
            "x-ncp-apigw-api-key-id": self.client_id,
            "x-ncp-apigw-api-key": self.client_secret,
            "Accept": "application/json",
        }

    def status(self) -> dict:
        return {
            "configured": self.online,
            "naver_configured": self.online,
            "tmap_configured": bool(getattr(self, "tmap_online", False)),
            "driving": "naver_directions5" if self.online else "estimated",
            "transit": ("tmap_transit" if getattr(self, "tmap_online", False)
                        else "estimated_haversine"),
            "walking": "estimated_haversine",
        }

    def route_provider(self, mode: str) -> str:
        if mode == "driving" and self.online:
            return "naver_directions5"
        if mode == "transit" and getattr(self, "tmap_online", False):
            return "tmap_transit"
        return f"estimated_haversine_{mode}"

    def has_live_route(self, mode: str) -> bool:
        return self.route_provider(mode) in {"naver_directions5", "tmap_transit"}

    def geocode(self, query: str) -> dict:
        query = str(query or "").strip()
        if not query:
            return {"ok": False, "reason": "empty_query", "query": query}

        known = KNOWN_LANDMARKS.get(query.lower()) or KNOWN_LANDMARKS.get(query)
        if known:
            return {"ok": True, "query": query, **known,
                    "source": "audited_landmark_catalog"}

        if self.online:
            try:
                response = requests.get(
                    GEOCODE_URL, headers=self.headers,
                    params={"query": query, "count": 5}, timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                payload = response.json()
                addresses = payload.get("addresses") or []
                if addresses:
                    row = addresses[0]
                    return {
                        "ok": True, "query": query, "lat": float(row["y"]),
                        "lng": float(row["x"]),
                        "address": row.get("roadAddress") or row.get("jibunAddress"),
                        "road_address": row.get("roadAddress"),
                        "jibun_address": row.get("jibunAddress"),
                        "source": "naver_geocoding",
                    }
            except requests.RequestException:
                # An unresolved condition is safer than failing the whole agent
                # turn or inventing a coordinate during a provider outage.
                pass
        return {
            "ok": False, "query": query, "reason": "address_not_found",
            "message": "장소명 대신 도로명주소나 지번주소를 입력해 주세요.",
        }

    def reverse_geocode(self, lat: float, lng: float) -> dict:
        if not self.online:
            return {"ok": False, "reason": "credentials_not_configured"}
        response = requests.get(
            REVERSE_GEOCODE_URL, headers=self.headers,
            params={
                "coords": f"{float(lng)},{float(lat)}", "output": "json",
                "orders": "roadaddr,addr,admcode,legalcode",
            }, timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results") or []
        return {"ok": bool(results), "lat": float(lat), "lng": float(lng),
                "results": results, "source": "naver_reverse_geocoding"}

    def static_map(self, *, lat: float, lng: float, width: int = 640,
                   height: int = 360, zoom: int = 15,
                   markers: Optional[list[tuple[float, float]]] = None) -> tuple[bytes, str]:
        if not self.online:
            raise RuntimeError("NAVER Maps credentials are not configured")
        width = max(100, min(int(width), 1024))
        height = max(100, min(int(height), 1024))
        params: list[tuple[str, str | int]] = [
            ("w", width), ("h", height), ("center", f"{lng},{lat}"),
            ("level", max(0, min(int(zoom), 20))), ("format", "png"),
        ]
        points = markers or [(lat, lng)]
        marker_value = "type:d|size:mid|color:0xFFBC00|pos:" + "|pos:".join(
            f"{point_lng} {point_lat}" for point_lat, point_lng in points[:100]
        )
        params.append(("markers", marker_value))
        response = requests.get(STATIC_MAP_URL, headers=self.headers, params=params,
                                timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.content, response.headers.get("content-type", "image/png")

    def travel_time(self, start, goal, mode: str = "transit") -> dict:
        mode = mode if mode in _AVG_SPEED_KMH else "transit"
        provider = self.route_provider(mode)
        if provider in {"naver_directions5", "tmap_transit"}:
            cached = self._get_cached_route(start, goal, mode)
            if cached is not None:
                return cached
            try:
                result = (self._naver_driving_time(start, goal)
                          if provider == "naver_directions5"
                          else self._tmap_transit_time(start, goal))
                self._cache_route(start, goal, mode, result)
                return result
            except RouteNotFoundError:
                return {
                    "minutes": 1_000_000.0,
                    "distance_km": round(haversine_km(start, goal), 2),
                    "mode": mode,
                    "source": f"{provider}_no_route",
                    "estimated": False,
                    "route_found": False,
                    "reason": "provider_returned_no_route",
                }
            except Exception as exc:
                result = self._estimated_travel_time(start, goal, mode)
                result["fallback_reason"] = type(exc).__name__
                result["attempted_provider"] = provider
                response = getattr(exc, "response", None)
                if response is not None:
                    result["provider_status_code"] = getattr(
                        response, "status_code", None)
                return result
        return self._estimated_travel_time(start, goal, mode)

    @staticmethod
    def _cache_key(start, goal, mode: str) -> tuple:
        return (mode, *(round(float(value), 6)
                        for value in (start[0], start[1], goal[0], goal[1])))

    def _get_cached_route(self, start, goal, mode: str) -> dict | None:
        ttl = getattr(self, "route_cache_ttl_seconds", 0)
        lock = getattr(self, "_route_cache_lock", None)
        cache = getattr(self, "_route_cache", None)
        if ttl <= 0 or lock is None or cache is None:
            return None
        key = self._cache_key(start, goal, mode)
        with lock:
            entry = cache.get(key)
            if not entry or time.monotonic() - entry[0] > ttl:
                if entry:
                    cache.pop(key, None)
                return None
            result = dict(entry[1])
        result["cached"] = True
        return result

    def _cache_route(self, start, goal, mode: str, result: dict) -> None:
        ttl = getattr(self, "route_cache_ttl_seconds", 0)
        lock = getattr(self, "_route_cache_lock", None)
        cache = getattr(self, "_route_cache", None)
        if ttl <= 0 or lock is None or cache is None:
            return
        with lock:
            cache[self._cache_key(start, goal, mode)] = (
                time.monotonic(), dict(result))

    def _estimated_travel_time(self, start, goal, mode: str) -> dict:
        straight = haversine_km(start, goal)
        road_factor = 1.25 if mode == "walking" else 1.30
        route_distance = straight * road_factor
        minutes = route_distance / _AVG_SPEED_KMH[mode] * 60
        if mode == "transit":
            minutes += 8.0
        return {
            "minutes": round(minutes, 1), "distance_km": round(route_distance, 2),
            "mode": mode, "source": f"estimated_haversine_{mode}",
            "estimated": True,
            "disclaimer": "실시간 노선 결과가 아닌 거리·평균속도 기반 예상 시간입니다.",
        }

    def estimate_travel_time(self, start, goal, mode: str = "transit") -> dict:
        """Network-free estimate for bulk candidate filtering.

        Directions 5 is intentionally reserved for individual routes; calling
        it once per database row would be slow and expensive for large maps.
        """
        mode = mode if mode in _AVG_SPEED_KMH else "transit"
        return self._estimated_travel_time(start, goal, mode)

    # Backward-compatible name used by older tests/callers.
    _mock_travel_time = _estimated_travel_time

    def _naver_driving_time(self, start, goal) -> dict:
        response = self._route_request_with_retry(
            requests.get,
            DIRECTIONS_URL, headers=self.headers,
            params={
                "start": f"{start[1]},{start[0]}",
                "goal": f"{goal[1]},{goal[0]}", "option": "trafast",
            }, timeout=self.timeout_seconds,
        )
        payload = response.json()
        summary = payload["route"]["trafast"][0]["summary"]
        return {
            "minutes": round(summary["duration"] / 60000, 1),
            "distance_km": round(summary["distance"] / 1000, 2),
            "mode": "driving", "source": "naver_directions5",
            "estimated": False,
        }

    def _tmap_transit_time(self, start, goal) -> dict:
        """Return the fastest current TMAP public-transport itinerary.

        The summary endpoint is sufficient for filtering properties and avoids
        downloading every leg/polyline for every candidate.  Coordinates are
        WGS84 and TMAP expects longitude in X and latitude in Y.
        """
        response = self._route_request_with_retry(
            requests.post,
            TMAP_TRANSIT_SUMMARY_URL,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "appKey": self.tmap_app_key,
            },
            json={
                "startX": str(float(start[1])),
                "startY": str(float(start[0])),
                "endX": str(float(goal[1])),
                "endY": str(float(goal[0])),
                "count": 3,
                "lang": 0,
                "format": "json",
            },
            timeout=self.timeout_seconds,
        )
        payload = response.json()
        metadata = payload.get("metaData") or {}
        itineraries = ((metadata.get("plan") or {}).get("itineraries") or [])
        valid = [item for item in itineraries
                 if isinstance(item, dict) and item.get("totalTime") is not None]
        if not valid:
            raise RouteNotFoundError("TMAP returned no public-transport itinerary")
        best = min(valid, key=lambda item: float(item["totalTime"]))
        fare = (((best.get("fare") or {}).get("regular") or {})
                .get("totalFare"))
        return {
            "minutes": round(float(best["totalTime"]) / 60.0, 1),
            "distance_km": round(float(best.get("totalDistance") or 0) / 1000.0, 2),
            "walk_minutes": round(float(best.get("totalWalkTime") or 0) / 60.0, 1),
            "walk_distance_m": int(best.get("totalWalkDistance") or 0),
            "transfer_count": int(best.get("transferCount") or 0),
            "fare_krw": float(fare) if fare is not None else None,
            "path_type": best.get("pathType"),
            "route_options_returned": len(valid),
            "mode": "transit", "source": "tmap_transit",
            "estimated": False, "route_found": True,
        }

    def _route_request_with_retry(self, requester, url: str, **kwargs):
        """Retry only throttling and transient upstream failures.

        Authentication and invalid-request errors are not retried.  The final
        exception is handled by :meth:`travel_time` and remains auditable.
        """
        response = None
        for attempt in range(3):
            response = requester(url, **kwargs)
            status = getattr(response, "status_code", None)
            transient = isinstance(status, int) and (
                status == 429 or status in {500, 502, 503, 504})
            if not transient or attempt == 2:
                response.raise_for_status()
                return response
            retry_after = (getattr(response, "headers", {}) or {}).get(
                "Retry-After")
            try:
                delay = min(3.0, max(0.2, float(retry_after)))
            except (TypeError, ValueError):
                delay = 0.4 * (2 ** attempt)
            time.sleep(delay)
        raise RuntimeError("unreachable route retry state")

    # Backward-compatible name.
    def _naver_travel_time(self, start, goal, mode: str) -> dict:
        return self.travel_time(start, goal, mode)

    def regions_within(self, goal, minutes: float, mode: str = "transit",
                       candidates: Optional[dict] = None) -> list[dict]:
        out = []
        for (sido, gugun), center in (candidates or SIDO_GUGUN_CENTROIDS).items():
            travel = self.travel_time(center, goal, mode)
            if travel["minutes"] <= minutes:
                out.append({"sido": sido, "gugun": gugun,
                            "minutes": travel["minutes"],
                            "distance_km": travel["distance_km"],
                            "source": travel["source"]})
        return sorted(out, key=lambda item: item["minutes"])


if __name__ == "__main__":
    tool = MapTool()
    print(tool.status())
    print(tool.geocode("아주대학교"))
