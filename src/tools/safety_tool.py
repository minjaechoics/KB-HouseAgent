"""
치안/안전 판단 모듈 (Agent Tool).

``data/downloaded/safety``에 현재 들어 있는 공공데이터 원본 파일명과 한글 컬럼을 직접
인식한다. CCTV/안전비상벨은 WGS84 좌표로 실제 반경 집계하고, CCTV는 시설 행 수가
아닌 ``카메라대수``를 합산한다. 경찰 치안센터/소방서는 원본에 주소만 있으므로
``data/generated/safety_geocoded/{police,fire_station}.csv`` 좌표 캐시가 있으면 실제
반경 집계한다. 경찰/소방 좌표 캐시가 없으면 NAVER API HUB 지역검색 결과를
좌표 거리로 재검증한다. 어떤 데이터원도 사용할 수 없으면 숫자를 만들지 않고
``None/unavailable``로 반환한다. 편의점은 생활안전지도 IF_0039 캐시를 공유한다.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.tools.convenience_tool import ConvenienceTool
from src.tools.naver_local_tool import NaverLocalSearchTool

SAFETY_DIR = config.DATA_RAW / "safety"
GEOCODED_DIR = config.DATA_GEN / "safety_geocoded"
RADIUS_M = 300

SAFETY_WEIGHTS = {
    "cctv": 1.0,
    "emergency_bell": 2.0,
    "police": 15.0,
    "fire_station": 10.0,
    "convenience_24h": 3.0,
}

SAFETY_SOURCES = {
    "cctv": {
        "files": ("CCTV정보.csv", "cctv.csv"),
        "lat": ("WGS84위도", "위도", "lat", "latitude"),
        "lng": ("WGS84경도", "경도", "lng", "longitude"),
        "weight": ("카메라대수", "count", "weight"),
        "address": ("소재지도로명주소", "소재지지번주소", "주소", "address"),
    },
    "emergency_bell": {
        "files": ("안전비상벨위치정보.csv", "emergency_bell.csv"),
        "lat": ("WGS84위도", "위도", "lat", "latitude"),
        "lng": ("WGS84경도", "경도", "lng", "longitude"),
        "address": ("소재지도로명주소", "소재지지번주소", "주소", "address"),
    },
    "police": {
        "files": (
            "police_gg.csv",
            "경찰청_전국 치안센터 주소 현황_20251231.csv",
            "police.csv",
        ),
        "lat": ("WGS84위도", "위도", "lat", "latitude"),
        "lng": ("WGS84경도", "경도", "lng", "longitude"),
        "address": ("주소", "소재지도로명주소", "address"),
    },
    "fire_station": {
        "files": (
            "fire_station_gg.csv",
            "소방청_시도 소방서 현황_20250701.csv",
            "fire_station.csv",
        ),
        "lat": ("WGS84위도", "위도", "lat", "latitude"),
        "lng": ("WGS84경도", "경도", "lng", "longitude"),
        "address": ("주소", "소재지도로명주소", "address"),
    },
}


def _first_existing(columns, aliases):
    return next((name for name in aliases if name in columns), None)


def _distance_mask(df: pd.DataFrame, lat: float, lng: float,
                   radius_m: int) -> np.ndarray:
    lat2 = np.radians(df["lat"].to_numpy(dtype=float))
    lng2 = np.radians(df["lng"].to_numpy(dtype=float))
    lat1, lng1 = math.radians(lat), math.radians(lng)
    dlat, dlng = lat2 - lat1, lng2 - lng1
    a = np.sin(dlat / 2.0) ** 2 + math.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2.0) ** 2
    distance = 6371000.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return distance <= radius_m


class SafetyTool:
    def __init__(self, data_dir: Path = SAFETY_DIR,
                 convenience_tool: ConvenienceTool | None = None,
                 local_search: NaverLocalSearchTool | None = None):
        self.data_dir = Path(data_dir)
        self.convenience_tool = convenience_tool or ConvenienceTool()
        self.local_search = local_search or self.convenience_tool.local_search
        self._cache: dict[str, pd.DataFrame | None] = {}
        self._metadata: dict[str, dict] = {}
        self._paths = {key: self._find_path(spec["files"])
                       for key, spec in SAFETY_SOURCES.items()}
        self.online = any(path is not None for path in self._paths.values()) \
            or self.convenience_tool.safemap.available

    def _find_path(self, candidates) -> Path | None:
        if not self.data_dir.exists() and not GEOCODED_DIR.exists():
            return None
        by_lower = {path.name.lower(): path for path in self.data_dir.glob("*.csv")}
        by_lower.update({
            path.name.lower(): path for path in GEOCODED_DIR.glob("*.csv")
        })
        for name in candidates:
            path = self.data_dir / name
            if path.exists():
                return path
            generated_path = GEOCODED_DIR / name
            if generated_path.exists():
                return generated_path
            if name.lower() in by_lower:
                return by_lower[name.lower()]
        return None

    @staticmethod
    def _read_csv(path: Path, usecols=None) -> pd.DataFrame:
        last_error = None
        for encoding in ("cp949", "utf-8-sig", "utf-8"):
            try:
                return pd.read_csv(path, encoding=encoding, usecols=usecols,
                                   low_memory=False)
            except (UnicodeDecodeError, LookupError, ValueError) as exc:
                last_error = exc
        raise last_error  # type: ignore[misc]

    def _load(self, key: str) -> pd.DataFrame | None:
        if key in self._cache:
            return self._cache[key]
        spec = SAFETY_SOURCES[key]
        path = self._paths.get(key)
        if path is None:
            self._metadata[key] = {"path": None, "raw_rows": 0,
                                   "geocoded_rows": 0, "status": "missing"}
            self._cache[key] = None
            return None

        wanted = set(spec["lat"] + spec["lng"] + spec.get("weight", ())
                     + spec.get("address", ()))
        raw = self._read_csv(path, usecols=lambda column: column in wanted)
        lat_col = _first_existing(raw.columns, spec["lat"])
        lng_col = _first_existing(raw.columns, spec["lng"])
        raw_rows = len(raw)

        # 주소 전용 원본은 별도의 지오코딩 결과 캐시와 결합한다.
        if not lat_col or not lng_col:
            geocoded_path = GEOCODED_DIR / f"{key}.csv"
            if geocoded_path.exists():
                raw = self._read_csv(geocoded_path)
                lat_col = _first_existing(raw.columns, spec["lat"])
                lng_col = _first_existing(raw.columns, spec["lng"])
            if not lat_col or not lng_col:
                self._metadata[key] = {
                    "path": str(path), "raw_rows": raw_rows,
                    "geocoded_rows": 0, "status": "address_only",
                    "geocoded_cache": str(geocoded_path),
                }
                self._cache[key] = raw
                return raw

        normalized = pd.DataFrame({
            "lat": pd.to_numeric(raw[lat_col], errors="coerce"),
            "lng": pd.to_numeric(raw[lng_col], errors="coerce"),
        })
        weight_col = _first_existing(raw.columns, spec.get("weight", ()))
        if weight_col:
            normalized["weight"] = pd.to_numeric(raw[weight_col], errors="coerce").fillna(1)
        else:
            normalized["weight"] = 1
        normalized = normalized.dropna(subset=["lat", "lng"])
        normalized = normalized[
            normalized["lat"].between(32.0, 39.5)
            & normalized["lng"].between(123.0, 133.0)
        ]
        self._metadata[key] = {
            "path": str(path), "raw_rows": raw_rows,
            "geocoded_rows": len(normalized), "status": "ready",
        }
        self._cache[key] = normalized
        return normalized

    def _count_real(self, key: str, lat: float, lng: float,
                    radius_m: int, exclude_origin: bool = False) -> int | None:
        df = self._load(key)
        if df is None or "lat" not in df.columns or "lng" not in df.columns:
            return None
        dlat = radius_m / 111000.0
        dlng = radius_m / (111000.0 * math.cos(math.radians(lat)) + 1e-9)
        nearby = df[
            df["lat"].between(lat - dlat, lat + dlat)
            & df["lng"].between(lng - dlng, lng + dlng)
        ]
        if exclude_origin and not nearby.empty:
            nearby = nearby[
                (nearby["lat"].sub(lat).abs() > 1e-7)
                | (nearby["lng"].sub(lng).abs() > 1e-7)
            ]
        if nearby.empty:
            return 0
        mask = _distance_mask(nearby, lat, lng, radius_m)
        return int(nearby.loc[mask, "weight"].sum())

    def geocoding_templates(self) -> dict[str, str]:
        """주소 전용 원본을 lat/lng 보강용 CSV 템플릿으로 내보낸다."""
        GEOCODED_DIR.mkdir(parents=True, exist_ok=True)
        result = {}
        for key in ("police", "fire_station"):
            df = self._load(key)
            meta = self._metadata.get(key, {})
            if df is None or meta.get("status") != "address_only":
                continue
            spec = SAFETY_SOURCES[key]
            address_col = _first_existing(df.columns, spec["address"])
            if not address_col:
                continue
            out = GEOCODED_DIR / f"{key}.csv"
            pd.DataFrame({"address": df[address_col], "lat": "", "lng": ""}) \
                .drop_duplicates("address").to_csv(out, index=False, encoding="utf-8-sig")
            result[key] = str(out)
        return result

    def assess(self, lat: float, lng: float, radius_m: int = RADIUS_M,
               context: str = "", exclude_cctv_anchor: bool = False) -> dict:
        counts, sources, places = {}, {}, {}
        for key in SAFETY_SOURCES:
            count = self._count_real(
                key, lat, lng, radius_m,
                exclude_origin=bool(exclude_cctv_anchor and key == "cctv"),
            )
            if count is None and key in ("police", "fire_station"):
                keyword = "경찰" if key == "police" else "소방서"
                result = self.local_search.search(
                    lat, lng, context, keyword, radius_m)
                count = result["count"]
                sources[key] = result["source"]
                places[key] = result["places"]
            elif count is None:
                meta = self._metadata.get(key, {})
                if meta.get("status") == "address_only":
                    sources[key] = f"unavailable(unlocated_raw:{meta.get('raw_rows', 0)})"
                else:
                    sources[key] = "unavailable"
                places[key] = []
            else:
                sources[key] = "raw"
                places[key] = []
            counts[key] = count

        (counts["convenience_24h"], sources["convenience_24h"],
         places["convenience_24h"]) = self.convenience_tool.count_convenience_stores(
            lat, lng, radius_m, context=context, allow_local=True)

        available = [key for key, value in counts.items() if value is not None]
        source = "real" if len(available) == len(counts) else (
            "partial" if available else "unavailable")
        raw_score = sum(counts[key] * SAFETY_WEIGHTS[key] for key in available)
        score = round(min(100.0, raw_score / 120.0 * 100), 1) if available else None
        grade = ("조회 불가" if score is None else
                 ("안전" if score >= 60 else ("보통" if score >= 30 else "주의")))
        return {
            "radius_m": radius_m,
            "counts": counts,
            "safety_score": score,
            "grade": grade,
            "source": source,
            "sources": sources,
            "places": places,
            "coverage": {"available": len(available), "total": len(counts)},
            "raw_data": dict(self._metadata),
            "cctv_anchor_excluded": bool(exclude_cctv_anchor),
            "detail_ko": {
                "cctv": "방범 CCTV", "emergency_bell": "안전 비상벨",
                "police": "치안센터/파출소", "fire_station": "소방서",
                "convenience_24h": "편의점(24시간 여부 미확인)",
            },
        }


if __name__ == "__main__":
    tool = SafetyTool()
    print(tool.assess(37.4784, 126.9516))
