"""전국 합성 매물용 시도·시군구 목록과 대표 좌표를 만든다.

지역 목록은 프로젝트에 내려받은 HUG 시군구 사고현황을 기준으로 하고, 대표 좌표는
전국 CCTV 공개데이터의 주소/좌표 중앙값으로 계산한다. CCTV 매칭이 없는 지역은 시도
중심점 주변의 결정론적 합성 좌표를 사용한다.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.data_augmentation.region_stats import load_region_accident_stats


EXPECTED_SIDOS = (
    "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
    "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
)

SIDO_ALIASES = {
    "서울특별시": "서울", "서울시": "서울",
    "부산광역시": "부산", "부산시": "부산",
    "대구광역시": "대구", "대구시": "대구",
    "인천광역시": "인천", "인천시": "인천",
    "광주광역시": "광주", "광주시": "광주",
    "대전광역시": "대전", "대전시": "대전",
    "울산광역시": "울산", "울산시": "울산",
    "세종특별자치시": "세종", "세종시": "세종",
    "경기도": "경기",
    "강원특별자치도": "강원", "강원도": "강원",
    "충청북도": "충북", "충청남도": "충남",
    "전북특별자치도": "전북", "전라북도": "전북",
    "전라남도": "전남", "경상북도": "경북", "경상남도": "경남",
    "제주특별자치도": "제주", "제주도": "제주",
}

# 시도청 인근 중심점과 시군구 fallback 좌표의 위·경도 분산 범위.
SIDO_GEOMETRY = {
    "서울": (37.5665, 126.9780, 0.42, 0.55),
    "부산": (35.1796, 129.0756, 0.42, 0.62),
    "대구": (35.8714, 128.6014, 0.48, 0.62),
    "인천": (37.4563, 126.7052, 0.72, 1.05),
    "광주": (35.1595, 126.8526, 0.34, 0.42),
    "대전": (36.3504, 127.3845, 0.35, 0.45),
    "울산": (35.5384, 129.3114, 0.52, 0.60),
    "세종": (36.4800, 127.2890, 0.24, 0.30),
    "경기": (37.4138, 127.5183, 1.35, 1.80),
    "강원": (37.8228, 128.1555, 1.85, 2.25),
    "충북": (36.6357, 127.4917, 1.20, 1.25),
    "충남": (36.5184, 126.8000, 1.15, 1.45),
    "전북": (35.7175, 127.1530, 1.10, 1.40),
    "전남": (34.8161, 126.4629, 1.25, 1.85),
    "경북": (36.4919, 128.8889, 1.65, 1.85),
    "경남": (35.4606, 128.2132, 1.25, 1.80),
    "제주": (33.4996, 126.5312, 0.60, 1.15),
}


def _fallback_center(sido: str, gugun: str) -> tuple[float, float]:
    lat, lng, lat_span, lng_span = SIDO_GEOMETRY[sido]
    digest = hashlib.blake2b(f"{sido}|{gugun}".encode("utf-8"), digest_size=4).digest()
    a = int.from_bytes(digest[:2], "big") / 65535 - 0.5
    b = int.from_bytes(digest[2:], "big") / 65535 - 0.5
    return lat + a * lat_span, lng + b * lng_span


def _address_region(address: str, allowed: dict[str, set[str]]) -> tuple[str, str] | None:
    text = " ".join(str(address).replace("(", " ").split())
    if not text or text == "nan":
        return None
    sido = None
    rest = text
    for alias in sorted(SIDO_ALIASES, key=len, reverse=True):
        if text.startswith(alias):
            sido = SIDO_ALIASES[alias]
            rest = text[len(alias):].strip()
            break
    if sido is None:
        return None
    tokens = rest.split()
    candidates: list[str] = []
    if tokens:
        candidates.append(tokens[0])
    if len(tokens) >= 2:
        city, district = tokens[0], tokens[1]
        candidates.extend([
            district,
            city + district,
            city.removesuffix("시") + district,
        ])
    if sido == "세종":
        candidates.extend(["세종시", "세종"])
    for candidate in candidates:
        if candidate in allowed.get(sido, set()):
            return sido, candidate
    return None


def _cctv_coordinate_points(regions: pd.DataFrame) -> pd.DataFrame:
    """Return valid public CCTV WGS84 points matched to canonical districts."""
    path = config.DATA_RAW / "safety" / "CCTV정보.csv"
    if not path.exists():
        return pd.DataFrame(columns=["sido", "gugun", "lat", "lng", "road_address", "jibun_address"])
    allowed = {
        sido: set(part["gugun"].astype(str))
        for sido, part in regions.groupby("sido", sort=False)
    }
    samples: list[pd.DataFrame] = []
    usecols = ["소재지도로명주소", "소재지지번주소", "WGS84위도", "WGS84경도"]
    encoding = "utf-8-sig"
    try:
        pd.read_csv(path, encoding=encoding, usecols=usecols, nrows=1)
    except UnicodeDecodeError:
        encoding = "cp949"
    for chunk in pd.read_csv(
        path, encoding=encoding, usecols=usecols, chunksize=100_000,
        low_memory=False,
    ):
        address = chunk["소재지도로명주소"].fillna(chunk["소재지지번주소"]).fillna("")
        matched = [_address_region(value, allowed) for value in address]
        valid = np.array([item is not None for item in matched])
        if not valid.any():
            continue
        coordinates = chunk.loc[valid, ["WGS84위도", "WGS84경도"]].copy()
        coordinates["sido"] = [item[0] for item in matched if item is not None]
        coordinates["gugun"] = [item[1] for item in matched if item is not None]
        coordinates["lat"] = pd.to_numeric(coordinates.pop("WGS84위도"), errors="coerce")
        coordinates["lng"] = pd.to_numeric(coordinates.pop("WGS84경도"), errors="coerce")
        coordinates["road_address"] = chunk.loc[valid, "소재지도로명주소"].fillna("").astype(str).to_numpy()
        coordinates["jibun_address"] = chunk.loc[valid, "소재지지번주소"].fillna("").astype(str).to_numpy()
        coordinates = coordinates[
            coordinates["lat"].between(32.5, 39.5)
            & coordinates["lng"].between(124.0, 132.0)
        ]
        samples.append(coordinates)
    if not samples:
        return pd.DataFrame(columns=["sido", "gugun", "lat", "lng", "road_address", "jibun_address"])
    return pd.concat(samples, ignore_index=True)[
        ["sido", "gugun", "lat", "lng", "road_address", "jibun_address"]
    ]


def _cctv_centroids(regions: pd.DataFrame) -> pd.DataFrame:
    points = _cctv_coordinate_points(regions)
    if points.empty:
        return pd.DataFrame(columns=["sido", "gugun", "lat", "lng", "coordinate_samples"])
    result = points.groupby(["sido", "gugun"], as_index=False).agg(
        lat=("lat", "median"), lng=("lng", "median"), coordinate_samples=("lat", "size"),
    )
    return result


def load_nationwide_coordinate_anchors(
    refresh: bool = False, max_points_per_region: int = 1200,
) -> pd.DataFrame:
    """Cache privacy-safe real-map coordinate anchors for synthetic listings.

    Only public WGS84 coordinates and canonical region labels are retained;
    CCTV identifiers and addresses are deliberately omitted.  Generation adds
    a small spatial jitter, so an output coordinate is not an actual facility
    or listing address.
    """
    output = config.DATA_GEN / "nationwide_coordinate_anchors.csv"
    if output.exists() and not refresh:
        cached = pd.read_csv(output, encoding="utf-8-sig")
        required = {
            "anchor_id", "sido", "gugun", "lat", "lng", "coordinate_source",
            "road_address", "jibun_address",
        }
        if required.issubset(cached.columns) and len(cached):
            cached["road_address"] = cached["road_address"].fillna("")
            cached["jibun_address"] = cached["jibun_address"].fillna("")
            return cached

    regions = load_region_accident_stats()[["sido", "gugun"]].drop_duplicates()
    points = _cctv_coordinate_points(regions)
    if points.empty:
        return pd.DataFrame(columns=[
            "anchor_id", "sido", "gugun", "lat", "lng", "coordinate_source",
            "road_address", "jibun_address",
        ])

    capped: list[pd.DataFrame] = []
    for (sido, gugun), part in points.groupby(["sido", "gugun"], sort=True):
        if len(part) > max_points_per_region:
            seed = int.from_bytes(
                hashlib.blake2b(f"{sido}|{gugun}".encode("utf-8"), digest_size=4).digest(),
                "big",
            )
            part = part.sample(n=max_points_per_region, random_state=seed)
        capped.append(part)
    anchors = pd.concat(capped, ignore_index=True).drop_duplicates(
        ["sido", "gugun", "lat", "lng"]
    )
    anchors.insert(0, "anchor_id", np.arange(1, len(anchors) + 1))
    anchors["coordinate_source"] = "전국 CCTV 공개 WGS84 좌표"
    output.parent.mkdir(parents=True, exist_ok=True)
    anchors.to_csv(output, index=False, encoding="utf-8-sig")
    return anchors


def load_nationwide_region_catalog(refresh: bool = False) -> pd.DataFrame:
    """17개 시도와 HUG 원천의 모든 시군구 검색 단위를 반환한다."""
    output = config.DATA_GEN / "nationwide_region_catalog.csv"
    stats = load_region_accident_stats()
    regions = stats[["sido", "gugun"]].drop_duplicates().sort_values(
        ["sido", "gugun"]
    ).reset_index(drop=True)
    missing_sidos = set(EXPECTED_SIDOS) - set(regions["sido"])
    if missing_sidos:
        raise ValueError(f"전국 지역 원천에서 누락된 시도: {sorted(missing_sidos)}")

    if output.exists() and not refresh:
        cached = pd.read_csv(output, encoding="utf-8-sig")
        cached_pairs = set(map(tuple, cached[["sido", "gugun"]].astype(str).to_numpy()))
        expected_pairs = set(map(tuple, regions.astype(str).to_numpy()))
        if cached_pairs == expected_pairs:
            return cached.sort_values("region_id").reset_index(drop=True)

    centers = _cctv_centroids(regions)
    catalog = regions.merge(centers, how="left", on=["sido", "gugun"])
    fallback = [
        _fallback_center(row.sido, row.gugun)
        for row in catalog.itertuples(index=False)
    ]
    missing = catalog["lat"].isna() | catalog["lng"].isna()
    catalog.loc[missing, "lat"] = [fallback[i][0] for i in np.flatnonzero(missing)]
    catalog.loc[missing, "lng"] = [fallback[i][1] for i in np.flatnonzero(missing)]
    catalog["coordinate_samples"] = catalog["coordinate_samples"].fillna(0).astype(int)
    catalog["coordinate_source"] = np.where(
        catalog["coordinate_samples"] > 0, "전국 CCTV 주소 좌표 중앙값", "시도 중심 결정론 합성좌표"
    )
    catalog.insert(0, "region_id", np.arange(1, len(catalog) + 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(output, index=False, encoding="utf-8-sig")
    return catalog
