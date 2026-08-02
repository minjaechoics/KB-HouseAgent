"""팔달구 활성 DB의 경찰서/소방서 좌표 캐시와 mart_walk_minutes를 채운다.

SAFEMAP_SERVICE_KEY(공공데이터)가 비어 있는 환경에서는 안전/편의 조건이
NAVER API HUB 지역검색으로 폴백하는데, 그 폴백이 요구하는 좌표 캐시
(data/generated/safety_geocoded/{police,fire_station}.csv)와 매물별
mart_walk_minutes 컬럼이 비어 있으면 검색 조건이 항상 0건으로 실패한다.

이 스크립트는 그 두 가지를 한 번 실행으로 채운다. NAVER_API_HUB_CLIENT_ID/
SECRET이 .env(.production)에 설정되어 있어야 한다. 재실행해도 안전하다
(기존 캐시/컬럼을 덮어쓸 뿐 스키마를 바꾸지 않는다).

사용법:
    python -m scripts.backfill_local_facility_cache
"""
from __future__ import annotations

import csv
import math
import sqlite3
from pathlib import Path

from src import config
from src.tools.naver_local_tool import NaverLocalSearchTool

GEOCODED_DIR = config.DATA_GEN / "safety_geocoded"
WALK_SPEED_KMH = 4.5
ROAD_FACTOR = 1.25

# 팔달구 원도심 대략 중심(매교동). radius_m을 크게 잡아 지역명 키워드로만
# 후보를 특정하고, 실제 거리는 매물별로 다시 계산한다.
CENTER = (37.2698, 127.0140)

STATION_QUERIES = {
    "police": ["수원팔달경찰서", "수원남부경찰서", "팔달지구대", "매산지구대",
               "지동파출소", "화서지구대", "수원중부경찰서", "행궁동파출소"],
    "fire_station": ["수원팔달소방서", "권선소방서", "매산119안전센터",
                     "지동119안전센터"],
}

DONGS = ["인계동", "화서동", "고등동", "우만동", "지동", "매산로3가", "매산로2가",
         "매교동", "교동", "장안동", "신풍동", "매산로1가", "남창동", "팔달로2가",
         "북수동", "남수동", "팔달로3가", "매향동", "팔달로1가", "중동", "구천동"]


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat, dlng = math.radians(lat2 - lat1), math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dlng / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(a))


def _write_station_cache(tool: NaverLocalSearchTool) -> None:
    GEOCODED_DIR.mkdir(parents=True, exist_ok=True)
    for key, keywords in STATION_QUERIES.items():
        rows: dict[tuple[float, float], dict] = {}
        for keyword in keywords:
            result = tool.search(CENTER[0], CENTER[1], "", keyword, radius_m=10000)
            for place in result.get("places", []):
                rows[(place["lat"], place["lng"])] = place
        path = GEOCODED_DIR / f"{key}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["name", "address", "lat", "lng"])
            for place in rows.values():
                writer.writerow(
                    [place["name"], place["address"], place["lat"], place["lng"]])
        print(f"{key}: {len(rows)}곳 -> {path}")


def _collect_mart_convenience(tool: NaverLocalSearchTool) -> list[tuple[float, float]]:
    rows: dict[tuple[float, float], dict] = {}
    for dong in DONGS:
        for keyword in (f"수원 팔달구 {dong} 편의점", f"수원 팔달구 {dong} 마트"):
            result = tool.search(CENTER[0], CENTER[1], "", keyword, radius_m=50000)
            for place in result.get("places", []):
                rows[(place["lat"], place["lng"])] = place
    path = GEOCODED_DIR / "paldal_mart_convenience.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "address", "lat", "lng"])
        for place in rows.values():
            writer.writerow(
                [place["name"], place["address"], place["lat"], place["lng"]])
    print(f"편의점/마트: {len(rows)}곳 -> {path}")
    return [(lat, lng) for lat, lng in rows.keys()]


def _backfill_mart_walk_minutes(stores: list[tuple[float, float]]) -> None:
    if not stores:
        print("편의점/마트 위치를 하나도 못 찾아 mart_walk_minutes를 건너뜁니다.")
        return
    con = sqlite3.connect(str(config.DB_PATH))
    try:
        cur = con.cursor()
        cur.execute("SELECT property_id, lat, lng FROM properties")
        updates = []
        for property_id, lat, lng in cur.fetchall():
            if lat is None or lng is None:
                continue
            nearest_km = min(_haversine_km(lat, lng, slat, slng)
                             for slat, slng in stores)
            minutes = round(nearest_km / WALK_SPEED_KMH * 60.0 * ROAD_FACTOR, 1)
            updates.append((minutes, property_id))
        cur.executemany(
            "UPDATE properties SET mart_walk_minutes = ? WHERE property_id = ?",
            updates,
        )
        con.commit()
        print(f"mart_walk_minutes 갱신: {len(updates)}건")
    finally:
        con.close()


def main() -> None:
    tool = NaverLocalSearchTool()
    if not tool.configured:
        raise SystemExit(
            "NAVER_API_HUB_CLIENT_ID/SECRET이 설정되어 있지 않습니다. "
            ".env(.production)에 값을 채운 뒤 다시 실행하세요."
        )
    tool.begin_request(max_calls=120)
    _write_station_cache(tool)
    stores = _collect_mart_convenience(tool)
    _backfill_mart_walk_minutes(stores)
    print(f"총 API 호출 수: {tool.calls}")


if __name__ == "__main__":
    main()
