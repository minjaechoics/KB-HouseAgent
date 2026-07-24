"""V-World 활용모델 방식의 공공 전기차 충전소 조회.

V-World 예제는 지도/클러스터 표현을 담당하고 실제 충전기 데이터는
공공데이터포털 EvInfoServiceV2에서 받는다. 대용량 원천을 매 요청마다 내려받지
않고 동기화 후 SQLite에서 반경 검색한다.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import requests

from src import config
from src.real_estate_feeds.storage import ensure_feed_schema, utc_now


SOURCE_PAGE = "https://v-world.github.io/Utilization-Model/"
SOURCE_CODE = (
    "https://github.com/V-world/Utilization-Model/blob/master/"
    "utilization-model/%5B23.07%5D%EC%A0%84%EA%B8%B0%EC%B0%A8%EC%B6%A9%EC%A0%84%EC%86%8C/index.html"
)

STATUS_NAMES = {
    "1": "충전 가능", "2": "충전 중", "3": "고장/점검", "4": "통신 장애",
    "5": "통신 미연결", "9": "상태 미확인",
}


def _float(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _pick(row: dict, *keys: str) -> Any:
    lowered = {str(key).lower(): value for key, value in row.items()}
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
        value = lowered.get(key.lower())
        if value not in (None, ""):
            return value
    return None


def sync_ev_chargers(
    db_path: Path = config.DB_PATH, per_page: int = 10000,
    service_key: str | None = None,
) -> dict:
    ensure_feed_schema(db_path)
    key = (service_key if service_key is not None else config.EV_CHARGER_SERVICE_KEY).strip()
    if not key:
        raise RuntimeError("EV_CHARGER_SERVICE_KEY가 설정되지 않았습니다.")
    fetched_at = utc_now()
    page, total, written = 1, None, 0
    while total is None or (page - 1) * per_page < total:
        response = requests.get(
            config.EV_CHARGER_API_URL,
            params={"page": page, "perPage": per_page, "serviceKey": key},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") not in (None, 0):
            raise RuntimeError(f"EvInfoServiceV2 오류: {payload.get('code')} {payload.get('msg')}")
        rows = payload.get("data") or []
        total = int(payload.get("totalCount") or len(rows))
        values = []
        for row in rows:
            lat = _float(_pick(row, "lat", "latitude"))
            lng = _float(_pick(row, "longi", "lng", "longitude"))
            if lat is None or lng is None or not (32.5 <= lat <= 39.5 and 124 <= lng <= 132):
                continue
            station = str(_pick(row, "csNm", "stationName") or "전기차 충전소")
            charger = str(_pick(row, "cpNm", "chargerName") or "")
            source_id = str(_pick(row, "cpId", "chargerId", "id") or "")
            stable = source_id or f"{station}|{charger}|{lat:.7f}|{lng:.7f}"
            charger_id = "EV_" + hashlib.sha256(stable.encode()).hexdigest()[:32]
            status = str(_pick(row, "cpStat", "status") or "9")
            values.append((
                charger_id, station, charger, str(_pick(row, "addr", "location", "address") or ""),
                lat, lng, status, STATUS_NAMES.get(status, "상태 미확인"),
                str(_pick(row, "cpTp", "chargerMethod") or ""),
                str(_pick(row, "chargeTp", "chargeType") or ""),
                str(_pick(row, "statUpdDt", "updatedAt") or ""), fetched_at,
                config.EV_CHARGER_API_URL,
                json.dumps(row, ensure_ascii=False, sort_keys=True, default=str),
            ))
        with sqlite3.connect(db_path) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO ev_chargers VALUES(" + ",".join("?" for _ in range(14)) + ")",
                values,
            )
        written += len(values)
        if not rows:
            break
        page += 1
    return {"written": written, "total_count": total or 0, "fetched_at": fetched_at}


def _distance_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6_371_000 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class EVChargerTool:
    def __init__(self, db_path: Path = config.DB_PATH):
        self.db_path = db_path

    def nearby(self, lat: float, lng: float, radius_m: int = 1500, limit: int = 20) -> dict:
        ensure_feed_schema(self.db_path)
        radius_m = max(100, min(int(radius_m), 10_000))
        lat_delta, lng_delta = radius_m / 111_000, radius_m / (111_000 * max(0.2, math.cos(math.radians(lat))))
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = [dict(row) for row in conn.execute(
                "SELECT * FROM ev_chargers WHERE latitude BETWEEN ? AND ? "
                "AND longitude BETWEEN ? AND ?",
                (lat - lat_delta, lat + lat_delta, lng - lng_delta, lng + lng_delta),
            )]
        for row in rows:
            row["distance_m"] = round(_distance_m(lat, lng, row["latitude"], row["longitude"]))
            row.pop("raw_json", None)
        rows = sorted((row for row in rows if row["distance_m"] <= radius_m), key=lambda x: x["distance_m"])
        selected = rows[:max(1, min(limit, 50))]
        available_count = sum(row.get("status_code") == "1" for row in rows)
        return {
            "available": bool(rows) or bool(config.EV_CHARGER_SERVICE_KEY),
            "configured": bool(config.EV_CHARGER_SERVICE_KEY),
            "radius_m": radius_m, "charger_count": len(rows),
            "available_charger_count": available_count,
            "stations": selected,
            "source": "V-World 전기차충전소 활용모델 / 공공데이터포털 EvInfoServiceV2",
            "source_url": SOURCE_PAGE, "source_code": SOURCE_CODE,
            "notice": (
                "상태는 수집 시점 스냅샷이며 현장 도착 전 운영기관 앱에서 다시 확인해야 합니다."
                if rows else
                "EvInfoServiceV2 별도 활용신청 키 또는 동기화 데이터가 없어 주변 충전소를 표시하지 못했습니다."
            ),
        }
