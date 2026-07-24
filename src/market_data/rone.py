"""한국부동산원 R-ONE Open API 동기화 및 지역시장 조회.

R-ONE의 738개 통계표 메타데이터는 모두 보존한다. 실제 관측값은 청년의 주택
의사결정에 직접 쓰이는 가격지수·평균가격·전세가율·수급동향만 선별해 최근
시계열을 저장한다. 키는 서버 환경변수에서만 읽고 응답이나 로그에 노출하지 않는다.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src import config
from src.real_estate_feeds.storage import ensure_feed_schema, utc_now


RONE_SOURCE_URL = "https://www.reb.or.kr/r-one/portal/openapi/openApiIntroPage.do"

# 유형별 매매/전세/월세 가격지수. 현재 R-ONE의 기준시점 개편 계열이다.
PRICE_INDEX_TABLES = {
    ("주택종합", "매매"): "A_2024_00016",
    ("주택종합", "전세"): "A_2024_00019",
    ("주택종합", "월세"): "A_2024_00022",
    ("아파트", "매매"): "A_2024_00045",
    ("아파트", "전세"): "A_2024_00050",
    ("아파트", "월세"): "A_2024_00055",
    ("연립/다세대", "매매"): "A_2024_00080",
    ("연립/다세대", "전세"): "A_2024_00085",
    ("연립/다세대", "월세"): "A_2024_00090",
    ("단독주택", "매매"): "A_2024_00114",
    ("단독주택", "전세"): "A_2024_00119",
    ("단독주택", "월세"): "A_2024_00124",
    ("오피스텔", "매매"): "A_2024_00615",
    ("오피스텔", "전세"): "A_2024_00618",
    ("오피스텔", "월세"): "A_2024_00621",
}

SUPPLY_TABLES = {
    ("아파트", "매매"): "A_2024_00076",
    ("아파트", "전세"): "A_2024_00077",
    ("아파트", "월세"): "A_2024_00078",
    ("주택종합", "매매"): "A_2024_00041",
    ("주택종합", "전세"): "A_2024_00042",
    ("주택종합", "월세"): "A_2024_00043",
}

# 금액 수준과 전세 위험 맥락에 유용한 보조 통계.
AUXILIARY_TABLES = {
    "A_2024_00060",  # 아파트 평균매매가격
    "A_2024_00064",  # 아파트 평균전세가격
    "A_2024_00069",  # 아파트 평균월세가격
    "A_2024_00072",  # 아파트 평균 매매가격 대비 전세가격
}

CURATED_TABLE_IDS = set(PRICE_INDEX_TABLES.values()) | set(SUPPLY_TABLES.values()) | AUXILIARY_TABLES


def _session() -> requests.Session:
    retry = Retry(
        total=3, connect=3, read=3, backoff_factor=0.35,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8))
    return session


def _rows(payload: dict, service: str) -> tuple[list[dict], int]:
    blocks = payload.get(service) or []
    if not isinstance(blocks, list) or len(blocks) < 2:
        return [], 0
    head = blocks[0].get("head") or []
    total = 0
    if head and isinstance(head, list):
        total = int((head[0] or {}).get("list_total_count") or 0)
        result = (head[-1] or {}).get("RESULT") or {}
        code = str(result.get("CODE") or "")
        if code and code != "INFO-000":
            raise RuntimeError(f"R-ONE 응답 오류: {code} {result.get('MESSAGE') or ''}".strip())
    rows = blocks[1].get("row") or []
    return (rows if isinstance(rows, list) else [rows]), total


class RoneClient:
    def __init__(self, api_key: str | None = None, timeout: int = 30):
        self.api_key = (api_key if api_key is not None else config.RONE_API_KEY).strip()
        self.timeout = timeout
        self.session = _session()

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _get(self, service: str, **params: Any) -> tuple[list[dict], int]:
        if not self.api_key:
            raise RuntimeError("RONE_API_KEY가 설정되지 않았습니다.")
        query = {
            "KEY": self.api_key, "Type": "json", "pIndex": 1,
            "pSize": int(params.pop("pSize", 1000)), **params,
        }
        response = self.session.get(
            f"{config.RONE_BASE_URL}/{service}.do", params=query,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return _rows(response.json(), service)

    def tables(self) -> list[dict]:
        rows, total = self._get("SttsApiTbl", pSize=1000)
        if total > len(rows):
            raise RuntimeError(f"R-ONE 통계표 목록이 잘렸습니다: {len(rows)}/{total}")
        return rows

    def table_data(self, statbl_id: str, cycle: str, period_id: str) -> list[dict]:
        rows, total = self._get(
            "SttsApiTblData", pSize=1000, STATBL_ID=statbl_id,
            DTACYCLE_CD=cycle, WRTTIME_IDTFR_ID=period_id,
        )
        if total > len(rows):
            # 한 월·통계표가 1000건을 넘는 경우 페이지를 끝까지 읽는다.
            all_rows = list(rows)
            for page in range(2, (total + 999) // 1000 + 1):
                query = {
                    "KEY": self.api_key, "Type": "json", "pIndex": page,
                    "pSize": 1000, "STATBL_ID": statbl_id,
                    "DTACYCLE_CD": cycle, "WRTTIME_IDTFR_ID": period_id,
                }
                response = self.session.get(
                    f"{config.RONE_BASE_URL}/SttsApiTblData.do",
                    params=query, timeout=self.timeout,
                )
                response.raise_for_status()
                part, _ = _rows(response.json(), "SttsApiTblData")
                all_rows.extend(part)
            return all_rows
        return rows


def month_ids(end_ym: str | None = None, months: int = 30) -> list[str]:
    if end_ym:
        year, month = int(end_ym[:4]), int(end_ym[4:6])
    else:
        today = date.today()
        year, month = today.year, today.month
    result: list[str] = []
    for offset in range(max(1, months)):
        ordinal = year * 12 + month - 1 - offset
        result.append(f"{ordinal // 12:04d}{ordinal % 12 + 1:02d}")
    return list(reversed(result))


def _number(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def sync_rone_market_data(
    db_path: Path = config.DB_PATH, months: int = 30,
    end_ym: str | None = None, max_workers: int = 6,
    client: RoneClient | None = None,
) -> dict:
    """모든 메타데이터와 최근 핵심 시장통계를 증분 upsert한다."""
    ensure_feed_schema(db_path)
    client = client or RoneClient()
    fetched_at = utc_now()
    tables = client.tables()
    table_by_id = {str(row.get("STATBL_ID")): row for row in tables}
    table_sql = (
        "INSERT OR REPLACE INTO rone_stat_tables(" + ",".join((
            "statbl_id", "statbl_name", "data_cycle_code", "data_cycle_name",
            "data_start_year", "data_end_year", "representative_unit",
            "top_org_name", "open_state", "source_url", "fetched_at", "raw_json",
        )) + ") VALUES(" + ",".join("?" for _ in range(12)) + ")"
    )
    with sqlite3.connect(db_path) as conn:
        conn.executemany(table_sql, [(
            str(row.get("STATBL_ID") or ""), str(row.get("STATBL_NM") or ""),
            row.get("DTACYCLE_CD"), row.get("DTACYCLE_NM"), row.get("DATA_START_YY"),
            row.get("DATA_END_YY"), row.get("RPSTUI_NM"), row.get("TOP_ORG_NM"),
            row.get("OPEN_STATE"), RONE_SOURCE_URL, fetched_at,
            json.dumps(row, ensure_ascii=False, sort_keys=True),
        ) for row in tables if row.get("STATBL_ID")])

    tasks = [
        (table_id, table_by_id.get(table_id, {}).get("DTACYCLE_CD") or "MM", ym)
        for table_id in sorted(CURATED_TABLE_IDS) for ym in month_ids(end_ym, months)
    ]
    results: list[tuple[str, str, list[dict]]] = []
    errors: list[str] = []

    def fetch(task: tuple[str, str, str]):
        table_id, cycle, period = task
        # R-ONE 호출을 짧게 분산시켜 공급자에 순간 부하가 몰리지 않게 한다.
        time.sleep(0.015)
        return table_id, period, client.table_data(table_id, cycle, period)

    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, 8))) as executor:
        future_map = {executor.submit(fetch, task): task for task in tasks}
        for future in as_completed(future_map):
            try:
                results.append(future.result())
            except Exception as exc:  # retain partial success and report only redacted task IDs
                table_id, _, period = future_map[future]
                errors.append(f"{table_id}/{period}: {type(exc).__name__}")

    values: list[tuple] = []
    for table_id, period, rows in results:
        table = table_by_id.get(table_id, {})
        table_name = str(table.get("STATBL_NM") or table_id)
        cycle = str(table.get("DTACYCLE_CD") or "MM")
        for row in rows:
            stable = "|".join(str(row.get(key) or "") for key in (
                "STATBL_ID", "WRTTIME_IDTFR_ID", "GRP_ID", "CLS_ID", "ITM_ID",
            ))
            oid = "RONE_" + hashlib.sha256(stable.encode()).hexdigest()[:32]
            values.append((
                oid, table_id, table_name, cycle,
                str(row.get("WRTTIME_IDTFR_ID") or period), row.get("WRTTIME_DESC"),
                row.get("GRP_ID"), row.get("GRP_NM"), row.get("CLS_ID"),
                row.get("CLS_NM"), row.get("CLS_FULLNM"), row.get("ITM_ID"),
                row.get("ITM_NM"), row.get("ITM_FULLNM"), _number(row.get("DTA_VAL")),
                row.get("UI_NM"), fetched_at, RONE_SOURCE_URL,
            ))
    sql = (
        "INSERT OR REPLACE INTO rone_stat_observations(" + ",".join((
            "observation_id", "statbl_id", "statbl_name", "data_cycle_code",
            "period_id", "period_description", "group_id", "group_name", "class_id",
            "class_name", "class_full_name", "item_id", "item_name", "item_full_name",
            "value", "unit_name", "fetched_at", "source_url",
        )) + ") VALUES(" + ",".join("?" for _ in range(18)) + ")"
    )
    with sqlite3.connect(db_path) as conn:
        conn.executemany(sql, values)
    return {
        "metadata_tables": len(tables), "curated_tables": len(CURATED_TABLE_IDS),
        "periods_requested": len(month_ids(end_ym, months)),
        "observations_written": len(values), "errors": errors[:30],
        "fetched_at": fetched_at,
    }


def _house_group(value: str) -> str:
    if value in {"다세대주택", "연립주택", "연립/다세대"}:
        return "연립/다세대"
    if value in {"단독주택", "다가구주택", "단독/다가구"}:
        return "단독주택"
    if value in {"아파트", "오피스텔"}:
        return value
    return "주택종합"


def _region_score(full_name: str, sido: str, gugun: str) -> int:
    tokens = [token.strip() for token in str(full_name or "").split(">")]
    score = 0
    if gugun and gugun in tokens:
        score += 10
    elif gugun and any(gugun in token for token in tokens):
        score += 6
    aliases = {"서울": "서울", "경기": "경기", "인천": "인천", "대전": "대전",
               "부산": "부산", "대구": "대구", "광주": "광주", "울산": "울산",
               "세종": "세종", "강원": "강원", "충북": "충북", "충남": "충남",
               "전북": "전북", "전남": "전남", "경북": "경북", "경남": "경남", "제주": "제주"}
    short = next((alias for alias in aliases if str(sido).startswith(alias)), str(sido))
    if short and any(token.startswith(short) for token in tokens):
        score += 3
    if tokens and tokens[-1] == "전국":
        score += 1
    return score


class RoneMarketTool:
    def __init__(self, db_path: Path = config.DB_PATH):
        self.db_path = db_path

    def _series(self, table_id: str, sido: str, gugun: str, limit: int = 24) -> dict:
        ensure_feed_schema(self.db_path)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            table = conn.execute(
                "SELECT * FROM rone_stat_tables WHERE statbl_id=?", (table_id,)
            ).fetchone()
            rows = [dict(row) for row in conn.execute(
                "SELECT period_id,period_description,class_name,class_full_name,item_name,"
                "value,unit_name FROM rone_stat_observations WHERE statbl_id=? "
                "AND value IS NOT NULL ORDER BY period_id DESC", (table_id,)
            )]
        if not rows:
            return {"available": False, "table_id": table_id, "series": []}
        by_period: dict[str, list[dict]] = {}
        for row in rows:
            by_period.setdefault(str(row["period_id"]), []).append(row)
        series: list[dict] = []
        match_name = ""
        match_score = 0
        for period in sorted(by_period, reverse=True):
            candidates = sorted(
                by_period[period],
                key=lambda row: _region_score(row.get("class_full_name"), sido, gugun),
                reverse=True,
            )
            row = candidates[0]
            score = _region_score(row.get("class_full_name"), sido, gugun)
            if score <= 0:
                continue
            match_name, match_score = str(row.get("class_full_name") or row.get("class_name") or ""), score
            series.append({
                "period": period, "period_label": row.get("period_description") or period,
                "value": round(float(row["value"]), 4), "unit": row.get("unit_name"),
            })
            if len(series) >= limit:
                break
        series.reverse()
        if not series:
            return {"available": False, "table_id": table_id, "series": []}
        latest = series[-1]["value"]
        prior = series[-2]["value"] if len(series) > 1 else None
        return {
            "available": True, "table_id": table_id,
            "table_name": table["statbl_name"] if table else table_id,
            "representative_unit": table["representative_unit"] if table else None,
            "region_match": match_name, "region_match_score": match_score,
            "latest_value": latest, "latest_period": series[-1]["period_label"],
            "change_1m": round(latest - prior, 4) if prior is not None else None,
            "series": series, "source": "한국부동산원 R-ONE Open API",
            "source_url": RONE_SOURCE_URL,
        }

    def market(self, prop: dict) -> dict:
        house = _house_group(str(prop.get("house_type") or ""))
        transaction = str(prop.get("transaction_type") or "매매")
        if transaction not in {"매매", "전세", "월세"}:
            transaction = "매매"
        sido, gugun = str(prop.get("sido") or ""), str(prop.get("gugun") or "")
        price_table = PRICE_INDEX_TABLES.get((house, transaction)) or PRICE_INDEX_TABLES[("주택종합", transaction)]
        supply_table = SUPPLY_TABLES.get((house, transaction)) or SUPPLY_TABLES.get(("주택종합", transaction))
        price = self._series(price_table, sido, gugun)
        supply = self._series(supply_table, sido, gugun) if supply_table else {"available": False, "series": []}
        available = bool(price.get("available") or supply.get("available"))
        return {
            "available": available, "house_group": house, "transaction_type": transaction,
            "region": f"{sido} {gugun}".strip(), "price_index": price,
            "supply_demand": supply,
            "interpretation": (
                "지수는 기준시점 대비 상대 변화이며 개별 매물 가격이 아닙니다. "
                "수급지수는 해당 통계표 정의에 따라 해석해야 합니다."
            ),
            "source": "한국부동산원 R-ONE Open API", "source_url": RONE_SOURCE_URL,
        }
