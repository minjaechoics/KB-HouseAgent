"""Import downloaded MOLIT RTMS transactions and expose local price history."""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from src import config
from src.real_estate_feeds.storage import ensure_feed_schema, utc_now


RTMS_SOURCE_URL = "https://rt.molit.go.kr/"
DATASETS = {
    "rtms_apt_trade.csv": ("아파트", "매매"),
    "rtms_apt_trade_dev.csv": ("아파트", "매매"),
    "rtms_offi_trade.csv": ("오피스텔", "매매"),
    "rtms_offi_rent.csv": ("오피스텔", "임대"),
    "rtms_sh_trade.csv": ("단독/다가구", "매매"),
    "rtms_sh_rent.csv": ("단독/다가구", "임대"),
}


def _num(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return None


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _observation(row: dict, filename: str, house_type: str,
                 transaction_family: str, ingested_at: str) -> tuple:
    ym = int(_num(row.get("deal_ym")) or 0)
    year = int(_num(row.get("dealYear")) or ym // 100)
    month = int(_num(row.get("dealMonth")) or ym % 100)
    day = int(_num(row.get("dealDay")) or 1)
    observed_date = f"{year:04d}-{month:02d}-{max(1, min(day, 31)):02d}"
    deposit = _num(row.get("deposit"))
    monthly = _num(row.get("monthlyRent"))
    transaction = transaction_family
    price = _num(row.get("dealAmount"))
    if transaction_family == "임대":
        transaction = "전세" if not monthly else "월세"
        price = None
    complex_name = _text(row.get("aptNm") or row.get("offiNm"))
    area = _num(row.get("excluUseAr") or row.get("totalFloorAr"))
    lawd = str(int(_num(row.get("lawd_cd") or row.get("sggCd")) or 0)).zfill(5)
    stable = {
        "dataset": filename, "lawd": lawd, "ym": ym, "day": day,
        "complex": complex_name, "dong": _text(row.get("umdNm")),
        "area": area, "floor": _num(row.get("floor")), "price": price,
        "deposit": deposit, "monthly": monthly,
    }
    raw = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(json.dumps(stable, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    cancelled = int(bool(_text(row.get("cdealDay"))))
    return (
        "RTMS_" + digest[:32], "MOLIT_RTMS", filename, RTMS_SOURCE_URL,
        observed_date, ym, lawd, lawd, _text(row.get("sggNm")),
        _text(row.get("umdNm")), complex_name, house_type, transaction,
        price, deposit, monthly, area, int(_num(row.get("floor")) or 0) or None,
        int(_num(row.get("buildYear")) or 0) or None, cancelled, ingested_at,
        # The source CSV remains the immutable raw artifact.  Repeating its full
        # JSON payload for every row would add hundreds of MB to the runtime DB.
        hashlib.sha256(raw.encode()).hexdigest(), None,
    )


def import_rtms_downloads(db_path: Path = config.DB_PATH,
                          source_dir: Path | None = None,
                          chunk_size: int = 20_000,
                          dataset_names: set[str] | None = None) -> dict:
    ensure_feed_schema(db_path)
    source_dir = source_dir or config.DATA_RAW / "real_estate"
    ingested_at = utc_now()
    sql = (
        "INSERT OR REPLACE INTO property_price_observations("
        "observation_id,source,source_dataset,source_url,observed_date,deal_ym,"
        "lawd_cd,legal_dong_code,region_name,dong,complex_name,house_type,"
        "transaction_type,price_manwon,deposit_manwon,monthly_rent_manwon,"
        "area_m2,floor,build_year,cancelled,ingested_at,content_hash,raw_json) "
        "VALUES(" + ",".join("?" for _ in range(23)) + ")"
    )
    total = 0
    details = []
    with sqlite3.connect(db_path) as conn:
        for filename, (house_type, transaction) in DATASETS.items():
            if dataset_names is not None and filename not in dataset_names:
                continue
            path = source_dir / filename
            if not path.exists():
                details.append({"dataset": filename, "status": "missing", "rows": 0})
                continue
            count = 0
            for frame in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
                records = frame.to_dict("records")
                values = [_observation(row, filename, house_type, transaction, ingested_at)
                          for row in records]
                conn.executemany(sql, values)
                if filename == "rtms_apt_trade_dev.csv":
                    detail_rows = [(
                        value[0], _text(row.get("aptSeq")), _text(row.get("aptDong")),
                        _text(row.get("roadNm")), _text(row.get("roadNmBonbun")),
                        _text(row.get("roadNmBubun")), _text(row.get("umdCd")),
                        _text(row.get("bonbun")), _text(row.get("bubun")),
                        _text(row.get("buyerGbn")), _text(row.get("slerGbn")),
                        _text(row.get("estateAgentSggNm")), _text(row.get("rgstDate")),
                        _text(row.get("landLeaseholdGbn")), filename,
                    ) for row, value in zip(records, values)]
                    conn.executemany(
                        "INSERT OR REPLACE INTO rtms_transaction_details VALUES(" +
                        ",".join("?" for _ in range(15)) + ")", detail_rows,
                    )
                count += len(values)
            total += count
            details.append({"dataset": filename, "status": "imported", "rows": count})
    return {"written": total, "datasets": details, "ingested_at": ingested_at}


class RTMSPriceHistoryTool:
    def __init__(self, db_path: Path = config.DB_PATH):
        self.db_path = db_path

    def history(self, prop: dict, months: int = 36) -> dict:
        ensure_feed_schema(self.db_path)
        raw_code = str(prop.get("legal_dong_code") or "")
        house = str(prop.get("house_type") or "")
        transaction = str(prop.get("transaction_type") or prop.get("lease_type") or "")
        if house not in {"아파트", "오피스텔"}:
            house = "단독/다가구"
        if transaction not in {"매매", "전세", "월세"}:
            transaction = "매매"
        code = raw_code.zfill(5)[:5] if raw_code[:5].isdigit() else ""
        if not code:
            # Synthetic candidates deliberately do not reuse an exact source
            # address. Resolve their comparison SGG by public region labels.
            prefix = {
                "서울": "11", "서울특별시": "11", "부산": "26", "부산광역시": "26",
                "대구": "27", "대구광역시": "27", "인천": "28", "인천광역시": "28",
                "광주": "29", "광주광역시": "29", "대전": "30", "대전광역시": "30",
                "울산": "31", "울산광역시": "31", "세종": "36", "세종특별자치시": "36",
                "경기": "41", "경기도": "41", "충북": "43", "충청북도": "43",
                "충남": "44", "충청남도": "44", "전북": "52", "전북특별자치도": "52",
                "전라북도": "45", "전남": "46", "전라남도": "46", "경북": "47",
                "경상북도": "47", "경남": "48", "경상남도": "48", "제주": "50",
                "제주특별자치도": "50", "강원": "51", "강원특별자치도": "51",
            }.get(str(prop.get("sido") or ""), "")
            gugun = str(prop.get("gugun") or "")
            with sqlite3.connect(self.db_path) as resolver:
                candidates = resolver.execute(
                    "SELECT lawd_cd, COUNT(*) n FROM property_price_observations "
                    "WHERE region_name=? AND lawd_cd LIKE ? GROUP BY lawd_cd ORDER BY n DESC",
                    (gugun, prefix + "%"),
                ).fetchall()
                if not candidates and prefix:
                    candidates = resolver.execute(
                        "SELECT lawd_cd, COUNT(*) n FROM property_price_observations "
                        "WHERE lawd_cd LIKE ? GROUP BY lawd_cd ORDER BY n DESC",
                        (prefix + "%",),
                    ).fetchall()
            code = str(candidates[0][0]) if candidates else ""
        if not code:
            return self._unavailable("시군구를 국토교통부 지역 코드에 연결하지 못했습니다.")
        value_column = {
            "매매": "price_manwon", "전세": "deposit_manwon",
            "월세": "monthly_rent_manwon",
        }[transaction]
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT deal_ym, AVG({value_column}) mean_value, "
                f"MIN({value_column}) min_value, MAX({value_column}) max_value, "
                "COUNT(*) sample_count, AVG(deposit_manwon) mean_deposit "
                "FROM property_price_observations WHERE lawd_cd=? AND house_type=? "
                f"AND transaction_type=? AND cancelled=0 AND {value_column} IS NOT NULL "
                "GROUP BY deal_ym ORDER BY deal_ym DESC LIMIT ?",
                (code, house, transaction, max(2, min(months, 120))),
            ).fetchall()
        series = [dict(row) for row in reversed(rows)]
        for row in series:
            ym = int(row.pop("deal_ym"))
            row["date"] = f"{ym // 100:04d}-{ym % 100:02d}-01"
            row["price_manwon"] = round(float(row.pop("mean_value")), 1)
            row["min_manwon"] = round(float(row["min_value"]), 1)
            row["max_manwon"] = round(float(row["max_value"]), 1)
            row["mean_deposit_manwon"] = (
                round(float(row["mean_deposit"]), 1)
                if row["mean_deposit"] is not None else None)
            row.pop("min_value", None); row.pop("max_value", None); row.pop("mean_deposit", None)
        if len(series) < 2:
            return self._unavailable("동일 시군구·주택유형·거래유형의 월별 표본이 부족합니다.")
        latest, prior = series[-1]["price_manwon"], series[-2]["price_manwon"]
        first = series[0]["price_manwon"]
        return {
            "available": True, "source": "국토교통부 실거래가 공개시스템",
            "source_url": RTMS_SOURCE_URL, "match_type": "sgg_house_transaction",
            "lawd_cd": code, "house_type": house, "transaction_type": transaction,
            "value_label": {"매매": "평균 실거래가", "전세": "평균 전세보증금",
                            "월세": "평균 월세"}[transaction],
            "latest_price_manwon": latest, "as_of": series[-1]["date"],
            "change_1m": round(latest / prior - 1, 4) if prior else None,
            "change_period": round(latest / first - 1, 4) if first else None,
            "series": series, "unit": "만원",
            "warning": "단지 호가가 아니라 같은 시군구·주택유형의 신고 실거래 월평균입니다.",
        }

    @staticmethod
    def _unavailable(reason: str) -> dict:
        return {"available": False, "source": "국토교통부 실거래가 공개시스템",
                "source_url": RTMS_SOURCE_URL, "reason": reason, "series": []}
