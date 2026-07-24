"""Download MOLIT single/multi-family rent transaction data to CSV.

API:
    https://apis.data.go.kr/1613000/RTMSDataSvcSHRent/getRTMSDataSvcSHRent

The API requires a data.go.kr service key with usage approval for
"국토교통부_단독/다가구 전월세 실거래가 자료".

Example:
    $env:MOLIT_SERVICE_KEY = "..."
    py -3 scripts/download_rtms_sh_rent.py --months 202401 --num-rows 100
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "downloaded" / "real_estate"
DEFAULT_OUT = OUT_DIR / "rtms_sh_rent.csv"
DEFAULT_META = OUT_DIR / "rtms_sh_rent_download_meta.json"

ENDPOINT = "https://apis.data.go.kr/1613000/RTMSDataSvcSHRent/getRTMSDataSvcSHRent"
SERVICE_START_YM = "202407"

SEOUL_LAWD_CODES = [
    "11110", "11140", "11170", "11200", "11215",
    "11230", "11260", "11290", "11305", "11320",
    "11350", "11380", "11410", "11440", "11470",
    "11500", "11530", "11545", "11560", "11590",
    "11620", "11650", "11680", "11710", "11740",
]

SEOUL_LAWD_NAMES = {
    "11110": "종로구", "11140": "중구", "11170": "용산구", "11200": "성동구",
    "11215": "광진구", "11230": "동대문구", "11260": "중랑구", "11290": "성북구",
    "11305": "강북구", "11320": "도봉구", "11350": "노원구", "11380": "은평구",
    "11410": "서대문구", "11440": "마포구", "11470": "양천구", "11500": "강서구",
    "11530": "구로구", "11545": "금천구", "11560": "영등포구", "11590": "동작구",
    "11620": "관악구", "11650": "서초구", "11680": "강남구", "11710": "송파구",
    "11740": "강동구",
}

EXPECTED_COLUMNS = [
    "lawd_cd", "deal_ym", "buildYear", "contractTerm", "contractType",
    "dealDay", "dealMonth", "dealYear", "deposit", "houseType", "monthlyRent",
    "preDeposit", "preMonthlyRent", "sggCd", "sggNm", "umdNm", "useRRRight",
]


def complete_months_back(n: int, today: date | None = None) -> list[str]:
    today = today or date.today()
    year, month = today.year, today.month - 1
    if month == 0:
        year -= 1
        month = 12
    values = []
    for _ in range(n):
        values.append(f"{year:04d}{month:02d}")
        month -= 1
        if month == 0:
            year -= 1
            month = 12
    return [ym for ym in reversed(values) if ym >= SERVICE_START_YM]


def text_of(parent: ET.Element | None, tag: str) -> str | None:
    if parent is None:
        return None
    found = parent.find(tag)
    return found.text.strip() if found is not None and found.text is not None else None


def parse_items(xml_text: str) -> tuple[list[dict], dict]:
    root = ET.fromstring(xml_text)
    header = root.find(".//header")
    result = {
        "resultCode": text_of(header, "resultCode"),
        "resultMsg": text_of(header, "resultMsg"),
        "totalCount": text_of(root, ".//totalCount"),
    }
    rows = []
    for item in root.findall(".//item"):
        row = {}
        for child in list(item):
            row[child.tag] = child.text.strip() if child.text else ""
        rows.append(row)
    return rows, result


def fetch_page(
    service_key: str,
    lawd_cd: str,
    deal_ym: str,
    page_no: int,
    num_rows: int,
    timeout: int,
) -> tuple[list[dict], dict, str, int]:
    params = {
        "serviceKey": service_key,
        "LAWD_CD": lawd_cd,
        "DEAL_YMD": deal_ym,
        "pageNo": str(page_no),
        "numOfRows": str(num_rows),
    }
    headers = {
        "Accept": "application/xml,text/xml,*/*",
        "User-Agent": "jeonse-helper/1.0",
    }
    resp = requests.get(ENDPOINT, params=params, headers=headers, timeout=timeout)
    if resp.status_code != 200:
        return [], {"http_status": resp.status_code, "body": resp.text[:1000]}, resp.url, resp.status_code
    rows, result = parse_items(resp.text)
    return rows, result, resp.url, resp.status_code


def download(
    service_key: str,
    lawd_codes: Iterable[str],
    months: Iterable[str],
    num_rows: int,
    sleep_sec: float,
    timeout: int,
) -> tuple[pd.DataFrame, list[dict]]:
    all_rows: list[dict] = []
    events: list[dict] = []

    for lawd_cd in lawd_codes:
        for deal_ym in months:
            page_no = 1
            while True:
                rows, info, url, status = fetch_page(
                    service_key=service_key,
                    lawd_cd=lawd_cd,
                    deal_ym=deal_ym,
                    page_no=page_no,
                    num_rows=num_rows,
                    timeout=timeout,
                )
                events.append({
                    "lawd_cd": lawd_cd,
                    "deal_ym": deal_ym,
                    "page_no": page_no,
                    "http_status": status,
                    "resultCode": info.get("resultCode"),
                    "resultMsg": info.get("resultMsg"),
                    "totalCount": info.get("totalCount"),
                    "rows": len(rows),
                    "url_without_key": url.split("ServiceKey=", 1)[0] + "ServiceKey=***" if "ServiceKey=" in url else url,
                    "error_body": info.get("body"),
                })

                if status != 200:
                    return pd.DataFrame(all_rows), events

                for row in rows:
                    row["lawd_cd"] = lawd_cd
                    row["deal_ym"] = deal_ym
                    row.setdefault("sggNm", SEOUL_LAWD_NAMES.get(str(lawd_cd), ""))
                all_rows.extend(rows)

                total = int(info.get("totalCount") or 0)
                if page_no * num_rows >= total or not rows:
                    break
                page_no += 1
                if sleep_sec:
                    time.sleep(sleep_sec)

            if sleep_sec:
                time.sleep(sleep_sec)

    return pd.DataFrame(all_rows), events


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-key", default=os.environ.get("MOLIT_SERVICE_KEY", ""))
    parser.add_argument("--lawd-cd", nargs="*", default=SEOUL_LAWD_CODES)
    parser.add_argument("--months", nargs="*", default=None, help="YYYYMM values. Overrides --last-months.")
    parser.add_argument("--last-months", type=int, default=12)
    parser.add_argument("--num-rows", type=int, default=100)
    parser.add_argument("--sleep-sec", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    args = parser.parse_args()

    if not args.service_key:
        raise SystemExit("MOLIT_SERVICE_KEY is required or pass --service-key.")

    months = args.months or complete_months_back(args.last_months)
    months = [ym for ym in months if ym >= SERVICE_START_YM]
    if not months:
        raise SystemExit(f"No months to download. {ENDPOINT} starts at {SERVICE_START_YM}.")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    df, events = download(
        service_key=args.service_key,
        lawd_codes=args.lawd_cd,
        months=months,
        num_rows=args.num_rows,
        sleep_sec=args.sleep_sec,
        timeout=args.timeout,
    )

    if df.empty:
        df = pd.DataFrame(columns=EXPECTED_COLUMNS)
    else:
        leading = [c for c in EXPECTED_COLUMNS if c in df.columns]
        rest = [c for c in df.columns if c not in leading]
        df = df[leading + rest]

    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    meta = {
        "endpoint": ENDPOINT,
        "out": str(args.out),
        "rows": int(len(df)),
        "lawd_codes": list(args.lawd_cd),
        "months": list(months),
        "events": events,
    }
    args.meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[rtms-sh-rent] rows={len(df)} out={args.out}")
    print(f"[rtms-sh-rent] meta={args.meta}")
    if events and any(e.get("http_status") != 200 for e in events):
        print("[rtms-sh-rent] stopped because the API returned a non-200 response.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
