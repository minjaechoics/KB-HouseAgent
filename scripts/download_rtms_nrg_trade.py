"""Download MOLIT commercial/office real-estate transaction data to CSV.

API:
    https://apis.data.go.kr/1613000/RTMSDataSvcNrgTrade/getRTMSDataSvcNrgTrade

The API requires a data.go.kr service key with usage approval for
"국토교통부_상업업무용 부동산 매매 실거래가 자료".

Example:
    $env:MOLIT_SERVICE_KEY = "..."
    py -3 scripts/download_rtms_nrg_trade.py --last-months 12
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
DEFAULT_OUT = OUT_DIR / "rtms_nrg_trade.csv"
DEFAULT_META = OUT_DIR / "rtms_nrg_trade_download_meta.json"

ENDPOINT = "https://apis.data.go.kr/1613000/RTMSDataSvcNrgTrade/getRTMSDataSvcNrgTrade"

# Seoul district LAWD_CD values. Add other regions with --lawd-cd as needed.
SEOUL_LAWD_CODES = [
    "11110", "11140", "11170", "11200", "11215",
    "11230", "11260", "11290", "11305", "11320",
    "11350", "11380", "11410", "11440", "11470",
    "11500", "11530", "11545", "11560", "11590",
    "11620", "11650", "11680", "11710", "11740",
]

EXPECTED_COLUMNS = [
    "lawd_cd", "deal_ym", "거래금액", "거래유형", "건물면적", "건물주용도",
    "건축년도", "년", "대지면적", "법정동", "시군구", "용도지역", "월",
    "유형", "일", "중개사소재지", "지번", "지역코드", "해제사유발생일", "해제여부",
]


def complete_months_back(n: int, today: date | None = None) -> list[str]:
    """Return the last n completed YYYYMM values."""
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
    return list(reversed(values))


def text_of(parent: ET.Element, tag: str) -> str | None:
    found = parent.find(tag)
    return found.text.strip() if found is not None and found.text is not None else None


def parse_items(xml_text: str) -> tuple[list[dict], dict]:
    root = ET.fromstring(xml_text)
    header = root.find(".//header")
    result = {
        "resultCode": text_of(header, "resultCode") if header is not None else None,
        "resultMsg": text_of(header, "resultMsg") if header is not None else None,
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
        # Some data.go.kr RTMS endpoints are case-sensitive depending on
        # parameter combinations; the technical docs commonly show ServiceKey.
        "ServiceKey": service_key,
        "LAWD_CD": lawd_cd,
        "DEAL_YMD": deal_ym,
        "pageNo": str(page_no),
        "numOfRows": str(num_rows),
    }
    resp = requests.get(ENDPOINT, params=params, timeout=timeout)
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
                    "url_without_key": url.split("serviceKey=", 1)[0] + "serviceKey=***" if "serviceKey=" in url else url,
                    "error_body": info.get("body"),
                })

                if status != 200:
                    return pd.DataFrame(all_rows), events

                for row in rows:
                    row["lawd_cd"] = lawd_cd
                    row["deal_ym"] = deal_ym
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
    parser.add_argument("--num-rows", type=int, default=1000)
    parser.add_argument("--sleep-sec", type=float, default=0.05)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    args = parser.parse_args()

    if not args.service_key:
        raise SystemExit("MOLIT_SERVICE_KEY is required or pass --service-key.")

    months = args.months or complete_months_back(args.last_months)
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

    print(f"[rtms] rows={len(df)} out={args.out}")
    print(f"[rtms] meta={args.meta}")
    if events and any(e.get("http_status") != 200 for e in events):
        print("[rtms] stopped because the API returned a non-200 response.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
