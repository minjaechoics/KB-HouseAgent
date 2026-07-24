"""Download MOLIT apartment pre-sale right transaction data to CSV.

API:
    https://apis.data.go.kr/1613000/RTMSDataSvcSilvTrade/getRTMSDataSvcSilvTrade

Example:
    $env:MOLIT_SERVICE_KEY = "..."
    py -3 scripts/download_rtms_silv_trade.py --start 202407
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "downloaded" / "real_estate"
DEFAULT_OUT = OUT_DIR / "rtms_silv_trade.csv"
DEFAULT_META = OUT_DIR / "rtms_silv_trade_download_meta.json"

ENDPOINT = "https://apis.data.go.kr/1613000/RTMSDataSvcSilvTrade/getRTMSDataSvcSilvTrade"
SERVICE_START_YM = "202407"

SEOUL_LAWD_CODES = [
    "11110", "11140", "11170", "11200", "11215",
    "11230", "11260", "11290", "11305", "11320",
    "11350", "11380", "11410", "11440", "11470",
    "11500", "11530", "11545", "11560", "11590",
    "11620", "11650", "11680", "11710", "11740",
]

EXPECTED_COLUMNS = [
    "lawd_cd", "deal_ym", "aptNm", "buyerGbn", "cdealDay", "cdealType",
    "dealAmount", "dealDay", "dealMonth", "dealYear", "dealingGbn",
    "estateAgentSggNm", "excluUseAr", "floor", "jibun", "ownershipGbn",
    "sggCd", "sggNm", "slerGbn", "umdNm",
]


def complete_month() -> str:
    today = date.today()
    year, month = today.year, today.month - 1
    if month == 0:
        year -= 1
        month = 12
    return f"{year:04d}{month:02d}"


def month_range(start: str, end: str) -> list[str]:
    y, m = int(start[:4]), int(start[4:])
    ey, em = int(end[:4]), int(end[4:])
    values: list[str] = []
    while (y, m) <= (ey, em):
        ym = f"{y:04d}{m:02d}"
        if ym >= SERVICE_START_YM:
            values.append(ym)
        m += 1
        if m == 13:
            y += 1
            m = 1
    return values


def text_of(parent: ET.Element | None, tag: str) -> str | None:
    if parent is None:
        return None
    found = parent.find(tag)
    return found.text.strip() if found is not None and found.text is not None else None


def parse_xml(xml_text: str) -> tuple[list[dict], dict]:
    root = ET.fromstring(xml_text)
    header = root.find(".//header")
    info = {
        "resultCode": text_of(header, "resultCode"),
        "resultMsg": text_of(header, "resultMsg"),
        "totalCount": text_of(root, ".//totalCount"),
    }
    rows: list[dict] = []
    for item in root.findall(".//item"):
        row = {}
        for child in list(item):
            row[child.tag] = child.text.strip() if child.text else ""
        rows.append(row)
    return rows, info


def safe_url(url: str) -> str:
    for sep in ("serviceKey=", "ServiceKey="):
        if sep in url:
            return url.split(sep, 1)[0] + sep + "***"
    return url


def fetch_task(
    session: requests.Session,
    service_key: str,
    lawd_cd: str,
    deal_ym: str,
    num_rows: int,
    timeout: int,
    max_retries: int,
    retry_sleep_sec: float,
) -> tuple[list[dict], list[dict]]:
    params = {
        "serviceKey": service_key,
        "LAWD_CD": lawd_cd,
        "DEAL_YMD": deal_ym,
        "pageNo": "1",
        "numOfRows": str(num_rows),
    }
    # This service can return HTTP 403 with custom User-Agent/Accept headers.
    # Keep the request shape close to data.go.kr's reference examples.
    for attempt in range(max_retries + 1):
        resp = session.get(ENDPOINT, params=params, timeout=timeout)
        if resp.status_code != 429 or attempt == max_retries:
            break
        time.sleep(retry_sleep_sec * (attempt + 1))
    if resp.status_code != 200:
        return [], [{
            "lawd_cd": lawd_cd, "deal_ym": deal_ym, "page_no": 1,
            "http_status": resp.status_code, "resultCode": None, "resultMsg": None,
            "totalCount": None, "rows": 0, "url_without_key": safe_url(resp.url),
            "error_body": resp.text[:1000],
        }]

    rows, info = parse_xml(resp.text)
    total = int(info.get("totalCount") or 0)
    events = [{
        "lawd_cd": lawd_cd, "deal_ym": deal_ym, "page_no": 1,
        "http_status": resp.status_code, "resultCode": info.get("resultCode"),
        "resultMsg": info.get("resultMsg"), "totalCount": info.get("totalCount"),
        "rows": len(rows), "url_without_key": safe_url(resp.url), "error_body": None,
    }]
    all_rows = rows

    page = 2
    while (page - 1) * num_rows < total:
        params["pageNo"] = str(page)
        for attempt in range(max_retries + 1):
            resp = session.get(ENDPOINT, params=params, timeout=timeout)
            if resp.status_code != 429 or attempt == max_retries:
                break
            time.sleep(retry_sleep_sec * (attempt + 1))
        if resp.status_code != 200:
            events.append({
                "lawd_cd": lawd_cd, "deal_ym": deal_ym, "page_no": page,
                "http_status": resp.status_code, "resultCode": None, "resultMsg": None,
                "totalCount": str(total), "rows": 0, "url_without_key": safe_url(resp.url),
                "error_body": resp.text[:1000],
            })
            break
        page_rows, page_info = parse_xml(resp.text)
        events.append({
            "lawd_cd": lawd_cd, "deal_ym": deal_ym, "page_no": page,
            "http_status": resp.status_code, "resultCode": page_info.get("resultCode"),
            "resultMsg": page_info.get("resultMsg"), "totalCount": page_info.get("totalCount"),
            "rows": len(page_rows), "url_without_key": safe_url(resp.url), "error_body": None,
        })
        all_rows.extend(page_rows)
        page += 1

    for row in all_rows:
        row["lawd_cd"] = lawd_cd
        row["deal_ym"] = deal_ym
    return all_rows, events


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-key", default=os.environ.get("MOLIT_SERVICE_KEY", ""))
    parser.add_argument("--start", default=SERVICE_START_YM)
    parser.add_argument("--end", default=complete_month())
    parser.add_argument("--include-current-month", action="store_true")
    parser.add_argument("--lawd-cd", nargs="*", default=SEOUL_LAWD_CODES)
    parser.add_argument("--months", nargs="*", default=None, help="Explicit YYYYMM values. Overrides --start/--end.")
    parser.add_argument("--num-rows", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-sleep-sec", type=float, default=1.0)
    parser.add_argument("--retry-failed-meta", type=Path, default=None)
    parser.add_argument("--merge-existing", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    args = parser.parse_args()

    if not args.service_key:
        raise SystemExit("MOLIT_SERVICE_KEY is required or pass --service-key.")

    end = date.today().strftime("%Y%m") if args.include_current_month else args.end
    months = args.months or month_range(args.start, end)
    months = [ym for ym in months if ym >= SERVICE_START_YM]
    if not months:
        raise SystemExit(f"No months to download. {ENDPOINT} starts at {SERVICE_START_YM}.")

    if args.retry_failed_meta:
        previous_meta = json.loads(args.retry_failed_meta.read_text(encoding="utf-8"))
        tasks = sorted({
            (str(e["lawd_cd"]), str(e["deal_ym"]))
            for e in previous_meta.get("failures", [])
            if e.get("lawd_cd") and e.get("deal_ym")
        })
        months = sorted({ym for _, ym in tasks})
    else:
        tasks = [(lawd_cd, ym) for lawd_cd in args.lawd_cd for ym in months]
    rows: list[dict] = []
    events: list[dict] = []
    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(
                    fetch_task,
                    session,
                    args.service_key,
                    lawd_cd,
                    ym,
                    args.num_rows,
                    args.timeout,
                    args.max_retries,
                    args.retry_sleep_sec,
                ): (lawd_cd, ym)
                for lawd_cd, ym in tasks
            }
            done = 0
            for fut in as_completed(future_map):
                task_rows, task_events = fut.result()
                rows.extend(task_rows)
                events.extend(task_events)
                done += 1
                if done % 50 == 0 or done == len(tasks):
                    print(f"[rtms-silv-trade] {done}/{len(tasks)} tasks complete")

    df = pd.DataFrame(rows)
    if args.merge_existing and args.out.exists():
        existing = pd.read_csv(args.out, encoding="utf-8-sig")
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates()
    if df.empty:
        df = pd.DataFrame(columns=EXPECTED_COLUMNS)
    else:
        leading = [c for c in EXPECTED_COLUMNS if c in df.columns]
        rest = [c for c in df.columns if c not in leading]
        df = df[leading + rest]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    events = sorted(events, key=lambda e: (e["deal_ym"], e["lawd_cd"], e["page_no"]))
    failures = [e for e in events if e["http_status"] != 200 or e.get("resultCode") not in (None, "000")]
    meta = {
        "endpoint": ENDPOINT,
        "out": str(args.out),
        "rows": int(len(df)),
        "lawd_codes": list(args.lawd_cd),
        "months": list(months),
        "events": events,
        "failures": failures,
        "updated_at_epoch": time.time(),
    }
    args.meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[rtms-silv-trade] rows={len(df)} out={args.out}")
    print(f"[rtms-silv-trade] meta={args.meta}")
    print(f"[rtms-silv-trade] failures={len(failures)}")
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
