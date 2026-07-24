"""Download Seoul single/multi-family RTMS trade and rent data concurrently.

This script is intentionally focused on the APIs used for the current project:

* RTMSDataSvcSHTrade: single/multi-family sale transactions
* RTMSDataSvcSHRent: single/multi-family rent transactions

It writes successful CSVs to data/downloaded/real_estate.
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
SERVICE_START_YM = "202407"

SEOUL_LAWD_NAMES = {
    "11110": "종로구", "11140": "중구", "11170": "용산구", "11200": "성동구",
    "11215": "광진구", "11230": "동대문구", "11260": "중랑구", "11290": "성북구",
    "11305": "강북구", "11320": "도봉구", "11350": "노원구", "11380": "은평구",
    "11410": "서대문구", "11440": "마포구", "11470": "양천구", "11500": "강서구",
    "11530": "구로구", "11545": "금천구", "11560": "영등포구", "11590": "동작구",
    "11620": "관악구", "11650": "서초구", "11680": "강남구", "11710": "송파구",
    "11740": "강동구",
}

SERVICES = {
    "trade": {
        "endpoint": "https://apis.data.go.kr/1613000/RTMSDataSvcSHTrade/getRTMSDataSvcSHTrade",
        "out": OUT_DIR / "rtms_sh_trade.csv",
        "meta": OUT_DIR / "rtms_sh_trade_download_meta.json",
        "columns": [
            "lawd_cd", "deal_ym", "buildYear", "buyerGbn", "cdealDay", "cdealType",
            "dealAmount", "dealDay", "dealMonth", "dealYear", "dealingGbn",
            "estateAgentSggNm", "houseType", "jibun", "plottageAr", "sggCd",
            "sggNm", "slerGbn", "totalFloorAr", "umdNm",
        ],
    },
    "rent": {
        "endpoint": "https://apis.data.go.kr/1613000/RTMSDataSvcSHRent/getRTMSDataSvcSHRent",
        "out": OUT_DIR / "rtms_sh_rent.csv",
        "meta": OUT_DIR / "rtms_sh_rent_download_meta.json",
        "columns": [
            "lawd_cd", "deal_ym", "buildYear", "contractTerm", "contractType",
            "dealDay", "dealMonth", "dealYear", "deposit", "houseType",
            "monthlyRent", "preDeposit", "preMonthlyRent", "sggCd", "sggNm",
            "umdNm", "useRRRight", "totalFloorAr",
        ],
    },
}


def month_range(start: str, end: str) -> list[str]:
    y, m = int(start[:4]), int(start[4:])
    ey, em = int(end[:4]), int(end[4:])
    out = []
    while (y, m) <= (ey, em):
        ym = f"{y:04d}{m:02d}"
        if ym >= SERVICE_START_YM:
            out.append(ym)
        m += 1
        if m == 13:
            y += 1
            m = 1
    return out


def complete_month() -> str:
    today = date.today()
    y, m = today.year, today.month - 1
    if m == 0:
        y -= 1
        m = 12
    return f"{y:04d}{m:02d}"


def text_of(parent: ET.Element | None, tag: str) -> str | None:
    if parent is None:
        return None
    found = parent.find(tag)
    return found.text.strip() if found is not None and found.text is not None else None


def parse_xml(xml_text: str) -> tuple[list[dict], dict]:
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


def safe_url(url: str) -> str:
    for sep in ("serviceKey=", "ServiceKey="):
        if sep in url:
            return url.split(sep, 1)[0] + sep + "***"
    return url


def fetch_task(session: requests.Session, service: str, lawd_cd: str, deal_ym: str, key: str, num_rows: int, timeout: int):
    spec = SERVICES[service]
    params = {
        "serviceKey": key,
        "LAWD_CD": lawd_cd,
        "DEAL_YMD": deal_ym,
        "pageNo": "1",
        "numOfRows": str(num_rows),
    }
    resp = session.get(spec["endpoint"], params=params, timeout=timeout)
    first_url = resp.url
    if resp.status_code != 200:
        return [], [{
            "service": service, "lawd_cd": lawd_cd, "deal_ym": deal_ym, "page_no": 1,
            "http_status": resp.status_code, "resultCode": None, "resultMsg": None,
            "totalCount": None, "rows": 0, "url_without_key": safe_url(first_url),
            "error_body": resp.text[:1000],
        }]

    rows, info = parse_xml(resp.text)
    total = int(info.get("totalCount") or 0)
    events = [{
        "service": service, "lawd_cd": lawd_cd, "deal_ym": deal_ym, "page_no": 1,
        "http_status": resp.status_code, "resultCode": info.get("resultCode"),
        "resultMsg": info.get("resultMsg"), "totalCount": info.get("totalCount"),
        "rows": len(rows), "url_without_key": safe_url(first_url), "error_body": None,
    }]
    all_rows = rows
    page = 2
    while (page - 1) * num_rows < total:
        params["pageNo"] = str(page)
        resp = session.get(spec["endpoint"], params=params, timeout=timeout)
        if resp.status_code != 200:
            events.append({
                "service": service, "lawd_cd": lawd_cd, "deal_ym": deal_ym, "page_no": page,
                "http_status": resp.status_code, "resultCode": None, "resultMsg": None,
                "totalCount": total, "rows": 0, "url_without_key": safe_url(resp.url),
                "error_body": resp.text[:1000],
            })
            break
        page_rows, page_info = parse_xml(resp.text)
        events.append({
            "service": service, "lawd_cd": lawd_cd, "deal_ym": deal_ym, "page_no": page,
            "http_status": resp.status_code, "resultCode": page_info.get("resultCode"),
            "resultMsg": page_info.get("resultMsg"), "totalCount": page_info.get("totalCount"),
            "rows": len(page_rows), "url_without_key": safe_url(resp.url), "error_body": None,
        })
        all_rows.extend(page_rows)
        page += 1

    for row in all_rows:
        row["lawd_cd"] = lawd_cd
        row["deal_ym"] = deal_ym
        row.setdefault("sggNm", SEOUL_LAWD_NAMES.get(lawd_cd, ""))
    return all_rows, events


def write_outputs(service: str, rows: list[dict], events: list[dict], months: list[str], lawd_codes: list[str]) -> None:
    spec = SERVICES[service]
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=spec["columns"])
    else:
        leading = [c for c in spec["columns"] if c in df.columns]
        rest = [c for c in df.columns if c not in leading]
        df = df[leading + rest]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(spec["out"], index=False, encoding="utf-8-sig")
    meta = {
        "endpoint": spec["endpoint"],
        "out": str(spec["out"]),
        "rows": int(len(df)),
        "lawd_codes": lawd_codes,
        "months": months,
        "events": events,
        "updated_at_epoch": time.time(),
    }
    spec["meta"].write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-key", default=os.environ.get("MOLIT_SERVICE_KEY", ""))
    parser.add_argument("--start", default=SERVICE_START_YM)
    parser.add_argument("--end", default=complete_month())
    parser.add_argument("--include-current-month", action="store_true")
    parser.add_argument("--services", nargs="*", choices=sorted(SERVICES), default=["trade", "rent"])
    parser.add_argument("--lawd-cd", nargs="*", default=list(SEOUL_LAWD_NAMES))
    parser.add_argument("--months", nargs="*", default=None, help="Explicit YYYYMM values. Overrides --start/--end.")
    parser.add_argument("--num-rows", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    if not args.service_key:
        raise SystemExit("MOLIT_SERVICE_KEY is required or pass --service-key.")

    end = date.today().strftime("%Y%m") if args.include_current_month else args.end
    months = args.months or month_range(args.start, end)
    months = [ym for ym in months if ym >= SERVICE_START_YM]
    if not months:
        raise SystemExit(f"No months to download. Services start at {SERVICE_START_YM}.")

    tasks = [
        (service, lawd_cd, ym)
        for service in args.services
        for lawd_cd in args.lawd_cd
        for ym in months
    ]
    rows_by_service = {service: [] for service in args.services}
    events_by_service = {service: [] for service in args.services}

    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(
                    fetch_task, session, service, lawd_cd, ym,
                    args.service_key, args.num_rows, args.timeout,
                ): (service, lawd_cd, ym)
                for service, lawd_cd, ym in tasks
            }
            done = 0
            for fut in as_completed(future_map):
                service, lawd_cd, ym = future_map[fut]
                rows, events = fut.result()
                rows_by_service[service].extend(rows)
                events_by_service[service].extend(events)
                done += 1
                if done % 50 == 0 or done == len(tasks):
                    print(f"[rtms-sh-all] {done}/{len(tasks)} tasks complete")

    bad = 0
    for service in args.services:
        events = sorted(events_by_service[service], key=lambda e: (e["deal_ym"], e["lawd_cd"], e["page_no"]))
        rows = rows_by_service[service]
        write_outputs(service, rows, events, months, args.lawd_cd)
        failures = [e for e in events if e["http_status"] != 200 or e.get("resultCode") not in (None, "000")]
        bad += len(failures)
        print(f"[rtms-sh-all] {service}: rows={len(rows)} events={len(events)} failures={len(failures)}")

    return 2 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
