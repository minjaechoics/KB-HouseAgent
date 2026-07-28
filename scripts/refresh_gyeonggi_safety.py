"""Refresh normalized Gyeonggi police/fire coordinate caches."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config


OUTPUT_DIR = config.DATA_GEN / "safety_geocoded"


def _rows(url: str, service: str, key: str, sigun: str) -> list[dict]:
    params = {
        "KEY": key,
        "Type": "json",
        "pIndex": 1,
        "pSize": 1000,
        "SIGUN_NM": sigun,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()[service]
    head = payload[0]["head"]
    code = str(head[1]["RESULT"]["CODE"])
    if code != "INFO-000":
        raise RuntimeError(
            f"{service}: {code} {head[1]['RESULT'].get('MESSAGE', '')}"
        )
    return list(payload[1].get("row", []))


def refresh(key: str, sigun: str = "수원시") -> dict:
    if not key:
        raise RuntimeError("GYEONGGI_OPENAPI_KEY가 비어 있습니다.")
    police_rows = _rows(
        config.GYEONGGI_POLICE_URL,
        "Ptrldvsnsubpolcstus",
        key,
        sigun,
    )
    combined_rows = _rows(
        config.GYEONGGI_SAFETY_FACILITY_URL,
        "FiresttnPolcsttnM",
        key,
        sigun,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    police = pd.DataFrame([{
        "name": row.get("GOVOFC_NM") or row.get("POLCSTTN_NM"),
        "subcategory": row.get("DIV_NM"),
        "address": (
            row.get("REFINE_ROADNM_ADDR")
            or row.get("REFINE_LOTNO_ADDR")
        ),
        "lat": row.get("REFINE_WGS84_LAT"),
        "lng": row.get("REFINE_WGS84_LOGT"),
        "source": "경기도 파출소·지구대 현황 Open API",
        "source_url": config.GYEONGGI_POLICE_URL,
    } for row in police_rows])
    fire = pd.DataFrame([{
        "name": row.get("INST_NM"),
        "subcategory": row.get("FACLT_DIV_NM"),
        "address": (
            row.get("REFINE_ROADNM_ADDR")
            or row.get("REFINE_LOTNO_ADDR")
        ),
        "lat": row.get("REFINE_WGS84_LAT"),
        "lng": row.get("REFINE_WGS84_LOGT"),
        "source": "경기도 소방·경찰 시설 현황 Open API",
        "source_url": config.GYEONGGI_SAFETY_FACILITY_URL,
    } for row in combined_rows if "소방" in str(row.get("FACLT_DIV_NM") or "")])

    outputs = {}
    for name, frame in (("police_gg.csv", police), ("fire_station_gg.csv", fire)):
        frame["lat"] = pd.to_numeric(frame["lat"], errors="coerce")
        frame["lng"] = pd.to_numeric(frame["lng"], errors="coerce")
        frame = frame.dropna(subset=["lat", "lng"]).drop_duplicates(
            subset=["name", "lat", "lng"]
        )
        path = OUTPUT_DIR / name
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        outputs[name] = {"path": str(path), "rows": len(frame)}
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", default=os.environ.get("GYEONGGI_OPENAPI_KEY"))
    parser.add_argument("--sigun", default="수원시")
    args = parser.parse_args()
    print(refresh(args.key or "", args.sigun))


if __name__ == "__main__":
    main()
