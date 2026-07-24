"""Geocode official police-center and fire-station CSVs once and cache them.

The runtime report then performs local radius queries; it does not spend a map
API call per property.  Four concurrent calls are used and progress is saved so
the job can resume safely.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
from src.tools.map_tool import MapTool


SOURCES = {
    "police": {
        "path": config.DATA_RAW / "safety" / "경찰청_전국 치안센터 주소 현황_20251231.csv",
        "name": "치안센터명",
        "address": "주소",
        "source_url": "https://www.data.go.kr/",
    },
    "fire_station": {
        "path": config.DATA_RAW / "safety" / "소방청_시도 소방서 현황_20250701.csv",
        "name": "소방서",
        "address": "주소",
        "source_url": "https://www.data.go.kr/",
    },
}
OUT_DIR = config.DATA_GEN / "safety_geocoded"


def _read(path: Path) -> pd.DataFrame:
    for encoding in ("cp949", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"인코딩을 판별할 수 없습니다: {path}")


def _one(tool: MapTool, name: str, address: str, source_url: str) -> dict:
    result = tool.geocode(address)
    if not result.get("ok") and "(" in address:
        result = tool.geocode(address.split("(", 1)[0].strip())
    return {
        "name": name,
        "address": address,
        "lat": result.get("lat"),
        "lng": result.get("lng"),
        "geocode_source": result.get("source"),
        "source_url": source_url,
    }


def geocode(key: str, workers: int = 4) -> Path:
    spec = SOURCES[key]
    raw = _read(spec["path"])
    rows = raw[[spec["name"], spec["address"]]].dropna().drop_duplicates().rename(
        columns={spec["name"]: "name", spec["address"]: "address"})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / f"{key}.csv"
    existing: dict[str, dict] = {}
    if output.exists():
        cached = _read(output)
        for row in cached.to_dict("records"):
            if pd.notna(row.get("lat")) and pd.notna(row.get("lng")):
                existing[str(row.get("address"))] = row
    pending = [row for row in rows.to_dict("records")
               if str(row["address"]) not in existing]
    tool = MapTool()
    if not tool.online:
        raise RuntimeError("NAVER_MAP_CLIENT_ID/SECRET가 설정되지 않았습니다.")
    completed = dict(existing)
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 4))) as pool:
        futures = {
            pool.submit(_one, tool, str(row["name"]), str(row["address"]), spec["source_url"]): row
            for row in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            item = future.result()
            completed[item["address"]] = item
            if index % 50 == 0:
                pd.DataFrame(completed.values()).to_csv(output, index=False, encoding="utf-8-sig")
                print(f"{key}: {index}/{len(pending)}")
            time.sleep(0.02)
    result = pd.DataFrame(completed.values())
    result = result.dropna(subset=["lat", "lng"]).drop_duplicates("address")
    if result.empty:
        raise RuntimeError(
            f"{key}: Geocoding 응답을 받지 못했습니다. 기존 캐시는 유지합니다.")
    result.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"{key}: saved {len(result):,}/{len(rows):,} -> {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=[*SOURCES, "all"], default="all")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    for selected in (SOURCES if args.type == "all" else [args.type]):
        geocode(selected, args.workers)
