"""Download official nationwide hospital/pharmacy/mart CSVs from LOCALDATA."""
from __future__ import annotations

import argparse
import io
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config

REGIONS = [
    "6110000_ALL", "6260000_ALL", "6270000_ALL", "6280000_ALL",
    "6130000_ALL", "6300000_ALL", "6310000_ALL", "5690000_ALL",
    "6410000_ALL", "6530000_ALL", "6430000_ALL", "6440000_ALL",
    "6540000_ALL", "6470000_ALL", "6480000_ALL", "6500000_ALL",
]
TARGETS = {
    "건강_병원.csv": ("hospital", "https://www.data.go.kr/data/15096293/standard.do"),
    "건강_약국.csv": ("pharmacy", "https://www.data.go.kr/data/15096290/standard.do"),
    "생활_대규모점포.csv": ("mart", "https://www.data.go.kr/data/15114138/standard.do"),
    "식품_일반음식점.csv": ("restaurant", "https://www.data.go.kr/data/15045016/fileData.do"),
    "식품_휴게음식점.csv": ("cafe", "https://www.data.go.kr/data/15154921/openapi.do"),
}
PAGE = "https://file.localdata.go.kr/file/pharmacies/info"
DOWNLOAD = "https://file.localdata.go.kr/file/download-all"
OUT = config.DATA_RAW / "facilities" / "public_facilities.csv"


def _zip_name(info: zipfile.ZipInfo) -> str:
    try:
        return info.filename.encode("cp437").decode("euc-kr")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return info.filename


def _read_csv(raw: bytes) -> pd.DataFrame:
    for encoding in ("cp949", "euc-kr", "utf-8-sig"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise UnicodeError("공식 시설 CSV 인코딩을 판별하지 못했습니다.")


def _active(frame: pd.DataFrame) -> pd.DataFrame:
    state = frame.get("영업상태명", pd.Series("", index=frame.index)).fillna("").astype(str)
    detail = frame.get("상세영업상태명", pd.Series("", index=frame.index)).fillna("").astype(str)
    bad = state.str.contains("폐업|취소|말소", regex=True) | detail.str.contains("폐업|취소|말소", regex=True)
    return frame[~bad].copy()


def _normalize(frame: pd.DataFrame, category: str, source_url: str,
               transformer: Transformer) -> pd.DataFrame:
    frame = _active(frame)
    x = pd.to_numeric(frame.get("좌표정보(X)"), errors="coerce")
    y = pd.to_numeric(frame.get("좌표정보(Y)"), errors="coerce")
    valid = x.notna() & y.notna()
    frame, x, y = frame.loc[valid].copy(), x.loc[valid], y.loc[valid]
    if frame.empty:
        return pd.DataFrame()
    lng, lat = transformer.transform(x.to_numpy(), y.to_numpy())
    address = frame.get("도로명주소", pd.Series("", index=frame.index)).fillna("").astype(str)
    lot = frame.get("지번주소", pd.Series("", index=frame.index)).fillna("").astype(str)
    address = address.where(address.str.strip().ne(""), lot)
    subcategory = frame.get(
        "점포구분명", frame.get(
            "의료기관종별명", frame.get(
                "업태구분명", pd.Series("", index=frame.index)))
    ).fillna("").astype(str)
    if category == "cafe":
        names = frame.get("사업장명", pd.Series("", index=frame.index)).fillna("").astype(str)
        cafe_mask = (subcategory.str.contains("커피|카페|다방|제과", regex=True)
                     | names.str.contains("커피|카페|CAFE|COFFEE", case=False, regex=True))
        frame, address, subcategory, lat, lng = (
            frame.loc[cafe_mask], address.loc[cafe_mask], subcategory.loc[cafe_mask],
            lat[cafe_mask.to_numpy()], lng[cafe_mask.to_numpy()])
    out = pd.DataFrame({
        "category": category,
        "name": frame.get("사업장명", pd.Series("", index=frame.index)).fillna("").astype(str),
        "subcategory": subcategory,
        "address": address,
        "lat": lat,
        "lng": lng,
        "source_url": source_url,
        "data_updated_at": frame.get("최종수정시점", pd.Series("", index=frame.index)).fillna("").astype(str),
    })
    return out[out["lat"].between(32.0, 39.5) & out["lng"].between(123.0, 133.5)]


def _download_region(session: requests.Session, region: str) -> bytes:
    session.get("https://file.localdata.go.kr/file/validate/download-count", timeout=30).raise_for_status()
    response = session.get(DOWNLOAD, params={"orgCode": region}, timeout=240)
    response.raise_for_status()
    if "zip" not in response.headers.get("content-type", "").lower():
        raise RuntimeError(f"{region}: ZIP 응답이 아닙니다.")
    return response.content


def run(output: Path = OUT, regions: list[str] | None = None) -> pd.DataFrame:
    selected_regions = regions or REGIONS
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; DdolddolhanChoi/1.0; public-data-cache)",
        "Referer": "https://www.data.go.kr/",
    })
    session.get(PAGE, timeout=30).raise_for_status()
    session.headers["Referer"] = PAGE
    transformer = Transformer.from_crs("EPSG:5174", "EPSG:4326", always_xy=True)
    rows: list[pd.DataFrame] = []
    for index, region in enumerate(selected_regions, start=1):
        for attempt in range(3):
            try:
                raw_zip = _download_region(session, region)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(3 * (attempt + 1))
        with zipfile.ZipFile(io.BytesIO(raw_zip)) as archive:
            for info in archive.infolist():
                name = _zip_name(info)
                if name not in TARGETS:
                    continue
                category, source_url = TARGETS[name]
                rows.append(_normalize(_read_csv(archive.read(info)), category,
                                       source_url, transformer))
        print(f"[{index}/{len(selected_regions)}] {region}: selected official CSVs")
        time.sleep(0.35)
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    result = result.drop_duplicates(subset=["category", "name", "address", "lat", "lng"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".tmp.csv")
    result.to_csv(temp, index=False, encoding="utf-8-sig")
    temp.replace(output)
    print(result.groupby("category").size().to_string())
    print(f"saved: {output} ({len(result):,} rows)")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--region", action="append", dest="regions")
    args = parser.parse_args()
    run(args.output, args.regions)
