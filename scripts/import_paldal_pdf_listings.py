"""팔달구 매물 목록을 properties 테이블로 가져온다.

PDF의 오른쪽 주소는 중개사무소 주소다. 매물 주소는 읍면동까지만 제공되므로
지도 좌표는 동 대표점 주변의 결정론적 분석 좌표이며 실제 주소로 표시하지 않는다.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import shutil
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src import config
from src.data_augmentation.property_schema import BROKER_LISTING_COLUMNS
from src.fraud_risk.infer import FraudRiskScorer
from src.real_estate_feeds.storage import PROPERTY_PROVENANCE_COLUMNS, ensure_feed_schema


SOURCE_DATE = "2026-07-25"
SIDO = "경기"
GUGUN = "수원시 팔달구"
HOUSE_TYPES = {"단독", "다가구", "상가주택", "아파트", "빌라", "다세대", "연립", "오피스텔", "원룸"}
TRANSACTIONS = {"매매", "전세", "월세"}
HOUSE_TYPE_MAP = {
    "단독": "단독주택", "다가구": "다가구주택", "상가주택": "상가주택",
    "아파트": "아파트", "빌라": "다세대주택", "다세대": "다세대주택",
    "연립": "연립주택", "오피스텔": "오피스텔", "원룸": "원룸",
}
DONG_CENTERS = {
    "고등동": (37.2732, 127.0103), "교동": (37.2748, 127.0126),
    "구천동": (37.2745, 127.0188), "남수동": (37.2812, 127.0175),
    "매교동": (37.2655, 127.0157), "매산로1가": (37.2669, 127.0018),
    "매산로2가": (37.2678, 127.0060), "매산로3가": (37.2678, 127.0093),
    "매향동": (37.2841, 127.0212), "북수동": (37.2851, 127.0158),
    "신풍동": (37.2826, 127.0113), "영동": (37.2763, 127.0174),
    "우만동": (37.2827, 127.0350), "인계동": (37.2654, 127.0287),
    "장안동": (37.2855, 127.0121), "중동": (37.2759, 127.0161),
    "지동": (37.2791, 127.0245), "팔달로1가": (37.2787, 127.0171),
    "팔달로2가": (37.2768, 127.0163), "팔달로3가": (37.2748, 127.0152),
    "화서동": (37.2834, 126.9991),
}
DEFAULT_CENTER = (37.2784, 127.0162)
FACILITY_ANCHOR_CSV = config.DATA_RAW / "facilities" / "public_facilities.csv"
_ADDRESS_ANCHORS: dict[str, list[dict]] | None = None


def _number(text: object) -> float | None:
    value = re.sub(r"[^0-9.\-]", "", str(text or ""))
    try:
        return float(value) if value not in {"", ".", "-"} else None
    except ValueError:
        return None


def _clean_dong(text: str) -> str:
    text = re.sub(r"\s+", "", text or "")
    return text.split("(", 1)[0].strip()


def _join(words: list[dict], xmin: float, xmax: float) -> str:
    selected = sorted((w for w in words if xmin <= w["x"] < xmax), key=lambda w: (w["y"], w["x"]))
    lines: list[list[str]] = []
    last_y: float | None = None
    for word in selected:
        if last_y is None or abs(word["y"] - last_y) > 3.5:
            lines.append([])
            last_y = word["y"]
        lines[-1].append(word["text"])
    return " ".join(" ".join(line) for line in lines).strip()


def _nearest_number(words: list[dict], xmin: float, xmax: float, y: float, tolerance: float = 14) -> float | None:
    candidates = [(abs(w["y"] - y), _number(w["text"])) for w in words if xmin <= w["x"] < xmax and abs(w["y"] - y) <= tolerance]
    candidates = [(d, n) for d, n in candidates if n is not None]
    return min(candidates, default=(0, None), key=lambda item: item[0])[1]


def _floor(text: str) -> tuple[int | None, int | None]:
    numbers = [int(x) for x in re.findall(r"\d+", text or "")]
    return (numbers[0] if numbers else None, numbers[1] if len(numbers) > 1 else None)


def _coordinate(dong: str, signature: str) -> tuple[float, float]:
    lat, lng = DONG_CENTERS.get(dong, DEFAULT_CENTER)
    digest = hashlib.sha256(signature.encode("utf-8")).digest()
    angle = int.from_bytes(digest[:2], "big") / 65535 * math.tau
    radius = 0.00025 + int.from_bytes(digest[2:4], "big") / 65535 * 0.0020
    return round(lat + math.sin(angle) * radius, 6), round(lng + math.cos(angle) * radius, 6)


def _load_address_anchors() -> dict[str, list[dict]]:
    """공개 시설 데이터에서 팔달구의 실재 도로주소와 좌표를 동별로 읽는다."""
    global _ADDRESS_ANCHORS
    if _ADDRESS_ANCHORS is not None:
        return _ADDRESS_ANCHORS
    anchors: dict[str, dict[str, dict]] = {}
    if FACILITY_ANCHOR_CSV.exists():
        for chunk in pd.read_csv(
            FACILITY_ANCHOR_CSV,
            usecols=["address", "lat", "lng"],
            chunksize=100_000,
            low_memory=False,
        ):
            selected = chunk[
                chunk["address"].astype(str).str.contains("수원시 팔달구", na=False)
            ]
            for row in selected.itertuples(index=False):
                raw = str(row.address or "").strip()
                match = re.search(r"\(([^()]*(?:동|가))\)", raw)
                dong = _clean_dong(match.group(1)) if match else ""
                if dong not in DONG_CENTERS:
                    continue
                road = re.sub(r"\s*\([^)]*\)\s*$", "", raw).split(",", 1)[0].strip()
                try:
                    lat, lng = float(row.lat), float(row.lng)
                except (TypeError, ValueError):
                    continue
                if not road or not (33 <= lat <= 39 and 124 <= lng <= 132):
                    continue
                anchors.setdefault(dong, {}).setdefault(
                    road, {"road_address": road, "lat": lat, "lng": lng}
                )
    _ADDRESS_ANCHORS = {
        dong: sorted(values.values(), key=lambda item: item["road_address"])
        for dong, values in anchors.items()
    }
    return _ADDRESS_ANCHORS


def _representative_address(dong: str, signature: str) -> dict:
    candidates = _load_address_anchors().get(dong) or []
    if candidates:
        return dict(candidates[int(signature[:12], 16) % len(candidates)])
    lat, lng = _coordinate(dong, signature)
    return {"road_address": f"{SIDO} {GUGUN} {dong}", "lat": lat, "lng": lng}


def _parse_page(words: list[dict], pdf_name: str, page: int) -> list[dict]:
    anchors = sorted((w for w in words if w["x"] < 90 and w["text"] in HOUSE_TYPES), key=lambda w: w["y"])
    records: list[dict] = []
    for index, anchor in enumerate(anchors):
        previous_y = anchors[index - 1]["y"] if index else None
        next_y = anchors[index + 1]["y"] if index + 1 < len(anchors) else 550.0
        top = (previous_y + anchor["y"]) / 2 if previous_y is not None else 95.0
        group = [w for w in words if top <= w["y"] < (anchor["y"] + next_y) / 2]
        tx = [w for w in group if w["x"] < 90 and w["text"] in TRANSACTIONS and abs(w["y"] - anchor["y"]) < 24]
        if not tx:
            continue
        transaction = min(tx, key=lambda w: abs(w["y"] - anchor["y"]))["text"]
        y = anchor["y"]
        dong = _clean_dong(_join(group, 140, 220))
        if not dong:
            continue
        current_floor, total_floors = _floor(_join(group, 292, 345))
        records.append({
            "pdf_name": pdf_name, "page": page, "house_type_raw": anchor["text"], "transaction_type": transaction,
            "area1": _nearest_number(group, 100, 140, y - 15.7, 9), "area2": _nearest_number(group, 100, 140, y + 5.3, 9), "area3": _nearest_number(group, 100, 140, y + 27.0, 9),
            "dong": dong, "building_name": _join(group, 220, 292), "current_floor": current_floor, "total_floors": total_floors,
            "description": _join(group, 345, 585), "sale_price": _nearest_number(group, 585, 650, y - 10.6, 9),
            "monthly_rent": _nearest_number(group, 670, 705, y - 10.6, 9), "deposit": _nearest_number(group, 585, 650, y + 22.4, 9),
            "loan": _nearest_number(group, 670, 705, y + 22.4, 9), "broker_text": _join(group, 705, 842),
        })
    return records


def parse_pdf(pdf_path: Path, pdftotext: str) -> list[dict]:
    completed = subprocess.run([pdftotext, "-tsv", "-enc", "UTF-8", str(pdf_path), "-"], check=True, capture_output=True, text=True, encoding="utf-8")
    frame = pd.read_csv(io.StringIO(completed.stdout), sep="\t")
    frame = frame[(frame["level"] == 5) & frame["text"].notna()].copy()
    records: list[dict] = []
    for page, page_frame in frame.groupby("page_num", sort=True):
        words = [{"x": float(row.left), "y": float(row.top), "text": str(row.text).strip()} for row in page_frame.itertuples() if str(row.text).strip()]
        records.extend(_parse_page(words, pdf_path.name, int(page)))
    return records


def _signature(record: dict) -> str:
    fields = ("house_type_raw", "transaction_type", "area1", "area2", "area3", "dong", "building_name", "current_floor", "total_floors", "description", "sale_price", "monthly_rent", "deposit", "loan", "broker_text")
    raw = json.dumps([record.get(key) for key in fields], ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_pdf_directory(source_dir: Path, pdftotext: str) -> tuple[list[dict], dict]:
    pdfs = sorted(source_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"PDF가 없습니다: {source_dir}")
    unique: dict[str, dict] = {}
    parsed_count = 0
    file_counts: dict[str, int] = {}
    for pdf in pdfs:
        records = parse_pdf(pdf, pdftotext)
        file_counts[pdf.name] = len(records)
        parsed_count += len(records)
        for record in records:
            unique.setdefault(_signature(record), record)
    summary = {"source_directory": str(source_dir), "source_date": SOURCE_DATE, "pdf_count": len(pdfs), "parsed_count": parsed_count, "deduplicated_count": len(unique), "duplicate_count": parsed_count - len(unique), "file_counts": file_counts}
    return list(unique.values()), summary


def _positive(value: float | None) -> float | None:
    return value if value is not None and value > 0 else None


def _first_int(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _maintenance(text: str) -> float | None:
    match = re.search(r"관리비\s*(?:는|가|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*만", text)
    if match:
        return float(match.group(1))
    if "관리비 포함" in text or "관리비포함" in text:
        return 0.0
    return None


def _broker_fields(text: str) -> dict:
    phones = re.findall(r"0\d{1,2}-\d{3,4}-\d{4}", text)
    first_phone = text.find(phones[0]) if phones else -1
    address_at = text.find("경기도")
    end_name = first_phone if first_phone >= 0 else address_at
    name = text[:end_name].strip() if end_name and end_name > 0 else None
    address = text[address_at:].strip() if address_at >= 0 else None
    return {"name": name, "phone": " / ".join(dict.fromkeys(phones)) or None, "address": address}


def record_to_row(record: dict) -> dict:
    signature = _signature(record)
    property_id = f"PALDAL-PDF-{signature[:16].upper()}"
    description = re.sub(r"\s+", " ", record.get("description") or "").strip()
    broker = _broker_fields(record.get("broker_text") or "")
    transaction = record["transaction_type"]
    sale_price = _positive(record.get("sale_price")) if transaction == "매매" else None
    deposit = _positive(record.get("deposit")) if transaction in {"전세", "월세"} else None
    monthly = _positive(record.get("monthly_rent")) if transaction == "월세" else None
    areas = [_positive(record.get(key)) for key in ("area2", "area1", "area3")]
    area = next((value for value in areas if value is not None), None)
    location = _representative_address(record["dong"], signature)
    lat, lng = float(location["lat"]), float(location["lng"])
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    house_type = HOUSE_TYPE_MAP[record["house_type_raw"]]
    joined = "보증" in description and any(token in description for token in ("가입", "가능", "HUG", "HF"))
    row = {column: None for column in BROKER_LISTING_COLUMNS}
    row.update({column: None for column in PROPERTY_PROVENANCE_COLUMNS})
    row.update({
        "property_id": property_id, "listing_id": property_id,
        "is_synthetic": 0,
        "synthetic_notice": None,
        "source_type": "paldal_listing_catalog",
        "generator_version": "paldal-listing-importer-2",
        "generated_at": now,
        "source_dataset": record["pdf_name"],
        "source_record_hash": signature,
        "source_transaction_type": transaction,
        "generation_method": "PDF 고정 좌표 표 추출; 값 자체는 생성하지 않음",
        "price_estimation_method": "PDF 표 기재 금액 원문",
        "region_assignment_method": "원문 읍면동 + 공개 시설 도로주소 대표점",
        "region_coordinate_source": "공공데이터 기반 동별 실재 도로주소 대표점",
        "coordinate_distribution_method": "동일 행정동 공개 시설 주소에 결정론적 매칭; 실제 매물 좌표 아님",
        "listing_status": "2026-07-25 매물 후보(현재 거래 가능 여부 재확인 필요)",
        "listing_created_at": SOURCE_DATE, "listing_updated_at": SOURCE_DATE,
        "sido": SIDO, "gugun": GUGUN, "dong": record["dong"],
        "road_address": location["road_address"],
        "address_detail_public": "위치 참고용 대표 도로주소이며 실제 매물 상세주소가 아닙니다.",
        "lat": lat, "lng": lng,
        "transaction_type": transaction, "lease_type": transaction,
        "asking_price_manwon": sale_price or deposit,
        "sale_price_manwon": sale_price,
        "deposit_manwon": deposit,
        "monthly_rent_manwon": monthly,
        "maintenance_fee_manwon": _maintenance(description),
        "price_negotiable": "협의" in description,
        "move_in_negotiable": "입주협의" in description or "협의입주" in description,
        "available_from_date": "즉시" if "즉시입주" in description.replace(" ", "") else None,
        "property_type": house_type, "house_type": house_type,
        "building_name": record.get("building_name") or f"{record['dong']} {house_type}",
        "building_use": house_type,
        "current_floor": record.get("current_floor"), "total_floors": record.get("total_floors"),
        "area_m2": area, "exclusive_area_m2": _positive(record.get("area2")) or area,
        "supply_area_m2": _positive(record.get("area1")), "contract_area_m2": _positive(record.get("area3")),
        "room_count": _first_int(r"(?:방|룸)\s*([0-9]+)", description),
        "bathroom_count": _first_int(r"(?:욕실|화장실)\s*([0-9]+)", description),
        "parking_total": 1 if "주차가능" in description.replace(" ", "") else None,
        "elevator_count": 1 if "엘리베이터" in description or "승강기" in description else None,
        "cooling_facility": "에어컨" if "에어컨" in description else None,
        "built_in_appliances": ", ".join(x for x in ("에어컨", "냉장고", "세탁기", "인덕션") if x in description) or None,
        "furnished": any(token in description for token in ("풀옵션", "기본옵션", "옵션")),
        "pet_allowed": True if "반려동물 가능" in description or "애완동물 가능" in description else None,
        "loan_available": True if "대출가능" in description.replace(" ", "") else None,
        "rental_deposit_guarantee_joined": True if joined else None,
        "rental_deposit_guarantee_details": "PDF 설명에 보증 가입/가능 문구 있음" if joined else None,
        "market_price_manwon": sale_price or deposit,
        "mortgage_max_claim_manwon": _positive(record.get("loan")),
        "fraud_label": None, "fraud_score": None,
        "advertisement_medium": "팔달구 매물 목록",
        "advertisement_title": f"{record['dong']} {house_type} {transaction}",
        "advertisement_description": description or None,
        "broker_office_name": broker["name"], "broker_office_address": broker["address"],
        "broker_phone": broker["phone"], "advertisement_confirmed_at": SOURCE_DATE,
        "explanation_completed": False,
        "explanation_notes": "상세주소·등기·건축물대장·보증·현재 거래상태는 계약 전 별도 확인 필요",
        "source_provider": "팔달구 매물 목록",
        "source_url": None, "source_captured_at": f"{SOURCE_DATE}T00:00:00+09:00",
        "source_expires_at": None, "source_authorized": 1,
        "source_license_reference": "프로토타입 내부 검토용 매물 목록",
        "last_verified_at": SOURCE_DATE,
    })
    return row


def enrich_market_prices_and_risk(frame: pd.DataFrame) -> pd.DataFrame:
    """매매 비교사례로 임대주택가액을 추정하고 전세 위험점수를 계산한다."""
    frame = frame.copy()
    sale = frame[
        (frame["transaction_type"] == "매매")
        & (pd.to_numeric(frame["sale_price_manwon"], errors="coerce") > 0)
        & (pd.to_numeric(frame["area_m2"], errors="coerce") > 0)
    ].copy()
    sale["unit_price"] = (
        pd.to_numeric(sale["sale_price_manwon"], errors="coerce")
        / pd.to_numeric(sale["area_m2"], errors="coerce")
    )
    exact_stats = sale.groupby(["dong", "house_type"])["unit_price"].agg(["median", "count"])
    type_stats = sale.groupby("house_type")["unit_price"].median().to_dict()
    global_rate = float(sale["unit_price"].median()) if not sale.empty else 400.0

    for index, row in frame.iterrows():
        if row["transaction_type"] == "매매":
            frame.at[index, "market_price_manwon"] = row["sale_price_manwon"]
            continue
        area = float(row.get("area_m2") or 0)
        deposit = float(row.get("deposit_manwon") or 0)
        key = (row.get("dong"), row.get("house_type"))
        if key in exact_stats.index and int(exact_stats.loc[key, "count"]) >= 2:
            rate = float(exact_stats.loc[key, "median"])
            method = "동·주택유형 매매 비교사례 ㎡당 중위가격"
        elif row.get("house_type") in type_stats:
            rate = float(type_stats[row.get("house_type")])
            method = "팔달구 주택유형 매매 비교사례 ㎡당 중위가격"
        else:
            rate = global_rate
            method = "팔달구 전체 매매 비교사례 ㎡당 중위가격"
        estimate = max(rate * max(area, 1.0), deposit * 1.08 if deposit else 0)
        frame.at[index, "market_price_manwon"] = round(estimate, 1)
        frame.at[index, "price_estimation_method"] = (
            f"{method}; 계약 전 감정가·공시가격 재확인 필요"
        )

    scorer = FraudRiskScorer()
    for index in frame.index[frame["transaction_type"] == "전세"]:
        try:
            result = scorer.score(frame.loc[index].to_dict())
            frame.at[index, "fraud_score"] = result["fraud_score"]
            frame.at[index, "explanation_notes"] = (
                "HF 실제 보증사고 연구 공개계수 기반 추정치. "
                "등기·선순위채권·임대인 정보 미확인 항목은 계약 전 별도 검증 필요"
            )
        except Exception as exc:
            frame.at[index, "explanation_notes"] = f"위험도 계산 보류: {type(exc).__name__}"
    frame["fraud_score"] = pd.to_numeric(frame["fraud_score"], errors="coerce")
    frame["market_price_manwon"] = pd.to_numeric(
        frame["market_price_manwon"], errors="coerce"
    )
    return frame


def _write_csv(frame: pd.DataFrame, output_csv: Path) -> Path | None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    if output_csv.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = output_csv.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup = backup_dir / f"properties_before_paldal_{stamp}.csv"
        shutil.move(str(output_csv), str(backup))
    temporary = output_csv.with_suffix(".tmp.csv")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    temporary.replace(output_csv)
    return backup


def _replace_properties(frame: pd.DataFrame, db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        frame.to_sql("properties_import_staging", conn, if_exists="replace", index=False)
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute("DROP TABLE IF EXISTS properties_previous")
            exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='properties'").fetchone()
            if exists:
                conn.execute("ALTER TABLE properties RENAME TO properties_previous")
            conn.execute("ALTER TABLE properties_import_staging RENAME TO properties")
            conn.execute("DROP TABLE IF EXISTS properties_previous")
            conn.execute("CREATE UNIQUE INDEX idx_prop_property_id ON properties(property_id)")
            conn.execute("CREATE INDEX idx_prop_region ON properties(sido, gugun, dong)")
            conn.execute("CREATE INDEX idx_prop_search ON properties(transaction_type, house_type)")
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    ensure_feed_schema(db_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="팔달구 매물 목록으로 properties DB를 교체")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--db-path", type=Path, default=config.DB_PATH)
    parser.add_argument("--output-csv", type=Path, default=config.DATA_GEN / "properties.csv")
    parser.add_argument("--pdftotext", default=r"C:\texlive\2025\bin\windows\pdftotext.exe")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--vacuum", action="store_true")
    args = parser.parse_args()

    records, summary = load_pdf_directory(args.source_dir.resolve(), args.pdftotext)
    rows = [record_to_row(record) for record in records]
    columns = [*BROKER_LISTING_COLUMNS, *PROPERTY_PROVENANCE_COLUMNS]
    frame = enrich_market_prices_and_risk(pd.DataFrame(rows, columns=columns))
    summary.update({
        "region": f"{SIDO} {GUGUN}",
        "transaction_counts": frame["transaction_type"].value_counts().to_dict(),
        "house_type_counts": frame["house_type"].value_counts().to_dict(),
        "dong_counts": frame["dong"].value_counts().to_dict(),
        "database_replaced": not args.dry_run,
        "limitations": [
            "매물 상세주소가 없어 같은 동의 실재 도로주소 대표점을 위치 참고용으로 사용",
            "현재 거래 가능 여부와 가격은 중개사에게 재확인 필요",
            "전세 위험도는 공개모델 추정치이며 등기·선순위권리·보증 가입 여부는 별도 확인",
        ],
    })
    if not args.dry_run:
        backup = _write_csv(frame, args.output_csv)
        _replace_properties(frame, args.db_path)
        summary["csv_backup"] = str(backup) if backup else None
        if args.vacuum:
            with sqlite3.connect(args.db_path) as conn:
                conn.execute("VACUUM")
    summary_path = config.DATA_GEN / "paldal_pdf_import_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
