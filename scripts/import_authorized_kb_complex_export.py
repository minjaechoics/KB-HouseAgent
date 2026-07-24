"""허가받은 KB 단지 시세 CSV를 별도 참조 테이블로 적재한다.

이 스크립트는 KB 화면을 크롤링하지 않는다. 제휴 API나 서면 이용허락 범위에서
제공받은 파일만 입력해야 하며, 투자테이블 통계는 현재 매물 DB로 위장하지 않는다.
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src import config


REQUIRED = {"complex_name", "sido", "gugun", "exclusive_area_m2",
            "observed_date", "sale_price_manwon"}


def import_export(csv_path: Path, db_path: Path, license_reference: str) -> int:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"필수 열 누락: {', '.join(sorted(missing))}")
        rows = list(reader)
    now = datetime.now(timezone.utc).isoformat()
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS kb_authorized_complex_prices (
                complex_name TEXT NOT NULL,
                sido TEXT NOT NULL,
                gugun TEXT NOT NULL,
                dong TEXT,
                exclusive_area_m2 REAL NOT NULL,
                observed_date TEXT NOT NULL,
                sale_price_manwon REAL NOT NULL,
                source_record_id TEXT,
                license_reference TEXT NOT NULL,
                imported_at TEXT NOT NULL,
                PRIMARY KEY(complex_name, sido, gugun, exclusive_area_m2, observed_date)
            )
        """)
        values = [(
            row["complex_name"].strip(), row["sido"].strip(), row["gugun"].strip(),
            row.get("dong", "").strip() or None, float(row["exclusive_area_m2"]),
            row["observed_date"].strip(), float(row["sale_price_manwon"]),
            row.get("source_record_id", "").strip() or None, license_reference, now,
        ) for row in rows]
        connection.executemany("""
            INSERT INTO kb_authorized_complex_prices VALUES(?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(complex_name,sido,gugun,exclusive_area_m2,observed_date)
            DO UPDATE SET sale_price_manwon=excluded.sale_price_manwon,
                          source_record_id=excluded.source_record_id,
                          license_reference=excluded.license_reference,
                          imported_at=excluded.imported_at
        """, values)
        connection.commit()
        return len(values)
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    parser.add_argument("--license-reference", required=True,
                        help="서면 허가/제휴 계약/공식 제공건 식별자")
    parser.add_argument("--confirm-authorized", action="store_true",
                        help="해당 파일의 저장·가공 권한을 확인했음을 명시")
    args = parser.parse_args()
    if not args.confirm_authorized:
        raise SystemExit("--confirm-authorized가 필요합니다. 권한 없는 파일은 적재하지 않습니다.")
    count = import_export(args.csv_path, args.db, args.license_reference)
    print(f"authorized KB complex price rows imported: {count}")


if __name__ == "__main__":
    main()
