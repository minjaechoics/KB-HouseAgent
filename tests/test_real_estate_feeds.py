from __future__ import annotations

import sqlite3

import pytest

from src.real_estate_feeds.brightdata import (
    FeedAuthorizationError, assert_source_authorized, import_authorized_snapshot,
)
from src.real_estate_feeds.rtms import RTMSPriceHistoryTool
from src.real_estate_feeds.storage import ensure_feed_schema, feed_status


PROPERTY_COLUMNS = [
    "property_id TEXT PRIMARY KEY", "listing_id TEXT", "is_synthetic INTEGER",
    "synthetic_notice TEXT", "source_type TEXT", "source_dataset TEXT",
    "listing_status TEXT", "listing_created_at TEXT", "listing_updated_at TEXT",
    "sido TEXT", "gugun TEXT", "dong TEXT", "legal_dong_code TEXT",
    "road_address TEXT", "jibun_address TEXT", "lat REAL", "lng REAL",
    "transaction_type TEXT", "lease_type TEXT", "house_type TEXT",
    "property_type TEXT", "building_name TEXT", "asking_price_manwon REAL",
    "sale_price_manwon REAL", "deposit_manwon REAL", "monthly_rent_manwon REAL",
    "maintenance_fee_manwon REAL", "area_m2 REAL", "room_count REAL",
    "bathroom_count REAL", "current_floor REAL", "total_floors REAL",
    "build_year REAL", "advertisement_title TEXT", "broker_office_name TEXT",
    "broker_phone TEXT", "photo_count INTEGER",
]


def make_db(path):
    with sqlite3.connect(path) as conn:
        conn.execute(f"CREATE TABLE properties ({','.join(PROPERTY_COLUMNS)})")
    ensure_feed_schema(path)


def test_restricted_domain_requires_explicit_permission_reference():
    with pytest.raises(FeedAuthorizationError):
        assert_source_authorized(
            "https://fin.land.naver.com/listing/1", rights_confirmed=True,
            license_reference="self-asserted")
    assert assert_source_authorized(
        "https://fin.land.naver.com/listing/1", rights_confirmed=True,
        license_reference="permission:NAVER-TEST") == "fin.land.naver.com"


def test_authorized_snapshot_is_searchable_and_has_expiry(tmp_path):
    db = tmp_path / "feed.db"
    make_db(db)
    result = import_authorized_snapshot([{
        "id": "A-1", "url": "https://partner.example/listing/A-1",
        "lat": 37.5, "lng": 127.0, "transaction_type": "전세",
        "road_address": "서울 테스트로 1", "sido": "서울특별시",
        "gugun": "테스트구", "house_type": "아파트",
        "deposit_manwon": 30000, "area_m2": 59.8,
    }], provider="licensed-partner", source_url="https://partner.example/feed",
        license_reference="contract:ABC-123", rights_confirmed=True,
        db_path=db, captured_at="2020-07-23T00:00:00+00:00", ttl_hours=24)
    assert result["written"] == 1
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT is_synthetic,source_authorized,listing_status,source_provider "
            "FROM properties"
        ).fetchone()
    assert row == (0, 1, "active", "licensed-partner")
    assert feed_status(db)["live_listings"]["count"] == 0  # fixture is expired now


def test_rtms_history_is_transaction_specific(tmp_path):
    db = tmp_path / "history.db"
    make_db(db)
    with sqlite3.connect(db) as conn:
        values = []
        for idx, (ym, price) in enumerate(((202601, 50000), (202602, 51000))):
            values.append((
                f"o{idx}", "MOLIT_RTMS", "apt", "https://rt.molit.go.kr/",
                f"{ym//100:04d}-{ym%100:02d}-15", ym, "41117", "41117", "", "",
                "테스트", "아파트", "매매", price, None, None, 59.8, 10, 2020,
                0, "2026-07-23T00:00:00+00:00", f"h{idx}", None,
            ))
        conn.executemany(
            "INSERT INTO property_price_observations VALUES("
            + ",".join("?" for _ in range(23)) + ")", values)
    history = RTMSPriceHistoryTool(db).history({
        "legal_dong_code": "41117", "house_type": "아파트",
        "transaction_type": "매매",
    })
    assert history["available"] is True
    assert history["latest_price_manwon"] == 51000
    assert history["change_1m"] == 0.02
