"""SQLite schema and read models for externally sourced real-estate data.

The existing ``properties`` table contains research/synthetic candidates.  An
authorized live-listing feed may upsert a small, compatible row into that table,
while its complete provenance is retained in ``live_property_listings``.
Official RTMS transactions are stored separately as immutable observations.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src import config


PROPERTY_PROVENANCE_COLUMNS = {
    "source_provider": "TEXT",
    "source_url": "TEXT",
    "source_captured_at": "TEXT",
    "source_expires_at": "TEXT",
    "source_authorized": "INTEGER DEFAULT 0",
    "source_license_reference": "TEXT",
    "last_verified_at": "TEXT",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_feed_schema(db_path: Path = config.DB_PATH) -> None:
    """Create additive feed tables/migrations without rebuilding user data."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(properties)")}
        if existing:
            for name, sql_type in PROPERTY_PROVENANCE_COLUMNS.items():
                if name not in existing:
                    conn.execute(f'ALTER TABLE properties ADD COLUMN "{name}" {sql_type}')
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_prop_property_id "
                "ON properties(property_id)"
            )
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS listing_sources (
                source_id TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                source_domain TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                authorized INTEGER NOT NULL CHECK (authorized IN (0, 1)),
                license_reference TEXT,
                terms_url TEXT,
                created_at TEXT NOT NULL,
                last_sync_at TEXT
            );

            CREATE TABLE IF NOT EXISTS live_property_listings (
                property_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL REFERENCES listing_sources(source_id),
                source_listing_id TEXT NOT NULL,
                source_url TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                active INTEGER NOT NULL CHECK (active IN (0, 1)),
                normalized_json TEXT NOT NULL,
                raw_json TEXT,
                UNIQUE(source_id, source_listing_id)
            );
            CREATE INDEX IF NOT EXISTS idx_live_active_expiry
              ON live_property_listings(active, expires_at);

            CREATE TABLE IF NOT EXISTS property_price_observations (
                observation_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                source_dataset TEXT NOT NULL,
                source_url TEXT NOT NULL,
                observed_date TEXT NOT NULL,
                deal_ym INTEGER NOT NULL,
                lawd_cd TEXT NOT NULL,
                legal_dong_code TEXT,
                region_name TEXT,
                dong TEXT,
                complex_name TEXT,
                house_type TEXT NOT NULL,
                transaction_type TEXT NOT NULL,
                price_manwon REAL,
                deposit_manwon REAL,
                monthly_rent_manwon REAL,
                area_m2 REAL,
                floor INTEGER,
                build_year INTEGER,
                cancelled INTEGER NOT NULL DEFAULT 0,
                ingested_at TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                raw_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_price_obs_lookup
              ON property_price_observations(
                lawd_cd, house_type, transaction_type, deal_ym
              );
            CREATE INDEX IF NOT EXISTS idx_price_obs_complex
              ON property_price_observations(complex_name, area_m2, deal_ym);

            CREATE TABLE IF NOT EXISTS rtms_transaction_details (
                observation_id TEXT PRIMARY KEY REFERENCES property_price_observations(observation_id),
                apartment_sequence TEXT,
                apartment_dong TEXT,
                road_name TEXT,
                road_main_no TEXT,
                road_sub_no TEXT,
                legal_dong_subcode TEXT,
                lot_main_no TEXT,
                lot_sub_no TEXT,
                buyer_type TEXT,
                seller_type TEXT,
                estate_agent_region TEXT,
                registration_date TEXT,
                land_leasehold_flag TEXT,
                source_dataset TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_rtms_detail_apt
              ON rtms_transaction_details(apartment_sequence, road_name);

            CREATE TABLE IF NOT EXISTS feed_sync_runs (
                run_id TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                source_ref TEXT,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL,
                records_seen INTEGER NOT NULL DEFAULT 0,
                records_written INTEGER NOT NULL DEFAULT 0,
                records_rejected INTEGER NOT NULL DEFAULT 0,
                message TEXT
            );

            CREATE TABLE IF NOT EXISTS rone_stat_tables (
                statbl_id TEXT PRIMARY KEY,
                statbl_name TEXT NOT NULL,
                data_cycle_code TEXT,
                data_cycle_name TEXT,
                data_start_year TEXT,
                data_end_year TEXT,
                representative_unit TEXT,
                top_org_name TEXT,
                open_state TEXT,
                source_url TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                raw_json TEXT
            );

            CREATE TABLE IF NOT EXISTS rone_stat_observations (
                observation_id TEXT PRIMARY KEY,
                statbl_id TEXT NOT NULL,
                statbl_name TEXT NOT NULL,
                data_cycle_code TEXT NOT NULL,
                period_id TEXT NOT NULL,
                period_description TEXT,
                group_id TEXT,
                group_name TEXT,
                class_id TEXT,
                class_name TEXT,
                class_full_name TEXT,
                item_id TEXT,
                item_name TEXT,
                item_full_name TEXT,
                value REAL,
                unit_name TEXT,
                fetched_at TEXT NOT NULL,
                source_url TEXT NOT NULL,
                UNIQUE(statbl_id, period_id, class_id, item_id, group_id)
            );
            CREATE INDEX IF NOT EXISTS idx_rone_market_lookup
              ON rone_stat_observations(statbl_id, period_id, class_full_name);

            CREATE TABLE IF NOT EXISTS ev_chargers (
                charger_id TEXT PRIMARY KEY,
                station_name TEXT,
                charger_name TEXT,
                address TEXT,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                status_code TEXT,
                status_name TEXT,
                charger_method TEXT,
                charge_type TEXT,
                updated_at TEXT,
                fetched_at TEXT NOT NULL,
                source_url TEXT NOT NULL,
                raw_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_ev_chargers_coordinates
              ON ev_chargers(latitude, longitude);
            """
        )


def expire_stale_listings(db_path: Path = config.DB_PATH, now: str | None = None) -> int:
    ensure_feed_schema(db_path)
    now = now or utc_now()
    with sqlite3.connect(db_path) as conn:
        rows = [row[0] for row in conn.execute(
            "SELECT property_id FROM live_property_listings "
            "WHERE active=1 AND expires_at < ?", (now,)
        )]
        if not rows:
            return 0
        conn.executemany(
            "UPDATE live_property_listings SET active=0 WHERE property_id=?",
            [(value,) for value in rows],
        )
        conn.executemany(
            "UPDATE properties SET listing_status='expired' WHERE property_id=?",
            [(value,) for value in rows],
        )
        return len(rows)


def feed_status(db_path: Path = config.DB_PATH) -> dict[str, Any]:
    ensure_feed_schema(db_path)
    expire_stale_listings(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        live = conn.execute(
            "SELECT COUNT(*) count, MAX(captured_at) latest FROM live_property_listings "
            "WHERE active=1"
        ).fetchone()
        price = conn.execute(
            "SELECT COUNT(*) count, MIN(observed_date) first_date, "
            "MAX(observed_date) latest_date FROM property_price_observations"
        ).fetchone()
        coverage = conn.execute(
            "SELECT COUNT(DISTINCT lawd_cd) sgg_count, "
            "COUNT(DISTINCT SUBSTR(lawd_cd,1,2)) province_prefix_count "
            "FROM property_price_observations"
        ).fetchone()
        sources = [dict(row) for row in conn.execute(
            "SELECT source_id, provider, source_domain, source_kind, authorized, "
            "license_reference, last_sync_at FROM listing_sources ORDER BY provider"
        )]
        rone = conn.execute(
            "SELECT COUNT(*) observation_count, COUNT(DISTINCT statbl_id) table_count, "
            "MIN(period_id) first_period, MAX(period_id) latest_period "
            "FROM rone_stat_observations"
        ).fetchone()
        ev = conn.execute(
            "SELECT COUNT(*) charger_count, MAX(fetched_at) latest_sync FROM ev_chargers"
        ).fetchone()
    return {
        "brightdata": {
            "configured": bool(config.BRIGHTDATA_API_TOKEN),
            "dataset_id_configured": bool(config.BRIGHTDATA_DATASET_ID),
            "active_authorized_feed": bool(live["count"]),
        },
        "live_listings": dict(live),
        "price_observations": dict(price),
        "price_coverage": dict(coverage),
        "sources": sources,
        "rone": {
            **dict(rone),
            "configured": bool(config.RONE_API_KEY),
            "source": "한국부동산원 R-ONE Open API",
        },
        "vworld_ev_model": {
            "implemented": True,
            "realtime_api_configured": bool(config.EV_CHARGER_SERVICE_KEY),
            "source": "V-World 전기차충전소 활용모델 / 공공데이터포털 EvInfoServiceV2",
            **dict(ev),
        },
        "policy": {
            "synthetic_is_never_presented_as_live": True,
            "authorized_source_required": True,
            "expired_listings_hidden": True,
        },
    }


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
