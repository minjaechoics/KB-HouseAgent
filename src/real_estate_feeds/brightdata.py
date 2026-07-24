"""Bright Data client and licensed listing-snapshot importer.

Bright Data is a transport/provider, not a grant of rights to third-party
content.  Consequently imports require an explicit rights assertion and a
non-empty license/contract reference.  NAVER/KB domains are additionally
blocked unless that exact source authorization is documented.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Iterable

import requests

from src import config
from src.real_estate_feeds.storage import ensure_feed_schema, json_text, utc_now


API_BASE = "https://api.brightdata.com"
RESTRICTED_DOMAINS = {"naver.com", "land.naver.com", "fin.land.naver.com", "kbland.kr"}


class FeedAuthorizationError(ValueError):
    pass


class BrightDataClient:
    def __init__(self, api_token: str | None = None, timeout: float = 30.0):
        self.api_token = (api_token or os.getenv("BRIGHTDATA_API_TOKEN", "")).strip()
        self.timeout = timeout
        if not self.api_token:
            raise ValueError("BRIGHTDATA_API_TOKEN is not configured")

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_token}"}

    def list_datasets(self) -> list[dict]:
        response = requests.get(
            f"{API_BASE}/datasets/list", headers=self.headers, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def list_zones(self) -> list[dict]:
        response = requests.get(
            f"{API_BASE}/zone/get_all_zones", headers=self.headers, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, list) else payload.get("zones", [])

    def download_snapshot(self, snapshot_id: str, output_format: str = "json") -> Any:
        if not snapshot_id.startswith(("s_", "snap_", "sd_")):
            raise ValueError("invalid Bright Data snapshot id")
        response = requests.get(
            f"{API_BASE}/datasets/v3/snapshot/{snapshot_id}",
            params={"format": output_format}, headers=self.headers,
            timeout=max(self.timeout, 120.0),
        )
        response.raise_for_status()
        if output_format == "json":
            return response.json()
        return response.content

    def account_capabilities(self) -> dict:
        datasets = self.list_datasets()
        zones = self.list_zones()
        real_estate = [row for row in datasets if any(
            token in str(row.get("name", "")).lower()
            for token in ("real estate", "property", "naver real")
        )]
        return {
            "authenticated": True,
            "zone_count": len(zones),
            "dataset_count": len(datasets),
            "real_estate_dataset_matches": real_estate,
        }


def _domain(source_url: str) -> str:
    host = (urlparse(source_url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def assert_source_authorized(source_url: str, *, rights_confirmed: bool,
                             license_reference: str) -> str:
    domain = _domain(source_url)
    if not domain:
        raise FeedAuthorizationError("source_url must be an absolute HTTP(S) URL")
    if not rights_confirmed:
        raise FeedAuthorizationError("source-use rights were not explicitly confirmed")
    if len(license_reference.strip()) < 6:
        raise FeedAuthorizationError("a concrete license/contract reference is required")
    restricted = any(domain == blocked or domain.endswith("." + blocked)
                     for blocked in RESTRICTED_DOMAINS)
    if restricted and not license_reference.lower().startswith(("contract:", "permission:")):
        raise FeedAuthorizationError(
            f"{domain} requires an explicit contract:/permission: reference")
    return domain


def _first(row: dict, *keys: str, default=None):
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def _float(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", "").replace("만원", "").strip())
    except (TypeError, ValueError):
        return None


def normalize_listing(row: dict, *, provider: str, source_url: str,
                      captured_at: str, ttl_hours: int = 24) -> dict:
    listing_id = str(_first(row, "listing_id", "article_id", "id", "item_id", default="")).strip()
    if not listing_id:
        raise ValueError("listing id missing")
    lat = _float(_first(row, "lat", "latitude"))
    lng = _float(_first(row, "lng", "longitude", "lon"))
    if lat is None or lng is None or not (32 <= lat <= 39 and 124 <= lng <= 132):
        raise ValueError("valid Korean coordinates are required")
    transaction = str(_first(row, "transaction_type", "trade_type", "deal_type", default="")).strip()
    aliases = {"sale": "매매", "trade": "매매", "jeonse": "전세", "rent": "월세", "monthly": "월세"}
    transaction = aliases.get(transaction.lower(), transaction)
    if transaction not in {"매매", "전세", "월세"}:
        raise ValueError("transaction_type must be 매매/전세/월세")
    captured = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
    if captured.tzinfo is None:
        captured = captured.replace(tzinfo=timezone.utc)
    normalized = {
        "source_listing_id": listing_id,
        "listing_id": f"{provider}:{listing_id}",
        "property_id": "LIVE_" + hashlib.sha256(
            f"{provider}|{listing_id}".encode()).hexdigest()[:24],
        "source_url": str(_first(row, "source_url", "url", default=source_url)),
        "captured_at": captured.replace(microsecond=0).isoformat(),
        "expires_at": (captured + timedelta(hours=max(1, ttl_hours))).replace(microsecond=0).isoformat(),
        "sido": str(_first(row, "sido", "province", default="")),
        "gugun": str(_first(row, "gugun", "district", "city", default="")),
        "dong": str(_first(row, "dong", "neighborhood", default="")),
        "legal_dong_code": str(_first(row, "legal_dong_code", "lawd_cd", default="")),
        "road_address": str(_first(row, "road_address", "address", default="")),
        "jibun_address": str(_first(row, "jibun_address", default="")),
        "lat": lat, "lng": lng,
        "transaction_type": transaction,
        "lease_type": transaction if transaction in {"전세", "월세"} else None,
        "house_type": str(_first(row, "house_type", "property_type", default="주택")),
        "property_type": str(_first(row, "property_type", "house_type", default="주택")),
        "building_name": str(_first(row, "building_name", "complex_name", "name", default="")),
        "asking_price_manwon": _float(_first(row, "asking_price_manwon", "price_manwon", "price")),
        "sale_price_manwon": _float(_first(row, "sale_price_manwon")),
        "deposit_manwon": _float(_first(row, "deposit_manwon", "deposit")),
        "monthly_rent_manwon": _float(_first(row, "monthly_rent_manwon", "monthly_rent")),
        "maintenance_fee_manwon": _float(_first(row, "maintenance_fee_manwon", "maintenance_fee")),
        "area_m2": _float(_first(row, "area_m2", "exclusive_area_m2", "area")),
        "room_count": _float(_first(row, "room_count", "rooms")),
        "bathroom_count": _float(_first(row, "bathroom_count", "bathrooms")),
        "current_floor": _float(_first(row, "current_floor", "floor")),
        "total_floors": _float(_first(row, "total_floors")),
        "build_year": _float(_first(row, "build_year")),
        "advertisement_title": str(_first(row, "advertisement_title", "title", default="")),
        "broker_office_name": str(_first(row, "broker_office_name", "agency_name", default="")),
        "broker_phone": str(_first(row, "broker_phone", default="")),
        "photo_count": int(_float(_first(row, "photo_count", default=0)) or 0),
    }
    if transaction == "매매" and normalized["sale_price_manwon"] is None:
        normalized["sale_price_manwon"] = normalized["asking_price_manwon"]
    if not normalized["road_address"] and not normalized["jibun_address"]:
        raise ValueError("public address is required")
    return normalized


def import_authorized_snapshot(rows: Iterable[dict], *, provider: str,
                               source_url: str, license_reference: str,
                               rights_confirmed: bool, db_path: Path = config.DB_PATH,
                               captured_at: str | None = None,
                               ttl_hours: int = 24) -> dict:
    domain = assert_source_authorized(
        source_url, rights_confirmed=rights_confirmed,
        license_reference=license_reference)
    ensure_feed_schema(db_path)
    captured_at = captured_at or utc_now()
    source_id = "brightdata:" + hashlib.sha256(
        f"{provider}|{domain}|{license_reference}".encode()).hexdigest()[:20]
    seen = written = rejected = 0
    errors: list[str] = []
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO listing_sources(source_id,provider,source_domain,source_kind,"
            "authorized,license_reference,terms_url,created_at,last_sync_at) "
            "VALUES(?,?,?,?,1,?,?,?,?) ON CONFLICT(source_id) DO UPDATE SET "
            "last_sync_at=excluded.last_sync_at, authorized=1, "
            "license_reference=excluded.license_reference",
            (source_id, provider, domain, "brightdata_authorized_snapshot",
             license_reference, source_url, captured_at, captured_at),
        )
        for row in rows:
            seen += 1
            try:
                normalized = normalize_listing(
                    row, provider=provider, source_url=source_url,
                    captured_at=captured_at, ttl_hours=ttl_hours)
                raw_text = json_text(row)
                content_hash = hashlib.sha256(raw_text.encode()).hexdigest()
                conn.execute(
                    "INSERT INTO live_property_listings(property_id,source_id,source_listing_id,"
                    "source_url,content_hash,captured_at,expires_at,last_seen_at,active,"
                    "normalized_json,raw_json) VALUES(?,?,?,?,?,?,?,?,1,?,?) "
                    "ON CONFLICT(property_id) DO UPDATE SET content_hash=excluded.content_hash,"
                    "captured_at=excluded.captured_at,expires_at=excluded.expires_at,"
                    "last_seen_at=excluded.last_seen_at,active=1,"
                    "normalized_json=excluded.normalized_json,raw_json=excluded.raw_json",
                    (normalized["property_id"], source_id, normalized["source_listing_id"],
                     normalized["source_url"], content_hash, normalized["captured_at"],
                     normalized["expires_at"], normalized["captured_at"],
                     json_text(normalized), raw_text),
                )
                columns = {
                    "property_id": normalized["property_id"],
                    "listing_id": normalized["listing_id"], "is_synthetic": 0,
                    "synthetic_notice": None, "source_type": "authorized_live_listing",
                    "source_dataset": source_id, "listing_status": "active",
                    "listing_created_at": normalized["captured_at"],
                    "listing_updated_at": normalized["captured_at"],
                    "source_provider": provider, "source_url": normalized["source_url"],
                    "source_captured_at": normalized["captured_at"],
                    "source_expires_at": normalized["expires_at"], "source_authorized": 1,
                    "source_license_reference": license_reference,
                    "last_verified_at": normalized["captured_at"],
                    **{key: value for key, value in normalized.items() if key in {
                        "sido", "gugun", "dong", "legal_dong_code", "road_address",
                        "jibun_address", "lat", "lng", "transaction_type", "lease_type",
                        "house_type", "property_type", "building_name", "asking_price_manwon",
                        "sale_price_manwon", "deposit_manwon", "monthly_rent_manwon",
                        "maintenance_fee_manwon", "area_m2", "room_count", "bathroom_count",
                        "current_floor", "total_floors", "build_year", "advertisement_title",
                        "broker_office_name", "broker_phone", "photo_count"
                    }},
                }
                names = list(columns)
                placeholders = ",".join("?" for _ in names)
                updates = ",".join(f'"{name}"=excluded."{name}"' for name in names if name != "property_id")
                conn.execute(
                    f'INSERT INTO properties({",".join(chr(34)+n+chr(34) for n in names)}) '
                    f'VALUES({placeholders}) ON CONFLICT(property_id) DO UPDATE SET {updates}',
                    [columns[name] for name in names],
                )
                written += 1
            except (ValueError, TypeError) as exc:
                rejected += 1
                if len(errors) < 20:
                    errors.append(f"row {seen}: {exc}")
    return {"seen": seen, "written": written, "rejected": rejected,
            "errors": errors, "source_id": source_id}


def load_snapshot_file(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, dict):
        payload = payload.get("data") or payload.get("results") or [payload]
    if not isinstance(payload, list):
        raise ValueError("snapshot must contain a JSON array or NDJSON objects")
    return payload
