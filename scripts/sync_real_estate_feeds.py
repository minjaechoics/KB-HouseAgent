"""Synchronize official RTMS history or import a licensed Bright Data snapshot.

Examples:
  python scripts/sync_real_estate_feeds.py rtms
  python scripts/sync_real_estate_feeds.py brightdata-status
  python scripts/sync_real_estate_feeds.py snapshot licensed.json \
      --provider partner --source-url https://partner.example/data \
      --license-reference contract:ABC-123 --confirm-rights
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.real_estate_feeds.brightdata import (
    BrightDataClient, import_authorized_snapshot, load_snapshot_file,
)
from src.real_estate_feeds.rtms import import_rtms_downloads
from src.real_estate_feeds.storage import feed_status


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("rtms")
    sub.add_parser("status")
    sub.add_parser("brightdata-status")
    snapshot = sub.add_parser("snapshot")
    snapshot.add_argument("path", type=Path)
    snapshot.add_argument("--provider", required=True)
    snapshot.add_argument("--source-url", required=True)
    snapshot.add_argument("--license-reference", required=True)
    snapshot.add_argument("--confirm-rights", action="store_true")
    snapshot.add_argument("--ttl-hours", type=int, default=24)
    args = parser.parse_args()

    if args.command == "rtms":
        result = import_rtms_downloads()
    elif args.command == "status":
        result = feed_status()
    elif args.command == "brightdata-status":
        result = BrightDataClient(config.BRIGHTDATA_API_TOKEN).account_capabilities()
    else:
        rows = load_snapshot_file(args.path)
        result = import_authorized_snapshot(
            rows, provider=args.provider, source_url=args.source_url,
            license_reference=args.license_reference,
            rights_confirmed=args.confirm_rights, ttl_hours=args.ttl_hours,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
