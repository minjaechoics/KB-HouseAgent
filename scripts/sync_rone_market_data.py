"""Sync official R-ONE regional market statistics into the runtime DB."""
from __future__ import annotations

import argparse
import json

from src.market_data.rone import sync_rone_market_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=30)
    parser.add_argument("--end-ym", default=None)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(sync_rone_market_data(
        months=args.months, end_ym=args.end_ym, max_workers=args.workers,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
