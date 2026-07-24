"""Print non-secret coverage counts for the official-data integration."""
from __future__ import annotations

import json
import sqlite3

from src import config


def main() -> None:
    with sqlite3.connect(config.DB_PATH) as conn:
        result = {
            "rtms_apt_trade_dev_rows": conn.execute(
                "SELECT COUNT(*) FROM property_price_observations "
                "WHERE source_dataset='rtms_apt_trade_dev.csv'"
            ).fetchone()[0],
            "rtms_transaction_detail_rows": conn.execute(
                "SELECT COUNT(*) FROM rtms_transaction_details"
            ).fetchone()[0],
            "rtms_apt_trade_dev_lawd_count": conn.execute(
                "SELECT COUNT(DISTINCT lawd_cd) FROM property_price_observations "
                "WHERE source_dataset='rtms_apt_trade_dev.csv'"
            ).fetchone()[0],
            "rone_observations": conn.execute(
                "SELECT COUNT(*) FROM rone_stat_observations"
            ).fetchone()[0],
            "properties": conn.execute("SELECT COUNT(*) FROM properties").fetchone()[0],
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
