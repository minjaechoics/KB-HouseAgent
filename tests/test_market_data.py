from __future__ import annotations

import sqlite3

from src.market_data.rone import RoneMarketTool
from src.real_estate_feeds.storage import ensure_feed_schema
from src.report.lifestyle import estimate_monthly_lifestyle
from src.tools.ev_charger_tool import EVChargerTool


class _NoRouteMap:
    def geocode(self, _query):
        return {"ok": True, "lat": 37.5, "lng": 127.0, "address": "test"}

    def travel_time(self, *_args):
        return {"minutes": 20, "distance_km": 10, "source": "test", "estimated": False}


def test_rone_market_selects_matching_region(tmp_path):
    db = tmp_path / "market.db"
    ensure_feed_schema(db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO rone_stat_tables VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            ("A_2024_00045", "(월) 매매가격지수_아파트", "MM", "매월",
             "2003", "2026", "2026.01=100", "한국부동산원", "Y",
             "https://example.test", "now", "{}"),
        )
        for index, period, value in ((1, "202605", 99.2), (2, "202606", 100.3)):
            conn.execute(
                "INSERT INTO rone_stat_observations VALUES(" + ",".join("?" for _ in range(18)) + ")",
                (f"o{index}", "A_2024_00045", "(월) 매매가격지수_아파트", "MM",
                 period, period, None, None, "41117", "영통구", "경기>수원시>영통구",
                 "100001", "지수", "지수", value, "지수", "now", "https://example.test"),
            )
    result = RoneMarketTool(db).market({
        "sido": "경기", "gugun": "영통구", "house_type": "아파트",
        "transaction_type": "매매",
    })
    assert result["price_index"]["available"] is True
    assert result["price_index"]["latest_value"] == 100.3
    assert result["price_index"]["change_1m"] == 1.1


def test_ev_charger_radius_query(tmp_path):
    db = tmp_path / "ev.db"
    ensure_feed_schema(db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO ev_chargers VALUES(" + ",".join("?" for _ in range(14)) + ")",
            ("EV1", "테스트 충전소", "급속1", "수원시", 37.5, 127.0, "1",
             "충전 가능", "급속", "DC", "now", "now", "https://example.test", "{}"),
        )
    result = EVChargerTool(db).nearby(37.5001, 127.0001, radius_m=500)
    assert result["charger_count"] == 1
    assert result["available_charger_count"] == 1


def test_ev_transport_uses_kwh_cost():
    result = estimate_monthly_lifestyle(
        {"monthly_living_cost_manwon": 0}, {"lat": 37.4, "lng": 127.1},
        {
            "use_itemized_budget": True, "transport_mode": "driving",
            "vehicle_powertrain": "ev", "ev_efficiency_km_per_kwh": 5,
            "ev_electricity_krw_per_kwh": 300,
            "destinations": [{"query": "학교", "visits_per_month": 10}],
        },
        _NoRouteMap(),
    )
    # 10km one-way * 2 * 10 visits / 5km/kWh * 300won = 12,000won.
    assert result["breakdown_krw"]["transport"] == 12000
