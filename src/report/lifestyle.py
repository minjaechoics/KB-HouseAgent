"""선택 주택 기준 월 생활비와 이동비를 근거와 함께 계산한다.

금액 입력은 브라우저에서 이해하기 쉬운 원 단위를 사용하고, 자산 모델에는
만원 단위로 전달한다. 목적지별 실경로 호출은 한 리포트 재계산당 최대 5회다.
"""
from __future__ import annotations

from typing import Any


# 가입 경로·요금제·프로모션에 따라 달라질 수 있는 편집 가능한 참고값이다.
SUBSCRIPTION_CATALOG = [
    {"name": "넷플릭스", "monthly_price_krw": 13500},
    {"name": "YouTube Premium", "monthly_price_krw": 14900},
    {"name": "쿠팡 와우", "monthly_price_krw": 7890},
    {"name": "네이버플러스 멤버십", "monthly_price_krw": 4900},
    {"name": "디즈니+", "monthly_price_krw": 9900},
    {"name": "티빙", "monthly_price_krw": 9500},
    {"name": "웨이브", "monthly_price_krw": 7900},
    {"name": "왓챠", "monthly_price_krw": 7900},
    {"name": "멜론", "monthly_price_krw": 7900},
    {"name": "Apple Music", "monthly_price_krw": 8900},
    {"name": "iCloud+", "monthly_price_krw": 1100},
    {"name": "Google One", "monthly_price_krw": 2400},
]


def _money(value: Any) -> float:
    try:
        return max(0.0, float(value or 0))
    except (TypeError, ValueError):
        return 0.0


def _fallback_transit_fare(distance_km: float) -> float:
    """노선 운임이 없을 때만 쓰는 명시적 거리 기반 추정치."""
    return 1500.0 + max(0.0, distance_km - 10.0) * 100.0


def _estimated_taxi_fare(distance_km: float) -> float:
    """사용자 예산용 보수적 택시비 근사값(실제 미터기 요금 아님)."""
    return 4800.0 + max(0.0, distance_km - 1.6) * 1000.0


def estimate_monthly_lifestyle(user: dict, prop: dict, inputs: dict | None,
                               map_tool) -> dict:
    raw = dict(inputs or {})
    destinations = list(raw.get("destinations") or [])[:5]
    mode = str(raw.get("transport_mode") or "transit")
    if mode not in {"transit", "driving"}:
        mode = "transit"
    start = (float(prop.get("lat") or 0), float(prop.get("lng") or 0))
    route_rows: list[dict] = []
    variable_transport = 0.0
    total_roundtrip_km = 0.0

    for destination in destinations:
        query = str(destination.get("query") or "").strip()
        if not query:
            continue
        visits = max(0.0, min(_money(destination.get("visits_per_month")), 62.0))
        geocode = map_tool.geocode(query)
        if not geocode.get("ok"):
            route_rows.append({
                "label": destination.get("label") or query, "query": query,
                "visits_per_month": visits, "resolved": False,
                "reason": geocode.get("reason") or "geocode_failed",
            })
            continue
        goal = (float(geocode["lat"]), float(geocode["lng"]))
        route = map_tool.travel_time(start, goal, mode)
        distance = _money(route.get("distance_km"))
        monthly_roundtrip_km = distance * visits * 2.0
        total_roundtrip_km += monthly_roundtrip_km
        row_cost = 0.0
        fare_source = route.get("source")
        if mode == "transit":
            taxi_ratio = max(0.0, min(_money(raw.get("transit_taxi_ratio_pct")), 100.0)) / 100.0
            transit_fare = route.get("fare_krw")
            fare_estimated = transit_fare is None
            transit_fare = (_fallback_transit_fare(distance)
                            if transit_fare is None else _money(transit_fare))
            taxi_fare = _estimated_taxi_fare(distance)
            row_cost = visits * 2.0 * (
                transit_fare * (1.0 - taxi_ratio) + taxi_fare * taxi_ratio)
            fare_source = ("tmap_fare" if not fare_estimated
                           else "distance_based_transit_fare_estimate")
        variable_transport += row_cost
        route_rows.append({
            "label": destination.get("label") or query, "query": query,
            "category": destination.get("category") or "frequent",
            "visits_per_month": visits, "resolved": True,
            "address": geocode.get("address"), "minutes_one_way": route.get("minutes"),
            "distance_km_one_way": round(distance, 2),
            "monthly_roundtrip_distance_km": round(monthly_roundtrip_km, 1),
            "monthly_cost_krw": round(row_cost), "route_source": route.get("source"),
            "fare_source": fare_source, "estimated": bool(route.get("estimated", False)),
        })

    fixed_transport = _money(raw.get("extra_transport_monthly_krw"))
    if mode == "driving":
        powertrain = str(raw.get("vehicle_powertrain") or "gasoline")
        if powertrain == "ev":
            efficiency = max(1.0, _money(raw.get("ev_efficiency_km_per_kwh")) or 5.5)
            energy_price = _money(raw.get("ev_electricity_krw_per_kwh")) or 320.0
            variable_transport = total_roundtrip_km / efficiency * energy_price
        else:
            efficiency = max(1.0, _money(raw.get("car_fuel_efficiency_km_per_liter")) or 12.0)
            fuel_price = _money(raw.get("car_fuel_price_krw_per_liter")) or 1700.0
            variable_transport = total_roundtrip_km / efficiency * fuel_price
        fixed_transport += sum(_money(raw.get(key)) for key in (
            "car_insurance_monthly_krw", "car_maintenance_monthly_krw",
            "car_parking_toll_monthly_krw",
        ))

    transport = variable_transport + fixed_transport
    food = _money(raw.get("daily_food_krw")) * 30.4
    subscriptions = [
        {"name": str(item.get("name") or "사용자 구독").strip(),
         "monthly_price_krw": round(_money(item.get("monthly_price_krw")))}
        for item in list(raw.get("subscriptions") or [])[:30]
        if _money(item.get("monthly_price_krw")) > 0
    ]
    subscription_total = sum(item["monthly_price_krw"] for item in subscriptions)
    breakdown = {
        "transport": transport,
        "other_insurance": _money(raw.get("other_insurance_monthly_krw")),
        "food": food,
        "subscriptions": subscription_total,
        "telecom": _money(raw.get("telecom_monthly_krw")),
        "internet": _money(raw.get("internet_monthly_krw")),
        "leisure": _money(raw.get("leisure_monthly_krw")),
        "other": _money(raw.get("other_living_monthly_krw")),
    }
    itemized_total = sum(breakdown.values())
    active = bool(raw.get("use_itemized_budget"))
    legacy = _money(user.get("monthly_living_cost_manwon")) * 10000.0
    effective_total = itemized_total if active else legacy
    return {
        "use_itemized_budget": active,
        "transport_mode": mode,
        "vehicle_powertrain": str(raw.get("vehicle_powertrain") or "gasoline"),
        "destinations": route_rows,
        "route_api_call_limit": 5,
        "route_api_calls": len([row for row in route_rows if row.get("resolved")]),
        "monthly_roundtrip_distance_km": round(total_roundtrip_km, 1),
        "subscriptions": subscriptions,
        "catalog": SUBSCRIPTION_CATALOG,
        "catalog_notice": (
            "표시 금액은 편집 가능한 참고값입니다. 가입 경로·요금제·프로모션에 따라 "
            "실제 청구액이 달라지므로 본인 결제내역을 확인하세요."
        ),
        "breakdown_krw": {key: round(value) for key, value in breakdown.items()},
        "itemized_total_krw": round(itemized_total),
        "legacy_monthly_living_cost_krw": round(legacy),
        "effective_monthly_living_cost_krw": round(effective_total),
        "effective_monthly_living_cost_manwon": round(effective_total / 10000.0, 4),
        "calculation_notice": (
            "목적지별 월 방문 횟수는 왕복으로 계산합니다. TMAP 운임이 없을 때의 대중교통비와 "
            "택시비는 거리 기반 예산 추정치이며 실제 결제액이 아닙니다."
        ),
    }
