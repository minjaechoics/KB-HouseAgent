"""Atomic property retrieval for the map-first web experience.

Each enabled condition is executed independently against the property DB.  The
result ID sets are intersected, then the same validated predicates are composed
into the final parameterized query.  This makes per-condition RAG and the final
intersection visible in the debug trace.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

from src import config
from src.scheduling import (
    PortfolioScheduler, compile_condition_graph, condition_resource_limits,
)
from src.tools.map_tool import MapTool


DISPLAY_COLUMNS = [
    "property_id", "listing_id", "is_synthetic", "synthetic_notice",
    "source_type", "source_provider", "source_url", "source_captured_at",
    "source_expires_at", "source_authorized", "last_verified_at",
    "sido", "gugun", "dong", "road_address", "jibun_address",
    "address_detail_public", "region_coordinate_source",
    "coordinate_distribution_method", "lat", "lng",
    "transaction_type", "lease_type", "house_type", "property_type",
    "asking_price_manwon", "sale_price_manwon", "deposit_manwon",
    "monthly_rent_manwon", "maintenance_fee_manwon", "area_m2",
    "room_count", "bathroom_count", "current_floor", "total_floors",
    "building_age_years", "parking_total", "elevator_count", "pet_allowed",
    "subway_walk_minutes", "bus_stop_walk_minutes", "available_from_date",
    "direction", "fraud_score", "guarantee_eligible", "photo_count",
    "advertisement_title", "broker_office_name", "broker_phone",
]

SORT_OPTIONS = {
    "recommended", "risk_asc", "risk_desc", "price_asc", "price_desc",
    "distance_asc",
}


def _price_sort_value(row: dict) -> tuple[float, float]:
    """Transaction-aware price key; monthly rent uses rent then deposit."""
    transaction = row.get("transaction_type")
    if transaction == "매매":
        return (float(row.get("sale_price_manwon") or row.get("asking_price_manwon") or 0), 0.0)
    if transaction == "전세":
        return (float(row.get("deposit_manwon") or 0), 0.0)
    return (
        float(row.get("monthly_rent_manwon") or 0),
        float(row.get("deposit_manwon") or 0),
    )


def _distance_km(row: dict, origin: tuple[float, float]) -> float:
    lat, lng = float(row.get("lat") or 0), float(row.get("lng") or 0)
    lat1, lng1 = map(math.radians, origin)
    lat2, lng2 = math.radians(lat), math.radians(lng)
    value = (math.sin((lat2 - lat1) / 2) ** 2
             + math.cos(lat1) * math.cos(lat2)
             * math.sin((lng2 - lng1) / 2) ** 2)
    return 6371.0088 * 2 * math.asin(min(1.0, math.sqrt(value)))


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    lat1, lng1, lat2, lng2 = map(math.radians, (lat1, lng1, lat2, lng2))
    value = (math.sin((lat2 - lat1) / 2) ** 2
             + math.cos(lat1) * math.cos(lat2) * math.sin((lng2 - lng1) / 2) ** 2)
    return 6371.0088 * 2 * math.asin(min(1.0, math.sqrt(value)))


def _sort_candidates(candidates: list[dict], sort_by: str,
                     origin: tuple[float, float] | None) -> list[dict]:
    if sort_by == "recommended":
        return candidates
    if sort_by.startswith("risk_"):
        reverse = sort_by.endswith("desc")
        present = [row for row in candidates if row.get("fraud_score") is not None]
        missing = [row for row in candidates if row.get("fraud_score") is None]
        present.sort(key=lambda row: float(row["fraud_score"]), reverse=reverse)
        return [*present, *missing]
    if sort_by.startswith("price_"):
        return sorted(
            candidates, key=_price_sort_value,
            reverse=sort_by.endswith("desc"),
        )
    if sort_by == "distance_asc":
        if origin is None:
            raise ValueError("distance_asc requires origin_lat and origin_lng")
        for row in candidates:
            row["distance_km"] = round(_distance_km(row, origin), 3)
        return sorted(candidates, key=lambda row: row["distance_km"])
    raise ValueError(f"unsupported sort_by: {sort_by}")


def _sort_sql(sort_by: str, origin: tuple[float, float] | None) -> tuple[str, list[float]]:
    price = (
        "CASE transaction_type "
        "WHEN '매매' THEN COALESCE(sale_price_manwon, asking_price_manwon) "
        "WHEN '전세' THEN deposit_manwon "
        "ELSE monthly_rent_manwon END"
    )
    if sort_by == "recommended":
        return "listing_updated_at DESC, property_id", []
    if sort_by in {"risk_asc", "risk_desc"}:
        direction = "ASC" if sort_by == "risk_asc" else "DESC"
        return (
            f"CASE WHEN fraud_score IS NULL THEN 1 ELSE 0 END ASC, "
            f"fraud_score {direction}, listing_updated_at DESC, property_id",
            [],
        )
    if sort_by in {"price_asc", "price_desc"}:
        direction = "ASC" if sort_by == "price_asc" else "DESC"
        return (
            f"CASE WHEN {price} IS NULL THEN 1 ELSE 0 END ASC, {price} {direction}, "
            f"CASE WHEN transaction_type='월세' THEN deposit_manwon ELSE 0 END {direction}, "
            "listing_updated_at DESC, property_id",
            [],
        )
    if sort_by == "distance_asc":
        if origin is None:
            raise ValueError("위치 정렬에는 지도 중심 좌표가 필요합니다.")
        lat, lng = origin
        lng_scale = math.cos(math.radians(lat)) ** 2
        return (
            "CASE WHEN lat IS NULL OR lng IS NULL THEN 1 ELSE 0 END ASC, "
            "((lat-?)*(lat-?) + (lng-?)*(lng-?)*?) ASC, property_id",
            [lat, lat, lng, lng, lng_scale],
        )
    raise ValueError(f"unsupported sort_by: {sort_by}")


def _atom_id(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return "c_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _display_value(value: Any) -> str:
    if isinstance(value, list):
        return " · ".join(map(str, value))
    if isinstance(value, float) and value.is_integer():
        return f"{int(value):,}"
    if isinstance(value, (int, float)):
        return f"{value:,}"
    return str(value)


def make_atom(*, field: str, operator: str, value: Any, label: str,
              source: str, extra: dict | None = None) -> dict:
    identity = {"field": field, "operator": operator, "value": value,
                **(extra or {})}
    return {
        "id": _atom_id(identity), "field": field, "operator": operator,
        "value": value, "label": label, "source": source, "enabled": True,
        "query_strategy": ("map_estimate" if field == "commute_minutes"
                           else "parameterized_sql"),
        **(extra or {}),
    }


def atoms_from_profile(profile: dict) -> list[dict]:
    atoms: list[dict] = []
    source = "기초정보"
    transaction_types = profile.get("transaction_types") or []
    if transaction_types:
        atoms.append(make_atom(
            field="transaction_type", operator="in", value=transaction_types,
            label=f"거래 { _display_value(transaction_types) }", source=source))
    house_types = profile.get("house_types") or []
    if house_types:
        atoms.append(make_atom(
            field="house_type", operator="in", value=house_types,
            label=f"주택 { _display_value(house_types) }", source=source))
    if profile.get("preferred_sido"):
        atoms.append(make_atom(
            field="sido", operator="eq", value=profile["preferred_sido"],
            label=f"{profile['preferred_sido']} 지역", source=source))
    if profile.get("preferred_gugun"):
        atoms.append(make_atom(
            field="gugun", operator="eq", value=profile["preferred_gugun"],
            label=f"{profile['preferred_gugun']} 지역", source=source))
    numeric = [
        ("max_deposit_manwon", "deposit_manwon", "lte", "보증금", "만원 이하"),
        ("max_sale_price_manwon", "sale_price_manwon", "lte", "매매가", "만원 이하"),
        ("max_monthly_rent_manwon", "monthly_rent_manwon", "lte", "월세", "만원 이하"),
        ("max_maintenance_manwon", "maintenance_fee_manwon", "lte", "관리비", "만원 이하"),
        ("min_area_m2", "area_m2", "gte", "전용면적", "㎡ 이상"),
    ]
    for input_key, field, operator, title, suffix in numeric:
        value = profile.get(input_key)
        if value is not None:
            atoms.append(make_atom(
                field=field, operator=operator, value=float(value),
                label=f"{title} {_display_value(float(value))}{suffix}", source=source))
    return atoms


def make_initial_scope_atom(profile_atoms: list[dict]) -> dict | None:
    """Collapse every non-empty setup condition into one immutable base scope.

    The component predicates are retained only as SQL metadata.  Search never
    materializes one property-ID set per setup condition; it runs their AND
    expression once and uses that intersection as the universal set for every
    later AI condition.
    """
    components = _dedupe_atoms(profile_atoms)
    if not components:
        return None
    labels = [str(atom.get("label") or "").strip() for atom in components]
    labels = [label for label in labels if label]
    summary = " · ".join(labels)
    identity = {
        "field": "initial_scope",
        "operator": "all",
        "conditions": [
            {key: atom.get(key) for key in ("field", "operator", "value")}
            for atom in components
        ],
    }
    return {
        "id": _atom_id(identity),
        "field": "initial_scope",
        "operator": "all",
        "value": None,
        "label": f"초기 조건 · {summary}",
        "summary": summary,
        "source": "기초정보",
        "enabled": True,
        "locked": True,
        "scope_role": "initial_universe",
        "query_strategy": "single_parameterized_sql_intersection",
        "conditions": components,
        "condition_count": len(components),
    }


def atoms_from_slots(slots: dict, source_text: str, map_tool: MapTool) -> tuple[list[dict], list[str]]:
    atoms: list[dict] = []
    notes: list[str] = []
    source = "AI 대화"
    value = slots.get("transaction_type") or slots.get("lease_type")
    if value:
        atoms.append(make_atom(field="transaction_type", operator="eq", value=value,
                               label=f"{value} 거래", source=source))
    value = slots.get("property_type")
    if value:
        atoms.append(make_atom(field="house_type", operator="contains", value=value,
                               label=f"{value} 유형", source=source))
    if slots.get("region_sido"):
        atoms.append(make_atom(field="sido", operator="eq", value=slots["region_sido"],
                               label=f"{slots['region_sido']} 지역", source=source))
    if slots.get("region_gugun") and not slots.get("region_dong"):
        # region_dong이 있으면 구/군 정보는 항상 그 동에 종속되므로 생략한다.
        # LLM이 짧은 구 이름("팔달구")만 뽑아내면 DB의 실제 값("수원시 팔달구")과
        # 불일치해 검색이 0건이 되므로, 동이 함께 있을 때는 이 atom 자체를 만들지 않는다.
        values = slots["region_gugun"]
        atoms.append(make_atom(field="gugun", operator="in", value=values,
                               label=f"{_display_value(values)} 지역", source=source))
    if slots.get("region_dong"):
        values = slots["region_dong"]
        atoms.append(make_atom(field="dong", operator="in", value=values,
                               label=f"{_display_value(values)} 지역", source=source))

    specs = [
        ("max_deposit_manwon", "deposit_manwon", "lte", "보증금", "만원 이하"),
        ("max_sale_price_manwon", "sale_price_manwon", "lte", "매매가", "만원 이하"),
        ("max_monthly_rent_manwon", "monthly_rent_manwon", "lte", "월세", "만원 이하"),
        ("max_maintenance_manwon", "maintenance_fee_manwon", "lte", "관리비", "만원 이하"),
        ("min_area_m2", "area_m2", "gte", "전용면적", "㎡ 이상"),
        ("max_building_age", "building_age_years", "lte", "건물연식", "년 이하"),
        ("min_safety_score", "safety_score", "gte", "치안점수", " 이상"),
        ("min_convenience_score", "convenience_score", "gte", "편의점수", " 이상"),
        ("max_subway_walk_min", "subway_walk_minutes", "lte", "접근성(지하철)", "분 이내"),
        ("max_facility_walk_min", "mart_walk_minutes", "lte", "편의성(편의시설)", "분 이내"),
        ("min_room_count", "room_count", "gte", "룸개수", "개 이상"),
    ]
    for slot, field, operator, title, suffix in specs:
        value = slots.get(slot)
        if value is not None and field not in {"safety_score", "convenience_score"}:
            atoms.append(make_atom(
                field=field, operator=operator, value=float(value),
                label=f"{title} {_display_value(float(value))}{suffix}", source=source))
    if slots.get("elevator_required"):
        atoms.append(make_atom(field="elevator_count", operator="gt", value=0,
                               label="엘리베이터 있음", source=source))
    if slots.get("pet_allowed_required"):
        atoms.append(make_atom(field="pet_allowed", operator="truthy", value=True,
                               label="반려동물 가능", source=source))
    if slots.get("max_police_distance_min") is not None:
        minutes = float(slots["max_police_distance_min"])
        atoms.append(make_atom(
            field="police_distance_minutes", operator="lte", value=minutes,
            label=f"안전성(경찰서) 도보 {_display_value(minutes)}분 이내", source=source))

    # Additional broker-schema conditions handled deterministically even when the
    # general planner schema does not expose them yet.
    patterns = [
        (r"(?:방|룸)\s*(\d+)\s*개?\s*(?:이상|넘)", "room_count", "gte", "방", "개 이상"),
        (r"지하철(?:역)?\s*(\d+)\s*분\s*(?:이내|이하)", "subway_walk_minutes", "lte", "지하철 도보", "분 이내"),
        (r"관리비\s*(\d+)\s*만?원?\s*(?:이내|이하)", "maintenance_fee_manwon", "lte", "관리비", "만원 이하"),
    ]
    for pattern, field, operator, title, suffix in patterns:
        match = re.search(pattern, source_text)
        if match:
            number = float(match.group(1))
            atoms.append(make_atom(field=field, operator=operator, value=number,
                                   label=f"{title} {_display_value(number)}{suffix}", source=source))
    if re.search(r"반려동물|반려견|반려묘|펫\s*가능", source_text):
        atoms.append(make_atom(field="pet_allowed", operator="truthy", value=True,
                               label="반려동물 가능", source=source))
    if re.search(r"주차\s*(?:가능|필수)", source_text):
        atoms.append(make_atom(field="parking_total", operator="gt", value=0,
                               label="주차 가능", source=source))
    if re.search(r"엘리베이터|승강기", source_text):
        atoms.append(make_atom(field="elevator_count", operator="gt", value=0,
                               label="엘리베이터 있음", source=source))

    minutes = slots.get("max_commute_min")
    landmark = (slots.get("workplace_landmark") or slots.get("_workplace_landmark")
                or _extract_landmark(source_text))
    if minutes is not None and landmark:
        mode = slots.get("commute_mode") or "transit"
        mode_label = {"transit": "대중교통", "walking": "도보", "driving": "자동차"}.get(
            mode, "대중교통"
        )
        resolved = map_tool.geocode(landmark)
        if resolved.get("ok"):
            atoms.append(make_atom(
                field="commute_minutes", operator="lte", value=float(minutes),
                label=f"{landmark} {mode_label} 예상 {_display_value(float(minutes))}분 이내",
                source=source,
                extra={"landmark": landmark, "destination_lat": resolved["lat"],
                       "destination_lng": resolved["lng"], "mode": mode,
                       "geocode_source": resolved.get("source"),
                       "route_provider": map_tool.route_provider(mode),
                       "estimated": not map_tool.has_live_route(mode)},
            ))
            if map_tool.has_live_route(mode):
                notes.append(f"{mode_label} 시간은 {map_tool.route_provider(mode)} 경로 API로 검증합니다.")
            else:
                notes.append(f"{mode_label} 시간은 거리·평균속도 기반 예상치입니다.")
        else:
            notes.append(f"'{landmark}'의 주소를 찾지 못했습니다. 도로명주소를 입력해 주세요.")
    return _dedupe_atoms(atoms), notes


def _extract_landmark(text: str) -> str | None:
    patterns = [
        r"(?:^|[,.]\s*)([^,.]{2,40}?)(?:에서|까지)\s*(?:대중교통|자동차|차|도보|걸어서)?(?:으로)?\s*\d+\s*분",
        r"([^,.]{2,40}?)\s*(?:기준|근처)\s*\d+\s*분",
    ]
    for pattern in patterns:
        match = re.search(pattern, text.strip())
        if match:
            return match.group(1).strip(" 은는이가을를부터")
    return None


def _dedupe_atoms(atoms: Iterable[dict]) -> list[dict]:
    by_group: dict[str, dict] = {}
    for atom in atoms:
        group = atom["field"]
        by_group[group] = atom
    return list(by_group.values())


def _map_price_bucket(row: dict) -> str:
    """지도 후보가 한 가격대에 몰리지 않도록 표시용 가격 구간을 만든다."""
    transaction = row.get("transaction_type")
    if transaction == "월세":
        value = float(row.get("monthly_rent_manwon") or 0)
        bounds = (30, 60, 100, 200)
    else:
        field = "sale_price_manwon" if transaction == "매매" else "deposit_manwon"
        value = float(row.get(field) or 0)
        bounds = (5000, 10000, 20000, 40000, 80000)
    return str(sum(value >= bound for bound in bounds))


def _diverse_map_results(candidates: list[dict], limit: int) -> list[dict]:
    """지역·주택·거래·가격 층을 round-robin으로 뽑아 지도 다양성을 보존한다."""
    if len(candidates) <= limit:
        return candidates
    groups: dict[tuple[str, ...], list[dict]] = {}
    for row in candidates:
        key = (
            str(row.get("sido") or ""), str(row.get("gugun") or ""),
            str(row.get("house_type") or ""), str(row.get("transaction_type") or ""),
            _map_price_bucket(row),
        )
        groups.setdefault(key, []).append(row)
    base_keys = sorted(
        groups,
        key=lambda key: hashlib.sha1("|".join(key).encode("utf-8")).hexdigest(),
    )
    # 첫 화면의 120건만으로도 17개 시도가 모두 보이도록 시도별 키를 교차 배치한다.
    keys_by_sido: dict[str, list[tuple[str, ...]]] = {}
    for key in base_keys:
        keys_by_sido.setdefault(key[0], []).append(key)
    sido_order = sorted(keys_by_sido)
    keys = [
        keys_by_sido[sido][depth]
        for depth in range(max(map(len, keys_by_sido.values())))
        for sido in sido_order
        if depth < len(keys_by_sido[sido])
    ]
    selected: list[dict] = []
    depth = 0
    while len(selected) < limit:
        added = False
        for key in keys:
            rows = groups[key]
            if depth < len(rows):
                selected.append(rows[depth])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        depth += 1
    return selected


class AtomicPropertySearch:
    def __init__(self, db_path: Path = config.DB_PATH, map_tool: MapTool | None = None,
                 safety_tool: "SafetyTool | None" = None):
        self.db_path = db_path
        self.map_tool = map_tool or MapTool()
        if safety_tool is None:
            from src.tools.safety_tool import SafetyTool
            safety_tool = SafetyTool()
        self.safety_tool = safety_tool
        self.scheduler = PortfolioScheduler(
            deadline_ms=max(
                10, int(getattr(config, "AGENT_SCHEDULER_DEADLINE_MS", 60))
            )
        )

    def _conn(self):
        uri = self.db_path.resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        return connection

    def regions(self) -> list[dict]:
        with self._conn() as connection:
            rows = connection.execute(
                "SELECT sido, gugun, COUNT(*) AS count FROM properties "
                "GROUP BY sido, gugun ORDER BY sido, gugun"
            ).fetchall()
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            grouped.setdefault(row["sido"], []).append(
                {"name": row["gugun"], "count": row["count"]})
        return [{"name": sido, "gugun": districts,
                 "count": sum(item["count"] for item in districts)}
                for sido, districts in grouped.items()]

    def search(self, atoms: list[dict], enabled_ids: set[str] | None = None,
               limit: int = 120, sort_by: str = "recommended",
               origin_lat: float | None = None,
               origin_lng: float | None = None,
               budget_caps: dict[str, dict[str, float]] | None = None) -> dict:
        enabled = [atom for atom in atoms
                   if atom.get("enabled", True)
                   and (atom.get("scope_role") == "initial_universe"
                        or enabled_ids is None or atom["id"] in enabled_ids)]
        limit = max(1, min(int(limit), 300))
        if sort_by not in SORT_OPTIONS:
            raise ValueError(f"unsupported sort_by: {sort_by}")
        origin = None
        if origin_lat is not None and origin_lng is not None:
            origin = (float(origin_lat), float(origin_lng))
            if not (-90 <= origin[0] <= 90 and -180 <= origin[1] <= 180):
                raise ValueError("invalid sort origin coordinates")
        order_sql, order_params = _sort_sql(sort_by, origin)
        traces: list[dict] = []
        sql_atoms: list[tuple[dict, str, list[Any]]] = []
        initial_atom = next(
            (atom for atom in enabled
             if atom.get("scope_role") == "initial_universe"
             or atom.get("field") == "initial_scope"),
            None,
        )
        additional_atoms = [atom for atom in enabled if atom is not initial_atom]
        commute_atoms = [atom for atom in additional_atoms
                         if atom.get("field") == "commute_minutes"]
        police_atoms = [atom for atom in additional_atoms
                        if atom.get("field") == "police_distance_minutes"]
        additional_sql_atoms = [atom for atom in additional_atoms
                                if atom.get("field") not in
                                ("commute_minutes", "police_distance_minutes")]
        route_api_calls_used: dict[str, int] = {"transit": 0, "driving": 0}
        schedule_candidate_limit = max(
            1, config.NAVER_DIRECTIONS_EXACT_CANDIDATE_LIMIT)
        unlimited_route_modes = {
            mode for mode in ("transit", "driving")
            if config.exact_route_candidate_limit(mode) is None
        }
        schedule_graph = compile_condition_graph(
            enabled,
            route_candidate_limit=schedule_candidate_limit,
            unlimited_route_modes=unlimited_route_modes,
        )
        schedule_limits = condition_resource_limits(
            sqlite_capacity=getattr(config, "CONDITION_SQL_MAX_WORKERS", 2),
            route_capacity=config.ROUTE_API_MAX_WORKERS,
        )
        execution_schedule = self.scheduler.schedule(
            schedule_graph, schedule_limits
        )
        scheduled_by_id = execution_schedule.task_map()

        def scheduled_start(atom):
            scheduled = scheduled_by_id.get(f"sql:{atom['id']}")
            return scheduled.start_ms if scheduled is not None else 10**12

        additional_sql_atoms.sort(
            key=lambda atom: (scheduled_start(atom), atom["id"])
        )

        with self._conn() as connection:
            base_components = list((initial_atom or {}).get("conditions") or [])
            base_sql_atoms: list[tuple[dict, str, list[Any]]] = []
            for component in base_components:
                if component.get("field") == "commute_minutes":
                    raise ValueError("initial scope cannot contain a live commute condition")
                clause, component_params = self._clause(component)
                base_sql_atoms.append((component, clause, component_params))
            sql_atoms.extend(base_sql_atoms)
            base_where = " AND ".join(
                f"({clause})" for _, clause, _ in base_sql_atoms
            ) or "1=1"
            base_where = f"({base_where}) AND (listing_status IS NULL OR listing_status!='expired')"
            base_params = [value for _, _, values in base_sql_atoms for value in values]
            if budget_caps:
                # "구매가능" 토글: 자기자금+최대 대출가능액 예산을 거래유형별로
                # 다른 컬럼과 비교해야 하므로 단일 atom으로 표현할 수 없다
                # (_sort_sql의 거래유형별 CASE 가격식과 동일한 패턴).
                sale_cap = budget_caps.get("매매", {}).get("sale_price_manwon")
                jeonse_cap = budget_caps.get("전세", {}).get("deposit_manwon")
                rent_cap = budget_caps.get("월세", {}).get("monthly_rent_manwon")
                rent_deposit_cap = budget_caps.get("월세", {}).get("deposit_manwon")
                budget_clause = (
                    "CASE transaction_type "
                    "WHEN '매매' THEN COALESCE(sale_price_manwon, asking_price_manwon, 0) <= ? "
                    "WHEN '전세' THEN deposit_manwon <= ? "
                    "ELSE (monthly_rent_manwon <= ? AND deposit_manwon <= ?) END"
                )
                budget_params = [
                    float(sale_cap) if sale_cap is not None else 0.0,
                    float(jeonse_cap) if jeonse_cap is not None else 0.0,
                    float(rent_cap) if rent_cap is not None else 0.0,
                    float(rent_deposit_cap) if rent_deposit_cap is not None else 0.0,
                ]
                base_where = f"({base_where}) AND ({budget_clause})"
                base_params = [*base_params, *budget_params]
                # base_where는 intersection 계산(초기/원자별 스코프 쿼리)에만 쓰이고
                # commute/police atom이 없을 때의 최종 SELECT는 sql_atoms만 읽으므로,
                # 예산 조건도 sql_atoms에 별도로 넣어야 최종 결과에 실제로 반영된다.
                sql_atoms.append((
                    {"field": "_budget_cap", "id": "_budget_cap"},
                    budget_clause, budget_params,
                ))

            intersection: set[str] | None = None
            if initial_atom:
                initial_sql = (
                    f"SELECT property_id FROM properties WHERE {base_where}"
                )
                initial_ids = {
                    row[0] for row in connection.execute(initial_sql, base_params)
                }
                intersection = initial_ids
                traces.append({
                    "atom_id": initial_atom["id"],
                    "label": initial_atom["label"],
                    "scope_role": "initial_universe",
                    "strategy": "single_parameterized_sql_intersection",
                    "sql": initial_sql,
                    "parameters": base_params,
                    "component_conditions": [
                        {"label": component.get("label"),
                         "field": component.get("field"),
                         "operator": component.get("operator"),
                         "value": component.get("value")}
                        for component in base_components
                    ],
                    "component_count": len(base_components),
                    "standalone_match_count": len(initial_ids),
                    "intersection_count_after": len(initial_ids),
                })
                initial_universe_count = len(initial_ids)
            else:
                initial_universe_count = int(connection.execute(
                    "SELECT COUNT(*) FROM properties"
                ).fetchone()[0])

            # Every AI SQL predicate is evaluated with the initial WHERE clause
            # physically present in the query. Independent atoms are dispatched
            # according to the portfolio schedule and intersected only after all
            # read-only queries finish.
            def execute_sql_atom(atom):
                clause, atom_params = self._clause(atom)
                scoped_where = f"({base_where}) AND ({clause})"
                scoped_params = [*base_params, *atom_params]
                sql = f"SELECT property_id FROM properties WHERE {scoped_where}"
                with self._conn() as atom_connection:
                    ids = {
                        row[0] for row in atom_connection.execute(
                            sql, scoped_params
                        )
                    }
                return atom, clause, atom_params, scoped_params, sql, ids

            sql_workers = min(
                schedule_limits.get("sqlite").capacity,
                len(additional_sql_atoms),
            )
            if sql_workers > 1:
                with ThreadPoolExecutor(
                    max_workers=sql_workers,
                    thread_name_prefix="condition-sql",
                ) as executor:
                    sql_results = list(
                        executor.map(execute_sql_atom, additional_sql_atoms)
                    )
            else:
                sql_results = [
                    execute_sql_atom(atom) for atom in additional_sql_atoms
                ]

            for (
                atom, clause, atom_params, scoped_params, sql, ids
            ) in sql_results:
                intersection = ids if intersection is None else intersection & ids
                sql_atoms.append((atom, clause, atom_params))
                traces.append({
                    "atom_id": atom["id"], "label": atom["label"],
                    "scope_role": "llm_refinement",
                    "universal_set": "initial_scope_intersection",
                    "universal_set_count": initial_universe_count,
                    "strategy": "parameterized_sql_with_initial_scope",
                    "sql": sql, "parameters": scoped_params,
                    "standalone_match_count": len(ids),
                    "intersection_count_after": len(intersection),
                })

            for atom in commute_atoms:
                destination = (float(atom["destination_lat"]),
                               float(atom["destination_lng"]))
                mode = atom.get("mode", "transit")
                route_api_limit = config.exact_route_candidate_limit(mode)
                max_minutes = float(atom["value"])
                live_route = self.map_tool.has_live_route(mode)
                if live_route:
                    # A broad spatial prefilter avoids excluding fast rail/road
                    # routes before the provider has evaluated them.
                    envelope_speeds = {"transit": 60.0, "driving": 90.0}
                    max_straight_km = max_minutes / 60.0 * envelope_speeds.get(mode, 30.0)
                else:
                    speeds = {"transit": 18.0, "walking": 4.5,
                              "driving": 22.0, "bicycling": 12.0}
                    usable_minutes = max(
                        0.0, max_minutes - (8.0 if mode == "transit" else 0.0))
                    road_factor = 1.25 if mode == "walking" else 1.30
                    max_straight_km = (usable_minutes / 60.0
                                       * speeds.get(mode, 18.0) / road_factor)
                lat_delta = max_straight_km / 110.574
                lng_scale = max(0.2, math.cos(math.radians(destination[0])))
                lng_delta = max_straight_km / (111.320 * lng_scale)
                rows = connection.execute(
                    "SELECT property_id, lat, lng FROM properties "
                    "WHERE lat BETWEEN ? AND ? AND lng BETWEEN ? AND ? "
                    f"AND ({base_where})",
                    [destination[0] - lat_delta, destination[0] + lat_delta,
                     destination[1] - lng_delta, destination[1] + lng_delta,
                     *base_params],
                ).fetchall()
                evaluation_rows = list(rows)
                if intersection is not None:
                    evaluation_rows = [row for row in evaluation_rows
                                       if row["property_id"] in intersection]
                evaluation_rows.sort(key=lambda row: math.hypot(
                    float(row["lat"]) - destination[0],
                    (float(row["lng"]) - destination[1])
                    * math.cos(math.radians(destination[0]))))
                candidate_limit_reached = False
                calls_used = route_api_calls_used.get(mode, 0)
                budget_before = (None if route_api_limit is None else
                                 max(0, route_api_limit - calls_used))
                if live_route:
                    if budget_before is not None:
                        candidate_limit_reached = len(evaluation_rows) > budget_before
                        evaluation_rows = evaluation_rows[:budget_before]
                    route_api_calls_used[mode] = calls_used + len(evaluation_rows)

                def evaluate_route(row):
                    travel = (self.map_tool.travel_time if live_route else
                              self.map_tool.estimate_travel_time)(
                                  (float(row["lat"]), float(row["lng"])),
                                  destination, mode)
                    return row, travel

                if live_route and len(evaluation_rows) > 1:
                    with ThreadPoolExecutor(
                            max_workers=min(config.ROUTE_API_MAX_WORKERS,
                                            len(evaluation_rows))) as executor:
                        evaluated_routes = executor.map(evaluate_route, evaluation_rows)
                        evaluated_routes = list(evaluated_routes)
                else:
                    evaluated_routes = [evaluate_route(row)
                                        for row in evaluation_rows]

                commute_ids = set()
                source_counts: dict[str, int] = {}
                fallback_count = 0
                for row, travel in evaluated_routes:
                    source = str(travel.get("source") or "unknown")
                    source_counts[source] = source_counts.get(source, 0) + 1
                    fallback_count += int(bool(travel.get("fallback_reason")))
                    if (travel.get("route_found", True)
                            and travel["minutes"] <= max_minutes):
                        commute_ids.add(row["property_id"])
                intersection = (commute_ids if intersection is None
                                else intersection & commute_ids)
                traces.append({
                    "atom_id": atom["id"], "label": atom["label"],
                    "scope_role": "llm_refinement",
                    "universal_set": "initial_scope_intersection",
                    "universal_set_count": initial_universe_count,
                    "strategy": ("indexed_bounding_box_then_live_route_api"
                                 if live_route else
                                 "map_bounding_box_haversine_estimate"),
                    "tool": ("MapTool.travel_time" if live_route else
                             "MapTool.estimate_travel_time"), "mode": mode,
                    "provider": self.map_tool.route_provider(mode),
                    "destination": {"lat": destination[0], "lng": destination[1],
                                    "landmark": atom.get("landmark")},
                    "bounding_box_candidate_count": len(rows),
                    "route_evaluated_candidate_count": len(evaluation_rows),
                    "candidate_limit": (route_api_limit if live_route else None),
                    "candidate_limit_policy": (
                        "unlimited_premium_tmap"
                        if live_route and route_api_limit is None
                        else "configured_cap" if live_route else "not_applicable"
                    ),
                    "candidate_limit_reached": candidate_limit_reached,
                    "provider_source_counts": source_counts,
                    "provider_fallback_count": fallback_count,
                    "standalone_match_count": len(commute_ids),
                    "intersection_count_after": len(intersection),
                    "estimated": not live_route or fallback_count > 0,
                })

            if police_atoms:
                # 경찰서는 목적지가 하나가 아니라 여러 곳 중 최소거리를 구하는
                # 문제라 commute_minutes와 반대 방향이다. 실시간 도보 경로 API가
                # 없으므로 하버사인 거리 기반 추정치만 계산한다(네트워크 호출 없음).
                station_df = self.safety_tool._load("police")
                station_coords: list[tuple[float, float]] = []
                if station_df is not None and {"lat", "lng"} <= set(station_df.columns):
                    station_coords = [
                        (float(lat), float(lng)) for lat, lng in zip(
                            station_df["lat"].tolist(), station_df["lng"].tolist())
                    ]
                candidate_rows = connection.execute(
                    f"SELECT property_id, lat, lng FROM properties WHERE ({base_where})",
                    base_params,
                ).fetchall()
                if intersection is not None:
                    candidate_rows = [row for row in candidate_rows
                                      if row["property_id"] in intersection]
                walk_speed_kmh, road_factor = 4.5, 1.25

                for atom in police_atoms:
                    max_minutes = float(atom["value"])
                    police_ids: set[str] = set()
                    for row in candidate_rows:
                        if not station_coords:
                            break
                        lat1, lng1 = float(row["lat"]), float(row["lng"])
                        nearest_km = min(
                            _haversine_km(lat1, lng1, slat, slng)
                            for slat, slng in station_coords
                        )
                        minutes = nearest_km / walk_speed_kmh * 60.0 * road_factor
                        if minutes <= max_minutes:
                            police_ids.add(row["property_id"])
                    intersection = (police_ids if intersection is None
                                    else intersection & police_ids)
                    traces.append({
                        "atom_id": atom["id"], "label": atom["label"],
                        "scope_role": "llm_refinement",
                        "universal_set": "initial_scope_intersection",
                        "universal_set_count": initial_universe_count,
                        "strategy": "estimated_haversine_nearest_of_many_walking",
                        "tool": "nearest_station_haversine_estimate",
                        "station_category": "police",
                        "station_count": len(station_coords),
                        "candidate_count": len(candidate_rows),
                        "standalone_match_count": len(police_ids),
                        "intersection_count_after": len(intersection),
                        "estimated": True,
                    })

            where = " AND ".join(f"({clause})" for _, clause, _ in sql_atoms) or "1=1"
            params = [value for _, _, values in sql_atoms for value in values]
            select_sql = (
                "SELECT " + ", ".join(DISPLAY_COLUMNS)
                + f" FROM properties WHERE {where} "
                + f"ORDER BY {order_sql} LIMIT ?"
            )
            multiplier = 20 if sort_by == "recommended" else (8 if sort_by == "distance_asc" else 3)
            fetch_limit = min(20000, max(limit * multiplier, 600))
            if commute_atoms or police_atoms:
                candidates = []
                selected_ids = sorted(intersection or ())[:fetch_limit]
                for offset in range(0, len(selected_ids), 800):
                    chunk = selected_ids[offset:offset + 800]
                    placeholders = ",".join("?" for _ in chunk)
                    rows = connection.execute(
                        "SELECT " + ", ".join(DISPLAY_COLUMNS)
                        + f" FROM properties WHERE property_id IN ({placeholders})",
                        chunk,
                    )
                    candidates.extend(dict(row) for row in rows)
            else:
                candidates = [dict(row) for row in connection.execute(
                    select_sql, [*params, *order_params, fetch_limit])]
            if sort_by == "recommended":
                results = _diverse_map_results(candidates, limit)
            else:
                results = _sort_candidates(candidates, sort_by, origin)[:limit]
            total = (len(intersection) if intersection is not None else int(
                connection.execute(
                    f"SELECT COUNT(*) FROM properties WHERE {where}", params
                ).fetchone()[0]
            ))

        enabled_atoms = []
        trace_by_id = {trace["atom_id"]: trace for trace in traces}
        for atom in enabled:
            enriched = dict(atom)
            enriched["match_count"] = trace_by_id.get(atom["id"], {}).get(
                "standalone_match_count", 0)
            enabled_atoms.append(enriched)
        return {
            "total": total, "returned": len(results), "properties": results,
            "enabled_atoms": enabled_atoms,
            "trace": {
                "pipeline": "initial_scope_intersection_then_llm_refinement",
                "initial_universe": {
                    "filter_id": initial_atom.get("id") if initial_atom else None,
                    "label": initial_atom.get("label") if initial_atom else "전체 매물",
                    "component_count": len(base_components),
                    "intersection_count": initial_universe_count,
                    "materialization": "single_combined_sql_query",
                },
                "llm_scope_policy": (
                    "all additional conditions are evaluated inside the initial "
                    "intersection and may only narrow it"
                ),
                "per_condition": traces,
                "final_intersection_count": total,
                "final_sql": (select_sql if not commute_atoms else
                              "SELECT <display columns> FROM properties "
                              "WHERE property_id IN (<validated commute intersection ids>)"),
                "final_parameters": ([*params, *order_params, fetch_limit] if not commute_atoms else
                                     {"validated_id_count": len(intersection or ())}),
                "sort": {
                    "sort_by": sort_by,
                    "origin": ({"lat": origin[0], "lng": origin[1]} if origin else None),
                    "risk_is_filter": False,
                },
                "display_sampling": (
                    "region_house_transaction_price_round_robin"
                    if sort_by == "recommended" else "explicit_global_sort"
                ),
                "display_candidate_pool": len(candidates),
                "route_api_call_budget": {
                    "max_per_search": (
                        None if any(
                            config.exact_route_candidate_limit(
                                atom.get("mode", "transit")) is None
                            for atom in commute_atoms)
                        else min(
                            config.exact_route_candidate_limit(
                                atom.get("mode", "transit"))
                            for atom in commute_atoms)
                        if commute_atoms else None
                    ),
                    "policy": (
                        "live_routes_unlimited" if any(
                            config.exact_route_candidate_limit(
                                atom.get("mode", "transit")) is None
                            for atom in commute_atoms)
                        else "provider_specific_cap"
                    ),
                    "used": sum(route_api_calls_used.values()),
                    "used_by_mode": route_api_calls_used,
                    "limits_by_mode": {
                        "transit": config.exact_route_candidate_limit("transit"),
                        "driving": config.exact_route_candidate_limit("driving"),
                    },
                },
                "map_time_policy": (
                    "indexed bounding-box prefilter; driving is validated by NAVER "
                    "Directions 5 and transit by TMAP when configured; provider errors "
                    "fall back to labelled haversine estimates"
                ),
                "condition_scheduler": execution_schedule.to_dict(),
            },
        }

    @staticmethod
    def _clause(atom: dict) -> tuple[str, list[Any]]:
        field = atom.get("field")
        operator = atom.get("operator")
        value = atom.get("value")
        allowed = {
            "transaction_type", "lease_type", "house_type", "sido", "gugun", "dong",
            "deposit_manwon", "monthly_rent_manwon", "maintenance_fee_manwon",
            "area_m2", "building_age_years", "room_count",
            "subway_walk_minutes", "mart_walk_minutes",
            "parking_total", "elevator_count", "pet_allowed",
        }
        if field == "sale_price_manwon":
            expression = "COALESCE(sale_price_manwon, asking_price_manwon, 0)"
        elif field in allowed:
            expression = field
        else:
            raise ValueError(f"unsupported condition field: {field}")

        if operator == "eq":
            return f"{expression} = ?", [value]
        if operator == "contains" and field == "house_type":
            return "house_type LIKE ?", [f"%{value}%"]
        if operator == "in" and isinstance(value, list) and value:
            marks = ",".join("?" for _ in value)
            return f"{expression} IN ({marks})", list(value)
        if operator == "lte":
            return f"{expression} <= ?", [float(value)]
        if operator == "gte":
            return f"{expression} >= ?", [float(value)]
        if operator == "gt":
            return f"{expression} > ?", [float(value)]
        if operator == "truthy":
            return f"LOWER(CAST({expression} AS TEXT)) IN ('1','true','y','yes','가능')", []
        raise ValueError(f"unsupported condition operator: {operator}")


def merge_atoms(existing: list[dict], incoming: list[dict]) -> list[dict]:
    """New conditions replace prior constraints on the same broker-schema field."""
    incoming_fields = {atom["field"] for atom in incoming}
    kept = [atom for atom in existing if atom["field"] not in incoming_fields]
    return _dedupe_atoms([*kept, *incoming])
