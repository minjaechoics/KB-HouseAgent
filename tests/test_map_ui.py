"""Map-first UI and atomic intersection regression tests."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

from src.server.property_search import (
    AtomicPropertySearch, atoms_from_profile, atoms_from_slots, make_atom,
    make_initial_scope_atom, merge_atoms,
)
from src.tools.map_tool import (
    DIRECTIONS_URL, TMAP_TRANSIT_SUMMARY_URL, MapTool,
)


class OfflineMap(MapTool):
    def __init__(self):
        self.client_id = ""
        self.client_secret = ""
        self.online = False
        self.timeout_seconds = 1


def test_directions_uses_current_maps_gateway_and_transit_is_estimated():
    assert DIRECTIONS_URL.startswith("https://maps.apigw.ntruss.com/")
    result = OfflineMap().travel_time(
        (37.28, 127.04), (37.282943, 127.043824), "transit")
    assert result["estimated"] is True
    assert result["source"] == "estimated_haversine_transit"


def test_driving_uses_naver_directions5_when_configured():
    tool = MapTool()
    tool.client_id = "naver-id"
    tool.client_secret = "naver-secret"
    tool.online = True
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "route": {"trafast": [{"summary": {
            "duration": 12 * 60 * 1000, "distance": 6400,
        }}]}}
    with patch("src.tools.map_tool.requests.get", return_value=response) as call:
        result = tool.travel_time((37.2, 127.0), (37.28, 127.04), "driving")
    assert result["source"] == "naver_directions5"
    assert result["minutes"] == 12.0
    assert call.call_args.args[0] == DIRECTIONS_URL


def test_transit_uses_tmap_summary_route_when_configured():
    tool = MapTool()
    tool.tmap_app_key = "tmap-app-key"
    tool.tmap_online = True
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"metaData": {"plan": {"itineraries": [{
        "totalTime": 590, "totalDistance": 3200, "totalWalkTime": 120,
        "totalWalkDistance": 140, "transferCount": 1, "pathType": 3,
        "fare": {"regular": {"totalFare": 1500}},
    }, {"totalTime": 720, "totalDistance": 3000}]}}}
    with patch("src.tools.map_tool.requests.post", return_value=response) as call:
        result = tool.travel_time((37.2, 127.0), (37.28, 127.04), "transit")
    assert result["source"] == "tmap_transit"
    assert result["estimated"] is False
    assert result["minutes"] == 9.8
    assert result["transfer_count"] == 1
    assert result["fare_krw"] == 1500.0
    assert call.call_args.args[0] == TMAP_TRANSIT_SUMMARY_URL
    assert call.call_args.kwargs["headers"]["appKey"] == "tmap-app-key"
    assert call.call_args.kwargs["json"]["startX"] == "127.0"


def test_tmap_failure_is_labelled_estimate_fallback():
    tool = MapTool()
    tool.tmap_app_key = "tmap-app-key"
    tool.tmap_online = True
    with patch("src.tools.map_tool.requests.post",
               side_effect=RuntimeError("temporary provider failure")):
        result = tool.travel_time((37.2, 127.0), (37.28, 127.04), "transit")
    assert result["estimated"] is True
    assert result["source"] == "estimated_haversine_transit"
    assert result["attempted_provider"] == "tmap_transit"
    assert result["fallback_reason"] == "RuntimeError"


def test_tmap_no_route_is_not_mislabelled_as_provider_failure():
    tool = MapTool()
    tool.tmap_app_key = "tmap-app-key"
    tool.tmap_online = True
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"metaData": {"plan": {"itineraries": []}}}
    with patch("src.tools.map_tool.requests.post", return_value=response):
        result = tool.travel_time((37.2, 127.0), (37.28, 127.04), "transit")
    assert result["source"] == "tmap_transit_no_route"
    assert result["route_found"] is False
    assert result["estimated"] is False
    assert "fallback_reason" not in result


def test_tmap_transient_429_is_retried_before_fallback():
    tool = MapTool()
    tool.tmap_app_key = "tmap-app-key"
    tool.tmap_online = True
    throttled = Mock(status_code=429, headers={"Retry-After": "0"})
    success = Mock(status_code=200, headers={})
    success.raise_for_status.return_value = None
    success.json.return_value = {"metaData": {"plan": {"itineraries": [{
        "totalTime": 600, "totalDistance": 3000,
    }]}}}
    with patch("src.tools.map_tool.requests.post",
               side_effect=[throttled, success]) as call:
        with patch("src.tools.map_tool.time.sleep"):
            result = tool.travel_time(
                (37.2, 127.0), (37.28, 127.04), "transit")
    assert call.call_count == 2
    assert result["source"] == "tmap_transit"
    assert result["estimated"] is False


def test_natural_language_landmark_becomes_atomic_commute_condition():
    atoms, notes = atoms_from_slots(
        {"max_commute_min": 20},
        "아주대학교에서 대중교통으로 20분 거리 이내였으면 좋겠어",
        OfflineMap(),
    )
    commute = next(atom for atom in atoms if atom["field"] == "commute_minutes")
    assert commute["value"] == 20
    assert commute["landmark"] == "아주대학교"
    assert commute["estimated"] is True
    assert notes


def test_initial_conditions_are_one_combined_intersection_scope():
    profile_atoms = atoms_from_profile({
        "preferred_sido": "경기",
        "preferred_gugun": "수원시 팔달구",
        "transaction_types": ["전세", "월세"],
        "house_types": ["아파트", "오피스텔"],
    })
    initial = make_initial_scope_atom(profile_atoms)
    assert initial is not None
    atoms = [initial]
    result = AtomicPropertySearch(map_tool=OfflineMap()).search(
        atoms, {atom["id"] for atom in atoms}, limit=10)

    assert result["total"] > 0
    assert result["returned"] <= 10
    assert all(
        row["sido"] == "경기" and row["gugun"] == "수원시 팔달구"
        for row in result["properties"]
    )
    trace = result["trace"]
    assert trace["pipeline"] == "initial_scope_intersection_then_llm_refinement"
    assert len(trace["per_condition"]) == 1
    initial_trace = trace["per_condition"][0]
    assert initial_trace["strategy"] == "single_parameterized_sql_intersection"
    assert initial_trace["component_count"] == len(profile_atoms)
    assert " AND " in initial_trace["sql"]
    assert trace["initial_universe"]["intersection_count"] == result["total"]


def test_llm_condition_query_is_scoped_inside_initial_intersection():
    initial = make_initial_scope_atom(atoms_from_profile({
        "preferred_sido": "대전", "transaction_types": ["전세", "월세"],
    }))
    added = make_atom(
        field="house_type", operator="eq", value="아파트",
        label="아파트 유형", source="AI 대화",
    )
    result = AtomicPropertySearch(map_tool=OfflineMap()).search(
        [initial, added], {initial["id"], added["id"]}, limit=10,
    )
    assert result["total"] <= result["trace"]["initial_universe"]["intersection_count"]
    added_trace = next(item for item in result["trace"]["per_condition"]
                       if item["atom_id"] == added["id"])
    assert added_trace["universal_set"] == "initial_scope_intersection"
    assert "sido = ?" in added_trace["sql"]
    assert "house_type = ?" in added_trace["sql"]
    assert all(row["sido"] == "대전" and row["house_type"] == "아파트"
               for row in result["properties"])


def test_llm_condition_cannot_replace_or_broaden_initial_scope():
    initial = make_initial_scope_atom(atoms_from_profile({
        "preferred_sido": "대전", "transaction_types": ["전세"],
    }))
    conflicting = make_atom(
        field="sido", operator="eq", value="서울",
        label="서울 지역", source="AI 대화",
    )
    merged = merge_atoms([initial], [conflicting])
    assert [atom["scope_role"] for atom in merged if atom.get("scope_role")] == [
        "initial_universe"
    ]
    assert len(merged) == 2
    result = AtomicPropertySearch(map_tool=OfflineMap()).search(
        merged, {atom["id"] for atom in merged}, limit=10,
    )
    assert result["total"] == 0
    added_trace = next(item for item in result["trace"]["per_condition"]
                       if item["atom_id"] == conflicting["id"])
    assert added_trace["parameters"][-2:] == ["대전", "서울"]


class CountingLiveMap(OfflineMap):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def has_live_route(self, mode: str) -> bool:
        return mode == "transit"

    def route_provider(self, mode: str) -> str:
        return "counting_tmap" if mode == "transit" else "estimate"

    def travel_time(self, origin, destination, mode="transit"):
        self.calls += 1
        return {"minutes": 10.0, "source": "counting_tmap",
                "estimated": False, "route_found": True}


def test_premium_tmap_transit_has_no_candidate_count_cap():
    tool = CountingLiveMap()
    commute = make_atom(
        field="commute_minutes", operator="lte", value=60,
        label="수원역 대중교통 60분 이내", source="AI 대화",
        extra={"destination_lat": 37.266, "destination_lng": 127.000,
               "mode": "transit", "landmark": "수원역"},
    )
    result = AtomicPropertySearch(map_tool=tool).search(
        [commute], {commute["id"]}, limit=10,
    )
    route_trace = next(item for item in result["trace"]["per_condition"]
                       if item["atom_id"] == commute["id"])
    assert tool.calls > 5
    assert tool.calls == route_trace["route_evaluated_candidate_count"]
    assert result["trace"]["route_api_call_budget"]["max_per_search"] is None
    assert result["trace"]["route_api_call_budget"]["policy"] == "live_routes_unlimited"
    assert result["trace"]["route_api_call_budget"]["used"] == tool.calls


class CountingDrivingMap(CountingLiveMap):
    def has_live_route(self, mode: str) -> bool:
        return mode == "driving"

    def route_provider(self, mode: str) -> str:
        return "counting_naver" if mode == "driving" else "estimate"


def test_naver_driving_has_no_candidate_count_cap():
    tool = CountingDrivingMap()
    commute = make_atom(
        field="commute_minutes", operator="lte", value=60,
        label="수원역 자동차 60분 이내", source="AI 대화",
        extra={"destination_lat": 37.266, "destination_lng": 127.000,
               "mode": "driving", "landmark": "수원역"},
    )
    result = AtomicPropertySearch(map_tool=tool).search(
        [commute], {commute["id"]}, limit=10,
    )
    budget = result["trace"]["route_api_call_budget"]
    route_trace = next(item for item in result["trace"]["per_condition"]
                       if item["atom_id"] == commute["id"])
    assert tool.calls > 5
    assert tool.calls == route_trace["route_evaluated_candidate_count"]
    assert budget["max_per_search"] is None
    assert budget["limits_by_mode"]["driving"] is None
    assert budget["policy"] == "live_routes_unlimited"


def test_blank_profile_does_not_create_any_filters():
    assert atoms_from_profile({
        "preferred_sido": None, "preferred_gugun": None,
        "transaction_types": [], "house_types": [],
        "max_deposit_manwon": None, "max_monthly_rent_manwon": None,
    }) == []


def test_risk_request_never_creates_a_filter_atom():
    atoms, _ = atoms_from_slots(
        {"max_fraud_score": 0.01, "safety_is_hard": True,
         "sort_by": "risk_asc"},
        "위험도가 낮은 집", OfflineMap(),
    )
    assert all(atom.get("field") != "fraud_score" for atom in atoms)


def test_risk_sort_keeps_same_universe_and_orders_scored_jeonse():
    initial = make_initial_scope_atom(atoms_from_profile({
        "preferred_sido": "대전", "transaction_types": ["전세"],
    }))
    search = AtomicPropertySearch(map_tool=OfflineMap())
    baseline = search.search([initial], {initial["id"]}, limit=30)
    sorted_result = search.search(
        [initial], {initial["id"]}, limit=30, sort_by="risk_asc",
    )
    scores = [row["fraud_score"] for row in sorted_result["properties"]
              if row["fraud_score"] is not None]
    assert sorted_result["total"] == baseline["total"]
    assert scores == sorted(scores)
    assert "fraud_score" not in sorted_result["trace"]["final_sql"].split(
        "WHERE", 1)[1].split("ORDER BY", 1)[0]
    assert sorted_result["trace"]["sort"]["risk_is_filter"] is False


def test_transaction_aware_price_and_map_center_distance_sorting():
    initial = make_initial_scope_atom(atoms_from_profile({
        "preferred_sido": "대전", "transaction_types": ["월세"],
    }))
    search = AtomicPropertySearch(map_tool=OfflineMap())
    by_price = search.search(
        [initial], {initial["id"]}, limit=25, sort_by="price_asc",
    )
    price_keys = [(row["monthly_rent_manwon"], row["deposit_manwon"])
                  for row in by_price["properties"]]
    assert price_keys == sorted(price_keys)

    by_distance = search.search(
        [initial], {initial["id"]}, limit=25, sort_by="distance_asc",
        origin_lat=36.35, origin_lng=127.38,
    )
    distances = [row["distance_km"] for row in by_distance["properties"]]
    assert distances == sorted(distances)


def test_commute_condition_uses_map_estimate_and_returns_candidates():
    atom = make_atom(
        field="commute_minutes", operator="lte", value=20,
        label="아주대학교 대중교통 예상 20분 이내", source="test",
        extra={"destination_lat": 37.282943, "destination_lng": 127.043824,
               "mode": "transit", "landmark": "아주대학교"},
    )
    result = AtomicPropertySearch(map_tool=OfflineMap()).search(
        [atom], {atom["id"]}, limit=5)
    assert result["total"] > 0
    assert result["trace"]["per_condition"][0]["estimated"] is True


def test_gui_contains_two_stage_map_and_condition_controls():
    gui = (Path(__file__).parents[1] / "src" / "server" / "gui.html").read_text(
        encoding="utf-8")
    for text in ("기초정보", "지도에서 후보 보기", "AI 조건 수정·추가", "조건 추가",
                 "초기 교집합 → AI 조건 축소", "ncpKeyId", "confirmConditionBtn",
                 "/api/conditions/remove", "pointerdown", "집을 찾는 청년 로고",
                 'id="sortBy"', "전세 위험도 낮은순", "property-detail"):
        assert text in gui
    assert 'name="transaction" type="checkbox" value="매매" checked' not in gui
    assert 'name="house" type="checkbox" value="아파트" checked' not in gui


def test_map_ui_exposes_editable_yellow_initial_scope_and_ai_applied_state():
    gui = (Path(__file__).parents[1] / "src" / "server" / "gui.html").read_text(
        encoding="utf-8")
    app = (Path(__file__).parents[1] / "src" / "server" / "app.py").read_text(
        encoding="utf-8")

    assert 'id="incomeDecile"' not in gui
    assert "소득분위</label>" not in gui
    assert ".filter-chip.base-scope{background:var(--kb-yellow)" in gui
    assert 'id="initialEditor"' in gui
    assert 'id="saveInitialBtn"' in gui
    assert "/api/conditions/initial" in gui
    assert '@app.post("/api/conditions/initial")' in app
    assert 'id="refinementStatus"' in gui
    assert "AI 조건 적용 완료" in gui
    assert 'class="ai-badge">AI 추가' in gui


def test_condition_chat_has_client_and_server_duplicate_response_guards():
    gui = (Path(__file__).parents[1] / "src" / "server" / "gui.html").read_text(
        encoding="utf-8")
    app = (Path(__file__).parents[1] / "src" / "server" / "app.py").read_text(
        encoding="utf-8")

    assert "conditionRequestInFlight" in gui
    assert "confirmInFlight" in gui
    assert "renderedResponseIds" in gui
    assert "request_id:id" in gui
    assert "condition_response_cache" in app
    assert "_cached_condition_response" in app


def test_new_category_slots_produce_expected_atoms():
    slots = {
        "region_dong": ["우만동"], "max_subway_walk_min": 10,
        "max_facility_walk_min": 15, "max_police_distance_min": 20,
        "min_room_count": 2, "elevator_required": True,
        "pet_allowed_required": True,
    }
    atoms, notes = atoms_from_slots(slots, "", OfflineMap())
    by_field = {atom["field"]: atom for atom in atoms}
    assert by_field["dong"]["operator"] == "in"
    assert by_field["dong"]["value"] == ["우만동"]
    assert by_field["subway_walk_minutes"]["operator"] == "lte"
    assert by_field["subway_walk_minutes"]["value"] == 10.0
    assert by_field["mart_walk_minutes"]["value"] == 15.0
    assert by_field["room_count"]["operator"] == "gte"
    assert by_field["room_count"]["value"] == 2.0
    assert by_field["elevator_count"]["operator"] == "gt"
    assert by_field["pet_allowed"]["operator"] == "truthy"
    assert by_field["police_distance_minutes"]["operator"] == "lte"
    assert by_field["police_distance_minutes"]["value"] == 20.0
    assert notes == []


def test_region_gugun_dropped_when_region_dong_present():
    """LLM이 구/군 없이 동만 인식하면 짧은 구 이름("팔달구")을 뽑아낼 수 있는데,
    DB의 실제 값은 "수원시 팔달구"라 형식이 어긋나 검색이 0건이 된다. 동이 이미
    구/군을 함의하므로 region_gugun atom 자체를 만들지 않아 이 불일치를 막는다."""
    slots = {"region_gugun": ["팔달구"], "region_dong": ["인계동"]}
    atoms, _ = atoms_from_slots(slots, "", OfflineMap())
    fields = {atom["field"] for atom in atoms}
    assert "gugun" not in fields
    assert "dong" in fields

    slots_no_dong = {"region_gugun": ["팔달구"]}
    atoms_no_dong, _ = atoms_from_slots(slots_no_dong, "", OfflineMap())
    assert any(atom["field"] == "gugun" for atom in atoms_no_dong)


def test_dong_and_mart_walk_minutes_are_queryable_via_clause():
    initial = make_initial_scope_atom(atoms_from_profile({
        "preferred_sido": "경기", "preferred_gugun": "수원시 팔달구",
    }))
    dong_atom = make_atom(field="dong", operator="in", value=["우만동"],
                          label="우만동 지역", source="test")
    mart_atom = make_atom(field="mart_walk_minutes", operator="lte", value=15,
                          label="편의성 15분 이내", source="test")
    search = AtomicPropertySearch(map_tool=OfflineMap())
    result = search.search([initial, dong_atom, mart_atom],
                           {initial["id"], dong_atom["id"], mart_atom["id"]},
                           limit=10)
    assert result["total"] >= 0  # 필드가 _clause에서 거부되지 않고 실행됨


def test_police_distance_condition_is_monotonic_and_estimated():
    initial = make_initial_scope_atom(atoms_from_profile({
        "preferred_sido": "경기", "preferred_gugun": "수원시 팔달구",
    }))
    search = AtomicPropertySearch(map_tool=OfflineMap())

    def total_within(minutes):
        atom = make_atom(field="police_distance_minutes", operator="lte",
                         value=minutes, label=f"안전성 {minutes}분", source="test")
        result = search.search([initial, atom], {initial["id"], atom["id"]}, limit=5)
        return result

    tight = total_within(3)
    loose = total_within(60)
    assert tight["total"] <= loose["total"]
    police_trace = next(
        t for t in loose["trace"]["per_condition"]
        if t.get("station_category") == "police")
    assert police_trace["estimated"] is True
    assert police_trace["station_count"] > 0


def test_budget_caps_filters_returned_rows_by_transaction_type():
    initial = make_initial_scope_atom(atoms_from_profile({
        "preferred_sido": "경기", "preferred_gugun": "수원시 팔달구",
    }))
    search = AtomicPropertySearch(map_tool=OfflineMap())
    caps = {
        "매매": {"sale_price_manwon": 30000.0},
        "전세": {"deposit_manwon": 10000.0},
        "월세": {"monthly_rent_manwon": 80.0, "deposit_manwon": 3000.0},
    }
    result = search.search([initial], {initial["id"]}, limit=200, budget_caps=caps)
    for row in result["properties"]:
        tx = row.get("transaction_type")
        if tx == "매매":
            price = row.get("sale_price_manwon") or row.get("asking_price_manwon") or 0
            assert price <= caps["매매"]["sale_price_manwon"]
        elif tx == "전세":
            assert (row.get("deposit_manwon") or 0) <= caps["전세"]["deposit_manwon"]
        else:
            assert (row.get("monthly_rent_manwon") or 0) <= caps["월세"]["monthly_rent_manwon"]
            assert (row.get("deposit_manwon") or 0) <= caps["월세"]["deposit_manwon"]


def test_reject_if_zero_match_blocks_conditions_with_no_backing_data():
    """subway_walk_minutes 등은 현재 매물 데이터에 전혀 없어 조건을 걸면 항상
    0건이 된다 — 그대로 반영하면 사용자가 조건이 적용됐다고 오해하므로,
    active_atoms에 추가하지 않고 확인할 수 없다는 안내로 대신해야 한다."""
    from src.server.app import _reject_if_zero_match

    initial = make_initial_scope_atom(atoms_from_profile({
        "preferred_sido": "경기", "preferred_gugun": "수원시 팔달구",
    }))
    ui = {"initial_scope": initial}
    empty_atom = make_atom(
        field="subway_walk_minutes", operator="lte", value=10.0,
        label="접근성(지하철) 10분 이내", source="AI 대화",
    )
    workflow = {}
    result = _reject_if_zero_match(
        ui, [initial], [empty_atom], workflow, slots={}, notes=[],
        decision={"decision": "ready_to_draft", "message": "", "_trace": {}},
        tool_events=[], sql_trace={},
    )
    assert result is not None
    assert result["status"] == "ask_clarification"
    assert "접근성(지하철) 10분 이내" in result["message"]
    assert "확인할 수 있는" in result["message"]
    assert workflow["state"] == "awaiting_clarification"


def test_reject_if_zero_match_allows_conditions_with_real_data():
    from src.server.app import _reject_if_zero_match

    initial = make_initial_scope_atom(atoms_from_profile({
        "preferred_sido": "경기", "preferred_gugun": "수원시 팔달구",
    }))
    ui = {"initial_scope": initial}
    real_atom = make_atom(field="dong", operator="in", value=["인계동"],
                          label="인계동 지역", source="AI 대화")
    workflow = {}
    result = _reject_if_zero_match(
        ui, [initial], [real_atom], workflow, slots={}, notes=[],
        decision={"decision": "ready_to_draft", "message": "", "_trace": {}},
        tool_events=[], sql_trace={},
    )
    assert result is None


def test_gui_shows_short_category_chips_and_affordability_toggle():
    gui = (Path(__file__).parents[1] / "src" / "server" / "gui.html").read_text(
        encoding="utf-8")
    assert "FIELD_CATEGORY" in gui
    assert "chipShortText" in gui
    assert "atomDetail" in gui
    assert "affordability-toggle" in gui
    assert "affordability_only" in gui
    assert "구매가능" in gui


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"  OK {name}")
