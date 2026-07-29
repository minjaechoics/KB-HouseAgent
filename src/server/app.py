"""
실서비스용 FastAPI 서버.

실행(개발):
    pip install fastapi uvicorn
    JEONSE_LLM=api uvicorn src.server.app:app --reload --port 8000

실행(운영, 예):
    gunicorn -k uvicorn.workers.UvicornWorker -w 4 src.server.app:app -b 0.0.0.0:8000

엔드포인트:
    POST /session            새 세션 생성(사용자 프로필) → session_id
    POST /chat               {session_id, text} → 에이전트 응답(2단계 대화)
    POST /fraud/score        단일 매물 위험도 추론
    GET  /health             헬스체크

세션 상태는 데모용으로 인메모리 저장. 운영에서는 Redis 등으로 교체(주석 참조).
"""
from __future__ import annotations
import logging
import copy
import time
import uuid
from typing import Optional
from pathlib import Path
from datetime import date
import pandas as pd
import requests

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
    from pydantic import BaseModel, Field
except ImportError as e:
    raise SystemExit("서버 실행에는 fastapi, uvicorn 설치 필요: pip install fastapi uvicorn")

from src.agent.harness import JeonseAgent
from src.audit import DecisionAuditStore
from src.agent.llm import MockLLM
from src.agent.planner import Planner
from src.fraud_risk.infer import FraudRiskScorer
from src.schemas import UserProfile, PropertyRecord
from src.server.property_search import (
    AtomicPropertySearch, atoms_from_profile, atoms_from_slots,
    make_initial_scope_atom, merge_atoms,
)
from src.report import PropertyReportService
from src.report.service import json_safe
from src.optimization import optimize_housing_choices
from src.preferences import normalize_preferences
from src.real_estate_feeds.storage import ensure_feed_schema, feed_status
from src import config
from src.owner_asset_ratio import OwnerAssetRatioPipeline
from src.senior_deposit import SeniorDepositPipeline

ensure_feed_schema()

app = FastAPI(title="청년 다가구주택 금융 도우미 API", version="1.0")
logger = logging.getLogger(__name__)

# 에이전트/스코어러는 프로세스당 1회 로드(모델 포함)
_agent = JeonseAgent(recommender_name="rule")
_scorer = FraudRiskScorer()
_property_search = AtomicPropertySearch(map_tool=_agent.map_tool)
_property_report = PropertyReportService(llm=_agent.llm, map_tool=_agent.map_tool)
_audit = DecisionAuditStore()
_owner_ratio_pipeline: OwnerAssetRatioPipeline | None = None
_owner_ratio_artifact = (
    config.MODELS_DIR / "owner_asset_ratio" / "owner_asset_ratio_actual.joblib")
_senior_deposit_pipeline: SeniorDepositPipeline | None = None
_senior_deposit_artifact = (
    config.MODELS_DIR / "senior_deposit" / "senior_deposit_actual.joblib")
_building_registry_cache: pd.DataFrame | None = None

# 데모용 인메모리 세션. 운영: Redis(session_id → state 직렬화)로 교체.
_SESSIONS: dict[str, dict] = {}

# This release is intentionally a single-region prototype.  Enforce the scope
# on the server as well as in the UI so direct API calls cannot escape it.
PROTOTYPE_SIDO = "경기"
PROTOTYPE_GUGUN = "수원시 팔달구"


def _prototype_profile(values: dict) -> dict:
    locked = dict(values)
    locked["preferred_sido"] = PROTOTYPE_SIDO
    locked["preferred_gugun"] = PROTOTYPE_GUGUN
    return locked


class ChildPlanIn(BaseModel):
    birth_year: int = Field(ge=2000, le=2100)


class PreferenceProfileIn(BaseModel):
    mode: str = Field(default="balanced", pattern="^(stable|balanced|growth)$")
    risk_tolerance: str = Field(default="balanced", pattern="^(stable|balanced|growth)$")
    weights: dict[str, float] = Field(default_factory=dict)
    approved: bool = True
    source: str = "setup_ui"


class SessionCreate(BaseModel):
    age: int = 29
    monthly_income_manwon: float = 300
    total_asset_manwon: float = 6000
    monthly_living_cost_manwon: float = 120
    income_decile: int = 5
    preferred_sido: Optional[str] = None
    preferred_gugun: Optional[str] = None
    transaction_types: list[str] = Field(default_factory=list)
    house_types: list[str] = Field(default_factory=list)
    max_deposit_manwon: Optional[float] = None
    max_sale_price_manwon: Optional[float] = None
    max_monthly_rent_manwon: Optional[float] = None
    max_maintenance_manwon: Optional[float] = None
    min_area_m2: Optional[float] = None
    # 금융상품 예비 자격 판정용. 모르는 값은 None으로 두고 심사 필요로 표시한다.
    employment_type: Optional[str] = None
    employment_months: Optional[int] = Field(default=None, ge=0, le=720)
    household_role: Optional[str] = None
    home_ownership_count: Optional[int] = Field(default=None, ge=0, le=99)
    marital_status: Optional[str] = None
    spouse_annual_income_manwon: Optional[float] = Field(default=None, ge=0)
    minor_children_count: Optional[int] = Field(default=None, ge=0, le=20)
    children_plans: list[ChildPlanIn] = Field(default_factory=list, max_length=20)
    expected_inheritance_manwon: Optional[float] = Field(default=None, ge=0)
    expected_inheritance_age: Optional[int] = Field(default=None, ge=18, le=100)
    workplace_or_school: Optional[str] = Field(default=None, max_length=200)
    is_korean_national: Optional[bool] = None
    has_income_proof: Optional[bool] = None
    contract_deposit_paid_5pct: Optional[bool] = None
    youth_life_stage: Optional[str] = Field(
        default=None, pattern="^(student|early_career|other)$"
    )
    housing_priorities: list[str] = Field(default_factory=list, max_length=12)
    has_hf_jeonse_loan_guarantee: Optional[bool] = None
    move_in_registration_possible: Optional[bool] = None
    fixed_date_possible: Optional[bool] = None
    senior_mortgage_established_date: Optional[date] = None
    wants_guarantee_insurance: Optional[bool] = None
    preferences: PreferenceProfileIn = Field(default_factory=PreferenceProfileIn)


class ChatIn(BaseModel):
    session_id: str
    text: str
    request_id: Optional[str] = None


class ConditionApplyIn(BaseModel):
    session_id: str
    atom_ids: Optional[list[str]] = None


class ConditionConfirmIn(BaseModel):
    session_id: str
    request_id: Optional[str] = None


class InitialConditionsUpdateIn(BaseModel):
    session_id: str
    preferred_sido: Optional[str] = None
    preferred_gugun: Optional[str] = None
    transaction_types: list[str] = Field(default_factory=list)
    house_types: list[str] = Field(default_factory=list)
    max_deposit_manwon: Optional[float] = None
    max_sale_price_manwon: Optional[float] = None
    max_monthly_rent_manwon: Optional[float] = None
    max_maintenance_manwon: Optional[float] = None
    min_area_m2: Optional[float] = None


class ConditionRemoveIn(BaseModel):
    session_id: str
    atom_id: str


class PropertySearchIn(BaseModel):
    session_id: str
    enabled_atom_ids: Optional[list[str]] = None
    limit: int = 120
    sort_by: str = "recommended"
    origin_lat: Optional[float] = None
    origin_lng: Optional[float] = None


class SimulationDestinationIn(BaseModel):
    label: Optional[str] = Field(default=None, max_length=80)
    query: str = Field(min_length=1, max_length=200)
    category: str = Field(default="frequent", max_length=30)
    visits_per_month: float = Field(default=20, ge=0, le=62)


class SimulationSubscriptionIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    monthly_price_krw: float = Field(ge=0, le=10_000_000)


class LifestyleSimulationIn(BaseModel):
    use_itemized_budget: bool = False
    transport_mode: str = "transit"
    destinations: list[SimulationDestinationIn] = Field(
        default_factory=list, max_length=5)
    transit_taxi_ratio_pct: float = Field(default=10, ge=0, le=100)
    extra_transport_monthly_krw: float = Field(default=0, ge=0, le=100_000_000)
    car_fuel_price_krw_per_liter: float = Field(default=1700, ge=0, le=100_000)
    car_fuel_efficiency_km_per_liter: float = Field(default=12, ge=1, le=100)
    car_insurance_monthly_krw: float = Field(default=0, ge=0, le=100_000_000)
    car_maintenance_monthly_krw: float = Field(default=0, ge=0, le=100_000_000)
    car_parking_toll_monthly_krw: float = Field(default=0, ge=0, le=100_000_000)
    vehicle_powertrain: str = Field(default="gasoline", max_length=20)
    ev_efficiency_km_per_kwh: float = Field(default=5.5, ge=1, le=20)
    ev_electricity_krw_per_kwh: float = Field(default=320, ge=0, le=10_000)
    other_insurance_monthly_krw: float = Field(default=0, ge=0, le=100_000_000)
    daily_food_krw: float = Field(default=0, ge=0, le=10_000_000)
    subscriptions: list[SimulationSubscriptionIn] = Field(
        default_factory=list, max_length=30)
    telecom_monthly_krw: float = Field(default=0, ge=0, le=100_000_000)
    internet_monthly_krw: float = Field(default=0, ge=0, le=100_000_000)
    leisure_monthly_krw: float = Field(default=0, ge=0, le=100_000_000)
    other_living_monthly_krw: float = Field(default=0, ge=0, le=100_000_000)
    requested_loan_amount_manwon: Optional[float] = Field(default=None, ge=0)


class PropertyReportIn(BaseModel):
    session_id: str
    property_id: str
    horizon_years: int = Field(default=10, ge=1, le=50)
    simulation_end_age: Optional[int] = Field(default=65, ge=18, le=100)
    income_growth_rate: float = Field(default=0.03, ge=-0.2, le=0.3)
    inflation_rate: float = Field(default=0.02, ge=-0.05, le=0.3)
    liquid_asset_return_rate: float = Field(default=0.03, ge=-0.3, le=0.5)
    selected_finance_program_id: Optional[str] = Field(default=None, max_length=200)
    requested_loan_amount_manwon: Optional[float] = Field(default=None, ge=0)
    monte_carlo_paths: int = Field(default=10_000, ge=500, le=20_000)
    simulation_seed: Optional[int] = Field(default=None, ge=0, le=2_147_483_647)
    enable_job_loss: bool = False
    annual_job_loss_probability: float = Field(default=0.05, ge=0, le=0.5)
    lifestyle: Optional[LifestyleSimulationIn] = None


class MetricExplanationIn(BaseModel):
    session_id: str
    decision_run_id: str = Field(min_length=8, max_length=100)


class OptimizationIn(BaseModel):
    session_id: str
    property_ids: Optional[list[str]] = Field(default=None, max_length=120)
    horizon_years: int = Field(default=10, ge=1, le=30)
    requested_loan_amount_manwon: Optional[float] = Field(default=None, ge=0)


class OwnerAssetRatioIn(BaseModel):
    session_id: str
    property_id: str
    samples: int = Field(default=20_000, ge=1_000, le=100_000)
    seed: int = Field(default=20260728, ge=0, le=2_147_483_647)
    occupancy_scenario: str = Field(
        default="baseline", pattern="^(low|baseline|high)$")
    random_effect_sigma: Optional[float] = Field(default=None, ge=0, le=1.0)


class SeniorDepositIn(BaseModel):
    session_id: str
    property_id: Optional[str] = None
    building_id: Optional[str] = None
    reference_date: date
    samples: int = Field(default=20_000, ge=1_000, le=100_000)
    seed: int = Field(default=20260728, ge=0, le=2_147_483_647)
    scenario: str = Field(
        default="conservative",
        pattern="^(conservative|probabilistic|scenario)$",
    )
    occupancy_scenario: str = Field(
        default="baseline", pattern="^(low|baseline|high)$")
    senior_probability: Optional[float] = Field(default=None, ge=0, le=1)
    random_effect_sigma: Optional[float] = Field(default=None, ge=0, le=1)
    target_rooms_excluded: int = Field(default=1, ge=0, le=10)


def _load_owner_ratio_pipeline() -> OwnerAssetRatioPipeline | None:
    global _owner_ratio_pipeline
    if _owner_ratio_pipeline is not None:
        return _owner_ratio_pipeline
    if not _owner_ratio_artifact.exists():
        return None
    _owner_ratio_pipeline = OwnerAssetRatioPipeline.load(
        _owner_ratio_artifact, allow_synthetic=False)
    return _owner_ratio_pipeline


def _load_senior_deposit_pipeline() -> SeniorDepositPipeline | None:
    global _senior_deposit_pipeline
    if _senior_deposit_pipeline is not None:
        return _senior_deposit_pipeline
    if not _senior_deposit_artifact.exists():
        return None
    _senior_deposit_pipeline = SeniorDepositPipeline.load(
        _senior_deposit_artifact, allow_synthetic=False)
    return _senior_deposit_pipeline


def _actual_building(building_id: str) -> dict | None:
    global _building_registry_cache
    if _building_registry_cache is None:
        path = (
            config.ROOT / "data" / "processed"
            / "owner_asset_ratio" / "buildings.csv")
        if not path.exists():
            return None
        _building_registry_cache = pd.read_csv(path, low_memory=False)
    matches = _building_registry_cache[
        _building_registry_cache["building_id"].astype(str).eq(
            str(building_id))]
    return None if matches.empty else matches.iloc[0].to_dict()


@app.get("/health")
def health():
    return {"status": "ok", "recommender": _agent.recommender_name,
            "llm": type(_agent.llm).__name__,
            "model": getattr(_agent.llm, "model", None),
            "agentic": bool(getattr(_agent.llm, "supports_agentic_calls", False)),
            "pipeline": ["plan", "text2sql", "tools", "validate", "synthesize"],
            "map": _agent.map_tool.status(),
            "owner_asset_ratio": {
                "available": _owner_ratio_artifact.exists(),
                "artifact_kind_required": "actual",
                "synthetic_artifact_allowed": False,
            },
            "senior_deposit": {
                "available": _senior_deposit_artifact.exists(),
                "model_mode": "scenario_only",
                "legal_seniority_confirmed": False,
                "synthetic_artifact_allowed": False,
            }}


@app.get("/api/data-sources/status")
def data_sources_status():
    """Public metadata only; credentials and license documents are never returned."""
    return feed_status()


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/gui", status_code=307)


@app.get("/gui", response_class=HTMLResponse)
def gui():
    """Small browser UI for manual Agent testing."""
    html_path = Path(__file__).with_name("gui.html")
    if not html_path.exists():
        raise HTTPException(404, "gui.html not found")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/assets/youth-home-hero-v1.png", include_in_schema=False)
def landing_hero():
    asset = Path(__file__).with_name("assets") / "youth-home-hero-v1.png"
    if not asset.exists():
        raise HTTPException(404, "화면 이미지를 준비하지 못했습니다.")
    return FileResponse(asset, media_type="image/png", headers={
        "Cache-Control": "public, max-age=86400",
    })


@app.get("/assets/youth-home-loader-sprite-v1.png", include_in_schema=False)
def report_loader_sprite():
    asset = Path(__file__).with_name("assets") / "youth-home-loader-sprite-v1.png"
    if not asset.exists():
        raise HTTPException(404, "로딩 이미지를 준비하지 못했습니다.")
    return FileResponse(asset, media_type="image/png", headers={
        "Cache-Control": "public, max-age=86400, immutable",
    })


@app.get("/assets/youth-home-loader-scene-v2.jpg", include_in_schema=False)
def report_loader_scene():
    asset = Path(__file__).with_name("assets") / "youth-home-loader-scene-v2.jpg"
    if not asset.exists():
        raise HTTPException(404, "로딩 이미지를 준비하지 못했습니다.")
    return FileResponse(asset, media_type="image/jpeg", headers={
        "Cache-Control": "public, max-age=31536000, immutable",
    })


@app.post("/session")
def create_session(body: SessionCreate):
    sid = uuid.uuid4().hex
    profile = _prototype_profile(body.model_dump())
    profile["preferences"] = normalize_preferences(profile.get("preferences"))
    user = dict(user_id=sid, **profile)
    session = _agent.new_session(user)
    profile_atoms = atoms_from_profile(profile)
    initial_scope = make_initial_scope_atom(profile_atoms)
    base_atoms = [initial_scope] if initial_scope else []
    session["map_ui"] = {
        "initial_scope": copy.deepcopy(initial_scope),
        "base_atoms": copy.deepcopy(base_atoms),
        "active_atoms": copy.deepcopy(base_atoms),
        "draft_atoms": [],
        "condition_chat": [],
        "condition_response_cache": {},
        "condition_workflow": {
            "state": "idle", "known_slots": {}, "proposed_slots": {},
            "last_question": None, "source_text": "", "audit": [],
        },
    }
    _SESSIONS[sid] = session
    return {
        "session_id": sid,
        "atoms": base_atoms,
        "initial_scope": initial_scope,
    }


@app.get("/api/client-config")
def client_config():
    """Only the public Dynamic Map client ID is exposed to the browser."""
    map_status = _agent.map_tool.status()
    return {
        "naver_map_client_id": _agent.map_tool.client_id or None,
        "map": map_status,
        "transit_notice": (
            "대중교통 시간은 TMAP 대중교통 경로 결과입니다. API 장애 시 예상치로 대체됩니다."
            if map_status.get("transit") == "tmap_transit"
            else "TMAP 키가 없어 대중교통 시간은 거리·평균속도 기반 예상치입니다."
        ),
    }


@app.get("/api/map/sdk.js", include_in_schema=False)
@app.get("/openapi/v3/maps.js", include_in_schema=False)
def naver_map_sdk():
    """Same-origin pass-through for networks that block the NAVER SDK host.

    The browser still receives and runs the official SDK response.  No client
    secret is involved; the Dynamic Map product authenticates with its public
    application Client ID and the registered page origin.
    """
    client_id = _agent.map_tool.client_id
    if not client_id:
        raise HTTPException(503, "NAVER Dynamic Map Client ID is not configured")
    try:
        response = requests.get(
            "https://oapi.map.naver.com/openapi/v3/maps.js",
            params={"ncpKeyId": client_id},
            headers={"User-Agent": "JeonseHelper/1.0"}, timeout=15,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.exception("NAVER Maps JavaScript SDK fetch failed")
        raise HTTPException(502, "NAVER Maps SDK upstream connection failed") from exc
    return Response(
        content=response.content, media_type="application/javascript",
        headers={"Cache-Control": "private, max-age=300"},
    )


@app.get("/api/regions")
def regions():
    available = _property_search.regions()
    sido_row = next((row for row in available if row.get("name") == PROTOTYPE_SIDO), {})
    gugun_row = next(
        (row for row in (sido_row.get("gugun") or []) if row.get("name") == PROTOTYPE_GUGUN),
        {},
    )
    return {"regions": [{
        "name": PROTOTYPE_SIDO,
        "count": int(sido_row.get("count") or 0),
        "gugun": [{"name": PROTOTYPE_GUGUN, "count": int(gugun_row.get("count") or 0)}],
        "locked": True,
    }], "prototype_locked": True}


def _reset_condition_workflow(workflow: dict) -> None:
    workflow.update({"state": "idle", "known_slots": {}, "proposed_slots": {},
                     "last_question": None, "source_text": ""})


def _cached_condition_response(ui: dict, request_id: Optional[str]) -> Optional[dict]:
    if not request_id:
        return None
    cached = (ui.get("condition_response_cache") or {}).get(request_id)
    return copy.deepcopy(cached) if cached is not None else None


def _remember_condition_response(ui: dict, request_id: Optional[str], response: dict) -> dict:
    response = dict(response)
    response["response_id"] = request_id or uuid.uuid4().hex
    if request_id:
        cache = ui.setdefault("condition_response_cache", {})
        cache[request_id] = copy.deepcopy(response)
        while len(cache) > 20:
            cache.pop(next(iter(cache)))
    return response


def _confirmed_condition_decision(workflow: dict) -> dict:
    """Convert an explicit UI-button approval into the internal tool decision."""
    confirmed_slots = dict(workflow.get("proposed_slots") or {})
    landmark_aliases = {"아주대": "아주대학교", "kaist": "카이스트"}
    landmark = str(confirmed_slots.get("workplace_landmark") or "").strip()
    if landmark:
        confirmed_slots["workplace_landmark"] = landmark_aliases.get(
            landmark.lower(), landmark)
    if (confirmed_slots.get("workplace_landmark")
            and confirmed_slots.get("commute_mode")
            and confirmed_slots.get("max_commute_min") is None):
        confirmed_slots["max_commute_min"] = {
            "transit": 20.0, "walking": 15.0, "driving": 20.0,
        }.get(confirmed_slots["commute_mode"], 20.0)
    confirmed_tools = ["text2sql_property_filter", "apply_ui_conditions"]
    if confirmed_slots.get("workplace_landmark"):
        confirmed_tools = ["geocode_landmark", "build_map_time_condition",
                           *confirmed_tools]
    return {
        "decision": "ready_to_draft",
        "message": "조건 추가 버튼으로 승인했습니다. 조건을 검증해 반영합니다.",
        "goal_summary": "UI 조건 추가 버튼 승인",
        "known_facts": ["사용자가 조건 추가 버튼으로 직전 제안을 승인함"],
        "uncertainties": [], "slots": confirmed_slots,
        "proposed_defaults": [], "tool_plan": confirmed_tools,
        "confidence": 1.0, "decision_reason": "명시적 UI 버튼 승인",
        "_trace": {
            "strategy": "deterministic_ui_confirmation_gate",
            "approval_channel": "condition_add_button", "fallback": False,
        },
    }


def _finalize_condition_draft(session: dict, ui: dict, history: list,
                              source_text: str, decision: dict) -> dict:
    """승인 뒤에만 지도 도구→Text2SQL→UI atom 순서로 실행한다."""
    workflow = ui.setdefault("condition_workflow", {})
    condition_source = str(workflow.get("source_text") or source_text)
    slots = {key: value for key, value in (decision.get("slots") or {}).items()
             if value is not None and not str(key).startswith("_")}
    requested_sort = slots.pop("sort_by", None)
    if requested_sort not in {
        None, "recommended", "risk_asc", "risk_desc",
        "price_asc", "price_desc", "distance_asc",
    }:
        requested_sort = None
    active = ui.get("active_atoms") or []
    if slots.get("max_commute_min") is not None and not slots.get("workplace_landmark"):
        previous = next((atom for atom in reversed(active)
                         if atom.get("field") == "commute_minutes"), None)
        if previous:
            slots["workplace_landmark"] = previous.get("landmark")
            slots["commute_mode"] = previous.get("mode", "transit")

    atoms, notes = atoms_from_slots(slots, condition_source, _agent.map_tool)
    commute_atom = next((atom for atom in atoms
                         if atom.get("field") == "commute_minutes"), None)
    tool_events: list[dict] = []
    web_sources: list[dict] = []
    if slots.get("workplace_landmark"):
        tool_events.append({
            "tool": "geocode_landmark", "status": "ok" if commute_atom else "failed",
            "input": slots.get("workplace_landmark"),
            "source": commute_atom.get("geocode_source") if commute_atom else None,
        })
        if not commute_atom:
            landmark = str(slots.get("workplace_landmark") or "").strip()
            web_result = _agent.llm.web_search(
                f"대한민국 장소 '{landmark}'의 공식 명칭과 도로명주소를 확인해줘",
                purpose="지도 지오코딩 실패 장소의 공식 주소 확인",
            )
            if web_result:
                web_sources = list(web_result.get("sources") or [])[:5]
                map_query = str(web_result.get("map_query") or "").strip()
                tool_events.append({
                    "tool": "internet_web_search", "status": "ok",
                    "query": landmark, "resolved_map_query": map_query or None,
                    "source_count": len(web_sources),
                    "strategy": "openai_responses_web_search",
                })
                if map_query:
                    retry_slots = {**slots, "workplace_landmark": map_query}
                    retry_atoms, retry_notes = atoms_from_slots(
                        retry_slots, condition_source, _agent.map_tool)
                    retry_commute = next((atom for atom in retry_atoms
                                          if atom.get("field") == "commute_minutes"), None)
                    if retry_commute:
                        slots = retry_slots
                        atoms, notes, commute_atom = retry_atoms, [*notes, *retry_notes], retry_commute
                        notes.append(
                            f"웹 검색으로 '{landmark}'의 공식 위치를 확인한 뒤 지도 주소로 재검증했습니다.")
            else:
                tool_events.append({
                    "tool": "internet_web_search", "status": "unavailable",
                    "query": landmark,
                    "reason": "provider_not_supported_or_search_failed",
                })
        if commute_atom:
            tool_events.append({
                "tool": "build_map_time_condition", "status": "ok",
                "mode": commute_atom.get("mode"), "minutes": commute_atom.get("value"),
                "estimated": commute_atom.get("estimated", True),
            })
        else:
            workflow.update({"state": "awaiting_clarification", "known_slots": slots,
                             "proposed_slots": {}, "last_question": (
                                 f"'{slots.get('workplace_landmark')}' 위치를 확인하지 못했습니다. "
                                 "정식 장소명이나 도로명주소를 알려주세요.")})
            ui["draft_atoms"] = []
            message = workflow["last_question"]
            history.append({"role": "assistant", "text": message})
            return {
                "status": "ask_clarification", "message": message,
                "draft_atoms": [], "notes": notes,
                "web_sources": web_sources,
                "trace": {"stage": "tool_validation_failed",
                          "property_search_executed": False,
                          "decision_record": {key: value for key, value in decision.items()
                                              if key != "_trace"},
                          "planner": decision.get("_trace") or {},
                          "tools": tool_events,
                          "text2sql": {"status": "not_run_due_to_tool_failure"}},
            }

    sql_trace = _agent.text2sql.compile_property_filter(condition_source, slots, limit=500)
    tool_events.append({
        "tool": "text2sql_property_filter",
        "status": "ok" if str(sql_trace.get("validation", "")).startswith("passed") else "fallback",
        "strategy": sql_trace.get("strategy"),
        "validation": sql_trace.get("validation"),
    })
    if not atoms and requested_sort is None:
        message = ("승인한 내용에서 실행 가능한 UI 조건을 만들지 못했습니다. "
                   "지역·가격·주택유형처럼 원하는 조건을 조금 더 구체적으로 알려주세요.")
        workflow.update({"state": "awaiting_clarification", "known_slots": slots,
                         "proposed_slots": {}, "last_question": message})
        history.append({"role": "assistant", "text": message})
        return {
            "status": "ask_clarification", "message": message,
            "draft_atoms": [], "notes": notes,
            "trace": {
                "stage": "no_executable_ui_condition",
                "workflow_state": workflow.get("state"),
                "property_search_executed": False,
                "decision_record": {key: value for key, value in decision.items()
                                    if key != "_trace"},
                "planner": decision.get("_trace") or {},
                "tools": tool_events, "text2sql": sql_trace,
            },
        }
    ui["draft_atoms"] = []
    ui["draft_sql_trace"] = sql_trace
    ui["active_atoms"] = merge_atoms(active, atoms)
    if requested_sort is not None:
        ui["sort_by"] = requested_sort
    workflow.update({"state": "idle", "known_slots": {},
                     "proposed_slots": {}, "last_question": None})
    tool_events.append({"tool": "apply_ui_conditions", "status": "ok",
                        "atom_count": len(atoms)})
    if requested_sort is not None:
        tool_events.append({"tool": "apply_result_sort", "status": "ok",
                            "sort_by": requested_sort,
                            "risk_is_filter": False})
    if atoms and requested_sort is not None:
        response_text = (
            f"확인한 내용으로 SQL 검증을 마쳤습니다. 승인한 {len(atoms)}개 조건과 "
            "정렬 기준을 적용해 후보를 다시 계산합니다."
        )
    elif atoms:
        response_text = (
            f"확인한 내용으로 지도 도구와 SQL 검증을 마쳤습니다. "
            f"승인한 {len(atoms)}개 조건을 검색 조건에 추가했습니다. 후보를 다시 계산합니다."
        )
    else:
        response_text = "매물을 제외하지 않고 요청한 정렬 기준만 적용합니다."
    if notes:
        response_text += " " + " ".join(notes)
    history.append({"role": "assistant", "text": response_text})
    return {
        "status": "applied", "message": response_text, "draft_atoms": [],
        "active_atoms": ui["active_atoms"], "applied_atoms": atoms, "notes": notes,
        "sort_by": requested_sort,
        "web_sources": web_sources,
        "trace": {
            "stage": "confirmed_tool_text2sql_to_ui_applied",
            "workflow_state": workflow.get("state"),
            "property_search_executed": False,
            "decision_record": {key: value for key, value in decision.items()
                                if key != "_trace"},
            "planner": decision.get("_trace") or {},
            "tools": tool_events, "text2sql": sql_trace,
        },
    }


@app.post("/api/conditions/draft")
def draft_conditions(body: ChatIn):
    """Negotiate ambiguity, confirm, call tools, compile SQL, then create UI atoms."""
    session = _SESSIONS.get(body.session_id)
    if session is None:
        raise HTTPException(404, "session not found. POST /session first.")
    ui = session.setdefault("map_ui", {})
    cached = _cached_condition_response(ui, body.request_id)
    if cached is not None:
        return cached
    history = ui.setdefault("condition_chat", [])
    active = ui.setdefault("active_atoms", [])
    workflow = ui.setdefault("condition_workflow", {
        "state": "idle", "known_slots": {}, "proposed_slots": {},
        "last_question": None, "source_text": "", "audit": [],
    })
    if workflow.get("state") == "draft_ready":
        workflow.update({"state": "idle", "known_slots": {}, "proposed_slots": {},
                         "last_question": None, "source_text": ""})
    initial_scope = ui.get("initial_scope") or next(
        (atom for atom in active if atom.get("scope_role") == "initial_universe"),
        None,
    )
    ai_active = [atom for atom in active
                 if atom.get("scope_role") != "initial_universe"]
    active_summary = ", ".join(atom.get("label", "") for atom in ai_active[-12:])
    history.append({"role": "user", "text": body.text})
    previous_source = str(workflow.get("source_text") or "").strip()
    workflow["source_text"] = (previous_source + "\n" + body.text).strip()

    context = {
        "state": workflow.get("state", "idle"),
        "known_slots": workflow.get("known_slots") or {},
        "proposed_slots": workflow.get("proposed_slots") or {},
        "last_question": workflow.get("last_question"),
        "active_conditions": active_summary,
        "initial_universe": (
            initial_scope.get("summary") if initial_scope else "초기 조건 없음(전체 매물)"
        ),
        "condition_scope_policy": (
            "AI 조건은 initial_universe의 교집합 안에서만 후보를 줄이며 "
            "초기 조건을 완화하거나 대체할 수 없음"
        ),
        "recent_dialogue": history[-8:],
        "approval_channel": "ui_condition_add_button_only",
        "chat_messages_are_condition_edits": True,
    }
    try:
        decision = _agent.llm.plan_condition_dialogue(body.text, context)
    except Exception as exc:
        logger.exception("condition dialogue planner failed")
        decision = MockLLM().plan_condition_dialogue(body.text, context)

    decision_name = decision.get("decision")
    slots = {key: value for key, value in (decision.get("slots") or {}).items()
             if value is not None and not str(key).startswith("_")}
    audit_record = {key: value for key, value in decision.items() if key != "_trace"}
    workflow.setdefault("audit", []).append(audit_record)
    workflow["audit"] = workflow["audit"][-20:]

    if decision_name == "ready_to_draft":
        decision_name = "ask_confirmation"
        decision["decision"] = decision_name
        decision["message"] = "아래 ‘조건 추가’ 버튼을 눌러 승인하거나 채팅으로 조건을 더 수정해 주세요."
        decision["tool_plan"] = []
    if decision_name == "cancel":
        _reset_condition_workflow(workflow)
        ui["draft_atoms"] = []
    elif decision_name == "ask_confirmation":
        workflow.update({"state": "awaiting_confirmation", "known_slots": slots,
                         "proposed_slots": slots,
                         "last_question": decision.get("message")})
    else:
        workflow.update({"state": "awaiting_clarification", "known_slots": slots,
                         "proposed_slots": {},
                         "last_question": decision.get("message")})
    ui["draft_atoms"] = []
    message = decision.get("message") or "조건을 조금 더 알려주세요."
    history.append({"role": "assistant", "text": message})
    response = {
        "status": decision_name, "message": message, "draft_atoms": [], "notes": [],
        "trace": {
            "stage": "condition_dialogue_decision",
            "workflow_state": workflow.get("state"),
            "property_search_executed": False,
            "decision_record": audit_record,
            "planner": decision.get("_trace") or {},
            "tools": [], "text2sql": {"status": "not_run_before_confirmation"},
        },
    }
    return _remember_condition_response(ui, body.request_id, response)


@app.post("/api/conditions/confirm")
def confirm_conditions(body: ConditionConfirmIn):
    """Approve the latest proposal only through the explicit UI button."""
    session = _SESSIONS.get(body.session_id)
    if session is None:
        raise HTTPException(404, "session not found. POST /session first.")
    ui = session.setdefault("map_ui", {})
    cached = _cached_condition_response(ui, body.request_id)
    if cached is not None:
        return cached
    workflow = ui.setdefault("condition_workflow", {})
    if workflow.get("state") != "awaiting_confirmation":
        raise HTTPException(409, "승인할 조건 제안이 없습니다. 채팅으로 조건을 먼저 정리해 주세요.")
    if not workflow.get("proposed_slots"):
        raise HTTPException(409, "실행 가능한 조건이 없어 승인할 수 없습니다.")
    history = ui.setdefault("condition_chat", [])
    decision = _confirmed_condition_decision(workflow)
    response = _finalize_condition_draft(
        session, ui, history, str(workflow.get("source_text") or ""), decision,
    )
    return _remember_condition_response(ui, body.request_id, response)


@app.post("/api/conditions/initial")
def update_initial_conditions(body: InitialConditionsUpdateIn):
    """Replace the setup intersection in-place while retaining later AI refinements."""
    session = _SESSIONS.get(body.session_id)
    if session is None:
        raise HTTPException(404, "session not found. POST /session first.")
    ui = session.setdefault("map_ui", {})
    updates = _prototype_profile(body.model_dump(exclude={"session_id"}))
    session.setdefault("user", {}).update(updates)

    profile_atoms = atoms_from_profile(session["user"])
    initial_scope = make_initial_scope_atom(profile_atoms)
    ai_atoms = [
        atom for atom in (ui.get("active_atoms") or [])
        if atom.get("scope_role") != "initial_universe"
    ]
    base_atoms = [initial_scope] if initial_scope else []
    ui["initial_scope"] = copy.deepcopy(initial_scope)
    ui["base_atoms"] = copy.deepcopy(base_atoms)
    ui["active_atoms"] = [*base_atoms, *ai_atoms]
    ui["draft_atoms"] = []
    ui.pop("draft_sql_trace", None)
    _reset_condition_workflow(ui.setdefault("condition_workflow", {}))
    return {
        "status": "updated",
        "atoms": ui["active_atoms"],
        "initial_scope": initial_scope,
        "profile": updates,
        "retained_ai_condition_count": len(ai_atoms),
    }


@app.post("/api/conditions/remove")
def remove_condition(body: ConditionRemoveIn):
    """Idempotently remove one active filter from the current map session."""
    session = _SESSIONS.get(body.session_id)
    if session is None:
        raise HTTPException(404, "session not found. POST /session first.")
    ui = session.setdefault("map_ui", {})
    before = ui.get("active_atoms") or []
    target = next((atom for atom in before if atom.get("id") == body.atom_id), None)
    if target and target.get("scope_role") == "initial_universe":
        raise HTTPException(
            409,
            "초기 조건은 AI 조건의 기준 집합이므로 지도에서 삭제할 수 없습니다. "
            "기초정보 화면으로 돌아가 새 검색을 시작해 주세요.",
        )
    after = [atom for atom in before if atom.get("id") != body.atom_id]
    ui["active_atoms"] = after
    ui["base_atoms"] = [atom for atom in (ui.get("base_atoms") or [])
                        if atom.get("id") != body.atom_id]
    ui["draft_atoms"] = [atom for atom in (ui.get("draft_atoms") or [])
                         if atom.get("id") != body.atom_id]
    return {"status": "removed", "removed": len(after) != len(before),
            "atoms": after}


@app.post("/api/conditions/apply")
def apply_conditions(body: ConditionApplyIn):
    session = _SESSIONS.get(body.session_id)
    if session is None:
        raise HTTPException(404, "session not found. POST /session first.")
    ui = session.setdefault("map_ui", {})
    draft = ui.get("draft_atoms") or []
    selected = set(body.atom_ids) if body.atom_ids is not None else None
    incoming = [atom for atom in draft if selected is None or atom["id"] in selected]
    ui["active_atoms"] = merge_atoms(ui.get("active_atoms") or [], incoming)
    ui["draft_atoms"] = []
    ui.pop("draft_sql_trace", None)
    _reset_condition_workflow(ui.setdefault("condition_workflow", {}))
    return {"status": "applied", "atoms": ui["active_atoms"],
            "applied_count": len(incoming)}


@app.post("/api/properties/search")
def search_properties(body: PropertySearchIn):
    session = _SESSIONS.get(body.session_id)
    if session is None:
        raise HTTPException(404, "session not found. POST /session first.")
    ui = session.setdefault("map_ui", {})
    atoms = ui.get("active_atoms") or []
    enabled = (set(body.enabled_atom_ids)
               if body.enabled_atom_ids is not None else None)
    initial_scope = ui.get("initial_scope")
    if initial_scope:
        if not any(atom.get("id") == initial_scope.get("id") for atom in atoms):
            atoms = [initial_scope, *atoms]
        if enabled is not None:
            enabled.add(initial_scope["id"])
    try:
        result = _property_search.search(
            atoms, enabled, body.limit,
            sort_by=body.sort_by,
            origin_lat=body.origin_lat,
            origin_lng=body.origin_lng,
        )
        ui["last_search_properties"] = copy.deepcopy(result.get("properties") or [])
        ui["last_search_trace"] = copy.deepcopy(result.get("trace") or {})
        return result
    except ValueError as exc:
        logger.info("property search validation rejected: %s", exc)
        raise HTTPException(400, "검색 조건을 다시 확인해 주세요.") from exc


@app.post("/api/properties/report")
def property_report(body: PropertyReportIn):
    """선택 매물의 금융·자산·치안·생활·계약안전 통합 리포트."""
    session = _SESSIONS.get(body.session_id)
    if session is None:
        raise HTTPException(404, "session not found. POST /session first.")
    try:
        result = _property_report.build(
            session["user"], body.property_id,
            assumptions={
                "horizon_years": body.horizon_years,
                "simulation_end_age": body.simulation_end_age,
                "income_growth_rate": body.income_growth_rate,
                "inflation_rate": body.inflation_rate,
                "liquid_asset_return_rate": body.liquid_asset_return_rate,
                "selected_finance_program_id": body.selected_finance_program_id,
                "monte_carlo_paths": body.monte_carlo_paths,
                "simulation_seed": body.simulation_seed,
                "enable_job_loss": body.enable_job_loss,
                "annual_job_loss_probability": body.annual_job_loss_probability,
                "requested_loan_amount_manwon": (
                    body.requested_loan_amount_manwon
                    if body.requested_loan_amount_manwon is not None else
                    (body.lifestyle.requested_loan_amount_manwon
                     if body.lifestyle is not None else None)
                ),
                "lifestyle_inputs": (
                    body.lifestyle.model_dump() if body.lifestyle is not None else None
                ),
            },
            session_id=body.session_id,
        )
        # The metric explanation is requested after the main report is already
        # visible. Keep only the latest report in this prototype session.
        session["last_property_report"] = result
        return result
    except KeyError as exc:
        _audit.fail_latest_running(
            session_id=body.session_id, property_id=body.property_id, error=exc)
        raise HTTPException(404, "property not found") from exc
    except Exception as exc:
        _audit.fail_latest_running(
            session_id=body.session_id, property_id=body.property_id, error=exc)
        logger.exception("property report failed")
        raise HTTPException(400, "매물 분석을 완료하지 못했습니다. 잠시 후 다시 시도해 주세요.") from exc


@app.post("/api/properties/report/metric-explanations")
def property_report_metric_explanations(body: MetricExplanationIn):
    """Explain the latest report's important numbers in one batched LLM call."""
    session = _SESSIONS.get(body.session_id)
    if session is None:
        raise HTTPException(404, "session not found. POST /session first.")
    report = session.get("last_property_report")
    if (
        not isinstance(report, dict)
        or str(report.get("decision_run_id") or "") != body.decision_run_id
    ):
        raise HTTPException(409, "latest property report changed")
    return _property_report.explain_metrics(report)


@app.post("/api/properties/report/news-assessment")
def property_report_news_assessment(body: MetricExplanationIn):
    """Upgrade the fast keyword preview with an asynchronous LLM judgement."""
    session = _SESSIONS.get(body.session_id)
    if session is None:
        raise HTTPException(404, "session not found. POST /session first.")
    report = session.get("last_property_report")
    if (
        not isinstance(report, dict)
        or str(report.get("decision_run_id") or "") != body.decision_run_id
    ):
        raise HTTPException(409, "latest property report changed")
    try:
        result = _property_report.explain_news(report)
        forecast = report.setdefault("forecast", {})
        forecast["news"] = copy.deepcopy(result["news"])
        forecast["market_assessment"] = copy.deepcopy(
            result["market_assessment"]
        )
        return result
    except Exception as exc:
        logger.exception("property report news assessment failed")
        raise HTTPException(
            503, "AI 뉴스 분석을 완료하지 못했습니다. 잠시 후 다시 시도해 주세요."
        ) from exc


@app.post("/api/properties/owner-asset-ratio")
def owner_asset_ratio(body: OwnerAssetRatioIn):
    """Estimate D/A only when a real-data artifact has been approved."""
    session = _SESSIONS.get(body.session_id)
    if session is None:
        raise HTTPException(404, "session not found. POST /session first.")
    pipeline = _load_owner_ratio_pipeline()
    if pipeline is None:
        raise HTTPException(
            409,
            "수원시 건축물대장과 가계금융복지조사 마이크로데이터로 "
            "학습·검증한 actual artifact가 아직 없습니다. 합성 smoke "
            "artifact는 사용자 위험도에 사용하지 않습니다.",
        )
    prop = _property_report.property(body.property_id)
    if prop is None:
        raise HTTPException(404, "property not found")
    run_id = _audit.start_run(
        session_id=body.session_id,
        property_id=body.property_id,
        input_snapshot={
            "samples": body.samples,
            "seed": body.seed,
            "occupancy_scenario": body.occupancy_scenario,
            "random_effect_sigma": body.random_effect_sigma,
        },
        simulation_seed=body.seed,
        model_versions={
            "owner_asset_ratio": pipeline.metadata.get("model_version")},
        data_version=pipeline.metadata.get("trained_at"),
    )
    started = time.perf_counter()
    try:
        result = pipeline.infer(
            prop,
            samples=body.samples,
            seed=body.seed,
            occupancy_scenario=body.occupancy_scenario,
            random_effect_sigma=body.random_effect_sigma,
        )
        result["decision_run_id"] = run_id
        _audit.record_step(
            run_id,
            stage="four_component_owner_asset_ratio",
            tool="monte_carlo",
            input_data={
                "property_id": body.property_id,
                "samples": body.samples,
                "seed": body.seed,
            },
            output_data={
                "ratio": result.get("deposit_to_total_assets_ratio"),
                "data_quality": result.get("data_quality"),
                "owner_prior_fallback_counts": result.get(
                    "owner_prior_fallback_counts"),
            },
            source_refs=[
                "building_registry", "rtms_sh_rent", "rtms_sh_trade",
                "household_finance_welfare_survey_prior",
            ],
        )
        _audit.complete_run(
            run_id, result,
            elapsed_ms=(time.perf_counter() - started) * 1000)
        return result
    except Exception as exc:
        _audit.fail_run(
            run_id, exc,
            elapsed_ms=(time.perf_counter() - started) * 1000)
        logger.exception("owner asset ratio inference failed")
        raise HTTPException(
            400, "전세금/집주인 총자산 비율 추정을 완료하지 못했습니다.") from exc


@app.post("/api/properties/senior-deposit")
def senior_deposit(body: SeniorDepositIn):
    """Estimate existing/senior deposits without claiming legal certainty."""
    if _SESSIONS.get(body.session_id) is None:
        raise HTTPException(404, "session not found. POST /session first.")
    if not body.property_id and not body.building_id:
        raise HTTPException(422, "property_id 또는 building_id가 필요합니다.")
    if body.building_id:
        pipeline = _load_senior_deposit_pipeline()
    else:
        integration = _property_report.senior_deposit
        pipeline = (
            integration._load_pipeline() if integration.available else None
        )
    if pipeline is None:
        raise HTTPException(
            409,
            "실제 건축HUB·RTMS로 학습한 선순위 보증금 artifact가 없습니다.",
        )
    if body.building_id:
        source = _actual_building(body.building_id)
        if source is None:
            raise HTTPException(404, "building_id not found")
        audit_property_id = body.property_id or body.building_id
    else:
        source = _property_report.property(body.property_id or "")
        if source is None:
            raise HTTPException(404, "property not found")
        audit_property_id = body.property_id

    run_id = _audit.start_run(
        session_id=body.session_id,
        property_id=audit_property_id,
        input_snapshot={
            "property_id": body.property_id,
            "building_id": body.building_id,
            "reference_date": body.reference_date.isoformat(),
            "samples": body.samples,
            "seed": body.seed,
            "scenario": body.scenario,
            "occupancy_scenario": body.occupancy_scenario,
            "senior_probability": body.senior_probability,
            "random_effect_sigma": body.random_effect_sigma,
            "target_rooms_excluded": body.target_rooms_excluded,
        },
        simulation_seed=body.seed,
        model_versions={
            "senior_deposit": pipeline.metadata.get("model_version")},
        data_version=pipeline.metadata.get("trained_at"),
    )
    started = time.perf_counter()
    try:
        if body.building_id:
            result = pipeline.infer(
                source,
                reference_date=body.reference_date.isoformat(),
                samples=body.samples,
                seed=body.seed,
                mode=body.scenario,
                occupancy_scenario=body.occupancy_scenario,
                senior_probability=body.senior_probability,
                random_effect_sigma=body.random_effect_sigma,
                target_rooms_excluded=body.target_rooms_excluded,
            )
            estimate = result
        else:
            result = _property_report.senior_deposit.analyze_property(
                source,
                reference_date=body.reference_date.isoformat(),
                samples=body.samples,
                seed=body.seed,
                mode=body.scenario,
                occupancy_scenario=body.occupancy_scenario,
                senior_probability=body.senior_probability,
                random_effect_sigma=body.random_effect_sigma,
                target_rooms_excluded=body.target_rooms_excluded,
            )
            estimate = result.get("estimate") or {}
        result["decision_run_id"] = run_id
        _audit.record_step(
            run_id,
            stage="senior_deposit_mvp",
            tool="monte_carlo",
            input_data={
                "reference_date": body.reference_date.isoformat(),
                "samples": body.samples,
                "scenario": body.scenario,
            },
            output_data={
                "available": result.get("available", True),
                "status": result.get("status", "estimated"),
                "match": result.get("match"),
                "senior_deposit": estimate.get("estimated_senior_deposit"),
                "conservative_upper": estimate.get(
                    "conservative_upper_deposit"),
                "data_quality": estimate.get("data_quality"),
                "model_mode": estimate.get("model_mode"),
            },
            source_refs=[
                "building_registry",
                "rtms_sh_rent_conditional_distribution",
                "declared_occupancy_prior",
                "declared_seniority_scenario",
            ],
            fallback=(
                {"reason": "verified legal-seniority labels unavailable"}
                if estimate.get("model_mode") == "scenario_only" else None
            ),
        )
        _audit.complete_run(
            run_id, result,
            elapsed_ms=(time.perf_counter() - started) * 1000)
        return result
    except Exception as exc:
        _audit.fail_run(
            run_id, exc,
            elapsed_ms=(time.perf_counter() - started) * 1000)
        logger.exception("senior deposit inference failed")
        raise HTTPException(
            400, "선순위 임차보증금 추정을 완료하지 못했습니다.") from exc


@app.post("/api/optimization/pareto")
def optimize_pareto(body: OptimizationIn):
    """현재 검색 교집합 안에서 매물·금융상품·대출액을 MILP로 선택한다."""
    session = _SESSIONS.get(body.session_id)
    if session is None:
        raise HTTPException(404, "session not found. POST /session first.")
    ui = session.setdefault("map_ui", {})
    properties = list(ui.get("last_search_properties") or [])
    if body.property_ids is not None:
        wanted = set(body.property_ids)
        properties = [row for row in properties if row.get("property_id") in wanted]
    if not properties:
        raise HTTPException(409, "먼저 지도에서 매물을 검색해 주세요.")
    user = session["user"]
    region = f"{user.get('preferred_sido', '')} {user.get('preferred_gugun', '')}".strip()
    programs = _agent.finance_tool.search(
        user_income_manwon=user.get("monthly_income_manwon"),
        user_age=user.get("age"), region=region, finance_mode="eligibility",
        user_profile=user, limit=50,
    )
    preference = normalize_preferences(user.get("preferences"))
    seed = uuid.uuid4().int % 2_147_483_647
    run_id = _audit.start_run(
        session_id=body.session_id, property_id=None,
        input_snapshot={
            "property_ids": [row.get("property_id") for row in properties],
            "preferences": preference,
            "horizon_years": body.horizon_years,
        },
        simulation_seed=seed,
        model_versions={"optimizer": "scipy_milp_pareto_v1"},
        data_version="current_search_intersection",
    )
    started = time.perf_counter()
    try:
        result = optimize_housing_choices(
            properties, programs, user, preference,
            {
                "horizon_years": body.horizon_years,
                "requested_loan_amount_manwon": body.requested_loan_amount_manwon,
            },
        )
        result["decision_run_id"] = run_id
        _audit.record_step(
            run_id, stage="milp_pareto_optimization", tool="scipy.optimize.milp",
            input_data={
                "property_count": len(properties), "finance_count": len(programs),
                "preference": preference,
            },
            output_data={
                "status": result.get("status"),
                "candidate_count": result.get("candidates_evaluated"),
                "pareto_count": result.get("pareto_candidate_count"),
                "representative_ids": [
                    row.get("option_id") for row in result.get("representatives") or []
                ],
            },
            source_refs=["current_search_intersection", "finance_programs"],
        )
        _audit.complete_run(
            run_id, result, elapsed_ms=(time.perf_counter() - started) * 1000)
        return result
    except Exception as exc:
        _audit.fail_run(
            run_id, exc, elapsed_ms=(time.perf_counter() - started) * 1000)
        logger.exception("MILP optimization failed")
        raise HTTPException(400, "최적 조합을 계산하지 못했습니다.") from exc


@app.get("/api/decisions/{decision_run_id}")
def decision_audit(decision_run_id: str):
    record = _audit.get(decision_run_id)
    if record is None:
        raise HTTPException(404, "decision run not found")
    return record


@app.get("/api/map/geocode")
def map_geocode(query: str = Query(min_length=1, max_length=200)):
    try:
        return _agent.map_tool.geocode(query)
    except Exception as exc:
        logger.exception("geocoding failed")
        raise HTTPException(502, "장소를 찾지 못했습니다. 주소를 확인해 주세요.") from exc


@app.get("/api/map/reverse-geocode")
def map_reverse_geocode(lat: float, lng: float):
    try:
        return _agent.map_tool.reverse_geocode(lat, lng)
    except Exception as exc:
        logger.exception("reverse geocoding failed")
        raise HTTPException(502, "지도 위치 정보를 확인하지 못했습니다.") from exc


@app.get("/api/map/static")
def static_map(lat: float, lng: float, width: int = 640, height: int = 360,
               zoom: int = 15):
    try:
        content, media_type = _agent.map_tool.static_map(
            lat=lat, lng=lng, width=width, height=height, zoom=zoom)
        return Response(content=content, media_type=media_type,
                        headers={"Cache-Control": "private, max-age=300"})
    except Exception as exc:
        logger.exception("static map failed")
        raise HTTPException(502, "지도 이미지를 준비하지 못했습니다.") from exc


@app.post("/chat")
def chat(body: ChatIn):
    session = _SESSIONS.get(body.session_id)
    if session is None:
        raise HTTPException(404, "session not found. POST /session first.")
    try:
        return _agent.handle(session, body.text)
    except Exception:
        logger.exception("agent chat request failed")
        raise HTTPException(500, "상담 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.")


def _enrich_advisor_recommendations(session: dict, result: dict) -> dict:
    """Agent 추천 ID를 지도/카드에서 바로 쓸 수 있는 현재 DB 행으로 보강한다."""
    groups = result.get("groups")
    if not isinstance(groups, dict):
        return result
    enriched_groups: dict = {}
    map_rows: list[dict] = []
    seen: set[str] = set()
    for key, values in groups.items():
        enriched = []
        for recommendation in values or []:
            property_id = str((recommendation or {}).get("property_id") or "")
            prop = _property_report.property(property_id) if property_id else None
            if not prop:
                enriched.append(recommendation)
                continue
            public = _property_report._public_property(prop)
            public.update(recommendation)
            enriched.append(public)
            if property_id not in seen:
                map_rows.append(public)
                seen.add(property_id)
        enriched_groups[key] = enriched
    result["groups"] = enriched_groups
    result["recommended_properties"] = map_rows
    if map_rows:
        ui = session.setdefault("map_ui", {})
        ui["last_advisor_properties"] = copy.deepcopy(map_rows)
    return result


@app.post("/api/advisor/chat")
def advisor_chat(body: ChatIn):
    """조건 편집과 분리된 의사결정·즉시 추천 채널."""
    session = _SESSIONS.get(body.session_id)
    if session is None:
        raise HTTPException(404, "session not found. POST /session first.")
    try:
        history = session.setdefault("advisor_chat", [])
        result = _agent.handle(
            session, body.text, direct_recommend=True,
            conversation_history=copy.deepcopy(history))
        result = _enrich_advisor_recommendations(session, result)
        result["advisor_channel"] = True
        result = json_safe(result)
        planner = (result.get("agent_trace") or {}).get("planner") or {}
        history.extend([
            {"role": "user", "text": body.text},
            {
                "role": "assistant",
                "text": (
                    result.get("answer") or result.get("message")
                    or "추천 결과를 지도에 표시했습니다."
                ),
                "status": result.get("status"),
                "intent": planner.get("intent"),
                "slots": planner.get("slots") or {},
                "qa_args": planner.get("qa_args") or {},
                "recommended_property_ids": [
                    row.get("property_id")
                    for row in result.get("recommended_properties") or []
                    if row.get("property_id")
                ],
            },
        ])
        result["conversation_memory"] = {
            "stored_entries": len(history),
            "prior_entries_used": max(0, len(history) - 2),
        }
        session["last_advisor_result"] = copy.deepcopy(result)
        return result
    except Exception:
        logger.exception("advisor chat request failed")
        raise HTTPException(
            500, "AI 추천·상담을 완료하지 못했습니다. 잠시 후 다시 시도해 주세요.")


@app.post("/fraud/score")
def fraud_score(prop: dict):
    """단일 매물 위험도. body는 PropertyRecord 필드 dict."""
    try:
        return _scorer.score(prop)
    except Exception:
        logger.exception("fraud score request failed")
        raise HTTPException(400, "매물 위험도 요청 형식을 확인해 주세요.")
