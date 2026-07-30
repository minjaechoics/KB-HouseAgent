"""Fair optimized-vs-naive experiment pipelines and grounded scoring."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from src import config
from src.agent.llm import BaseLLM
from src.experiments.cases import ExperimentCase
from src.experiments.naive_baseline import naive_decision
from src.server.property_search import (
    DISPLAY_COLUMNS,
    AtomicPropertySearch,
    _sort_sql,
    atoms_from_profile,
    atoms_from_slots,
    make_initial_scope_atom,
)
from src.tools.map_tool import MapTool


BASE_PROFILE = {
    "preferred_sido": "경기",
    "preferred_gugun": "수원시 팔달구",
    # Both modes receive the same non-restrictive personal context.
    "age": 29,
    "monthly_income_manwon": 320,
    "assets_manwon": 7000,
}


def _experiment_map_tool() -> MapTool:
    """Disable unrelated paid route/geocode traffic in this retrieval benchmark.

    None of the 50 answer keys contains a commute condition.  This also prevents
    a bad baseline extraction of a distractor number from turning into an unfair
    network timeout; that bad extraction is still visible and scored as wrong.
    """
    tool = MapTool(timeout_seconds=0.5)
    tool.online = False
    tool.tmap_online = False
    return tool

def database_fingerprint(db_path: Path = config.DB_PATH) -> str:
    stat = db_path.stat()
    raw = f"{db_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _compact_slots(slots: dict[str, Any] | None) -> dict[str, Any]:
    clean = {key: value for key, value in (slots or {}).items()
             if value is not None and value != [] and not key.startswith("_")}
    if not clean.get("transaction_type") and clean.get("lease_type"):
        clean["transaction_type"] = clean["lease_type"]
    clean.pop("lease_type", None)
    return clean


def _agentic_context() -> dict[str, Any]:
    """Identical planner context for optimized and NAIVE experiment arms."""
    return {
        "state": "idle",
        "known_slots": {},
        "proposed_slots": {},
        "initial_profile": dict(BASE_PROFILE),
        "initial_universe_policy": "fixed AND intersection",
    }


def agentic_decision_for_case(case: ExperimentCase, llm: BaseLLM) -> dict[str, Any]:
    """Run the exact production Agentic prompt/schema in both experiment arms."""
    decision = llm.plan_condition_dialogue(case.query, _agentic_context())
    if (llm.supports_agentic_calls
            and bool((decision.get("_trace") or {}).get("fallback"))):
        raise RuntimeError(
            "live Agentic LLM call failed; rule fallback is prohibited in experiments"
        )
    return decision


class NaivePropertySearch:
    """One serial parameterized query with no atom scheduling or intersections."""

    def __init__(self, db_path: Path = config.DB_PATH):
        self.db_path = Path(db_path)
        self._clause_builder = AtomicPropertySearch(db_path=self.db_path)

    def _conn(self):
        uri = self.db_path.resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        return connection

    def search(self, slots: dict[str, Any], source_text: str,
               limit: int = 5, parse_source_patterns: bool = False) -> dict[str, Any]:
        map_tool = _experiment_map_tool()
        base_atoms = atoms_from_profile(BASE_PROFILE)
        # The NAIVE baseline must not receive the production pipeline's second
        # deterministic extraction pass: conditions come only from its one LLM call.
        parsed_atoms, notes = atoms_from_slots(
            slots, source_text if parse_source_patterns else "", map_tool)
        # A one-shot SQL has one predicate per field; the latest parsed value wins.
        by_field = {atom["field"]: atom for atom in [*base_atoms, *parsed_atoms]
                    if atom.get("field") != "commute_minutes"}
        clauses: list[str] = []
        params: list[Any] = []
        for atom in by_field.values():
            clause, values = self._clause_builder._clause(atom)
            clauses.append(f"({clause})")
            params.extend(values)
        where = " AND ".join(clauses) or "1=1"
        where += " AND (listing_status IS NULL OR listing_status!='expired')"
        sort_by = str(slots.get("sort_by") or "recommended")
        if sort_by not in {"recommended", "risk_asc", "risk_desc", "price_asc", "price_desc"}:
            sort_by = "recommended"
        order_sql, order_params = _sort_sql(sort_by, None)
        select_sql = (
            "SELECT " + ", ".join(DISPLAY_COLUMNS)
            + f" FROM properties WHERE {where} ORDER BY {order_sql} LIMIT ?"
        )
        count_sql = f"SELECT COUNT(*) FROM properties WHERE {where}"
        started = time.perf_counter()
        with self._conn() as connection:
            total = int(connection.execute(count_sql, params).fetchone()[0])
            rows = [dict(row) for row in connection.execute(
                select_sql, [*params, *order_params, int(limit)])]
        elapsed = time.perf_counter() - started
        return {
            "total": total,
            "returned": len(rows),
            "properties": rows,
            "trace": {
                "pipeline": "naive_one_shot_single_serial_sql",
                "atomic_processing": False,
                "parallel_processing": False,
                "dependency_scheduling": False,
                "sql": select_sql,
                "parameters": [*params, *order_params, int(limit)],
                "condition_count": len(by_field),
                "db_elapsed_seconds": elapsed,
                "notes": notes,
            },
        }


def _optimized_run(case: ExperimentCase, llm: BaseLLM,
                   db_path: Path) -> dict[str, Any]:
    planning_started = time.perf_counter()
    decision = agentic_decision_for_case(case, llm)
    planning_elapsed = time.perf_counter() - planning_started
    slots = _compact_slots(decision.get("slots") or {})
    map_tool = _experiment_map_tool()
    initial = make_initial_scope_atom(atoms_from_profile(BASE_PROFILE))
    atoms, notes = atoms_from_slots(slots, case.query, map_tool)
    all_atoms = ([initial] if initial else []) + atoms
    retrieval_started = time.perf_counter()
    result = AtomicPropertySearch(db_path=db_path, map_tool=map_tool).search(
        all_atoms, limit=5, sort_by=str(slots.get("sort_by") or "recommended"),
    )
    retrieval_elapsed = time.perf_counter() - retrieval_started
    return {
        "llm_output": decision,
        "parsed_slots": slots,
        "recommendation": _recommendation_text(result.get("properties") or []),
        "search": result,
        "rag_trace": {
            "mode": "optimized",
            "planner": decision,
            "planner_prompt_policy": "identical_production_agentic_prompt",
            "atomic_conditions": all_atoms,
            "condition_notes": notes,
            "retrieval": result.get("trace"),
            "timing": {
                "planner_seconds": planning_elapsed,
                "retrieval_seconds": retrieval_elapsed,
            },
        },
    }


def _naive_run(case: ExperimentCase, llm: BaseLLM,
               db_path: Path) -> dict[str, Any]:
    planning_started = time.perf_counter()
    if getattr(llm, "experiment_shared_decision_replay", False):
        # alg_test.py is a separate controlled retrieval-only ablation.  It
        # intentionally replays the same decision into both search algorithms.
        decision = agentic_decision_for_case(case, llm)
        prompt_policy = "shared_agentic_decision_replay_for_alg_test"
    else:
        # speed_test.py's NAIVE arm receives the whole user prompt in one flat
        # LLM request and does not reuse the production Agentic prompt.
        decision = naive_decision(llm, case.query, context={
            "initial_profile": dict(BASE_PROFILE),
            "initial_universe_policy": "fixed AND intersection",
        })
        prompt_policy = "naive_whole_prompt_single_pass"
    planning_elapsed = time.perf_counter() - planning_started
    slots = _compact_slots(decision.get("slots") or {})
    retrieval_started = time.perf_counter()
    result = NaivePropertySearch(db_path).search(slots, case.query, limit=5)
    retrieval_elapsed = time.perf_counter() - retrieval_started
    return {
        "llm_output": decision,
        "parsed_slots": slots,
        "recommendation": _recommendation_text(result.get("properties") or []),
        "search": result,
        "rag_trace": {
            "mode": "naive",
            "planner": decision,
            "planner_prompt_policy": prompt_policy,
            "atomic_conditions": None,
            "retrieval": result.get("trace"),
            "timing": {
                "planner_seconds": planning_elapsed,
                "retrieval_seconds": retrieval_elapsed,
            },
        },
    }


def run_case(case: ExperimentCase, mode: str, llm: BaseLLM,
             db_path: Path = config.DB_PATH) -> dict[str, Any]:
    if mode not in {"optimized", "naive"}:
        raise ValueError(f"unsupported experiment mode: {mode}")
    started = time.perf_counter()
    try:
        result = (_optimized_run(case, llm, Path(db_path)) if mode == "optimized"
                  else _naive_run(case, llm, Path(db_path)))
        result["error"] = None
    except Exception as exc:
        result = {
            "llm_output": None, "parsed_slots": {}, "recommendation": "추천 실패",
            "search": {"total": 0, "returned": 0, "properties": []},
            "rag_trace": {"mode": mode, "error": f"{type(exc).__name__}: {exc}"},
            "error": f"{type(exc).__name__}: {exc}",
        }
    result.update({
        "case_id": case.case_id,
        "query": case.query,
        "mode": mode,
        "elapsed_seconds": time.perf_counter() - started,
    })
    return result


def _recommendation_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "조건을 만족하는 추천 매물을 찾지 못했습니다."
    lines = [f"DB 근거 추천 {len(rows)}건"]
    for index, row in enumerate(rows, 1):
        transaction = row.get("transaction_type")
        if transaction == "매매":
            price = f"매매 {float(row.get('sale_price_manwon') or row.get('asking_price_manwon') or 0):,.0f}만원"
        elif transaction == "전세":
            price = f"전세 {float(row.get('deposit_manwon') or 0):,.0f}만원"
        else:
            price = (f"월세 {float(row.get('deposit_manwon') or 0):,.0f}/"
                     f"{float(row.get('monthly_rent_manwon') or 0):,.0f}만원")
        lines.append(
            f"  {index}. [{row.get('property_id')}] {row.get('house_type')} · {price} · "
            f"{float(row.get('area_m2') or 0):.1f}㎡ · {row.get('road_address') or row.get('dong')}"
        )
    return "\n".join(lines)


def build_ground_truth(case: ExperimentCase,
                       db_path: Path = config.DB_PATH) -> dict[str, Any]:
    oracle = NaivePropertySearch(db_path).search(
        case.expected_slots, "", limit=2000,
    )
    return {
        "expected_slots": case.expected_slots,
        "rationale": case.rationale,
        "matching_property_count": oracle["total"],
        "valid_property_ids": [row["property_id"] for row in oracle["properties"]],
        "oracle_sql": oracle["trace"]["sql"],
        "oracle_parameters": oracle["trace"]["parameters"],
    }


def _canonical_text(field: str, value: Any) -> Any:
    if field == "region_sido":
        return str(value or "").replace("경기도", "경기").strip()
    if field == "property_type":
        aliases = {"다가구": "다가구주택", "다세대": "다세대주택",
                   "단독": "단독주택", "연립": "연립주택"}
        return aliases.get(str(value or "").strip(), str(value or "").strip())
    if field == "region_gugun":
        values = value if isinstance(value, list) else [value]
        return sorted(str(item).strip() for item in values if item)
    return value


def score_result(case: ExperimentCase, run: dict[str, Any],
                 ground_truth: dict[str, Any]) -> dict[str, Any]:
    parsed = run.get("parsed_slots") or {}
    slot_errors: list[str] = []
    for field, expected in case.expected_slots.items():
        actual = parsed.get(field)
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            try:
                equal = abs(float(actual) - float(expected)) < 1e-6
            except (TypeError, ValueError):
                equal = False
        else:
            equal = _canonical_text(field, actual) == _canonical_text(field, expected)
        if not equal:
            slot_errors.append(f"{field}: expected={expected!r}, actual={actual!r}")

    valid_ids = set(ground_truth["valid_property_ids"])
    rows = (run.get("search") or {}).get("properties") or []
    result_ids = [row.get("property_id") for row in rows]
    invalid_ids = [value for value in result_ids if value not in valid_ids]
    reasons: list[str] = []
    if run.get("error"):
        reasons.append("pipeline_error")
    if slot_errors:
        reasons.append("condition_extraction_mismatch")
    if ground_truth["matching_property_count"] <= 0:
        reasons.append("invalid_benchmark_case_no_oracle_match")
    if not rows:
        reasons.append("no_recommendation")
    if invalid_ids:
        reasons.append("recommendation_outside_ground_truth")

    # A sorted request must put the oracle's optimum first.  This catches a
    # plausible-looking but unsupported "best" claim.
    order_ok = True
    if rows and ground_truth["valid_property_ids"]:
        order_ok = result_ids[0] == ground_truth["valid_property_ids"][0]
        if not order_ok:
            reasons.append("wrong_top_rank")
    correct = not reasons
    return {
        "correct": correct,
        "slot_accuracy": round(
            (len(case.expected_slots) - len(slot_errors)) / len(case.expected_slots), 4),
        "slot_errors": slot_errors,
        "recommended_ids": result_ids,
        "invalid_recommended_ids": invalid_ids,
        "ranking_correct": order_ok,
        "reasons": reasons,
    }


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)
