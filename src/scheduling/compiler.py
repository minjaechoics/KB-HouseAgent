"""Compile condition atoms into the scheduler's tool-level DAG."""
from __future__ import annotations

from .core import ResourceLimit, ResourceLimits, TaskGraph, TaskNode


_SQL_REDUCTION = {
    "transaction_type": 0.65,
    "house_type": 0.55,
    "sido": 0.80,
    "gugun": 0.85,
    "deposit_manwon": 0.50,
    "sale_price_manwon": 0.50,
    "monthly_rent_manwon": 0.45,
    "maintenance_fee_manwon": 0.25,
    "area_m2": 0.30,
    "building_age_years": 0.20,
}


def condition_resource_limits(
    *,
    sqlite_capacity: int = 2,
    route_capacity: int = 3,
) -> ResourceLimits:
    return ResourceLimits(resources={
        "sqlite": ResourceLimit(capacity=max(1, sqlite_capacity)),
        "cpu": ResourceLimit(capacity=2),
        "naver_directions": ResourceLimit(capacity=max(1, route_capacity)),
        "tmap_transit": ResourceLimit(capacity=max(1, route_capacity)),
        "map_estimate": ResourceLimit(capacity=4),
    })


def compile_condition_graph(
    atoms: list[dict],
    *,
    route_candidate_limit: int = 5,
    unlimited_route_modes: set[str] | None = None,
) -> TaskGraph:
    tasks: list[TaskNode] = []
    initial = next(
        (
            atom for atom in atoms
            if atom.get("scope_role") == "initial_universe"
            or atom.get("field") == "initial_scope"
        ),
        None,
    )
    root_id = "sql:initial_scope"
    tasks.append(TaskNode(
        root_id, 12, "sqlite",
        candidate_reduction=0.75 if initial else 0.0,
        user_importance=2.0,
        metadata={"kind": "initial_scope", "atom_id": (initial or {}).get("id")},
    ))
    sql_ids: list[str] = []
    commute_atoms: list[dict] = []
    for atom in atoms:
        if atom is initial:
            continue
        if atom.get("field") == "commute_minutes":
            commute_atoms.append(atom)
            continue
        task_id = f"sql:{atom['id']}"
        sql_ids.append(task_id)
        tasks.append(TaskNode(
            task_id, 8, "sqlite", dependencies=(root_id,),
            candidate_reduction=_SQL_REDUCTION.get(atom.get("field"), 0.15),
            user_importance=1.5,
            metadata={
                "kind": "sql_atom", "atom_id": atom.get("id"),
                "field": atom.get("field"),
            },
        ))
    narrowing = tuple([root_id, *sql_ids])
    commute_join_ids: list[str] = []
    for atom in commute_atoms:
        atom_id = str(atom["id"])
        prefilter_id = f"spatial:{atom_id}"
        tasks.append(TaskNode(
            prefilter_id, 15, "sqlite", dependencies=narrowing,
            candidate_reduction=0.80, user_importance=2.0,
            metadata={"kind": "spatial_prefilter", "atom_id": atom_id},
        ))
        mode = str(atom.get("mode") or "transit")
        provider = (
            "tmap_transit" if mode == "transit"
            else "naver_directions" if mode == "driving"
            else "map_estimate"
        )
        unlimited = mode in (unlimited_route_modes or set())
        route_ids = []
        route_task_count = 1 if unlimited else max(1, int(route_candidate_limit))
        for index in range(route_task_count):
            route_id = f"route:{atom_id}:{index}"
            route_ids.append(route_id)
            tasks.append(TaskNode(
                route_id,
                1_500 if provider != "map_estimate" else 3,
                provider,
                dependencies=(prefilter_id,),
                monetary_cost=(0.001 if provider != "map_estimate" else 0.0),
                candidate_reduction=0.10,
                user_importance=2.0,
                metadata={
                    "kind": ("route_candidate_batch" if unlimited
                             else "route_candidate"),
                    "unlimited_candidate_batch": unlimited,
                    "atom_id": atom_id,
                    "candidate_index": index, "mode": mode,
                },
            ))
        join_id = f"intersection:{atom_id}"
        commute_join_ids.append(join_id)
        tasks.append(TaskNode(
            join_id, 4, "cpu", dependencies=tuple(route_ids),
            candidate_reduction=0.50, user_importance=2.0,
            metadata={"kind": "commute_intersection", "atom_id": atom_id},
        ))
    tasks.append(TaskNode(
        "sql:final_fetch", 18, "sqlite",
        dependencies=tuple([*narrowing, *commute_join_ids]),
        user_importance=2.0,
        metadata={"kind": "final_fetch"},
    ))
    return TaskGraph(tasks)


def compile_report_enrichment_graph() -> TaskGraph:
    """Independent LLM enrichments after the deterministic report is visible."""
    return TaskGraph([
        TaskNode(
            "llm:news_assessment", 15_000, "openai",
            expected_tokens=3_000, monetary_cost=0.003,
            user_importance=1.5,
        ),
        TaskNode(
            "llm:metric_explanations", 12_000, "openai",
            expected_tokens=4_000, monetary_cost=0.004,
            user_importance=1.2,
        ),
    ])


def report_resource_limits(openai_capacity: int = 6) -> ResourceLimits:
    return ResourceLimits(resources={
        "openai": ResourceLimit(
            capacity=max(1, openai_capacity),
            requests_per_minute=500,
            tokens_per_minute=200_000,
        )
    })
