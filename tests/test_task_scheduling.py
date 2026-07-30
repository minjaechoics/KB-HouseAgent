from __future__ import annotations

import threading

from src.agent.llm import MockLLM
from src.scheduling import (
    BranchAndBoundScheduler, CPSATScheduler, FIFOScheduler, HEFTScheduler,
    PortfolioScheduler, ResourceLimit, ResourceLimits,
    RollingHorizonScheduler, TaskGraph, TaskNode, TreewidthDPScheduler,
    approximate_treewidth, compile_condition_graph, validate_schedule,
)


def sample_graph() -> TaskGraph:
    return TaskGraph([
        TaskNode("parse", 10, "cpu"),
        TaskNode("sql_a", 40, "sqlite", ("parse",), candidate_reduction=.8),
        TaskNode("sql_b", 30, "sqlite", ("parse",), candidate_reduction=.5),
        TaskNode("route_a", 100, "route", ("sql_a", "sql_b")),
        TaskNode("route_b", 100, "route", ("sql_a", "sql_b")),
        TaskNode("join", 10, "cpu", ("route_a", "route_b")),
    ])


def sample_limits(sqlite_capacity: int = 2) -> ResourceLimits:
    return ResourceLimits(resources={
        "cpu": ResourceLimit(1),
        "sqlite": ResourceLimit(sqlite_capacity),
        "route": ResourceLimit(2),
    })


def test_all_schedulers_produce_valid_schedule():
    graph = sample_graph()
    limits = sample_limits()
    algorithms = [
        FIFOScheduler(), HEFTScheduler(), BranchAndBoundScheduler(),
        CPSATScheduler(), TreewidthDPScheduler(),
    ]
    for algorithm in algorithms:
        result = algorithm.schedule(graph, limits, deadline_ms=500)
        assert result.feasible
        assert not validate_schedule(graph, limits, result), (
            algorithm.name, result.to_dict()
        )


def test_exact_solvers_match_expected_parallel_makespan():
    graph = sample_graph()
    limits = sample_limits()
    branch = BranchAndBoundScheduler().schedule(
        graph, limits, deadline_ms=1_000
    )
    cp_sat = CPSATScheduler().schedule(graph, limits, deadline_ms=1_000)
    assert branch.makespan_ms == 160
    if cp_sat.status != "dependency_unavailable":
        assert cp_sat.makespan_ms == 160


def test_treewidth_dp_is_exact_for_unit_capacity_resources():
    graph = sample_graph()
    limits = sample_limits(sqlite_capacity=1)
    limits.resources["route"] = ResourceLimit(1)
    width, order = approximate_treewidth(graph)
    assert width <= 3
    assert set(order) == set(graph.tasks)
    result = TreewidthDPScheduler().schedule(
        graph, limits, deadline_ms=1_000
    )
    assert result.status == "optimal"
    assert result.optimal
    assert not validate_schedule(graph, limits, result)


def test_portfolio_retains_candidates_and_selects_valid_plan():
    result = PortfolioScheduler(deadline_ms=300).schedule(
        sample_graph(), sample_limits()
    )
    assert result.algorithm in {
        "topological_fifo", "cost_aware_heft", "branch_and_bound",
        "cp_sat", "treewidth_dp",
    }
    assert len(result.metadata["portfolio"]) == 5
    assert result.metadata["selection_rule"].startswith(
        "parallel algorithm race"
    )


def test_condition_compiler_preserves_expensive_route_precedence():
    atoms = [
        {
            "id": "initial", "field": "initial_scope",
            "scope_role": "initial_universe",
        },
        {"id": "rent", "field": "monthly_rent_manwon"},
        {
            "id": "commute", "field": "commute_minutes",
            "mode": "transit",
        },
    ]
    graph = compile_condition_graph(atoms, route_candidate_limit=5)
    assert graph.tasks["sql:rent"].dependencies == ("sql:initial_scope",)
    assert set(graph.tasks["spatial:commute"].dependencies) == {
        "sql:initial_scope", "sql:rent",
    }
    routes = [task for task in graph.tasks if task.startswith("route:commute")]
    assert len(routes) == 5
    assert all(
        graph.tasks[task].resource == "tmap_transit" for task in routes
    )
    assert set(graph.tasks["intersection:commute"].dependencies) == set(routes)


def test_condition_compiler_represents_unlimited_tmap_as_dynamic_batch():
    atoms = [{
        "id": "commute", "field": "commute_minutes", "mode": "transit",
    }]
    graph = compile_condition_graph(
        atoms, route_candidate_limit=5, unlimited_route_modes={"transit"},
    )
    routes = [task for task in graph.tasks if task.startswith("route:commute")]
    assert len(routes) == 1
    route = graph.tasks[routes[0]]
    assert route.metadata["kind"] == "route_candidate_batch"
    assert route.metadata["unlimited_candidate_batch"] is True


def test_rolling_horizon_only_repairs_unfinished_tasks():
    graph = sample_graph()
    limits = sample_limits()
    initial = HEFTScheduler().schedule(graph, limits)
    repaired = RollingHorizonScheduler(
        PortfolioScheduler(
            algorithms=[HEFTScheduler()], deadline_ms=20
        )
    ).repair(
        graph, limits, initial, {"parse", "sql_a", "sql_b"},
        actual_duration_ms={"route_a": 180},
        current_time_ms=80,
    )
    assert repaired.algorithm.startswith("rolling_horizon:")
    assert {task.task_id for task in repaired.tasks} == {
        "route_a", "route_b", "join",
    }
    assert min(task.start_ms for task in repaired.tasks) >= 80


def test_list_scheduler_respects_rpm_and_tpm_admission_windows():
    graph = TaskGraph([
        TaskNode("prompt_a", 1_000, "openai", expected_tokens=700),
        TaskNode("prompt_b", 1_000, "openai", expected_tokens=700),
    ])
    limits = ResourceLimits(resources={
        "openai": ResourceLimit(
            capacity=2, requests_per_minute=1, tokens_per_minute=1_000
        )
    })
    result = HEFTScheduler().schedule(graph, limits)
    starts = sorted(task.start_ms for task in result.tasks)
    assert starts == [0, 60_000]
    assert not validate_schedule(graph, limits, result)


def test_llm_audit_trace_is_isolated_between_parallel_threads():
    llm = MockLLM()
    barrier = threading.Barrier(2)
    observed = {}

    def worker(name):
        llm.last_trace = [{"worker": name}]
        barrier.wait()
        observed[name] = llm.last_trace

    first = threading.Thread(target=worker, args=("a",))
    second = threading.Thread(target=worker, args=("b",))
    first.start()
    second.start()
    first.join()
    second.join()
    assert observed == {
        "a": [{"worker": "a"}],
        "b": [{"worker": "b"}],
    }
