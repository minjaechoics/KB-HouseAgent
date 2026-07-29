"""Deterministic offline benchmark for the agent task-scheduling portfolio.

No external API is called. Durations are replay estimates in milliseconds.

Usage:
    python scripts/benchmark_agent_schedulers.py
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scheduling import (
    BranchAndBoundScheduler, CPSATScheduler, FIFOScheduler, HEFTScheduler,
    PortfolioScheduler, ResourceLimit, ResourceLimits, TaskGraph, TaskNode,
    TreewidthDPScheduler, approximate_treewidth, validate_schedule,
)


OUTPUT = ROOT / "reports" / "agent_scheduler_benchmark.json"


def fork_join(name: str, branches: int, capacity: int = 2):
    tasks = [TaskNode("parse", 20, "cpu")]
    for index in range(branches):
        tasks.append(TaskNode(
            f"sql_{index}", 20 + index * 2, "sqlite", ("parse",),
            candidate_reduction=max(.1, .8 - index * .05),
        ))
    sql_ids = tuple(f"sql_{index}" for index in range(branches))
    tasks.append(TaskNode("prefilter", 30, "sqlite", sql_ids))
    for index in range(min(5, branches + 1)):
        tasks.append(TaskNode(
            f"route_{index}", 900 + index * 50, "route", ("prefilter",),
            monetary_cost=.001,
        ))
    route_ids = tuple(
        task.id for task in tasks if task.id.startswith("route_")
    )
    tasks.append(TaskNode("join", 15, "cpu", route_ids))
    return name, TaskGraph(tasks), ResourceLimits(resources={
        "cpu": ResourceLimit(1),
        "sqlite": ResourceLimit(capacity),
        "route": ResourceLimit(min(3, capacity + 1)),
    })


def low_treewidth(name: str, length: int):
    tasks = [TaskNode("start", 10, "cpu")]
    previous = "start"
    for index in range(length):
        left = f"left_{index}"
        right = f"right_{index}"
        join = f"join_{index}"
        tasks.extend([
            TaskNode(left, 20 + index, "openai", (previous,)),
            TaskNode(right, 16 + index, "sqlite", (previous,)),
            TaskNode(join, 8, "cpu", (left, right)),
        ])
        previous = join
    return name, TaskGraph(tasks), ResourceLimits(resources={
        "cpu": ResourceLimit(1),
        "openai": ResourceLimit(1),
        "sqlite": ResourceLimit(1),
    })


def random_dag(name: str, count: int, seed: int):
    rng = random.Random(seed)
    resources = ("cpu", "sqlite", "openai", "route")
    tasks = []
    for index in range(count):
        possible = list(range(index))
        rng.shuffle(possible)
        dependencies = tuple(
            f"t{parent}" for parent in sorted(possible[:rng.randint(
                0, min(3, len(possible))
            )])
        )
        tasks.append(TaskNode(
            f"t{index}", rng.randint(10, 700), rng.choice(resources),
            dependencies,
            expected_tokens=rng.randint(0, 3500),
            candidate_reduction=rng.random(),
        ))
    return name, TaskGraph(tasks), ResourceLimits(resources={
        "cpu": ResourceLimit(2),
        "sqlite": ResourceLimit(2),
        "openai": ResourceLimit(6),
        "route": ResourceLimit(3),
    })


def main() -> None:
    cases = [
        fork_join("condition_small", 3),
        fork_join("condition_medium", 8),
        low_treewidth("series_parallel_small", 3),
        low_treewidth("series_parallel_medium", 6),
        random_dag("random_12", 12, 7),
        random_dag("random_20", 20, 11),
        random_dag("random_35", 35, 19),
    ]
    algorithms = [
        FIFOScheduler(), HEFTScheduler(), BranchAndBoundScheduler(),
        CPSATScheduler(), TreewidthDPScheduler(),
    ]
    rows = []
    for name, graph, limits in cases:
        reference = CPSATScheduler().schedule(graph, limits, deadline_ms=3_000)
        optimum = reference.makespan_ms
        width, _ = approximate_treewidth(graph)
        for algorithm in algorithms:
            result = algorithm.schedule(graph, limits, deadline_ms=300)
            errors = validate_schedule(graph, limits, result)
            rows.append({
                "case": name,
                "tasks": len(graph),
                "estimated_treewidth": width,
                "algorithm": algorithm.name,
                "status": result.status,
                "valid": not errors,
                "planning_time_ms": round(result.planning_time_ms, 3),
                "makespan_ms": result.makespan_ms,
                "reference_makespan_ms": optimum,
                "gap_to_reference": round(
                    max(0.0, (result.makespan_ms - optimum) / optimum), 6
                ) if optimum else 0.0,
            })
        portfolio = PortfolioScheduler(deadline_ms=100).schedule(graph, limits)
        rows.append({
            "case": name,
            "tasks": len(graph),
            "estimated_treewidth": width,
            "algorithm": "portfolio",
            "selected_algorithm": portfolio.algorithm,
            "status": portfolio.status,
            "valid": not validate_schedule(graph, limits, portfolio),
            "planning_time_ms": round(portfolio.planning_time_ms, 3),
            "makespan_ms": portfolio.makespan_ms,
            "reference_makespan_ms": optimum,
            "gap_to_reference": round(
                max(0.0, (portfolio.makespan_ms - optimum) / optimum), 6
            ) if optimum else 0.0,
        })
    summaries = {}
    for algorithm in [item.name for item in algorithms] + ["portfolio"]:
        selected = [row for row in rows if row["algorithm"] == algorithm]
        summaries[algorithm] = {
            "cases": len(selected),
            "valid_cases": sum(row["valid"] for row in selected),
            "mean_planning_time_ms": round(statistics.mean(
                row["planning_time_ms"] for row in selected
            ), 3),
            "p95_planning_time_ms": round(sorted(
                row["planning_time_ms"] for row in selected
            )[max(0, int(len(selected) * .95) - 1)], 3),
            "mean_gap_to_reference": round(statistics.mean(
                row["gap_to_reference"] for row in selected
            ), 6),
        }
    payload = {
        "benchmark": "agent_task_scheduler_v1",
        "external_api_calls": 0,
        "cases": len(cases),
        "summary": summaries,
        "results": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
