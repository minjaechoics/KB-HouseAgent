"""Scheduling algorithms used by the agent task portfolio."""
from __future__ import annotations

import math
import time
from functools import lru_cache
from itertools import combinations
from typing import Protocol

from .core import (
    ResourceLimit, ResourceLimits, ScheduleResult, ScheduledTask, TaskGraph, TaskNode,
    validate_schedule,
)

try:
    # Import once during application startup. Lazy importing inside the first
    # user search added about a second of avoidable cold-request latency.
    from ortools.sat.python import cp_model as _cp_model
except ImportError:  # optional fallback remains available for minimal installs
    _cp_model = None


class Scheduler(Protocol):
    name: str

    def schedule(
        self, graph: TaskGraph, limits: ResourceLimits,
        deadline_ms: int | None = None,
    ) -> ScheduleResult: ...


def _earliest_start(
    node: TaskNode,
    dependency_end: int,
    intervals: list[tuple[int, int, int, int]],
    limit: ResourceLimit,
) -> int:
    """Find the earliest resource-feasible non-preemptive start."""
    candidate = max(0, dependency_end)
    while True:
        end = candidate + node.duration_ms
        points = sorted(
            {candidate, end}
            | {
                value
                for left, right, _, _ in intervals
                if left < end and right > candidate
                for value in (max(candidate, left), min(end, right))
            }
        )
        conflict_end: int | None = None
        for left, right in zip(points, points[1:]):
            if left == right:
                continue
            demand = node.demand + sum(
                amount for start, finish, amount, _ in intervals
                if start < right and finish > left
            )
            if demand > limit.capacity:
                overlapping_ends = [
                    finish for start, finish, _, _ in intervals
                    if start < right and finish > left
                ]
                conflict_end = min(overlapping_ends) if overlapping_ends else right
                break
        if conflict_end is None:
            recent = [
                row for row in intervals
                if candidate - 60_000 < row[0] <= candidate
            ]
            request_exceeded = (
                limit.requests_per_minute is not None
                and len(recent) + 1 > limit.requests_per_minute
            )
            token_exceeded = (
                limit.tokens_per_minute is not None
                and sum(row[3] for row in recent) + node.expected_tokens
                > limit.tokens_per_minute
            )
            if not request_exceeded and not token_exceeded:
                return candidate
            candidate = min(row[0] + 60_000 for row in recent)
            continue
        candidate = max(candidate + 1, conflict_end)


def _schedule_priority_order(
    graph: TaskGraph,
    limits: ResourceLimits,
    priority_order: list[str],
) -> list[ScheduledTask]:
    ends: dict[str, int] = {}
    intervals: dict[str, list[tuple[int, int, int, int]]] = {}
    result: list[ScheduledTask] = []
    for task_id in priority_order:
        node = graph.tasks[task_id]
        dependency_end = max(
            (ends[dependency] for dependency in node.dependencies),
            default=0,
        )
        resource_intervals = intervals.setdefault(node.resource, [])
        start = _earliest_start(
            node, dependency_end, resource_intervals,
            limits.get(node.resource),
        )
        end = start + node.duration_ms
        resource_intervals.append(
            (start, end, node.demand, node.expected_tokens)
        )
        ends[task_id] = end
        result.append(ScheduledTask(task_id, start, end, node.resource))
    return sorted(result, key=lambda item: (item.start_ms, item.task_id))


def _critical_ranks(graph: TaskGraph) -> dict[str, float]:
    successors = graph.successors()

    @lru_cache(maxsize=None)
    def rank(task_id: str) -> float:
        node = graph.tasks[task_id]
        tail = max((rank(child) for child in successors[task_id]), default=0.0)
        return float(node.duration_ms) + tail

    return {task_id: rank(task_id) for task_id in graph.tasks}


def _priority_topological_order(
    graph: TaskGraph, score: dict[str, float]
) -> list[str]:
    indegree = {
        task_id: len(node.dependencies)
        for task_id, node in graph.tasks.items()
    }
    successors = graph.successors()
    ready = [task_id for task_id, value in indegree.items() if value == 0]
    order: list[str] = []
    while ready:
        ready.sort(key=lambda task_id: (-score[task_id], task_id))
        task_id = ready.pop(0)
        order.append(task_id)
        for child in successors[task_id]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    return order


class FIFOScheduler:
    name = "topological_fifo"

    def schedule(
        self, graph: TaskGraph, limits: ResourceLimits,
        deadline_ms: int | None = None,
    ) -> ScheduleResult:
        started = time.perf_counter()
        tasks = _schedule_priority_order(
            graph, limits, graph.topological_order()
        )
        elapsed = (time.perf_counter() - started) * 1000
        return ScheduleResult(
            self.name, tasks, max((x.end_ms for x in tasks), default=0), elapsed,
        )


class HEFTScheduler:
    """Critical-path list scheduling with cost/selectivity-aware tie breaking."""

    name = "cost_aware_heft"

    def schedule(
        self, graph: TaskGraph, limits: ResourceLimits,
        deadline_ms: int | None = None,
    ) -> ScheduleResult:
        started = time.perf_counter()
        ranks = _critical_ranks(graph)
        max_rank = max(ranks.values(), default=1.0)
        scores: dict[str, float] = {}
        for task_id, node in graph.tasks.items():
            critical = ranks[task_id] / max_rank
            reduction = max(0.0, min(1.0, node.candidate_reduction))
            token_penalty = min(1.0, node.expected_tokens / 10_000)
            cost_penalty = min(1.0, node.monetary_cost / 0.05)
            scores[task_id] = (
                0.45 * critical
                + 0.25 * reduction
                + 0.20 * max(0.0, min(2.0, node.user_importance)) / 2
                + 0.10 * len(graph.successors()[task_id]) / max(1, len(graph))
                - 0.05 * token_penalty
                - 0.05 * cost_penalty
            )
        order = _priority_topological_order(graph, scores)
        tasks = _schedule_priority_order(graph, limits, order)
        elapsed = (time.perf_counter() - started) * 1000
        return ScheduleResult(
            self.name, tasks, max((x.end_ms for x in tasks), default=0), elapsed,
            metadata={"priority_order": order},
        )


def _lower_bound(
    graph: TaskGraph,
    limits: ResourceLimits,
    scheduled: list[ScheduledTask],
) -> int:
    ends = {item.task_id: item.end_ms for item in scheduled}
    ranks = _critical_ranks(graph)
    precedence = max(
        (
            max(
                (ends.get(dep, 0) for dep in graph.tasks[task_id].dependencies),
                default=0,
            )
            + ranks[task_id]
            for task_id in graph.tasks
            if task_id not in ends
        ),
        default=max(ends.values(), default=0),
    )
    resource_bound = 0
    by_resource: dict[str, int] = {}
    for node in graph.tasks.values():
        by_resource[node.resource] = (
            by_resource.get(node.resource, 0)
            + node.duration_ms * node.demand
        )
    for resource, workload in by_resource.items():
        resource_bound = max(
            resource_bound,
            math.ceil(workload / limits.get(resource).capacity),
        )
    return int(max(precedence, resource_bound, max(ends.values(), default=0)))


class BranchAndBoundScheduler:
    """Anytime exact search over precedence-feasible priority lists."""

    name = "branch_and_bound"

    def __init__(self, max_tasks: int = 18):
        self.max_tasks = max_tasks

    def schedule(
        self, graph: TaskGraph, limits: ResourceLimits,
        deadline_ms: int | None = 150,
    ) -> ScheduleResult:
        started = time.perf_counter()
        seed = HEFTScheduler().schedule(graph, limits)
        if len(graph) > self.max_tasks:
            seed.algorithm = self.name
            seed.status = "size_fallback"
            seed.optimal = False
            seed.metadata["fallback_algorithm"] = "cost_aware_heft"
            seed.planning_time_ms = (time.perf_counter() - started) * 1000
            return seed
        deadline = started + max(1, int(deadline_ms or 150)) / 1000
        best_tasks = seed.tasks
        best_makespan = seed.makespan_ms
        successors = graph.successors()
        ranks = _critical_ranks(graph)
        explored = 0
        pruned = 0
        timed_out = False

        def visit(
            order: list[str],
            indegree: dict[str, int],
            ready: set[str],
        ) -> None:
            nonlocal best_tasks, best_makespan, explored, pruned, timed_out
            if time.perf_counter() >= deadline:
                timed_out = True
                return
            explored += 1
            partial = _schedule_priority_order(graph, limits, order)
            if _lower_bound(graph, limits, partial) >= best_makespan:
                pruned += 1
                return
            if len(order) == len(graph):
                makespan = max((item.end_ms for item in partial), default=0)
                if makespan < best_makespan:
                    best_makespan, best_tasks = makespan, partial
                return
            choices = sorted(
                ready, key=lambda task_id: (-ranks[task_id], task_id)
            )
            for task_id in choices:
                next_indegree = dict(indegree)
                next_ready = set(ready)
                next_ready.remove(task_id)
                for child in successors[task_id]:
                    next_indegree[child] -= 1
                    if next_indegree[child] == 0:
                        next_ready.add(child)
                visit([*order, task_id], next_indegree, next_ready)
                if timed_out:
                    return

        indegree = {
            task_id: len(node.dependencies)
            for task_id, node in graph.tasks.items()
        }
        visit([], indegree, {x for x, degree in indegree.items() if degree == 0})
        elapsed = (time.perf_counter() - started) * 1000
        lower = _lower_bound(graph, limits, [])
        gap = (
            max(0.0, (best_makespan - lower) / best_makespan)
            if best_makespan else 0.0
        )
        return ScheduleResult(
            self.name, best_tasks, best_makespan, elapsed,
            optimal=not timed_out, optimality_gap=(0.0 if not timed_out else gap),
            status=("optimal" if not timed_out else "time_limit"),
            metadata={"explored_states": explored, "pruned_states": pruned},
        )


def approximate_treewidth(graph: TaskGraph) -> tuple[int, list[str]]:
    """Deterministic min-fill elimination on the undirected precedence graph."""
    adjacency = {task_id: set() for task_id in graph.tasks}
    for node in graph.tasks.values():
        for dependency in node.dependencies:
            adjacency[node.id].add(dependency)
            adjacency[dependency].add(node.id)
    order: list[str] = []
    width = 0
    while adjacency:
        def fill_score(task_id: str) -> tuple[int, int, str]:
            neighbours = adjacency[task_id]
            missing = sum(
                1
                for left, right in combinations(sorted(neighbours), 2)
                if right not in adjacency[left]
            )
            return missing, len(neighbours), task_id

        current = min(adjacency, key=fill_score)
        neighbours = set(adjacency[current])
        width = max(width, len(neighbours))
        for left, right in combinations(neighbours, 2):
            adjacency[left].add(right)
            adjacency[right].add(left)
        for neighbour in neighbours:
            adjacency[neighbour].discard(current)
        del adjacency[current]
        order.append(current)
    return width, order


class TreewidthDPScheduler:
    """Exact state-compressed DP for low-width, unit-capacity resource graphs.

    End times are retained only for the dependency frontier (the current
    separator).  Completed tasks with no edge to the remaining graph are
    forgotten, which is the useful treewidth-driven compression.
    """

    name = "treewidth_dp"

    def __init__(self, max_treewidth: int = 6, max_tasks: int = 22):
        self.max_treewidth = max_treewidth
        self.max_tasks = max_tasks

    def schedule(
        self, graph: TaskGraph, limits: ResourceLimits,
        deadline_ms: int | None = 200,
    ) -> ScheduleResult:
        started = time.perf_counter()
        width, elimination = approximate_treewidth(graph)
        unsupported = (
            len(graph) > self.max_tasks
            or width > self.max_treewidth
            or any(
                limits.get(node.resource).capacity != 1 or node.demand != 1
                for node in graph.tasks.values()
            )
        )
        if unsupported:
            seed = HEFTScheduler().schedule(graph, limits)
            seed.algorithm = self.name
            seed.status = "structural_fallback"
            seed.metadata.update({
                "estimated_treewidth": width,
                "fallback_algorithm": "cost_aware_heft",
            })
            seed.planning_time_ms = (time.perf_counter() - started) * 1000
            return seed

        ids = sorted(graph.tasks)
        index = {task_id: position for position, task_id in enumerate(ids)}
        successor_map = {
            task_id: set(values)
            for task_id, values in graph.successors().items()
        }
        resources = sorted({node.resource for node in graph.tasks.values()})
        resource_index = {value: idx for idx, value in enumerate(resources)}
        elimination_rank = {
            task_id: idx for idx, task_id in enumerate(elimination)
        }
        all_mask = (1 << len(ids)) - 1
        deadline = started + max(1, int(deadline_ms or 200)) / 1000
        timed_out = False
        states = 0

        def frontier(mask: int) -> list[str]:
            return [
                task_id for task_id in ids
                if mask & (1 << index[task_id])
                and any(
                    not mask & (1 << index[child])
                    for child in successor_map[task_id]
                )
            ]

        @lru_cache(maxsize=None)
        def solve(
            mask: int,
            resource_available: tuple[int, ...],
            frontier_ends: tuple[int, ...],
        ) -> tuple[int, tuple[str, ...]]:
            nonlocal timed_out, states
            states += 1
            if time.perf_counter() >= deadline:
                timed_out = True
                return 10**15, ()
            if mask == all_mask:
                return max(resource_available, default=0), ()
            old_frontier = frontier(mask)
            end_map = dict(zip(old_frontier, frontier_ends))
            ready = [
                task_id for task_id in ids
                if not mask & (1 << index[task_id])
                and all(
                    mask & (1 << index[parent])
                    for parent in graph.tasks[task_id].dependencies
                )
            ]
            ready.sort(key=lambda task_id: (
                elimination_rank[task_id], task_id
            ))
            best = (10**15, ())
            for task_id in ready:
                node = graph.tasks[task_id]
                resource_slot = resource_index[node.resource]
                dependency_end = max(
                    (end_map[parent] for parent in node.dependencies),
                    default=0,
                )
                start_at = max(
                    dependency_end, resource_available[resource_slot]
                )
                end_at = start_at + node.duration_ms
                next_resources = list(resource_available)
                next_resources[resource_slot] = end_at
                next_mask = mask | (1 << index[task_id])
                next_frontier = frontier(next_mask)
                next_ends = tuple(
                    end_at if item == task_id else end_map[item]
                    for item in next_frontier
                )
                value, suffix = solve(
                    next_mask, tuple(next_resources), next_ends
                )
                if value < best[0]:
                    best = value, (task_id, *suffix)
            return best

        initial_resources = tuple(0 for _ in resources)
        makespan, order = solve(0, initial_resources, ())
        if timed_out or not order:
            seed = HEFTScheduler().schedule(graph, limits)
            seed.algorithm = self.name
            seed.status = "time_limit"
            seed.metadata.update({
                "estimated_treewidth": width,
                "states": states,
                "fallback_algorithm": "cost_aware_heft",
            })
            seed.planning_time_ms = (time.perf_counter() - started) * 1000
            return seed
        tasks = _schedule_priority_order(graph, limits, list(order))
        elapsed = (time.perf_counter() - started) * 1000
        return ScheduleResult(
            self.name, tasks, int(makespan), elapsed,
            optimal=True, optimality_gap=0.0, status="optimal",
            metadata={
                "estimated_treewidth": width,
                "elimination_order": elimination,
                "states": states,
            },
        )


class CPSATScheduler:
    name = "cp_sat"

    def schedule(
        self, graph: TaskGraph, limits: ResourceLimits,
        deadline_ms: int | None = 200,
    ) -> ScheduleResult:
        started = time.perf_counter()
        cp_model = _cp_model
        if cp_model is None:
            seed = HEFTScheduler().schedule(graph, limits)
            seed.algorithm = self.name
            seed.status = "dependency_unavailable"
            seed.metadata["fallback_algorithm"] = "cost_aware_heft"
            seed.planning_time_ms = (time.perf_counter() - started) * 1000
            return seed

        model = cp_model.CpModel()
        horizon = sum(node.duration_ms for node in graph.tasks.values())
        starts = {}
        ends = {}
        intervals: dict[str, list] = {}
        demands: dict[str, list[int]] = {}
        for task_id, node in graph.tasks.items():
            starts[task_id] = model.new_int_var(0, horizon, f"start_{task_id}")
            ends[task_id] = model.new_int_var(0, horizon, f"end_{task_id}")
            interval = model.new_interval_var(
                starts[task_id], node.duration_ms, ends[task_id],
                f"interval_{task_id}",
            )
            intervals.setdefault(node.resource, []).append(interval)
            demands.setdefault(node.resource, []).append(node.demand)
            for dependency in node.dependencies:
                model.add(starts[task_id] >= ends[dependency])
        for resource, values in intervals.items():
            model.add_cumulative(
                values, demands[resource], limits.get(resource).capacity
            )
        makespan = model.new_int_var(0, horizon, "makespan")
        model.add_max_equality(makespan, list(ends.values()))
        model.minimize(makespan)
        seed = HEFTScheduler().schedule(graph, limits)
        for item in seed.tasks:
            model.add_hint(starts[item.task_id], item.start_ms)
            model.add_hint(ends[item.task_id], item.end_ms)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max(
            0.001, int(deadline_ms or 200) / 1000
        )
        solver.parameters.num_search_workers = 1
        status = solver.solve(model)
        acceptable = {
            cp_model.OPTIMAL, cp_model.FEASIBLE,
        }
        if status not in acceptable:
            seed.algorithm = self.name
            seed.status = "solver_no_solution"
            seed.metadata["fallback_algorithm"] = "cost_aware_heft"
            seed.planning_time_ms = (time.perf_counter() - started) * 1000
            return seed
        tasks = sorted(
            [
                ScheduledTask(
                    task_id, solver.value(starts[task_id]),
                    solver.value(ends[task_id]), node.resource,
                )
                for task_id, node in graph.tasks.items()
            ],
            key=lambda item: (item.start_ms, item.task_id),
        )
        value = int(solver.value(makespan))
        bound = float(solver.best_objective_bound)
        gap = max(0.0, (value - bound) / value) if value else 0.0
        elapsed = (time.perf_counter() - started) * 1000
        return ScheduleResult(
            self.name, tasks, value, elapsed,
            optimal=status == cp_model.OPTIMAL,
            optimality_gap=(0.0 if status == cp_model.OPTIMAL else gap),
            status=("optimal" if status == cp_model.OPTIMAL else "feasible"),
            metadata={
                "best_bound_ms": bound,
                "conflicts": solver.num_conflicts,
                "branches": solver.num_branches,
            },
        )
