"""Anytime algorithm portfolio and rolling-horizon schedule repair."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Iterable

from .algorithms import (
    BranchAndBoundScheduler, CPSATScheduler, FIFOScheduler, HEFTScheduler,
    Scheduler, TreewidthDPScheduler,
)
from .core import (
    ResourceLimits, ScheduleResult, ScheduledTask, TaskGraph, TaskNode,
    validate_schedule,
)


class PortfolioScheduler:
    """Run bounded solvers and retain the best validated user-visible plan."""

    def __init__(
        self,
        algorithms: Iterable[Scheduler] | None = None,
        deadline_ms: int = 200,
    ):
        self.algorithms = list(algorithms or (
            FIFOScheduler(),
            HEFTScheduler(),
            BranchAndBoundScheduler(),
            CPSATScheduler(),
            TreewidthDPScheduler(),
        ))
        self.deadline_ms = max(1, int(deadline_ms))

    def schedule(
        self, graph: TaskGraph, limits: ResourceLimits,
        deadline_ms: int | None = None,
    ) -> ScheduleResult:
        budget = max(1, int(deadline_ms or self.deadline_ms))
        portfolio_started = time.perf_counter()

        def run(algorithm):
            result = algorithm.schedule(graph, limits, budget)
            errors = validate_schedule(graph, limits, result)
            if errors:
                result.feasible = False
                result.status = "invalid"
                result.metadata["validation_errors"] = errors
            return result

        # This is an algorithm race: HEFT provides an immediate upper bound
        # while exact/DP solvers improve it within the same wall-clock budget.
        with ThreadPoolExecutor(
            max_workers=max(1, len(self.algorithms)),
            thread_name_prefix="schedule-portfolio",
        ) as executor:
            results = list(executor.map(run, self.algorithms))
        portfolio_wall_ms = (time.perf_counter() - portfolio_started) * 1000
        eligible = [result for result in results if result.feasible]
        if not eligible:
            raise RuntimeError("no scheduler produced a feasible plan")
        # Solvers that only returned another algorithm's fallback do not win
        # merely by relabelling the same plan.
        genuine = [
            result for result in eligible
            if "fallback_algorithm" not in result.metadata
        ] or eligible
        best = min(
            genuine,
            key=lambda result: (
                result.makespan_ms,
                result.planning_time_ms,
                result.algorithm,
            ),
        )
        individual_planning_ms = best.planning_time_ms
        best.metadata = {
            **best.metadata,
            "portfolio": [
                {
                    "algorithm": result.algorithm,
                    "status": result.status,
                    "feasible": result.feasible,
                    "optimal": result.optimal,
                    "optimality_gap": result.optimality_gap,
                    "planning_time_ms": round(result.planning_time_ms, 3),
                    "predicted_makespan_ms": result.makespan_ms,
                    "selection_score_ms": round(
                        result.selection_score_ms, 3
                    ),
                }
                for result in results
            ],
            "selection_rule": (
                "parallel algorithm race; min makespan, then solver latency"
            ),
            "portfolio_wall_time_ms": round(portfolio_wall_ms, 3),
            "selected_solver_time_ms": round(individual_planning_ms, 3),
        }
        # All solvers raced before execution, so the wall time—not only the
        # selected solver's own CPU time—is visible to the user.
        best.planning_time_ms = portfolio_wall_ms
        return best


class RollingHorizonScheduler:
    """Repair only the unfinished suffix after duration/failure updates."""

    def __init__(self, portfolio: PortfolioScheduler | None = None):
        self.portfolio = portfolio or PortfolioScheduler()

    def repair(
        self,
        graph: TaskGraph,
        limits: ResourceLimits,
        previous: ScheduleResult,
        completed_task_ids: set[str],
        *,
        actual_duration_ms: dict[str, int] | None = None,
        current_time_ms: int | None = None,
    ) -> ScheduleResult:
        actual_duration_ms = actual_duration_ms or {}
        remaining_ids = set(graph.tasks) - set(completed_task_ids)
        if not remaining_ids:
            return ScheduleResult(
                "rolling_horizon", [], 0, 0.0, optimal=True,
                metadata={"repaired_from": previous.algorithm},
            )
        remaining_tasks = []
        for task_id in remaining_ids:
            node = graph.tasks[task_id]
            remaining_tasks.append(replace(
                node,
                duration_ms=max(
                    1, int(actual_duration_ms.get(task_id, node.duration_ms))
                ),
                dependencies=tuple(
                    dep for dep in node.dependencies if dep in remaining_ids
                ),
            ))
        repaired = self.portfolio.schedule(TaskGraph(remaining_tasks), limits)
        offset = (
            int(current_time_ms)
            if current_time_ms is not None
            else max(
                (
                    item.end_ms for item in previous.tasks
                    if item.task_id in completed_task_ids
                ),
                default=0,
            )
        )
        repaired.tasks = [
            ScheduledTask(
                item.task_id, item.start_ms + offset, item.end_ms + offset,
                item.resource,
            )
            for item in repaired.tasks
        ]
        repaired.makespan_ms += offset
        repaired.algorithm = "rolling_horizon:" + repaired.algorithm
        repaired.metadata["repaired_from"] = previous.algorithm
        repaired.metadata["completed_task_ids"] = sorted(completed_task_ids)
        return repaired
