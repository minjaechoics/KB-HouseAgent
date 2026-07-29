"""Shared task-graph and schedule result types for the agent runtime.

The scheduler never executes tools directly.  It produces an auditable static
plan that the caller may execute with its own async/thread runtime.  Durations
are integer milliseconds so exact solvers and deterministic tests share the
same representation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class TaskNode:
    id: str
    duration_ms: int
    resource: str
    dependencies: tuple[str, ...] = ()
    demand: int = 1
    expected_tokens: int = 0
    monetary_cost: float = 0.0
    candidate_reduction: float = 0.0
    user_importance: float = 1.0
    required: bool = True
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("task id is required")
        if self.duration_ms <= 0:
            raise ValueError(f"{self.id}: duration_ms must be positive")
        if self.demand <= 0:
            raise ValueError(f"{self.id}: demand must be positive")


@dataclass(frozen=True)
class ResourceLimit:
    capacity: int = 1
    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("resource capacity must be positive")


@dataclass
class ResourceLimits:
    resources: dict[str, ResourceLimit] = field(default_factory=dict)
    default_capacity: int = 1

    def get(self, resource: str) -> ResourceLimit:
        return self.resources.get(
            resource, ResourceLimit(capacity=self.default_capacity)
        )


class TaskGraph:
    def __init__(self, tasks: Iterable[TaskNode]):
        values = list(tasks)
        self.tasks = {task.id: task for task in values}
        if len(self.tasks) != len(values):
            raise ValueError("duplicate task id")
        self._validate()

    def _validate(self) -> None:
        for task in self.tasks.values():
            missing = set(task.dependencies) - self.tasks.keys()
            if missing:
                raise ValueError(
                    f"{task.id}: unknown dependencies {sorted(missing)}"
                )
            if task.id in task.dependencies:
                raise ValueError(f"{task.id}: self dependency")
        self.topological_order()

    def topological_order(self) -> list[str]:
        indegree = {
            task_id: len(task.dependencies)
            for task_id, task in self.tasks.items()
        }
        successors = self.successors()
        ready = sorted(task_id for task_id, value in indegree.items() if value == 0)
        order: list[str] = []
        while ready:
            task_id = ready.pop(0)
            order.append(task_id)
            for child in sorted(successors[task_id]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
                    ready.sort()
        if len(order) != len(self.tasks):
            raise ValueError("task graph contains a cycle")
        return order

    def successors(self) -> dict[str, list[str]]:
        result = {task_id: [] for task_id in self.tasks}
        for task in self.tasks.values():
            for dependency in task.dependencies:
                result[dependency].append(task.id)
        return result

    def subgraph(self, task_ids: set[str]) -> "TaskGraph":
        return TaskGraph(
            TaskNode(
                **{
                    **asdict(task),
                    "dependencies": tuple(
                        dep for dep in task.dependencies if dep in task_ids
                    ),
                }
            )
            for task_id, task in self.tasks.items()
            if task_id in task_ids
        )

    def __len__(self) -> int:
        return len(self.tasks)


@dataclass(frozen=True)
class ScheduledTask:
    task_id: str
    start_ms: int
    end_ms: int
    resource: str


@dataclass
class ScheduleResult:
    algorithm: str
    tasks: list[ScheduledTask]
    makespan_ms: int
    planning_time_ms: float
    feasible: bool = True
    optimal: bool = False
    optimality_gap: float | None = None
    status: str = "ok"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def selection_score_ms(self) -> float:
        # Planning blocks execution, so it is part of the user-visible latency.
        return float(self.makespan_ms) + float(self.planning_time_ms)

    def task_map(self) -> dict[str, ScheduledTask]:
        return {task.task_id: task for task in self.tasks}

    def parallel_groups(self) -> list[list[str]]:
        groups: dict[int, list[str]] = {}
        for task in self.tasks:
            groups.setdefault(task.start_ms, []).append(task.task_id)
        return [sorted(groups[start]) for start in sorted(groups)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "status": self.status,
            "feasible": self.feasible,
            "optimal": self.optimal,
            "optimality_gap": self.optimality_gap,
            "planning_time_ms": round(self.planning_time_ms, 3),
            "predicted_makespan_ms": self.makespan_ms,
            "selection_score_ms": round(self.selection_score_ms, 3),
            "parallel_groups": self.parallel_groups(),
            "tasks": [asdict(task) for task in self.tasks],
            "metadata": self.metadata,
        }


def validate_schedule(
    graph: TaskGraph,
    limits: ResourceLimits,
    result: ScheduleResult,
) -> list[str]:
    """Return validation errors instead of raising so benchmarks can retain them."""
    errors: list[str] = []
    scheduled = result.task_map()
    missing = set(graph.tasks) - scheduled.keys()
    extra = scheduled.keys() - set(graph.tasks)
    if missing:
        errors.append(f"missing tasks: {sorted(missing)}")
    if extra:
        errors.append(f"unknown tasks: {sorted(extra)}")
    for task_id, node in graph.tasks.items():
        current = scheduled.get(task_id)
        if not current:
            continue
        if current.end_ms - current.start_ms != node.duration_ms:
            errors.append(f"{task_id}: duration mismatch")
        for dependency in node.dependencies:
            parent = scheduled.get(dependency)
            if parent and current.start_ms < parent.end_ms:
                errors.append(f"{task_id}: precedence violation from {dependency}")
    resources: dict[str, list[tuple[int, int, int, str]]] = {}
    for task in result.tasks:
        node = graph.tasks.get(task.task_id)
        if node:
            resources.setdefault(task.resource, []).append(
                (task.start_ms, task.end_ms, node.demand, task.task_id)
            )
    for resource, intervals in resources.items():
        capacity = limits.get(resource).capacity
        points = sorted({value for row in intervals for value in row[:2]})
        for left, right in zip(points, points[1:]):
            if left == right:
                continue
            demand = sum(
                row[2] for row in intervals
                if row[0] < right and row[1] > left
            )
            if demand > capacity:
                errors.append(
                    f"{resource}: capacity {capacity} exceeded by {demand}"
                )
                break
        limit = limits.get(resource)
        if (
            limit.requests_per_minute is not None
            or limit.tokens_per_minute is not None
        ):
            starts = sorted(
                (
                    task.start_ms,
                    graph.tasks[task.task_id].expected_tokens,
                    task.task_id,
                )
                for task in result.tasks
                if task.resource == resource
            )
            for current_start, _, _ in starts:
                window = [
                    row for row in starts
                    if current_start - 60_000 < row[0] <= current_start
                ]
                if (
                    limit.requests_per_minute is not None
                    and len(window) > limit.requests_per_minute
                ):
                    errors.append(
                        f"{resource}: RPM {limit.requests_per_minute} exceeded"
                    )
                    break
                if (
                    limit.tokens_per_minute is not None
                    and sum(row[1] for row in window)
                    > limit.tokens_per_minute
                ):
                    errors.append(
                        f"{resource}: TPM {limit.tokens_per_minute} exceeded"
                    )
                    break
    return errors
