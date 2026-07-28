"""의도·조건·SQL·근거·지연시간을 분리해 측정하는 평가 실행기."""
from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from src.agent.planner import Planner
from .golden import build_golden_cases


ALLOWED_SLOTS = {
    "transaction_type", "lease_type", "property_type", "region_sido",
    "region_gugun", "max_sale_price_manwon", "max_deposit_manwon",
    "max_monthly_rent_manwon", "max_maintenance_manwon", "min_area_m2",
    "max_building_age", "sort_by", "min_safety_score",
    "min_convenience_score", "max_commute_min", "_workplace_landmark",
}


@dataclass(frozen=True)
class EvaluationThresholds:
    intent_accuracy: float = .90
    action_accuracy: float = .90
    required_slot_recall: float = .90
    invalid_condition_rate: float = .02
    p95_latency_ms: float = 200.0


class AgentEvaluator:
    def __init__(
        self,
        planner: Any | None = None,
        *,
        sql_probe: Callable[[dict], bool] | None = None,
        retrieval_probe: Callable[[dict], float] | None = None,
        answer_probe: Callable[[dict], dict] | None = None,
    ):
        self.planner = planner or Planner()
        self.sql_probe = sql_probe
        self.retrieval_probe = retrieval_probe
        self.answer_probe = answer_probe

    def run(self, cases: list[dict] | None = None) -> dict:
        cases = cases or build_golden_cases()
        rows: list[dict[str, Any]] = []
        latencies: list[float] = []
        for case in cases:
            started = time.perf_counter()
            plan = self.planner.plan(case["text"])
            latency = (time.perf_counter() - started) * 1000
            latencies.append(latency)
            required = case.get("required_slots") or {}
            matched = sum(plan.slots.get(key) == value for key, value in required.items())
            invalid = sorted(set(plan.slots) - ALLOWED_SLOTS)
            row = {
                "case_id": case["case_id"], "text": case["text"],
                "intent_ok": plan.intent == case["expected_intent"],
                "action_ok": plan.action == case["expected_action"],
                "required_slot_count": len(required), "required_slot_match": matched,
                "invalid_slots": invalid, "latency_ms": round(latency, 3),
            }
            if self.sql_probe:
                row["sql_execution_ok"] = bool(self.sql_probe(
                    {"case": case, "plan": plan}))
            if self.retrieval_probe:
                row["retrieval_recall"] = float(self.retrieval_probe(
                    {"case": case, "plan": plan}))
            if self.answer_probe:
                row.update(self.answer_probe({"case": case, "plan": plan}) or {})
            rows.append(row)
        required_total = sum(row["required_slot_count"] for row in rows)
        metrics: dict[str, Any] = {
            "case_count": len(rows),
            "intent_accuracy": _ratio(sum(row["intent_ok"] for row in rows), len(rows)),
            "action_accuracy": _ratio(sum(row["action_ok"] for row in rows), len(rows)),
            "required_slot_recall": _ratio(
                sum(row["required_slot_match"] for row in rows), required_total,
                empty_value=1.0),
            "invalid_condition_rate": _ratio(
                sum(bool(row["invalid_slots"]) for row in rows), len(rows)),
            "latency_ms": {
                "mean": round(statistics.fmean(latencies), 3),
                "p50": round(_percentile(latencies, .50), 3),
                "p95": round(_percentile(latencies, .95), 3),
            },
            "sql_execution_success_rate": _optional_rate(rows, "sql_execution_ok"),
            "retrieval_recall": _optional_mean(rows, "retrieval_recall"),
            "grounded_answer_rate": _optional_rate(rows, "grounded_answer"),
            "api_failure_recovery_rate": _optional_rate(rows, "api_recovered"),
            "request_cost_usd": _optional_mean(rows, "request_cost_usd"),
        }
        return {
            "schema_version": "agent_eval_v1", "metrics": metrics,
            "failures": [row for row in rows if not row["intent_ok"] or
                         not row["action_ok"] or row["invalid_slots"] or
                         row["required_slot_match"] < row["required_slot_count"]],
            "cases": rows,
        }

    @staticmethod
    def gate(report: dict, thresholds: EvaluationThresholds | None = None) -> dict:
        t = thresholds or EvaluationThresholds()
        m = report["metrics"]
        checks = {
            "intent_accuracy": m["intent_accuracy"] >= t.intent_accuracy,
            "action_accuracy": m["action_accuracy"] >= t.action_accuracy,
            "required_slot_recall": m["required_slot_recall"] >= t.required_slot_recall,
            "invalid_condition_rate": m["invalid_condition_rate"] <= t.invalid_condition_rate,
            "p95_latency_ms": m["latency_ms"]["p95"] <= t.p95_latency_ms,
        }
        return {"passed": all(checks.values()), "checks": checks,
                "thresholds": asdict(t)}

    @staticmethod
    def save(report: dict, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                          encoding="utf-8")
        return target


def _ratio(numerator: float, denominator: float, empty_value: float = 0.0) -> float:
    return round(float(numerator) / denominator, 6) if denominator else empty_value


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = (len(ordered) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - index) + ordered[high] * (index - low)


def _optional_rate(rows: list[dict], key: str) -> float | None:
    values = [bool(row[key]) for row in rows if key in row]
    return _ratio(sum(values), len(values)) if values else None


def _optional_mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return round(statistics.fmean(values), 6) if values else None
