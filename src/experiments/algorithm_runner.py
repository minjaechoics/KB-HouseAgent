"""Controlled algorithm-only comparison with shared Agentic LLM decisions."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from src import config
from src.agent.llm import BaseLLM
from src.agent.planner import Plan
from src.agent.prompts import (
    CONDITION_DECISION_JSON_SCHEMA,
    CONDITION_DIALOGUE_SYSTEM_PROMPT,
)
from src.experiments.cases import CASES, ExperimentCase
from src.experiments.console import FixedHeaderConsole, format_elapsed, wait_for_key
from src.experiments.pipeline import (
    agentic_decision_for_case,
    build_ground_truth,
    database_fingerprint,
    json_text,
    run_case,
    score_result,
)
from src.experiments.runner import (
    _create_llm,
    _observed_peak_concurrency,
    _preflight_live_llm,
    _save_report,
)


class _DecisionReplayLLM(BaseLLM):
    """Replay the exact same LLM decision into both retrieval algorithms."""

    experiment_shared_decision_replay = True

    def __init__(self, decisions: dict[str, dict[str, Any]]):
        super().__init__()
        self._decisions = decisions

    def plan(self, text: str, has_prior_region: bool = False,
             conversation_history=None) -> Plan:
        decision = self._decisions[text]
        return Plan(
            intent="recommend",
            slots=copy.deepcopy(decision.get("slots") or {}),
            tool_calls=[],
            action="confirm",
            reason="shared_agentic_decision_replay",
        )

    def plan_condition_dialogue(self, text: str, context: dict) -> dict:
        return copy.deepcopy(self._decisions[text])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="동일한 50개 Agentic 출력으로 Atomic 스케줄러와 단일 SQL 비교",
    )
    parser.add_argument("--workers", type=int, default=config.LLM_MAX_CONCURRENCY,
                        help="두 알고리즘에 동일 적용할 병렬 worker 수(기본 6)")
    parser.add_argument("--limit", type=int, default=50,
                        help="평가 질의 수(기본 50, 최대 50)")
    parser.add_argument("--order", choices=["optimized-first", "naive-first"],
                        default="optimized-first", help="DB 실행 순서")
    parser.add_argument("--mock", action="store_true",
                        help="화면·코드 점검용 로컬 규칙 LLM")
    parser.add_argument("--no-wait", action="store_true",
                        help="종료 시 키 입력을 기다리지 않음")
    parser.add_argument("--output-dir", default="reports/experiments")
    return parser


def _prompt_fingerprint() -> str:
    contract = (
        CONDITION_DIALOGUE_SYSTEM_PROMPT
        + json.dumps(CONDITION_DECISION_JSON_SCHEMA, sort_keys=True, ensure_ascii=False)
    )
    return hashlib.sha256(contract.encode("utf-8")).hexdigest()[:16]


def _plan_one(case: ExperimentCase, llm: BaseLLM,
              stage_started: float) -> dict[str, Any]:
    started = time.perf_counter()
    decision = agentic_decision_for_case(case, llm)
    completed = time.perf_counter()
    return {
        "case_id": case.case_id,
        "query": case.query,
        "decision": decision,
        "decision_fingerprint": hashlib.sha256(
            json.dumps(decision, ensure_ascii=False, sort_keys=True,
                       default=str).encode("utf-8")
        ).hexdigest()[:16],
        "elapsed_seconds": completed - started,
        "batch_timing": {
            "started_offset_seconds": started - stage_started,
            "completed_offset_seconds": completed - stage_started,
        },
    }


def _plan_all(cases: list[ExperimentCase], llm: BaseLLM,
              workers: int) -> tuple[list[dict[str, Any]], float]:
    stage_started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix="algorithm-agentic") as executor:
        futures = [executor.submit(_plan_one, case, llm, stage_started)
                   for case in cases]
        for future in as_completed(futures):
            rows.append(future.result())
    return rows, time.perf_counter() - stage_started


def _algorithm_one(case: ExperimentCase, mode: str, replay: BaseLLM,
                   truth: dict[str, Any], stage_started: float) -> dict[str, Any]:
    started = time.perf_counter()
    result = run_case(case, mode, replay)
    if result.get("error"):
        raise RuntimeError(f"{case.case_id} {mode}: {result['error']}")
    result["ground_truth"] = truth
    result["score"] = score_result(case, result, truth)
    completed = time.perf_counter()
    result["batch_timing"] = {
        "started_offset_seconds": started - stage_started,
        "completed_offset_seconds": completed - stage_started,
        "worker_elapsed_seconds": completed - started,
    }
    return result


def _run_algorithm_stage(cases: list[ExperimentCase], mode: str,
                         replay: BaseLLM, truths: dict[str, dict[str, Any]],
                         workers: int) -> tuple[list[dict[str, Any]], float]:
    stage_started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix=f"algorithm-{mode}") as executor:
        futures = [
            executor.submit(
                _algorithm_one, case, mode, replay, truths[case.case_id], stage_started
            )
            for case in cases
        ]
        for future in as_completed(futures):
            rows.append(future.result())
    return rows, time.perf_counter() - stage_started


def _branch_summary(rows: list[dict[str, Any]], batch_seconds: float,
                    workers: int) -> dict[str, Any]:
    retrieval = [float(row["rag_trace"]["timing"]["retrieval_seconds"])
                 for row in rows]
    correct = sum(bool(row["score"]["correct"]) for row in rows)
    ordered = sorted(retrieval)
    p95_index = min(len(ordered) - 1, int((len(ordered) - 1) * 0.95))
    return {
        "cases": len(rows),
        "configured_workers": workers,
        "observed_peak_concurrency": _observed_peak_concurrency(rows),
        "batch_seconds": batch_seconds,
        "throughput_queries_per_second": len(rows) / batch_seconds,
        "mean_retrieval_seconds": statistics.fmean(retrieval),
        "p50_retrieval_seconds": statistics.median(retrieval),
        "p95_retrieval_seconds": ordered[p95_index],
        "correct": correct,
        "accuracy": correct / len(rows),
        "mean_slot_accuracy": statistics.fmean(
            float(row["score"]["slot_accuracy"]) for row in rows
        ),
    }


def _print_pair(console: FixedHeaderConsole, index: int, total: int,
                case: ExperimentCase, decision: dict[str, Any],
                optimized: dict[str, Any], naive: dict[str, Any]) -> None:
    console.log("\n" + "=" * 78)
    console.log(f"[{index:02d}/{total:02d}] {case.case_id} | SHARED AGENTIC DECISION")
    console.log("-" * 78)
    console.log("[INPUT]")
    console.log(case.query)
    console.log("\n[IDENTICAL LLM OUTPUT USED BY BOTH]")
    console.log(json_text(decision))
    console.log("\n[OPTIMIZED RAG JSON]")
    console.log(json_text(optimized["rag_trace"]))
    console.log("\n[OPTIMIZED JUDGEMENT]")
    console.log(json_text(optimized["score"]))
    console.log("\n[NAIVE RAG JSON]")
    console.log(json_text(naive["rag_trace"]))
    console.log("\n[NAIVE JUDGEMENT]")
    console.log(json_text(naive["score"]))


def run_algorithm(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.mock:
        os.environ["JEONSE_LLM"] = "mock"
    cases = CASES[:max(1, min(int(args.limit), 50))]
    workers = max(1, min(int(args.workers), len(cases)))
    llm = _create_llm(args.no_wait)
    if llm is None:
        return 2
    try:
        _preflight_live_llm(llm)
    except Exception as exc:
        print(f"OpenAI 사전 점검 실패: {type(exc).__name__}: {exc}")
        return 2

    overall_started = time.perf_counter()
    console = FixedHeaderConsole(
        lambda: f"실행시간: {format_elapsed(time.perf_counter() - overall_started)}"
    )
    console.start()
    try:
        console.log("알고리즘 통제 실험: 동일 Agentic 출력 → 두 검색 알고리즘")
        console.log(f"공통 병렬도: workers={workers} | 질의: {len(cases)}개")
        console.log(f"Prompt fingerprint: {_prompt_fingerprint()}")

        plans, planning_batch_seconds = _plan_all(cases, llm, workers)
        decision_by_query = {row["query"]: row["decision"] for row in plans}
        decision_by_case = {row["case_id"]: row["decision"] for row in plans}
        replay = _DecisionReplayLLM(decision_by_query)

        # Ground truth is built outside both timed branches.
        truths = {case.case_id: build_ground_truth(case) for case in cases}
        stage_order = (["optimized", "naive"] if args.order == "optimized-first"
                       else ["naive", "optimized"])
        branch_rows: dict[str, list[dict[str, Any]]] = {}
        branch_times: dict[str, float] = {}
        for mode in stage_order:
            console.log(f"\n{mode.upper()} 검색 실행 중... workers={workers}")
            branch_rows[mode], branch_times[mode] = _run_algorithm_stage(
                cases, mode, replay, truths, workers,
            )

        by_mode_case = {
            mode: {row["case_id"]: row for row in rows}
            for mode, rows in branch_rows.items()
        }
        for index, case in enumerate(cases, 1):
            _print_pair(
                console, index, len(cases), case, decision_by_case[case.case_id],
                by_mode_case["optimized"][case.case_id],
                by_mode_case["naive"][case.case_id],
            )
    finally:
        console.stop()

    optimized_summary = _branch_summary(
        branch_rows["optimized"], branch_times["optimized"], workers)
    naive_summary = _branch_summary(
        branch_rows["naive"], branch_times["naive"], workers)
    credential = os.environ.get("OPENAI_API_KEY", "")
    report = {
        "metadata": {
            "experiment": "algorithm_only_controlled",
            "started_at": datetime.now().astimezone().isoformat(),
            "query_count": len(cases),
            "shared_query_workers": workers,
            "algorithm_order": args.order,
            "database_path": str(config.DB_PATH.resolve()),
            "database_fingerprint": database_fingerprint(),
            "llm_provider": getattr(llm, "provider", "local"),
            "llm_model": getattr(llm, "model", "rule"),
            "api_credential_fingerprint": (
                hashlib.sha256(credential.encode("utf-8")).hexdigest()[:12]
                if credential and getattr(llm, "supports_agentic_calls", False)
                else None
            ),
            "agentic_prompt_fingerprint": _prompt_fingerprint(),
            "controls": {
                "same_50_user_prompts": True,
                "llm_prompt_executed_once_per_case": True,
                "same_llm_decision_replayed_to_both": True,
                "same_parallel_workers": True,
                "same_ground_truth": True,
                "same_database_snapshot": True,
                "only_algorithm_differs": (
                    "atomic intersection and dependency scheduling vs single SQL"
                ),
            },
        },
        "common_agentic_stage": {
            "prompt_count": len(plans),
            "configured_workers": workers,
            "observed_peak_concurrency": _observed_peak_concurrency(plans),
            "batch_seconds": planning_batch_seconds,
            "mean_seconds": statistics.fmean(
                float(row["elapsed_seconds"]) for row in plans
            ),
            "decisions": sorted(plans, key=lambda row: row["case_id"]),
        },
        "optimized": {
            "summary": optimized_summary,
            "cases": sorted(branch_rows["optimized"], key=lambda row: row["case_id"]),
        },
        "naive": {
            "summary": naive_summary,
            "cases": sorted(branch_rows["naive"], key=lambda row: row["case_id"]),
        },
        "comparison": {
            "optimized_minus_naive_accuracy_points": 100 * (
                optimized_summary["accuracy"] - naive_summary["accuracy"]
            ),
            "retrieval_batch_speed_ratio_naive_over_optimized": (
                naive_summary["batch_seconds"] / optimized_summary["batch_seconds"]
            ),
        },
    }
    path = _save_report(args.output_dir, "algorithm", "paired", report)
    elapsed = time.perf_counter() - overall_started
    print("\n" + "=" * 78)
    print(f"최종 실행시간: {format_elapsed(elapsed)}")
    print(f"공통 LLM: {len(cases)}개 중 {len(plans)}개 | workers={workers} | "
          f"{planning_batch_seconds:.3f}s")
    print(
        "OPTIMIZED | "
        f"정확도 {optimized_summary['accuracy'] * 100:.1f}% | "
        f"검색 배치 {optimized_summary['batch_seconds']:.3f}s | "
        f"동시성 {optimized_summary['observed_peak_concurrency']}"
    )
    print(
        "NAIVE     | "
        f"정확도 {naive_summary['accuracy'] * 100:.1f}% | "
        f"검색 배치 {naive_summary['batch_seconds']:.3f}s | "
        f"동시성 {naive_summary['observed_peak_concurrency']}"
    )
    print(f"결과 JSON: {path.resolve()}")
    if not args.no_wait:
        wait_for_key()
    return 0


def main_algorithm() -> None:
    try:
        raise SystemExit(run_algorithm())
    except KeyboardInterrupt:
        print("\n사용자가 실험을 중단했습니다.")
        raise SystemExit(130)
