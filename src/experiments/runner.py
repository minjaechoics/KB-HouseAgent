"""Command-line orchestration shared by speed_test.py and hallucination.py."""
from __future__ import annotations

import argparse
import getpass
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src import config
from src.agent.llm import get_llm
from src.agent.prompts import (
    CONDITION_DECISION_JSON_SCHEMA,
    CONDITION_DIALOGUE_SYSTEM_PROMPT,
)
from src.experiments.cases import CASES
from src.experiments.console import FixedHeaderConsole, format_elapsed, wait_for_key
from src.experiments.pipeline import (
    build_ground_truth,
    database_fingerprint,
    json_text,
    run_case,
    score_result,
)
from src.experiments.naive_baseline import naive_prompt_fingerprint


def _parser(kind: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("50개 복합 질의의 추천 지연시간 비교" if kind == "speed"
                     else "50개 복합 질의의 근거 정답률 비교"),
    )
    parser.add_argument("--mode", choices=["optimized", "naive"],
                        help=("optimized=Agentic Atomic·병렬·스케줄링, "
                              "naive=전체 질문 단일 추출·직렬 단일 SQL"))
    parser.add_argument("--mock", action="store_true",
                        help="개발용: OpenAI 대신 로컬 규칙 LLM 사용")
    parser.add_argument("--no-wait", action="store_true",
                        help="자동 테스트용: 종료 전 키 입력을 기다리지 않음")
    parser.add_argument("--limit", type=int, default=50,
                        help=argparse.SUPPRESS)
    parser.add_argument(
        "--workers", type=int, default=None,
        help="optimized 동시 질의 수(기본 LLM_MAX_CONCURRENCY, naive는 항상 1)",
    )
    parser.add_argument("--output-dir", default="reports/experiments")
    return parser


def _choose_mode(provided: str | None) -> str:
    if provided:
        return provided
    print("실험할 파이프라인을 선택하세요.")
    print("  1. 현재 알고리즘 (Atomic + 병렬처리 + 의존성 스케줄링)")
    print("  2. NAIVE 기준선 (사용자 질문 전체를 한 번에 추출 + 직렬 단일 SQL)")
    while True:
        value = input("선택 [1/2]: ").strip().lower()
        if value in {"1", "optimized", "o"}:
            return "optimized"
        if value in {"2", "naive", "n"}:
            return "naive"
        print("1 또는 2를 입력해 주세요.")


def _metadata(mode: str, kind: str, total: int, llm: Any,
              workers: int) -> dict[str, Any]:
    credential = os.environ.get("OPENAI_API_KEY", "")
    prompt_contract = (
        CONDITION_DIALOGUE_SYSTEM_PROMPT
        + json.dumps(CONDITION_DECISION_JSON_SCHEMA, sort_keys=True, ensure_ascii=False)
    )
    return {
        "experiment": kind,
        "mode": mode,
        "query_count": total,
        "started_at": datetime.now().astimezone().isoformat(),
        "database_path": str(config.DB_PATH.resolve()),
        "database_fingerprint": database_fingerprint(),
        "llm_class": type(llm).__name__,
        "llm_provider": getattr(llm, "provider", "local"),
        "llm_model": getattr(llm, "model", "rule"),
        "api_credential_fingerprint": (
            hashlib.sha256(credential.encode("utf-8")).hexdigest()[:12]
            if credential and getattr(llm, "supports_agentic_calls", False)
            else None
        ),
        "planner_prompt_fingerprint": (
            hashlib.sha256(prompt_contract.encode("utf-8")).hexdigest()[:16]
            if mode == "optimized" else naive_prompt_fingerprint()
        ),
        "query_workers": workers,
        "execution_policy": (
            "parallel_thread_pool" if workers > 1 else "serial_baseline"
        ),
        "fairness_controls": {
            "same_query_set_and_submission_order": True,
            "same_ground_truth": True,
            "same_database_snapshot": True,
            "same_llm_instance_per_run": True,
            "same_system_prompt": False,
            "prompt_treatment": (
                "production_agentic_atomic_orchestration"
                if mode == "optimized" else "naive_whole_prompt_single_pass"
            ),
            "same_structured_output_schema": False,
            "same_scored_slot_contract": True,
            "llm_calls_per_query": 1,
            "result_limit": 5,
            "live_route_calls": False,
            "parallelism_is_treatment": True,
            "naive_forced_serial": mode == "naive",
        },
    }


def _create_llm(no_wait: bool):
    # AWS injects .env.production; local experiments may use this ignored file.
    private_env = config.ROOT / "deploy" / "OPENAI_KEYS.private.env"
    if not os.environ.get("OPENAI_API_KEY") and private_env.exists():
        for raw in private_env.read_text(encoding="utf-8-sig").splitlines():
            raw = raw.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            key, value = key.strip(), value.strip()
            if key in {
                "OPENAI_API_KEY", "OPENAI_BASE_URL", "LLM_MODEL",
                "LLM_FALLBACK_MODEL", "LLM_PROVIDER", "JEONSE_LLM",
            } and value:
                os.environ.setdefault(key, value)

    wants_api = os.environ.get("JEONSE_LLM", "api").lower() not in {
        "mock", "rule", "offline",
    }
    if (wants_api and not os.environ.get("OPENAI_API_KEY") and not no_wait
            and sys.stdin.isatty()):
        print("\n[OpenAI 실험 설정]")
        print("키는 화면에 표시되지 않으며 현재 프로세스에서만 사용됩니다.")
        key = getpass.getpass("OpenAI API Key (취소하려면 Enter): ").strip()
        if key:
            os.environ["OPENAI_API_KEY"] = key
            os.environ["JEONSE_LLM"] = "api"
            os.environ["LLM_PROVIDER"] = "openai"

    key = os.environ.get("OPENAI_API_KEY", "").strip()
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip().lower()
    is_official_openai = not base_url or "api.openai.com" in base_url
    if wants_api and is_official_openai and key:
        if (not key.startswith("sk-") or any(char.isspace() for char in key)
                or "/" in key or "\\" in key):
            print("\nOpenAI API 키 형식이 올바르지 않습니다.")
            print("파일 경로나 다른 문자열이 아니라 sk- 로 시작하는 API 키를 입력하세요.")
            return None

    try:
        return get_llm()
    except Exception as exc:
        print("\nOpenAI LLM을 초기화하지 못했습니다.")
        print("실행 중 키를 입력하거나 deploy/OPENAI_KEYS.private.env에 설정하세요.")
        print(f"원인: {type(exc).__name__}: {exc}")
        if not no_wait:
            wait_for_key()
        return None


def _preflight_live_llm(llm: Any) -> None:
    """Fail before 50 cases if auth/model/structured output is unusable."""
    if not getattr(llm, "supports_agentic_calls", False):
        return
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
    }
    value = llm.analyze_json(
        operation="experiment.preflight",
        system="Return the requested JSON only.",
        user='Return {"ok": true}.',
        schema=schema,
        schema_name="experiment_preflight",
        max_tokens=32,
    )
    if not isinstance(value, dict) or value.get("ok") is not True:
        raise RuntimeError("OpenAI preflight returned an invalid response")


def _assert_live_result_valid(result: dict[str, Any], llm: Any) -> None:
    """Prevent API errors from being reported as deceptively fast results."""
    if not getattr(llm, "supports_agentic_calls", False):
        return
    output = result.get("llm_output") or {}
    trace = output.get("_trace") or {}
    fallback = output.get("fallback") is True or trace.get("fallback") is True
    if result.get("error") or fallback:
        detail = result.get("error") or trace.get("error") or "rule fallback detected"
        raise RuntimeError(f"invalid live experiment result: {detail}")


def _save_report(output_dir: str, kind: str, mode: str,
                 report: dict[str, Any]) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = destination / f"{kind}_{mode}_{stamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")
    return path


def _print_case(console: FixedHeaderConsole, index: int, total: int,
                result: dict[str, Any]) -> None:
    console.log("\n" + "=" * 78)
    console.log(f"[{index:02d}/{total:02d}] {result['case_id']} | {result['mode'].upper()}")
    console.log("-" * 78)
    console.log("[INPUT]")
    console.log(result["query"])
    console.log("\n[LLM OUTPUT]")
    console.log(json_text(result.get("llm_output")))
    console.log("\n[HOUSE RECOMMENDATION]")
    console.log(result.get("recommendation") or "추천 없음")
    console.log("\n[RAW RAG JSON]")
    console.log(json_text(result.get("rag_trace")))
    console.log(f"\n[CASE ELAPSED] {format_elapsed(result['elapsed_seconds'])}")


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))
    return ordered[index]


def _effective_workers(mode: str, requested: int | None, total: int) -> int:
    """Optimized is concurrent; NAIVE remains the explicit serial baseline."""
    if mode == "naive":
        return 1
    configured = requested if requested is not None else config.LLM_MAX_CONCURRENCY
    return max(1, min(int(configured), max(1, int(total))))


def _timed_run_case(case, mode: str, llm, batch_started: float) -> dict[str, Any]:
    task_started = time.perf_counter()
    result = run_case(case, mode, llm)
    _assert_live_result_valid(result, llm)
    task_completed = time.perf_counter()
    result["batch_timing"] = {
        "queue_wait_seconds": task_started - batch_started,
        "started_offset_seconds": task_started - batch_started,
        "completed_offset_seconds": task_completed - batch_started,
        "worker_elapsed_seconds": task_completed - task_started,
    }
    return result


def _execute_speed_cases(cases, mode: str, llm, workers: int,
                         batch_started: float):
    if workers == 1:
        for completion, case in enumerate(cases, 1):
            result = _timed_run_case(case, mode, llm, batch_started)
            result["completion_order"] = completion
            yield completion, result
        return
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="openai-query"
    ) as executor:
        futures = [
            executor.submit(_timed_run_case, case, mode, llm, batch_started)
            for case in cases
        ]
        for completion, future in enumerate(as_completed(futures), 1):
            result = future.result()
            result["completion_order"] = completion
            yield completion, result


def _timed_evaluate_case(case, mode: str, llm,
                         batch_started: float) -> dict[str, Any]:
    task_started = time.perf_counter()
    ground_truth = build_ground_truth(case)
    result = run_case(case, mode, llm)
    _assert_live_result_valid(result, llm)
    score = score_result(case, result, ground_truth)
    task_completed = time.perf_counter()
    result["ground_truth"] = ground_truth
    result["score"] = score
    result["batch_timing"] = {
        "queue_wait_seconds": task_started - batch_started,
        "started_offset_seconds": task_started - batch_started,
        "completed_offset_seconds": task_completed - batch_started,
        "worker_elapsed_seconds": task_completed - task_started,
    }
    return result


def _execute_accuracy_cases(cases, mode: str, llm, workers: int,
                            batch_started: float):
    if workers == 1:
        for completion, case in enumerate(cases, 1):
            result = _timed_evaluate_case(case, mode, llm, batch_started)
            result["completion_order"] = completion
            yield completion, result
        return
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="openai-accuracy"
    ) as executor:
        futures = [
            executor.submit(_timed_evaluate_case, case, mode, llm, batch_started)
            for case in cases
        ]
        for completion, future in enumerate(as_completed(futures), 1):
            result = future.result()
            result["completion_order"] = completion
            yield completion, result


def _observed_peak_concurrency(results: list[dict[str, Any]]) -> int:
    events: list[tuple[float, int]] = []
    for result in results:
        timing = result.get("batch_timing") or {}
        if "started_offset_seconds" in timing and "completed_offset_seconds" in timing:
            events.append((float(timing["started_offset_seconds"]), 1))
            events.append((float(timing["completed_offset_seconds"]), -1))
    active = peak = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        peak = max(peak, active)
    return peak


def run_speed(argv: list[str] | None = None) -> int:
    args = _parser("speed").parse_args(argv)
    mode = _choose_mode(args.mode)
    if args.mock:
        os.environ["JEONSE_LLM"] = "mock"
    cases = CASES[:max(1, min(args.limit, 50))]
    llm = _create_llm(args.no_wait)
    if llm is None:
        return 2
    try:
        _preflight_live_llm(llm)
    except Exception as exc:
        print(f"OpenAI 사전 점검 실패: {type(exc).__name__}: {exc}")
        return 2
    workers = _effective_workers(mode, args.workers, len(cases))
    started = time.perf_counter()
    console = FixedHeaderConsole(
        lambda: f"실행시간: {format_elapsed(time.perf_counter() - started)}")
    console.start()
    metadata = _metadata(mode, "speed", len(cases), llm, workers)
    results: list[dict[str, Any]] = []
    try:
        console.log(f"모드: {mode} | LLM: {metadata['llm_provider']}/{metadata['llm_model']}")
        console.log(
            f"질의 실행: {'병렬' if workers > 1 else '직렬'} | workers={workers}"
        )
        console.log(f"DB fingerprint: {metadata['database_fingerprint']}")
        for completion, result in _execute_speed_cases(
                cases, mode, llm, workers, started):
            results.append(result)
            _print_case(console, completion, len(cases), result)
    finally:
        console.stop()

    wall_elapsed = time.perf_counter() - started
    latencies = [float(item["elapsed_seconds"]) for item in results]
    batch_pipeline_seconds = max(
        ((item.get("batch_timing") or {}).get(
            "completed_offset_seconds", 0.0) for item in results),
        default=0.0,
    )
    peak_concurrency = _observed_peak_concurrency(results)
    summary = {
        "completed": len(results),
        "wall_elapsed_seconds": wall_elapsed,
        "parallel_batch_seconds": batch_pipeline_seconds,
        "throughput_queries_per_second": (
            len(results) / batch_pipeline_seconds if batch_pipeline_seconds else 0.0
        ),
        "configured_workers": workers,
        "observed_peak_concurrency": peak_concurrency,
        "measured_pipeline_seconds": sum(latencies),
        "mean_seconds": statistics.fmean(latencies) if latencies else 0.0,
        "median_p50_seconds": statistics.median(latencies) if latencies else 0.0,
        "p95_seconds": _percentile(latencies, 0.95),
        "min_seconds": min(latencies, default=0.0),
        "max_seconds": max(latencies, default=0.0),
        "errors": sum(bool(item.get("error")) for item in results),
    }
    results.sort(key=lambda item: item["case_id"])
    report = {"metadata": metadata, "summary": summary, "cases": results}
    path = _save_report(args.output_dir, "speed", mode, report)
    print("\n" + "=" * 78)
    print(f"최종 실행시간: {format_elapsed(wall_elapsed)}")
    print(
        f"workers {workers} | 관측 최대 동시성 {peak_concurrency} | "
        f"배치 {batch_pipeline_seconds:.3f}s | "
        f"처리량 {summary['throughput_queries_per_second']:.3f} query/s"
    )
    print(f"평균 {summary['mean_seconds']:.3f}s | P50 {summary['median_p50_seconds']:.3f}s "
          f"| P95 {summary['p95_seconds']:.3f}s | 오류 {summary['errors']}건")
    print(f"결과 JSON: {path.resolve()}")
    if not args.no_wait:
        wait_for_key()
    return 0 if len(results) == len(cases) else 1


def _run_legacy_search_hallucination(argv: list[str] | None = None) -> int:
    args = _parser("hallucination").parse_args(argv)
    mode = _choose_mode(args.mode)
    if args.mock:
        os.environ["JEONSE_LLM"] = "mock"
    cases = CASES[:max(1, min(args.limit, 50))]
    llm = _create_llm(args.no_wait)
    if llm is None:
        return 2
    try:
        _preflight_live_llm(llm)
    except Exception as exc:
        print(f"OpenAI 사전 점검 실패: {type(exc).__name__}: {exc}")
        return 2
    workers = _effective_workers(mode, args.workers, len(cases))
    correct = 0
    completed = 0
    started = time.perf_counter()
    console = FixedHeaderConsole(
        lambda: f"실행시간: {correct:02d}/{len(cases):02d}  "
                f"({(100 * correct / len(cases) if cases else 0):05.1f}%)")
    console.start()
    metadata = _metadata(mode, "hallucination", len(cases), llm, workers)
    results: list[dict[str, Any]] = []
    try:
        console.log(f"모드: {mode} | LLM: {metadata['llm_provider']}/{metadata['llm_model']}")
        console.log(
            f"질의 실행: {'병렬' if workers > 1 else '직렬'} | workers={workers}"
        )
        console.log("정답 기준: 조건 추출 + DB 근거 + 최상위 정렬이 모두 맞아야 1점")
        for completion, result in _execute_accuracy_cases(
                cases, mode, llm, workers, started):
            ground_truth = result["ground_truth"]
            score = result["score"]
            results.append(result)
            completed += 1
            correct += int(score["correct"])
            _print_case(console, completion, len(cases), result)
            console.log("\n[ACTUAL ANSWER / GROUND TRUTH]")
            console.log(json_text({
                "expected_slots": ground_truth["expected_slots"],
                "rationale": ground_truth["rationale"],
                "matching_property_count": ground_truth["matching_property_count"],
                "top_valid_property_ids": ground_truth["valid_property_ids"][:10],
            }))
            console.log("\n[JUDGEMENT]")
            console.log(json_text(score))
    finally:
        console.stop()

    elapsed = time.perf_counter() - started
    accuracy = correct / completed if completed else 0.0
    peak_concurrency = _observed_peak_concurrency(results)
    summary = {
        "correct": correct,
        "total": completed,
        "accuracy": accuracy,
        "elapsed_seconds": elapsed,
        "mean_slot_accuracy": (statistics.fmean(
            item["score"]["slot_accuracy"] for item in results) if results else 0.0),
        "errors": sum(bool(item.get("error")) for item in results),
        "configured_workers": workers,
        "observed_peak_concurrency": peak_concurrency,
    }
    results.sort(key=lambda item: item["case_id"])
    report = {"metadata": metadata, "summary": summary, "cases": results}
    path = _save_report(args.output_dir, "hallucination", mode, report)
    print("\n" + "=" * 78)
    print(f"최종 정답률: {correct:02d}/{completed:02d} ({accuracy * 100:.1f}%)")
    print(f"workers {workers} | 관측 최대 동시성 {peak_concurrency}")
    print(f"평균 조건 추출 점수: {summary['mean_slot_accuracy'] * 100:.1f}%")
    print(f"결과 JSON(전체 실제 정답 포함): {path.resolve()}")
    if not args.no_wait:
        wait_for_key()
    return 0 if completed == len(cases) else 1


def run_hallucination(argv: list[str] | None = None) -> int:
    """Run the production advisor mixed-intent benchmark.

    Kept here as a compatibility entrypoint for existing imports.  The legacy
    search-only evaluator remains private for old report reproducibility.
    """
    from src.experiments.advisor_hallucination import (
        run_hallucination as run_advisor_hallucination,
    )
    return run_advisor_hallucination(argv)


def main_speed() -> None:
    try:
        raise SystemExit(run_speed())
    except KeyboardInterrupt:
        print("\n사용자가 실험을 중단했습니다.")
        raise SystemExit(130)


def main_hallucination() -> None:
    try:
        raise SystemExit(run_hallucination())
    except KeyboardInterrupt:
        print("\n사용자가 실험을 중단했습니다.")
        raise SystemExit(130)
