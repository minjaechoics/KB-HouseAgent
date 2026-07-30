from __future__ import annotations

import json

from src.experiments.algorithm_runner import run_algorithm


def test_algorithm_experiment_reuses_identical_decisions_and_workers(
        tmp_path, monkeypatch):
    monkeypatch.setenv("JEONSE_LLM", "mock")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code = run_algorithm([
        "--mock", "--limit", "4", "--workers", "2", "--no-wait",
        "--output-dir", str(tmp_path),
    ])
    assert code == 0
    reports = list(tmp_path.glob("algorithm_paired_*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["metadata"]["query_count"] == 4
    assert report["metadata"]["shared_query_workers"] == 2
    assert report["metadata"]["controls"][
        "same_llm_decision_replayed_to_both"
    ] is True
    assert report["common_agentic_stage"]["prompt_count"] == 4
    assert report["optimized"]["summary"]["configured_workers"] == 2
    assert report["naive"]["summary"]["configured_workers"] == 2
    optimized = {row["case_id"]: row["llm_output"]
                 for row in report["optimized"]["cases"]}
    naive = {row["case_id"]: row["llm_output"]
             for row in report["naive"]["cases"]}
    assert optimized == naive

