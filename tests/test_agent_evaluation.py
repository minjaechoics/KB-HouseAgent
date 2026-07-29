from src.evaluation import AgentEvaluator, build_golden_cases


def test_golden_set_has_165_cases():
    cases = build_golden_cases()
    assert len(cases) == 165
    assert len({case["case_id"] for case in cases}) == 165


def test_agent_evaluation_emits_separate_metrics():
    report = AgentEvaluator().run()
    metrics = report["metrics"]
    assert metrics["case_count"] == 165
    assert 0 <= metrics["intent_accuracy"] <= 1
    assert 0 <= metrics["required_slot_recall"] <= 1
    assert "sql_execution_success_rate" in metrics
    assert "retrieval_recall" in metrics
    assert metrics["latency_ms"]["p95"] >= 0
