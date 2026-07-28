from pathlib import Path

from src.audit import DecisionAuditStore
from src.report.budget import simulate
from src.simulation import simulate_probabilistic


def _fixture():
    user = {
        "age": 30, "monthly_income_manwon": 450,
        "total_asset_manwon": 7000, "monthly_living_cost_manwon": 120,
    }
    prop = {
        "property_id": "P1", "transaction_type": "매매",
        "sale_price_manwon": 15000, "maintenance_fee_manwon": 10,
    }
    forecast = {"annual_growth_rate": 0.02, "annual_low": -0.02,
                "annual_high": 0.06}
    programs = [{
        "program_id": "L1", "name": "청년 주택구입 대출", "category": "구입대출",
        "product_kind": "대출", "rate_pct": 3.5,
        "max_amount_manwon": 12000, "loan_period_text": "30년",
        "eligibility_status": "preliminarily_eligible",
    }]
    budget = simulate(user, prop, forecast, programs, {"horizon_years": 10})
    return user, prop, forecast, budget


def test_monte_carlo_is_reproducible_and_reports_10k_paths():
    user, prop, forecast, budget = _fixture()
    first = simulate_probabilistic(
        user, prop, forecast, budget, {"horizon_years": 10}, paths=10_000, seed=1729)
    second = simulate_probabilistic(
        user, prop, forecast, budget, {"horizon_years": 10}, paths=10_000, seed=1729)
    assert first["path_count"] == 10_000
    assert first["base"] == second["base"]
    assert len(first["base"]["yearly_percentiles"]) == 11
    assert first["base"]["ten_year_net_worth"]["p10"] <= first["base"][
        "ten_year_net_worth"]["p50"] <= first["base"]["ten_year_net_worth"]["p90"]


def test_rate_stress_does_not_improve_financing_outcome():
    user, prop, forecast, budget = _fixture()
    result = simulate_probabilistic(user, prop, forecast, budget, paths=2500, seed=99)
    assert result["rate_plus_2pp"]["terminal_net_worth"]["p50"] <= result["base"][
        "terminal_net_worth"]["p50"]
    assert result["rate_plus_2pp"]["repayment_distress_probability"] >= result[
        "base"]["repayment_distress_probability"]


def test_decision_audit_redacts_secret_and_replays_steps(tmp_path: Path):
    store = DecisionAuditStore(tmp_path / "audit.db")
    run_id = store.start_run(
        session_id="s1", property_id="P1", simulation_seed=42,
        input_snapshot={"income": 300, "api_key": "sk-never-store"},
        model_versions={"simulation": "v1"}, data_version="test",
    )
    store.record_step(
        run_id, stage="finance_sql", tool="sqlite",
        input_data={"query": "전세"}, output_data={"rows": 2},
        sql_text="SELECT * FROM finance_programs WHERE region_scope=?",
        sql_parameters=["전국"],
    )
    store.complete_run(run_id, {"answer": "ok"}, elapsed_ms=12.5)
    record = store.get(run_id)
    assert record is not None and record["status"] == "completed"
    assert record["input"]["api_key"] == "[REDACTED]"
    assert record["simulation_seed"] == 42
    assert record["steps"][0]["sql_text"].startswith("SELECT")
