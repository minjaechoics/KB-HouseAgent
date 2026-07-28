from src.optimization import optimize_housing_choices
from src.preferences import normalize_preferences, preferences_from_text


def _user():
    return {"age": 30, "monthly_income_manwon": 500,
            "total_asset_manwon": 8000, "monthly_living_cost_manwon": 120}


def _programs():
    return [{
        "program_id": "mortgage", "name": "KB 주택담보대출",
        "category": "담보대출", "product_kind": "대출",
        "rate_pct": 3.5, "max_amount_manwon": 20000,
        "loan_period_text": "30년", "eligibility_status": "preliminarily_eligible",
    }]


def test_preferences_are_normalized_and_natural_language_requires_confirmation():
    profile = normalize_preferences({"mode": "stable", "approved": True})
    assert abs(sum(profile["weights"].values()) - 1) < 1e-5
    draft = preferences_from_text("출퇴근이 조금 길어도 안전하고 대출은 적었으면 해")
    assert draft["requires_confirmation"] is True
    assert draft["approved"] is False
    assert "safety" in draft["detected_dimensions"]
    assert "debt_aversion" in draft["detected_dimensions"]


def test_milp_returns_feasible_pareto_representatives():
    properties = [
        {"property_id": "safe", "transaction_type": "매매", "house_type": "아파트",
         "sale_price_manwon": 13000, "maintenance_fee_manwon": 8,
         "fraud_score": .05, "commute_minutes": 35},
        {"property_id": "close", "transaction_type": "매매", "house_type": "아파트",
         "sale_price_manwon": 14000, "maintenance_fee_manwon": 12,
         "fraud_score": .30, "commute_minutes": 10},
    ]
    result = optimize_housing_choices(
        properties, _programs(), _user(),
        {"mode": "stable", "approved": True}, {"horizon_years": 10},
    )
    assert result["status"] == "ok"
    assert result["solver"] == "mixed_integer_linear_programming"
    assert result["pareto_candidate_count"] >= 1
    assert all(row["hard_constraints"]["funding"] for row in result["representatives"])
    assert all(row["monthly_payment_manwon"] <=
               (_user()["monthly_income_manwon"] - _user()["monthly_living_cost_manwon"]) * .35 + .01
               for row in result["representatives"])
    assert any(trace["status"] in {"optimal", "feasible"}
               for trace in result["solver_traces"])


def test_infeasible_financing_returns_explicit_result():
    result = optimize_housing_choices(
        [{"property_id": "too-expensive", "transaction_type": "매매",
          "house_type": "아파트", "sale_price_manwon": 200000}],
        [], _user(), {"mode": "balanced"},
    )
    assert result["status"] == "infeasible"
    assert "없습니다" in result["message"]
