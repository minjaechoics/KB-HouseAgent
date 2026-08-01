"""KB국민은행 사용자 제공 Excel 적재 및 예비 자격판정 회귀 테스트."""
from pathlib import Path
import sqlite3

from scripts.import_kb_loan_products import normalize_workbook
from src import config
from src.db.build_db import build_finance_db
from src.tools.finance_tool import FinanceTool


def test_kb_workbook_union_and_verified_detail_count():
    frame = normalize_workbook()
    assert len(frame) == 77
    assert int(frame["detail_verified"].sum()) == 11
    emergency = frame.loc[frame["name"] == "KB 비상금대출"].iloc[0]
    assert emergency["max_amount_manwon"] == 300
    assert emergency["age_min"] == 19
    assert emergency["rate_min_pct"] == 5.83
    car = frame.loc[frame["name"] == "KB 매직카대출(신차)"].iloc[0]
    assert car["max_amount_manwon"] == 6000
    assert car["min_employment_months_employee"] == 6
    assert car["min_employment_months_business"] == 12


def test_combined_finance_db_keeps_policy_and_bank_products(tmp_path: Path):
    db = tmp_path / "finance.db"
    with sqlite3.connect(db) as connection:
        build_finance_db(connection)
        total, kb = connection.execute(
            "SELECT COUNT(*), SUM(provider='KB국민은행') FROM finance_programs"
        ).fetchone()
    assert total == 83
    assert kb == 77


def test_kb_youth_jeonse_preliminary_eligibility_and_hard_filters(tmp_path: Path):
    db = tmp_path / "finance.db"
    with sqlite3.connect(db) as connection:
        build_finance_db(connection)
    tool = FinanceTool(db)
    eligible_profile = {
        "employment_type": "employee", "employment_months": 24,
        "marital_status": "single",
        "household_role": "head", "home_ownership_count": 0,
        "spouse_annual_income_manwon": 0, "is_korean_national": True,
        "has_income_proof": True, "contract_deposit_paid_5pct": True,
    }
    rows = tool.search(
        product_kind="대출", user_income_manwon=300, user_age=29,
        user_profile=eligible_profile, limit=200,
    )
    youth = next(row for row in rows if row["name"] == "KB 청년 맞춤형 전세자금대출")
    assert youth["eligibility_status"] == "preliminarily_eligible"

    too_old = tool.search(
        product_kind="대출", user_income_manwon=300, user_age=40,
        user_profile=eligible_profile, limit=200,
    )
    assert "KB 청년 맞춤형 전세자금대출" not in {row["name"] for row in too_old}
    assert "KB 신혼부부 전세자금대출" not in {row["name"] for row in rows}


def test_gui_collects_finance_precheck_fields():
    gui = (Path(__file__).parents[1] / "src" / "server" / "gui.html").read_text(
        encoding="utf-8"
    )
    for value in (
        "금융상품 사전 판정 정보", 'id="employmentType"', 'id="employmentMonths"',
        'id="householdRole"', 'id="homeCount"', 'id="spouseIncome"',
        'id="nationality"', 'id="incomeProof"', 'id="contractPaid"',
        "추가 심사 필요",
    ):
        assert value in gui


def test_gui_collects_credit_and_vehicle_loan_precheck_fields():
    gui = (Path(__file__).parents[1] / "src" / "server" / "gui.html").read_text(
        encoding="utf-8"
    )
    for value in (
        'id="creditGrade"', 'id="vehiclePurchaseType"', 'id="vehiclePrice"',
        "credit_grade:", "vehicle_purchase_type:", "vehicle_price_manwon:",
    ):
        assert value in gui


def test_kb_v2_rescrape_refreshed_rate_and_disclosure_data():
    """재수집한 KB 상세 페이지(Playwright)에서 얻은 금리·자격요건이
    kb_kookmin_loan_products.csv(정규화 워크북과 별개로, build_finance_db가
    직접 읽는 원본)에 반영되어 있는지 확인한다(과거 정적 스크래핑은 카테고리
    목록만 반복 저장해 상품별 실제 금리가 전혀 없었음)."""
    import pandas as pd
    frame = pd.read_csv(config.KB_LOAN_CSV, dtype={"program_id": str})
    house_loan = frame.loc[frame["name"] == "KB 주택담보대출"].iloc[0]
    assert house_loan["current_disclosure_verified"] == 1
    assert house_loan["rate_min_pct"] > 0
    assert house_loan["rate_max_pct"] > house_loan["rate_min_pct"]
    assert "담보" in house_loan["eligibility_text"]
    assert (frame["requires_business_registration"] == 1).sum() > 0


def test_business_registration_gate_excludes_non_business_employment(tmp_path: Path):
    db = tmp_path / "finance.db"
    with sqlite3.connect(db) as connection:
        build_finance_db(connection)
    tool = FinanceTool(db)
    employee_rows = tool.search(
        category="개인사업자대출", user_profile={"employment_type": "employee"},
        limit=200,
    )
    assert employee_rows == []

    business_rows = tool.search(
        category="개인사업자대출", user_profile={"employment_type": "business"},
        limit=200,
    )
    assert len(business_rows) > 0
    for row in business_rows:
        checks = {c["label"]: c for c in row["eligibility_checks"]}
        assert checks["사업자등록"]["status"] == "passed"

    unknown_rows = tool.search(category="개인사업자대출", user_profile={}, limit=200)
    assert len(unknown_rows) > 0
    assert any(
        "사업자 여부" in review
        for row in unknown_rows for review in row["eligibility_reviews"]
    )


def test_vehicle_price_check_flags_over_limit_auto_loan(tmp_path: Path):
    db = tmp_path / "finance.db"
    with sqlite3.connect(db) as connection:
        build_finance_db(connection)
    tool = FinanceTool(db)
    rows = tool.search(
        category="자동차대출", user_profile={"vehicle_price_manwon": 100_000},
        limit=200,
    )
    assert len(rows) > 0
    capped = [r for r in rows if r.get("max_amount_manwon")]
    assert capped, "at least one auto loan row should carry a numeric loan limit"
    for row in capped:
        check = next(
            c for c in row["eligibility_checks"] if c["label"] == "차량가액 대비 한도"
        )
        assert check["status"] == "failed"
