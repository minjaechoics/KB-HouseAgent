"""첨부 청년 주거·금융 정책의 정규화 및 DB 적재 회귀 테스트."""
from __future__ import annotations

import sqlite3

import pandas as pd

from src import config
from src.tools.finance_tool import FinanceTool


def test_attached_policy_csv_and_db_are_in_sync():
    frame = pd.read_csv(config.FINANCE_POLICY_CSV, dtype={"program_id": str})
    assert len(frame) == 6
    assert frame["program_id"].is_unique
    assert set(frame["region_scope"]) == {"전국", "울산", "충남", "세종"}

    with sqlite3.connect(config.DB_PATH) as con:
        db_ids = {row[0] for row in con.execute(
            "SELECT program_id FROM finance_programs"
        )}
    assert set(frame["program_id"]).issubset(db_ids)
    assert len(db_ids) == 83  # 기존 정책 6 + KB국민은행 정규화 상품 77


def test_policy_amounts_and_rich_fields_are_normalized():
    with sqlite3.connect(config.DB_PATH) as con:
        con.row_factory = sqlite3.Row
        guarantee = dict(con.execute(
            "SELECT * FROM finance_programs WHERE name LIKE '%보증료%' AND region_scope='전국'"
        ).fetchone())
        account = dict(con.execute(
            "SELECT * FROM finance_programs WHERE name='청년주택드림청약통장'"
        ).fetchone())

    assert guarantee["max_amount_manwon"] == 40
    assert guarantee["income_limit_manwon"] == 7500
    assert guarantee["support_content"] and guarantee["application_site"]
    assert account["max_amount_manwon"] == 40000
    assert account["rate_pct"] == 2.4
    assert account["age_min"] == 19 and account["age_max"] == 34


def test_finance_search_uses_annual_income_age_kind_and_region():
    tool = FinanceTool()
    rows = tool.search(
        product_kind="대출", region="서울", user_income_manwon=300,
        user_age=29, finance_mode="eligibility",
    )
    policy_rows = [row for row in rows if not row["program_id"].startswith("KB-")]
    assert [row["name"] for row in policy_rows] == ["청년주택드림청약통장"]
    assert any(row["provider"] == "KB국민은행" for row in rows)

    under_two = tool.search(
        product_kind="대출", max_rate_pct=2, finance_mode="catalog"
    )
    assert under_two == []


if __name__ == "__main__":
    test_attached_policy_csv_and_db_are_in_sync()
    test_policy_amounts_and_rich_fields_are_normalized()
    test_finance_search_uses_annual_income_age_kind_and_region()
    print("OK: attached finance policy import tests passed")
