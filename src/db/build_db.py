"""
(4) 주택 조사 데이터베이스 (Agent Tool) — SQLite 구축.

CSV(properties.csv)를 SQLite로 적재하고,
전세 물건에 대해 전세사기 위험 모델을 돌려 fraud_score 컬럼을 채운다.
(요구사항: "수집한 데이터가 전세라면 전세사기위험예측 모듈을 통해 점수 attribute 업데이트")

또한 (4) 금융서비스 DB의 테이블도 함께 만든다(청년 금융지원 제도).

실행:
    python -m src.db.build_db
"""
from __future__ import annotations
import sqlite3
import pandas as pd

from src import config
from src.fraud_risk.infer import FraudRiskScorer
from src.real_estate_feeds.storage import ensure_feed_schema


def build_property_db(conn: sqlite3.Connection):
    df = pd.read_csv(config.DATA_GEN / "properties.csv")

    # 전세 물건에 대해 fraud_score 채우기
    scorer = FraudRiskScorer()
    jeonse_mask = df["lease_type"] == "전세"
    if jeonse_mask.any():
        scores = scorer.score_batch(df[jeonse_mask])
        df.loc[jeonse_mask, "fraud_score"] = scores
    # 월세는 fraud_score NULL 유지

    df.to_sql("properties", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prop_region ON properties(sido, gugun)")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_prop_property_id ON properties(property_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prop_listing_order ON properties(listing_updated_at DESC, property_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prop_lease ON properties(lease_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prop_transaction ON properties(transaction_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prop_house_type ON properties(house_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prop_region_txn_house ON properties(sido, gugun, transaction_type, house_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prop_coordinates ON properties(lat, lng)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prop_deposit ON properties(deposit_manwon)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prop_monthly_rent ON properties(monthly_rent_manwon)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prop_sale_price ON properties(sale_price_manwon)")
    conn.commit()
    n_scored = int(jeonse_mask.sum())
    print(f"[db] properties 적재: {len(df)}건 (전세 {n_scored}건 fraud_score 계산)")


def build_finance_db(conn: sqlite3.Connection):
    """정부 정책과 사용자 제공 금융상품 CSV를 하나의 검색 DB에 적재한다."""
    if not config.FINANCE_POLICY_CSV.exists():
        raise FileNotFoundError(
            f"금융정책 CSV가 없습니다: {config.FINANCE_POLICY_CSV}\n"
            "scripts/import_finance_policies.py로 첨부 원문을 먼저 가져오세요."
        )
    sources = [config.FINANCE_POLICY_CSV]
    if config.KB_LOAN_CSV.exists():
        sources.append(config.KB_LOAN_CSV)
    frames = [pd.read_csv(path, dtype={"program_id": str}) for path in sources]
    policies = pd.concat(frames, ignore_index=True, sort=False)
    policies = policies.drop_duplicates(subset=["program_id"], keep="last")
    if "min_minor_children" not in policies.columns:
        policies["min_minor_children"] = pd.NA
    # Official KB disclosures checked on 2026-07-23.  These structured
    # overrides turn human-readable eligibility into auditable SQL columns.
    kb_overrides = {
        "KB-3B7357B6C926DF": {
            "requires_income_proof": 1, "requires_css_review": 1,
            "detail_verified": 1, "current_disclosure_verified": 1,
        },
        "KB-D747CF4A3A1620": {
            "requires_income_proof": 1, "requires_css_review": 1,
            "detail_verified": 1, "current_disclosure_verified": 1,
        },
        "KB-AE460567BDE87F": {
            "rate_min_pct": 3.61, "rate_max_pct": 5.48,
            "rate_pct": 5.48, "rate_as_of": "2026-07-18",
            "min_minor_children": 2, "requires_household_head": 1,
            "allows_prospective_household_head": 1, "max_home_count": 1,
            "requires_contract_5pct": 1, "requires_guarantee_review": 1,
            "requires_css_review": 1, "detail_verified": 1,
            "current_disclosure_verified": 1,
            "loan_period_text": "1년 이상 2년 이내",
            "repayment_method": "일시상환, 혼합상환",
            "loan_limit_text": "임차보증금의 90% 이내, 최고 2억원",
            "source_url": "https://obank.kbstar.com/quics?page=C103531",
        },
        "KB-33B8304E29CF5B": {
            "rate_min_pct": 4.69, "rate_max_pct": 6.09,
            "rate_pct": 6.09, "rate_as_of": "2026-07-20",
            "requires_household_head": 1,
            "allows_prospective_household_head": 1, "max_home_count": 1,
            "requires_contract_5pct": 1, "requires_guarantee_review": 1,
            "requires_css_review": 1, "detail_verified": 1,
            "current_disclosure_verified": 1,
            "loan_period_text": "1년 초과 2년 이내",
            "repayment_method": "일시상환, 혼합상환",
            "loan_limit_text": "임차보증금의 80% 이내, 최고 4억4,400만원",
            "source_url": (
                "https://obank.kbstar.com/quics?cc=b104363%3Ab104516"
                "&page=C103507&prcode=LN20001331"
            ),
        },
    }
    for program_id, values in kb_overrides.items():
        mask = policies["program_id"].astype(str).eq(program_id)
        for column, value in values.items():
            if column not in policies.columns:
                policies[column] = pd.NA
            policies.loc[mask, column] = value
    policies.to_sql("finance_programs", conn, if_exists="replace", index=False)
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_finance_program_id ON finance_programs(program_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_finance_category ON finance_programs(category)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_finance_region ON finance_programs(region_scope)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_finance_dates ON finance_programs(application_end_date)")
    columns = {row[1] for row in conn.execute("PRAGMA table_info(finance_programs)")}
    if "provider" in columns:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_finance_provider ON finance_programs(provider)")
    if "detail_verified" in columns:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_finance_detail ON finance_programs(detail_verified)")
    if "min_minor_children" in columns:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_finance_children ON finance_programs(min_minor_children)")
    conn.commit()
    print(f"[db] finance_programs 적재: {len(policies)}개 (정책 + 금융상품 {len(sources)}개 소스)")


    build_guarantee_db(conn)


def build_guarantee_db(conn: sqlite3.Connection):
    """보증상품의 공식 공개 요건을 별도 테이블로 관리한다."""
    rows = [
        {
            "product_id": "HUG-JEONSE-RETURN",
            "provider": "주택도시보증공사(HUG)",
            "name": "전세보증금반환보증",
            "capital_deposit_limit_manwon": 70000,
            "noncapital_deposit_limit_manwon": 50000,
            "value_limit_ratio": 0.90,
            "requires_linked_loan_guarantee": 0,
            "official_review_required": 1,
            "source_url": "https://www.khug.or.kr/hug/web/ig/dr/igdr000001.jsp",
            "rule_as_of": "2026-07-28",
        },
        {
            "product_id": "HF-JEONSE-RETURN",
            "provider": "한국주택금융공사(HF)",
            "name": "전세지킴보증",
            "capital_deposit_limit_manwon": 70000,
            "noncapital_deposit_limit_manwon": 50000,
            "value_limit_ratio": 0.90,
            "requires_linked_loan_guarantee": 1,
            "official_review_required": 1,
            "source_url": "https://hf.go.kr/ko/sub02/sub02_05_01.do",
            "rule_as_of": "2026-07-28",
        },
        {
            "product_id": "SGI-JEONSE-RETURN",
            "provider": "SGI서울보증",
            "name": "전세금보장신용보험",
            "capital_deposit_limit_manwon": None,
            "noncapital_deposit_limit_manwon": None,
            "value_limit_ratio": None,
            "requires_linked_loan_guarantee": 0,
            "official_review_required": 1,
            "source_url": (
                "https://www.sgic.co.kr/biz/ccg/index.html?"
                "p=CCGIRI020101F01"
            ),
            "rule_as_of": "2026-07-28",
        },
    ]
    pd.DataFrame(rows).to_sql(
        "guarantee_products", conn, if_exists="replace", index=False
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_guarantee_product_id "
        "ON guarantee_products(product_id)"
    )
    conn.commit()


def build_region_stats_db(conn: sqlite3.Connection):
    """실제 KHUG 지역 사고율을 참조 테이블로 저장(Text2SQL에서 조인 가능)."""
    from src.data_augmentation.region_stats import load_region_accident_stats
    load_region_accident_stats().to_sql(
        "region_accident_stats", conn, if_exists="replace", index=False
    )
    conn.commit()
    print("[db] region_accident_stats 적재 완료")


def main():
    conn = sqlite3.connect(config.DB_PATH)
    build_property_db(conn)
    build_finance_db(conn)
    build_guarantee_db(conn)
    build_region_stats_db(conn)
    conn.close()
    ensure_feed_schema(config.DB_PATH)
    conn = sqlite3.connect(config.DB_PATH)
    # 검증
    for t in ("properties", "finance_programs", "region_accident_stats"):
        n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  [ok] {t}: {n} rows")
    conn.close()
    print(f"[db] SQLite 생성 완료 → {config.DB_PATH}")


if __name__ == "__main__":
    main()
