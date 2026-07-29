"""
(4) 주택 조사 데이터베이스 Text2SQL 도구 (Agent Tool).

Agent가 사용자 조건(적정예산·지역·위험허용도)을 SQL로 바꿔 매물을 조회한다.
두 경로를 제공:
  1) build_query(slots):   결정론적 슬롯 기반 SQL 생성 (LLM 없이도 안전하게 동작).
  2) run_sql(sql):         LLM이 생성한 SQL을 안전 검증 후 실행 (SELECT만 허용).

안전장치(SQL injection/파괴 쿼리 방지):
  - SELECT 문만 허용, 세미콜론 다중구문 금지, DML/DDL 키워드 차단.
  - 화이트리스트 테이블/컬럼만 참조 가능.
"""
from __future__ import annotations
import re
import sqlite3
from typing import Optional

from src import config

ALLOWED_TABLES = {"properties", "finance_programs", "region_accident_stats"}
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|truncate|"
    r"vacuum|reindex|analyze|load_extension)\b",
    re.IGNORECASE,
)

SAFE_SQL_FUNCTIONS = {
    "abs", "avg", "coalesce", "count", "ifnull", "instr", "length", "like",
    "lower", "max", "min", "nullif", "round", "sum", "total", "trim", "upper",
}

PROPERTY_COLUMNS = [
    "property_id", "is_synthetic", "synthetic_notice", "sido", "gugun", "dong",
    "lat", "lng", "transaction_type",
    "lease_type", "property_type", "house_type", "asking_price_manwon",
    "sale_price_manwon",
    "deposit_manwon", "monthly_rent_manwon", "maintenance_fee_manwon",
    "onetime_fee_manwon", "market_price_manwon", "building_total_units",
    "my_priority_rank", "senior_deposit_sum_manwon", "senior_mortgage_manwon",
    "building_age_years", "area_m2", "fraud_label", "fraud_score",
]


class PropertyDBTool:
    def __init__(self, db_path=config.DB_PATH):
        self.db_path = db_path

    def _conn(self):
        # LLM SQL은 운영체제 수준에서도 읽기 전용 URI로 연다.
        uri = self.db_path.resolve().as_uri() + "?mode=ro"
        con = sqlite3.connect(uri, uri=True, timeout=3.0)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only=ON")
        con.execute("PRAGMA busy_timeout=3000")
        return con

    # ------------------------------------------------------------------
    def build_query(self, slots: dict) -> str:
        """
        슬롯 dict → 안전한 SELECT SQL 생성.

        지원 슬롯:
          lease_type: "전세"|"월세"
          sido, gugun: 문자열 또는 리스트 (지역 후보)
          max_deposit_manwon: 보증금 상한 (적정 계약금)
          max_monthly_rent_manwon: 월세 상한
          order_by: 위험도·가격 등 허용된 결과 정렬 기준
          min_priority_rank_first: True면 my_priority_rank=1 (최선순위만)
          order_by: 정렬 컬럼 (기본 deposit_manwon ASC)
          limit: 개수
        """
        where = ["1=1"]
        if lt := slots.get("lease_type"):
            where.append(f"lease_type = '{_san(lt)}'")

        if tt := slots.get("transaction_type"):
            where.append(f"transaction_type = '{_san(tt)}'")
        if slots.get("rental_only"):
            where.append("transaction_type IN ('전세', '월세')")
        if pt := slots.get("property_type"):
            safe_pt = _san(pt)
            where.append(
                f"(property_type = '{safe_pt}' OR house_type = '{safe_pt}' "
                f"OR house_type LIKE '%{safe_pt}%')"
            )

        for col in ("sido", "gugun"):
            val = slots.get(col)
            if isinstance(val, (list, tuple)) and val:
                joined = ",".join(f"'{_san(v)}'" for v in val)
                where.append(f"{col} IN ({joined})")
            elif isinstance(val, str) and val:
                where.append(f"{col} = '{_san(val)}'")

        if (v := slots.get("max_deposit_manwon")) is not None:
            where.append(f"deposit_manwon <= {float(v)}")
        if (v := slots.get("max_monthly_rent_manwon")) is not None:
            where.append(f"monthly_rent_manwon <= {float(v)}")
        if (v := slots.get("max_sale_price_manwon")) is not None:
            where.append(
                f"COALESCE(sale_price_manwon, asking_price_manwon, 0) <= {float(v)}"
            )
        if (v := slots.get("max_maintenance_manwon")) is not None:
            where.append(f"maintenance_fee_manwon <= {float(v)}")
        if (v := slots.get("min_area_m2")) is not None:
            where.append(f"area_m2 >= {float(v)}")
        if (v := slots.get("max_building_age")) is not None:
            where.append(f"building_age_years <= {float(v)}")
        if slots.get("min_priority_rank_first"):
            where.append("my_priority_rank = 1")

        order_by = slots.get("order_by", "deposit_manwon ASC")
        # order_by 컬럼 화이트리스트 검증
        col0 = order_by.split()[0]
        if col0 not in PROPERTY_COLUMNS:
            order_by = "deposit_manwon ASC"
        limit = max(1, min(int(slots.get("limit", 20)), 500))

        sql = (
            "SELECT property_id, is_synthetic, synthetic_notice, sido, gugun, dong, "
            "lat, lng, transaction_type, "
            "lease_type, property_type, house_type, asking_price_manwon, sale_price_manwon, "
            "deposit_manwon, monthly_rent_manwon, maintenance_fee_manwon, "
            "market_price_manwon, area_m2, building_age_years, my_priority_rank, building_total_units, "
            "fraud_score FROM properties WHERE "
            + " AND ".join(where)
            + f" ORDER BY {order_by} LIMIT {limit}"
        )
        return sql

    # ------------------------------------------------------------------
    def validate_sql(self, sql: str, allowed_tables: set[str] | None = None) -> tuple[bool, str]:
        if not isinstance(sql, str) or not sql.strip():
            return False, "빈 SQL"
        if len(sql) > 10000:
            return False, "SQL 길이 제한 초과"
        s = sql.strip().rstrip(";")
        if ";" in s:
            return False, "다중 구문(;) 금지"
        if "--" in s or "/*" in s or "*/" in s:
            return False, "SQL 주석 금지"
        if not re.match(r"(?is)^\s*select\b", s):
            return False, "SELECT 문만 허용"
        if FORBIDDEN.search(s):
            return False, "DML/DDL 키워드 차단"
        # 참조 테이블 화이트리스트 체크
        tables = set(re.findall(r"\bfrom\s+(\w+)|\bjoin\s+(\w+)", s, re.IGNORECASE))
        flat = {t for pair in tables for t in pair if t}
        permitted = allowed_tables or ALLOWED_TABLES
        if not flat or not flat.issubset(permitted):
            return False, f"허용되지 않은 테이블: {flat - permitted}"
        return True, "ok"

    def run_sql(self, sql: str, limit_cap: int = 500,
                allowed_tables: set[str] | None = None) -> list[dict]:
        permitted = allowed_tables or ALLOWED_TABLES
        ok, msg = self.validate_sql(sql, permitted)
        if not ok:
            raise ValueError(f"unsafe SQL rejected: {msg}")
        con = self._conn()
        try:
            table_columns = {
                table: {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
                for table in permitted
            }

            def authorizer(action, arg1, arg2, _db, _trigger):
                if action == sqlite3.SQLITE_SELECT:
                    return sqlite3.SQLITE_OK
                if action == sqlite3.SQLITE_READ:
                    return (sqlite3.SQLITE_OK
                            if arg1 in permitted and arg2 in table_columns.get(arg1, set())
                            else sqlite3.SQLITE_DENY)
                if action == sqlite3.SQLITE_FUNCTION:
                    return (sqlite3.SQLITE_OK if str(arg2).lower() in SAFE_SQL_FUNCTIONS
                            else sqlite3.SQLITE_DENY)
                return sqlite3.SQLITE_DENY

            con.set_authorizer(authorizer)
            # 비정상적으로 비싼 LLM 쿼리는 VM 명령 예산을 넘으면 중단한다.
            con.set_progress_handler(lambda: 1, 1_000_000)
            rows = con.execute(sql).fetchmany(limit_cap)
            return [dict(r) for r in rows]
        finally:
            con.close()

    def query(self, slots: dict) -> list[dict]:
        """슬롯 → SQL → 실행 (원스톱)."""
        return self.run_sql(self.build_query(slots))

    def schema_prompt(self, allowed_tables: set[str] | None = None) -> str:
        """현재 SQLite의 실제 컬럼을 읽어 허용된 테이블만 LLM에 주입한다."""
        permitted = set(allowed_tables or ALLOWED_TABLES)
        unknown = permitted - ALLOWED_TABLES
        if unknown:
            raise ValueError(f"schema prompt에 허용되지 않은 테이블: {sorted(unknown)}")
        descriptions = {
            "properties": (
                "모든 주택유형의 연구용 합성 매물. 금액 *_manwon은 만원, "
                "transaction_type/lease_type은 매매·전세·월세, fraud_score는 0~1이다."
            ),
            "finance_programs": (
                "청년 주거정책과 KB국민은행 대출상품. income_limit_manwon은 연소득 상한(만원), "
                "product_kind는 지원·주거공급·청약,대출 등이다. KB 상품의 상세조건은 "
                "employment_type_codes, eligible_marital_status_codes, min_employment_months_*, requires_household_head, "
                "max_home_count, spouse_income_limit_manwon 등의 정규화 컬럼에 있다."
            ),
            "region_accident_stats": "HUG 시군구별 보증사고 통계와 사고율.",
        }
        lines = ["현재 SQLite 실제 스키마(아래 테이블·컬럼만 사용):"]
        con = self._conn()
        try:
            for table in sorted(permitted):
                columns = con.execute(f"PRAGMA table_info({table})").fetchall()
                if not columns:
                    raise RuntimeError(f"DB에 테이블이 없습니다: {table}")
                rendered = ", ".join(
                    f"{row[1]} {row[2] or 'ANY'}" for row in columns
                )
                lines.append(f"테이블 {table}: {descriptions[table]}")
                lines.append(f"  컬럼: {rendered}")
        finally:
            con.close()
        lines.append("규칙: SELECT만 사용한다. WHERE에는 위 실제 컬럼만 사용한다.")
        return "\n".join(lines)


def _san(v: str) -> str:
    """작은따옴표 이스케이프(간단 방어)."""
    return str(v).replace("'", "''")


if __name__ == "__main__":
    tool = PropertyDBTool()
    slots = dict(lease_type="전세", sido="서울", max_deposit_manwon=3000,
                 order_by="fraud_score ASC", limit=5)
    sql = tool.build_query(slots)
    print("[build_query] SQL:\n ", sql)
    print("\n[query] 결과:")
    for r in tool.query(slots):
        print(f"  {r['sido']} {r['gugun']:6s} 보증금 {r['deposit_manwon']:.0f}만 "
              f"| 위험도 {r['fraud_score']:.3f} | 순위 {r['my_priority_rank']}")
    print("\n[validate] 위험 쿼리 차단 테스트:")
    for bad in ["DROP TABLE properties", "SELECT * FROM users; DELETE FROM properties",
                "SELECT * FROM secret_table"]:
        print(f"  {bad[:40]:40s} -> {tool.validate_sql(bad)}")
