"""LLM Text-to-SQL → 검증 → 읽기 전용 실행 → 수정 재시도 → 결정론 폴백."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from src.agent.llm import BaseLLM
from src.tools.finance_tool import FinanceTool
from src.tools.property_db_tool import PropertyDBTool


PROPERTY_REQUIRED_COLUMNS = {
    "property_id", "is_synthetic", "synthetic_notice", "sido", "gugun", "lat", "lng",
    "lease_type",
    "deposit_manwon", "monthly_rent_manwon", "maintenance_fee_manwon",
    "market_price_manwon", "my_priority_rank", "building_total_units", "fraud_score",
}


@dataclass
class SQLTrace:
    target: str
    strategy: str = "llm_text2sql"
    request_summary: str | None = None
    input_filters: dict = field(default_factory=dict)
    required_columns: list[str] = field(default_factory=list)
    attempts: list[dict] = field(default_factory=list)
    final_sql: str | None = None
    parameters: list = field(default_factory=list)
    validation: str = "not_run"
    row_count: int = 0
    result_preview: list[dict] = field(default_factory=list)
    fallback: bool = False
    fallback_reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class Text2SQLPipeline:
    def __init__(self, llm: BaseLLM, db_tool: PropertyDBTool,
                 finance_tool: FinanceTool):
        self.llm = llm
        self.db_tool = db_tool
        self.finance_tool = finance_tool

    def search_properties(self, user_text: str, slots: dict,
                          limit: int = 500) -> tuple[list[dict], dict]:
        safe_slots = dict(slots)
        safe_slots.pop("max_fraud_score", None)
        safe_slots.pop("safety_is_hard", None)
        sort_by = safe_slots.pop("sort_by", None)
        if sort_by == "risk_asc":
            safe_slots["order_by"] = "fraud_score ASC"
        elif sort_by == "risk_desc":
            safe_slots["order_by"] = "fraud_score DESC"
        elif sort_by == "price_asc":
            safe_slots["order_by"] = "deposit_manwon ASC"
        elif sort_by == "price_desc":
            safe_slots["order_by"] = "deposit_manwon DESC"
        safe_slots["limit"] = min(max(int(limit), 1), 500)
        fallback_sql = self.db_tool.build_query(safe_slots)
        trace = SQLTrace(
            target="properties",
            request_summary="확인된 매물 조건을 properties WHERE 절로 변환",
            input_filters=safe_slots,
            required_columns=sorted(PROPERTY_REQUIRED_COLUMNS),
        )

        request = (
            f"사용자 원문: {user_text}\n"
            f"검증된 검색 슬롯: {json.dumps(safe_slots, ensure_ascii=False, default=str)}\n"
            "properties에서 추천 후보를 조회하라. 검색 슬롯은 WHERE에 반영하되 "
            "fraud_score는 WHERE에 절대 쓰지 말고 ORDER BY에만 사용할 수 있다."
        )
        rows = self._try_llm_sql(
            request=request, allowed_tables={"properties"},
            required_columns=PROPERTY_REQUIRED_COLUMNS, trace=trace,
            required_slots=safe_slots,
        )
        if rows is not None:
            return rows, trace.to_dict()

        trace.strategy = "deterministic_slots"
        trace.fallback = True
        try:
            rows = self.db_tool.run_sql(
                fallback_sql, limit_cap=safe_slots["limit"],
                allowed_tables={"properties"},
            )
            trace.final_sql = fallback_sql
            trace.validation = "passed_deterministic_sql"
            trace.row_count = len(rows)
            trace.result_preview = _preview_rows(rows, "properties")
            return rows, trace.to_dict()
        except Exception as exc:
            trace.fallback_reason = _err(exc)
            trace.attempts.append({"stage": "deterministic_fallback", "ok": False,
                                   "error": _err(exc)})
            return [], trace.to_dict()

    def compile_property_filter(self, user_text: str, slots: dict,
                                limit: int = 500) -> dict:
        """확인된 UI 조건을 실행하지 않고 안전한 properties SQL로 컴파일한다."""
        normalized = dict(slots)
        if normalized.pop("region_sido", None):
            normalized["sido"] = slots["region_sido"]
        if normalized.pop("region_gugun", None):
            normalized["gugun"] = slots["region_gugun"]
        for key in (
            "workplace_landmark", "commute_mode", "max_commute_min",
            "min_safety_score", "min_convenience_score", "safety_is_hard",
            "max_fraud_score", "sort_by",
        ):
            normalized.pop(key, None)
        normalized["limit"] = min(max(int(limit), 1), 500)
        fallback_sql = self.db_tool.build_query(normalized)
        trace = SQLTrace(
            target="properties",
            request_summary="사용자가 최종 확인한 UI 조건을 properties WHERE 절로 사전 컴파일",
            input_filters={**normalized, "map_condition_separate": bool(
                slots.get("workplace_landmark"))},
            required_columns=sorted(PROPERTY_REQUIRED_COLUMNS),
        )
        request = (
            f"사용자 원문: {user_text}\n"
            f"최종 확인된 DB 검색 슬롯: {json.dumps(normalized, ensure_ascii=False, default=str)}\n"
            "properties에서 지도 후보를 조회할 읽기 전용 SQL을 생성하라. "
            "아직 실행하지 않으며, 모든 검색 슬롯을 WHERE에 포함하라. "
            "fraud_score는 WHERE에 절대 쓰지 않는다."
        )
        if self.llm.supports_agentic_calls:
            previous_error: str | None = None
            for attempt in range(1, 3):
                sql = None
                try:
                    generated = self.llm.generate_sql(
                        request, self.db_tool.schema_prompt({"properties"}),
                        previous_error=previous_error,
                    )
                    if not generated or not isinstance(generated.get("sql"), str):
                        raise ValueError("Text-to-SQL 응답에 sql이 없습니다")
                    sql = generated["sql"].strip()
                    ok, reason = self.db_tool.validate_sql(sql, {"properties"})
                    if not ok:
                        raise ValueError(reason)
                    _assert_no_risk_filter(sql)
                    _assert_slot_coverage(sql, normalized)
                    selected = _select_aliases(sql)
                    if "*" not in selected:
                        missing = PROPERTY_REQUIRED_COLUMNS - selected
                        if missing:
                            raise ValueError(f"필수 SELECT 컬럼 누락: {sorted(missing)}")
                    trace.attempts.append({
                        "repair_attempt": attempt, "ok": True,
                        "purpose": generated.get("purpose"),
                        "llm_attempts": list(getattr(self.llm, "last_trace", [])),
                    })
                    trace.final_sql = sql
                    trace.validation = "passed_readonly_validation_not_executed"
                    return trace.to_dict()
                except Exception as exc:
                    previous_error = _err(exc)
                    trace.attempts.append({
                        "repair_attempt": attempt, "ok": False,
                        "sql": sql, "error": previous_error,
                        "llm_attempts": list(getattr(self.llm, "last_trace", [])),
                    })
            trace.fallback_reason = previous_error
        else:
            trace.fallback_reason = "현재 LLM은 Text-to-SQL을 지원하지 않음"
        trace.strategy = "deterministic_slots"
        trace.fallback = True
        trace.final_sql = fallback_sql
        trace.validation = "passed_deterministic_sql_not_executed"
        return trace.to_dict()

    def search_finance(self, user_text: str, user: dict,
                       category: str | None = None,
                       max_rate_pct: float | None = None,
                       product_kind: str | None = None,
                       region: str | None = None,
                       finance_mode: str = "catalog",
                       limit: int = 10) -> tuple[list[dict], dict]:
        # 자연어 상위개념은 DB의 정확 카테고리 값이 아니므로 부분일치 조건으로 승격한다.
        if category in {"대출", "지원", "청약"}:
            product_kind = product_kind or category
            category = None
        trace = SQLTrace(
            target="finance_programs",
            request_summary="사용자 금융 질문과 프로필을 finance_programs WHERE 절로 변환",
            input_filters={
                "raw_query": user_text, "category": category,
                "max_rate_pct_exclusive": max_rate_pct,
                "product_kind": product_kind,
                "region": region,
                "finance_mode": finance_mode,
                "user_age": user.get("age"),
                "monthly_income_manwon": (
                    user.get("monthly_income_manwon")
                    if finance_mode == "eligibility" else None),
                "employment_type": user.get("employment_type"),
                "marital_status": user.get("marital_status"),
                "employment_months": user.get("employment_months"),
                "household_role": user.get("household_role"),
                "home_ownership_count": user.get("home_ownership_count"),
                "spouse_annual_income_manwon": user.get("spouse_annual_income_manwon"),
                "minor_children_count": user.get("minor_children_count"),
                "is_korean_national": user.get("is_korean_national"),
                "has_income_proof": user.get("has_income_proof"),
                "contract_deposit_paid_5pct": user.get("contract_deposit_paid_5pct"),
                "limit": limit,
            },
            required_columns=sorted({
                "program_id", "name", "category", "product_kind", "target",
                "support_content", "region_scope", "application_period",
                "application_status", "max_amount_manwon", "rate_pct",
                "income_limit_manwon", "source_url", "desc",
            }),
        )
        request = (
            f"사용자 원문: {user_text}\n"
            f"검색 모드: {finance_mode}, 사용자 나이: {user.get('age')}, 월소득: "
            f"{user.get('monthly_income_manwon') if finance_mode == 'eligibility' else '적용하지 않음'}, "
            f"희망 카테고리: {category or '전체'}, "
            f"상품 종류: {product_kind or '전체'}, 지역: {region or '전체'}, "
            f"배타적 금리 상한: {max_rate_pct}\n"
            "상품 종류는 product_kind에 쉼표 포함 문자열로 저장되어 있으므로 LIKE로 찾고, "
            "배타적 금리 상한은 rate_pct < 값으로 적용하고 rate_min_pct나 "
            "rate_max_pct로 바꾸지 않는다. "
            "지역을 지정하면 전국 정책 또는 region_scope/eligible_regions가 일치하는 정책을 찾는다. "
            "검색 모드가 catalog이면 income_limit_manwon 조건을 절대 넣지 말고 전체 제도를 탐색하라. "
            "검색 모드가 eligibility일 때만 월소득의 12배를 연소득 상한과 비교하고 나이 범위도 적용하라. "
            "age_min/age_max는 대부분 상품에서 NULL(연령 제한 없음)이므로 반드시 "
            "(age_min IS NULL OR age_min <= 나이) AND (age_max IS NULL OR age_max >= 나이) "
            "형태로 안전하게 비교하고, 절대 age_min <= 나이 AND age_max >= 나이 처럼 "
            "NULL 체크 없이 바로 비교하지 않는다(NULL 비교는 그 상품 전체를 제외시켜버린다). "
            f"eligibility 사용자 추가정보: 직업={user.get('employment_type')}, "
            f"재직개월={user.get('employment_months')}, 혼인={user.get('marital_status')}, "
            f"세대구분={user.get('household_role')}, "
            f"주택수={user.get('home_ownership_count')}, 배우자연소득={user.get('spouse_annual_income_manwon')}, "
            f"미성년자녀수={user.get('minor_children_count')}, "
            f"내국인={user.get('is_korean_national')}, 소득증빙={user.get('has_income_proof')}, "
            f"계약금5%={user.get('contract_deposit_paid_5pct')}. "
            "값이 NULL이면 해당 조건을 WHERE에 넣지 않는다. 값이 있으면 requires_korean_national, "
            "employment_type_codes, eligible_marital_status_codes, 직업별 min_employment_months_*, requires_household_head, "
            "allows_prospective_household_head, max_home_count, min_minor_children, spouse_income_limit_manwon, "
            "requires_income_proof, requires_contract_5pct 컬럼으로 명시적 불충족 상품을 제외한다. "
            f"finance_programs에서 모든 조건을 만족하는 상품을 금리 오름차순으로 최대 {limit}개 조회하라."
        )
        rows = self._try_llm_sql(
            request=request, allowed_tables={"finance_programs"},
            required_columns={"program_id", "name", "category", "product_kind", "target",
                              "support_content", "region_scope", "application_period",
                              "application_status", "max_amount_manwon", "rate_pct",
                              "income_limit_manwon", "source_url", "desc"},
            trace=trace,
            finance_filters={"max_rate_pct": max_rate_pct,
                              "product_kind": product_kind,
                              "region": region,
                              "finance_mode": finance_mode},
        )
        if rows is not None:
            rows = [self.finance_tool.annotate_eligibility(row, user) for row in rows[:limit]]
            if finance_mode == "eligibility":
                rows = [row for row in rows if row.get("eligibility_status") != "not_eligible"]
            return rows, trace.to_dict()

        trace.strategy = "parameterized_finance_query"
        trace.fallback = True
        try:
            rows = self.finance_tool.search(
                category=category, user_income_manwon=user.get("monthly_income_manwon"),
                user_age=user.get("age"), max_rate_pct=max_rate_pct,
                product_kind=product_kind, region=region,
                finance_mode=finance_mode, user_profile=user, limit=limit,
            )
            trace.final_sql, trace.parameters = self.finance_tool.build_query(
                category=category,
                user_income_manwon=user.get("monthly_income_manwon"),
                user_age=user.get("age"), max_rate_pct=max_rate_pct,
                product_kind=product_kind, region=region,
                finance_mode=finance_mode, user_profile=user, limit=limit,
            )
            trace.validation = "passed_parameterized_fallback"
            trace.row_count = len(rows)
            trace.result_preview = _preview_rows(rows, "finance_programs")
            return rows, trace.to_dict()
        except Exception as exc:
            trace.fallback_reason = _err(exc)
            trace.attempts.append({"stage": "finance_fallback", "ok": False,
                                   "error": _err(exc)})
            return [], trace.to_dict()

    def _try_llm_sql(self, *, request: str, allowed_tables: set[str],
                     required_columns: set[str], trace: SQLTrace,
                     required_slots: dict | None = None,
                     finance_filters: dict | None = None) -> list[dict] | None:
        if not self.llm.supports_agentic_calls:
            trace.fallback_reason = "현재 LLM은 Text-to-SQL을 지원하지 않음"
            return None

        previous_error: str | None = None
        # SQL 자체 오류는 오류 메시지를 다시 모델에 제공해 한 번 수정한다.
        for repair_attempt in range(1, 3):
            sql: str | None = None
            try:
                generated = self.llm.generate_sql(
                    request, self.db_tool.schema_prompt(allowed_tables),
                    previous_error=previous_error)
                if not generated or not isinstance(generated.get("sql"), str):
                    raise ValueError("Text-to-SQL 응답에 sql이 없습니다")
                sql = generated["sql"].strip()
                ok, reason = self.db_tool.validate_sql(sql, allowed_tables)
                if not ok:
                    raise ValueError(reason)
                if "properties" in allowed_tables:
                    _assert_no_risk_filter(sql)
                if required_slots:
                    _assert_slot_coverage(sql, required_slots)
                if finance_filters:
                    _assert_finance_coverage(sql, finance_filters)
                rows = self.db_tool.run_sql(
                    sql, limit_cap=500, allowed_tables=allowed_tables)
                present = set(rows[0]) if rows else _select_aliases(sql)
                # 결과가 0건이면 실제 row key를 볼 수 없다. SELECT *는 허용 테이블의
                # 전체 컬럼을 뜻하므로 필수 컬럼을 모두 선택한 것으로 처리한다.
                if "*" in present:
                    present = set(required_columns)
                missing = required_columns - present
                if missing:
                    raise ValueError(f"필수 SELECT 컬럼 누락: {sorted(missing)}")
                trace.attempts.append({
                    "repair_attempt": repair_attempt, "ok": True,
                    "purpose": generated.get("purpose"),
                    "sql_validation": "passed",
                    "llm_attempts": list(getattr(self.llm, "last_trace", [])),
                })
                trace.final_sql = sql
                trace.validation = "passed_readonly_execution"
                trace.row_count = len(rows)
                trace.result_preview = _preview_rows(rows, trace.target)
                return rows
            except Exception as exc:
                previous_error = _err(exc)
                trace.attempts.append({
                    "repair_attempt": repair_attempt, "ok": False,
                    "sql": sql, "error": previous_error,
                    "llm_attempts": list(getattr(self.llm, "last_trace", [])),
                })
        trace.fallback_reason = previous_error
        trace.validation = "failed_then_fallback"
        return None


def _select_aliases(sql: str) -> set[str]:
    """0행 결과에서도 단순 SELECT 컬럼 누락을 판별하기 위한 보조 파서."""
    import re
    match = re.search(r"(?is)^\s*select\s+(.*?)\s+from\s", sql)
    if not match:
        return set()
    columns = set()
    for raw in match.group(1).split(","):
        token = raw.strip().split()[-1].split(".")[-1].strip('"`[]')
        columns.add(token)
    return columns


def _assert_slot_coverage(sql: str, slots: dict) -> None:
    """LLM이 사용자가 확인한 핵심 필터를 조용히 누락하지 못하게 한다."""
    low = sql.lower()
    expected = {
        "lease_type": ("lease_type", "transaction_type"),
        "transaction_type": ("transaction_type", "lease_type"),
        "property_type": ("property_type", "house_type"),
        "sido": ("sido",), "gugun": ("gugun",),
        "max_deposit_manwon": ("deposit_manwon",),
        "max_sale_price_manwon": ("sale_price_manwon", "asking_price_manwon"),
        "max_monthly_rent_manwon": ("monthly_rent_manwon",),
        "max_maintenance_manwon": ("maintenance_fee_manwon",),
        "min_area_m2": ("area_m2", "exclusive_area_m2"),
        "max_building_age": ("building_age_years", "build_year"),
        "rental_only": ("transaction_type", "lease_type"),
    }
    missing = []
    for slot, alternatives in expected.items():
        if slots.get(slot) is not None and not any(col in low for col in alternatives):
            missing.append(slot)
    if missing:
        raise ValueError(f"확인된 검색 슬롯이 SQL에 누락됨: {missing}")


def _assert_no_risk_filter(sql: str) -> None:
    """위험도는 표시·정렬 전용이며 properties WHERE 조건으로 사용할 수 없다."""
    import re
    match = re.search(r"(?is)\bwhere\b(.*?)(?:\border\s+by\b|\blimit\b|$)", sql)
    if match and re.search(r"\bfraud_score\b", match.group(1)):
        raise ValueError("fraud_score는 필터가 아닌 표시·정렬 전용 컬럼입니다")


def _assert_finance_coverage(sql: str, filters: dict) -> None:
    """금리·상품종류가 LLM SQL에서 누락되거나 DB 값과 불일치하는 것을 차단한다."""
    import re
    low = sql.lower()
    if filters.get("max_rate_pct") is not None:
        if not re.search(r"\brate_pct\s*<", low):
            raise ValueError("배타적 금리 상한(rate_pct < 값)이 SQL에 누락됨")
    if filters.get("product_kind"):
        if "product_kind" not in low or not re.search(r"\b(like|instr)\b", low):
            raise ValueError("상품 종류는 product_kind LIKE 조건이어야 함")
    if filters.get("region") and "region_scope" not in low:
        raise ValueError("지역 조건에는 region_scope와 전국 정책 포함이 필요함")
    where_clause = low.split("where", 1)[1] if "where" in low else ""
    if filters.get("finance_mode") == "catalog" and "income_limit_manwon" in where_clause:
        raise ValueError("전체 제도 탐색(catalog)에 개인 소득 WHERE 조건을 넣을 수 없음")


def _err(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:800]


def _preview_rows(rows: list[dict], target: str, size: int = 3) -> list[dict]:
    """디버그 화면용 비민감 핵심 컬럼 미리보기."""
    if target == "finance_programs":
        keys = ("program_id", "name", "category", "product_kind", "region_scope",
                "rate_pct", "max_amount_manwon", "income_limit_manwon",
                "application_status")
    else:
        keys = ("property_id", "sido", "gugun", "lease_type", "property_type",
                "deposit_manwon", "monthly_rent_manwon", "sale_price_manwon",
                "fraud_score")
    return [{k: row.get(k) for k in keys if k in row} for row in rows[:size]]
