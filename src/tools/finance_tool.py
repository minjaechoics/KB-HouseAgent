"""
(4) 금융서비스 검색 모듈 / 데이터베이스 (Agent Tool).

첨부한 청년 주거·금융 정책을 조회한다. 정규화 CSV가 원본 역할을 하며
build_db.py 또는 refresh()가 finance_programs 테이블을 다시 만든다.

사용:
    from src.tools.finance_tool import FinanceTool
    tool = FinanceTool()
    tool.search(product_kind="대출", user_income_manwon=280)
    tool.refresh()   # 12h 스케줄러가 호출(정규화 CSV 재적재)
"""
from __future__ import annotations
import sqlite3
from typing import Optional

from src import config


class FinanceTool:
    def __init__(self, db_path=config.DB_PATH):
        self.db_path = db_path

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def build_query(self, category: Optional[str] = None,
                    user_income_manwon: Optional[float] = None,
                    user_age: Optional[int] = None,
                    max_rate_pct: Optional[float] = None,
                    product_kind: Optional[str] = None,
                    region: Optional[str] = None,
                    finance_mode: str = "eligibility",
                    user_profile: Optional[dict] = None,
                    limit: int = 10) -> tuple[str, list]:
        """Build the exact parameterized SQL used by search and debug traces."""
        q = "SELECT * FROM finance_programs WHERE 1=1"
        params: list = []
        if category:
            q += " AND category = ?"
            params.append(category)
        if product_kind:
            q += " AND (product_kind LIKE ? OR category LIKE ?)"
            params.extend([f"%{product_kind}%", f"%{product_kind}%"])
        if region:
            q += " AND (region_scope = '전국' OR region_scope = ? OR eligible_regions LIKE ?)"
            params.extend([region, f"%{region}%"])
        if max_rate_pct is not None:
            q += " AND rate_pct < ?"
            params.append(float(max_rate_pct))
        if finance_mode == "eligibility" and user_income_manwon is not None:
            q += " AND (income_limit_manwon IS NULL OR income_limit_manwon >= ?)"
            params.append(float(user_income_manwon) * 12)
        if finance_mode == "eligibility" and user_age is not None:
            q += " AND (age_min IS NULL OR age_min <= ?) AND (age_max IS NULL OR age_max >= ?)"
            params.extend([int(user_age), int(user_age)])
        profile = dict(user_profile or {})
        if user_age is not None:
            profile.setdefault("age", int(user_age))
        if user_income_manwon is not None:
            profile.setdefault("monthly_income_manwon", float(user_income_manwon))
        if finance_mode == "eligibility":
            nationality = profile.get("is_korean_national")
            if nationality is False:
                q += " AND (requires_korean_national IS NULL OR requires_korean_national = 0)"
            employment_type = profile.get("employment_type")
            if employment_type:
                q += (" AND (employment_type_codes IS NULL OR employment_type_codes = '' OR "
                      "instr(',' || replace(employment_type_codes, ' ', '') || ',', ',' || ? || ',') > 0)")
                params.append(employment_type)
                if employment_type != "business":
                    q += (" AND (requires_business_registration IS NULL OR "
                          "requires_business_registration = 0)")
            marital_status = profile.get("marital_status")
            if marital_status:
                q += (" AND (eligible_marital_status_codes IS NULL OR "
                      "eligible_marital_status_codes = '' OR "
                      "instr(',' || replace(eligible_marital_status_codes, ' ', '') || ',', ',' || ? || ',') > 0)")
                params.append(marital_status)
            minor_children = profile.get("minor_children_count")
            if minor_children is not None:
                q += " AND (min_minor_children IS NULL OR min_minor_children <= ?)"
                params.append(int(minor_children))
            employment_months = profile.get("employment_months")
            months_column = {
                "employee": "min_employment_months_employee",
                "business": "min_employment_months_business",
            }.get(employment_type)
            if months_column and employment_months is not None:
                q += f" AND ({months_column} IS NULL OR {months_column} <= ?)"
                params.append(int(employment_months))
            home_count = profile.get("home_ownership_count")
            if home_count is not None:
                q += " AND (max_home_count IS NULL OR max_home_count >= ?)"
                params.append(int(home_count))
            household_role = profile.get("household_role")
            if household_role == "member":
                q += " AND (requires_household_head IS NULL OR requires_household_head = 0)"
            elif household_role == "prospective_head":
                q += (" AND (requires_household_head IS NULL OR requires_household_head = 0 "
                      "OR allows_prospective_household_head = 1)")
            if profile.get("has_income_proof") is False:
                q += " AND (requires_income_proof IS NULL OR requires_income_proof = 0)"
            if profile.get("contract_deposit_paid_5pct") is False:
                q += " AND (requires_contract_5pct IS NULL OR requires_contract_5pct = 0)"
            spouse_income = profile.get("spouse_annual_income_manwon")
            annual_income = (float(user_income_manwon) * 12
                             if user_income_manwon is not None else None)
            if spouse_income is not None and annual_income is not None:
                q += " AND (spouse_income_limit_manwon IS NULL OR spouse_income_limit_manwon >= ?)"
                params.append(annual_income + float(spouse_income))
        q += (" ORDER BY CASE WHEN rate_pct IS NULL THEN 1 ELSE 0 END, rate_pct ASC, "
              "last_modified_date DESC LIMIT ?")
        params.append(limit)
        return q, params

    def search(self, category: Optional[str] = None,
               user_income_manwon: Optional[float] = None,
               user_age: Optional[int] = None,
               max_rate_pct: Optional[float] = None,
               product_kind: Optional[str] = None,
               region: Optional[str] = None,
               finance_mode: str = "eligibility",
               user_profile: Optional[dict] = None,
               limit: int = 10) -> list[dict]:
        """
        조건에 맞는 청년 금융지원 제도 조회.
        소득한도/연령 조건을 만족하는 제도만 필터.
        """
        profile = dict(user_profile or {})
        if user_age is not None:
            profile.setdefault("age", int(user_age))
        if user_income_manwon is not None:
            profile.setdefault("monthly_income_manwon", float(user_income_manwon))
        q, params = self.build_query(
            category=category, user_income_manwon=user_income_manwon,
            user_age=user_age, max_rate_pct=max_rate_pct,
            product_kind=product_kind, region=region,
            finance_mode=finance_mode, user_profile=profile, limit=limit,
        )

        con = self._conn()
        con.row_factory = sqlite3.Row
        rows = con.execute(q, params).fetchall()
        con.close()
        return [self.annotate_eligibility(dict(r), profile) for r in rows]

    @staticmethod
    def annotate_eligibility(row: dict, profile: dict | None = None) -> dict:
        """입력값과 공개조건을 비교해 예비 판정과 미확인 항목을 명시한다."""
        profile = profile or {}
        result = dict(row)
        failures: list[str] = []
        reviews: list[str] = []
        def _codes(value):
            return {part.strip() for part in str(value or "").split(",") if part.strip()}

        age = profile.get("age")
        if age is not None and result.get("age_min") is not None and int(age) < int(result["age_min"]):
            failures.append(f"최소 연령 {int(result['age_min'])}세 미달")
        if age is not None and result.get("age_max") is not None and int(age) > int(result["age_max"]):
            failures.append(f"최대 연령 {int(result['age_max'])}세 초과")
        marital = profile.get("marital_status")
        marital_codes = _codes(result.get("eligible_marital_status_codes"))
        if marital and marital_codes and marital not in marital_codes:
            failures.append("혼인 상태 조건 불충족")
        children = profile.get("minor_children_count")
        if result.get("min_minor_children") is not None:
            if children is None:
                reviews.append("미성년 자녀 수 미입력")
            elif int(children) < int(result["min_minor_children"]):
                failures.append(f"미성년 자녀 {int(result['min_minor_children'])}명 이상 조건 불충족")
        employment = profile.get("employment_type")
        employment_codes = _codes(result.get("employment_type_codes"))
        if employment and employment_codes and employment not in employment_codes:
            failures.append("직업 유형 조건 불충족")
        if result.get("requires_business_registration"):
            if employment is None:
                reviews.append("직업 유형(사업자 여부) 미입력")
            elif employment != "business":
                failures.append("개인사업자 대상 상품")
        if result.get("requires_korean_national") and profile.get("is_korean_national") is False:
            failures.append("내국인 조건 불충족")
        if result.get("requires_income_proof") and profile.get("has_income_proof") is False:
            failures.append("소득증빙 조건 불충족")
        if result.get("requires_contract_5pct") and profile.get("contract_deposit_paid_5pct") is False:
            failures.append("임대차 계약금 5% 지급 조건 불충족")
        if result.get("max_home_count") is not None and profile.get("home_ownership_count") is not None \
                and int(profile["home_ownership_count"]) > int(result["max_home_count"]):
            failures.append("주택 보유 수 조건 불충족")
        if result.get("requires_household_head") and profile.get("household_role") == "member":
            failures.append("세대주 또는 예비세대주 조건 불충족")
        if result.get("detail_verified") in (None, 0, 0.0, "0") and result.get("provider"):
            reviews.append("상세 신청자격이 수집되지 않아 상품 원문 확인 필요")
        if result.get("requires_css_review"):
            credit_grade = profile.get("credit_grade")
            reviews.append(
                f"KB CSS 신용심사 필요(입력 신용등급 {int(credit_grade)}등급 참고)"
                if credit_grade is not None else "KB CSS 신용심사 필요"
            )
        if result.get("requires_guarantee_review"):
            reviews.append("보증기관 보증서 발급 심사 필요")
        if result.get("requires_korean_national") and profile.get("is_korean_national") is None:
            reviews.append("내국인 여부 미입력")
        if result.get("requires_income_proof") and profile.get("has_income_proof") is None:
            reviews.append("소득증빙 가능 여부 미입력")
        if result.get("requires_household_head") and profile.get("household_role") is None:
            reviews.append("세대주/예비세대주 여부 미입력")
        if result.get("max_home_count") is not None and profile.get("home_ownership_count") is None:
            reviews.append("주택 보유 수 미입력")
        if result.get("requires_contract_5pct") and profile.get("contract_deposit_paid_5pct") is None:
            reviews.append("임대차 계약금 5% 이상 지급 여부 미입력")
        if result.get("employment_type_codes") and not profile.get("employment_type"):
            reviews.append("직업 유형 미입력")
        if result.get("eligible_marital_status_codes") and not profile.get("marital_status"):
            reviews.append("혼인 상태 미입력")
        if (result.get("spouse_income_limit_manwon") is not None
                and profile.get("marital_status") in {"married", "prospective_newlywed"}
                and profile.get("spouse_annual_income_manwon") is None):
            reviews.append("배우자 연소득 미입력")
        input_reviews = [item for item in reviews
                         if "미입력" in item or "상세 신청자격" in item]
        result["eligibility_status"] = (
            "not_eligible" if failures else
            "needs_review" if input_reviews else
            "preliminarily_eligible"
        )
        result["eligibility_failures"] = failures
        result["eligibility_reviews"] = list(dict.fromkeys(reviews))
        checks: list[dict] = []

        def add_check(label: str, known: bool, passed: bool, detail: str) -> None:
            checks.append({
                "label": label,
                "status": "passed" if known and passed else (
                    "failed" if known else "review"),
                "detail": detail,
            })

        if result.get("age_min") is not None or result.get("age_max") is not None:
            age_value = profile.get("age")
            age_ok = (
                age_value is not None
                and (result.get("age_min") is None or int(age_value) >= int(result["age_min"]))
                and (result.get("age_max") is None or int(age_value) <= int(result["age_max"]))
            )
            age_range = (
                f"{result.get('age_min') or 0}~{result.get('age_max') or '제한 없음'}세"
            )
            add_check("나이", age_value is not None, age_ok,
                      f"입력 {age_value if age_value is not None else '-'}세 · 기준 {age_range}")
        if result.get("income_limit_manwon") is not None:
            monthly_income = profile.get("monthly_income_manwon")
            annual_income = (
                float(monthly_income) * 12 if monthly_income is not None else None)
            add_check(
                "연소득", annual_income is not None,
                annual_income is not None
                and annual_income <= float(result["income_limit_manwon"]),
                f"입력 {annual_income:,.0f}만원" if annual_income is not None
                else "소득 미입력",
            )
        if marital_codes:
            add_check(
                "혼인상태", marital is not None,
                marital is not None and marital in marital_codes,
                f"입력 {marital or '-'} · 허용 {', '.join(sorted(marital_codes))}",
            )
        if result.get("min_minor_children") is not None:
            add_check(
                "미성년 자녀", children is not None,
                children is not None
                and int(children) >= int(result["min_minor_children"]),
                f"입력 {children if children is not None else '-'}명 · "
                f"기준 {int(result['min_minor_children'])}명 이상",
            )
        if employment_codes:
            add_check(
                "직업", employment is not None,
                employment is not None and employment in employment_codes,
                f"입력 {employment or '-'} · 허용 {', '.join(sorted(employment_codes))}",
            )
        if result.get("requires_korean_national"):
            nationality = profile.get("is_korean_national")
            add_check("국적", nationality is not None, nationality is True,
                      "대한민국 국민" if nationality is True else (
                          "외국 국적" if nationality is False else "미입력"))
        if result.get("max_home_count") is not None:
            home_count = profile.get("home_ownership_count")
            add_check(
                "주택보유", home_count is not None,
                home_count is not None
                and int(home_count) <= int(result["max_home_count"]),
                f"입력 {home_count if home_count is not None else '-'}채 · "
                f"기준 {int(result['max_home_count'])}채 이하",
            )
        if result.get("requires_household_head"):
            household_role = profile.get("household_role")
            allowed_roles = {"head"}
            if result.get("allows_prospective_household_head"):
                allowed_roles.add("prospective_head")
            add_check(
                "세대주", household_role is not None,
                household_role in allowed_roles,
                f"입력 {household_role or '-'}",
            )
        if result.get("requires_income_proof"):
            income_proof = profile.get("has_income_proof")
            add_check("소득증빙", income_proof is not None, income_proof is True,
                      "가능" if income_proof is True else (
                          "불가" if income_proof is False else "미입력"))
        if result.get("requires_contract_5pct"):
            contract_paid = profile.get("contract_deposit_paid_5pct")
            add_check(
                "계약금 5%", contract_paid is not None, contract_paid is True,
                "지급" if contract_paid is True else (
                    "미지급" if contract_paid is False else "미입력"),
            )
        if result.get("requires_business_registration"):
            add_check(
                "사업자등록", employment is not None, employment == "business",
                f"입력 {employment or '-'} · 개인사업자·사업소득자 대상",
            )
        if result.get("category") == "자동차대출" and result.get("max_amount_manwon") is not None:
            vehicle_price = profile.get("vehicle_price_manwon")
            add_check(
                "차량가액 대비 한도", vehicle_price is not None,
                vehicle_price is not None
                and float(vehicle_price) <= float(result["max_amount_manwon"]),
                f"입력 {vehicle_price:,.0f}만원" if vehicle_price is not None
                else "차량가액 미입력",
            )
        result["eligibility_checks"] = checks
        result["eligibility_disclaimer"] = "예비 판정이며 KB국민은행·보증기관의 최종 심사를 대체하지 않습니다."
        return result

    def categories(self) -> list[str]:
        con = self._conn()
        rows = con.execute(
            "SELECT DISTINCT category FROM finance_programs"
        ).fetchall()
        con.close()
        return [r[0] for r in rows]

    def refresh(self):
        """
        12시간 주기 갱신 훅.
        실서비스: 정부/기관 사이트 크롤링 → upsert.
        여기서는 정규화된 첨부 정책 CSV를 다시 적재한다.
        스케줄 예시(APScheduler):
            from apscheduler.schedulers.background import BackgroundScheduler
            sched = BackgroundScheduler()
            sched.add_job(tool.refresh, "interval", hours=12)
            sched.start()
        """
        from src.db.build_db import build_finance_db
        con = self._conn()
        build_finance_db(con)
        con.close()
        return "refreshed (attached policy CSV)"


if __name__ == "__main__":
    tool = FinanceTool()
    print("[finance_tool] 카테고리:", tool.categories())
    print("\n[search] 대출 연계 정책 (월소득 280만원):")
    for p in tool.search(product_kind="대출", user_income_manwon=280):
        print(f"  {p['name']} | 금리 {p['rate_pct']}% | 한도 {p['max_amount_manwon']}만원")
        print(f"    → {p['source_url']}")
