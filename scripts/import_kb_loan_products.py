"""사용자가 제공한 KB국민은행 대출상품 Excel을 정규화 CSV/SQLite로 적재한다.

상품 목록, 상세정보, 현행 공시실 시트를 합쳐 상품명을 기준으로 중복을 제거한다.
공개 자료만으로 가입 여부를 확정할 수 없는 조건은 NULL로 보존한다.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from src import config


SOURCE_DATE = "2026-07-23"


def _text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _number(text: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", _text(text).replace(",", ""))
    return float(match.group(1)) if match else None


def _limit_manwon(text: str) -> float | None:
    """절대 한도만 만원으로 변환한다. 비율/CSS 한도는 추측하지 않는다."""
    value = _text(text).replace(",", "")
    candidates: list[float] = []
    # 범위/괄호/세미콜론마다 별도 금액으로 보고, 1억5천만원 같은 복합표기는 합산한다.
    for clause in re.split(r"[~～;/()]", value):
        eok = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*억", clause)]
        cheonman = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*천만", clause)]
        # '천만원'에 속한 '만'은 숫자가 바로 앞에 없으므로 중복 집계되지 않는다.
        man = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*만원", clause)]
        if eok or cheonman or man:
            candidates.append(sum(eok) * 10000 + sum(cheonman) * 1000 + sum(man))
    return max(candidates) if candidates else None


def _id(name: str) -> str:
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:14].upper()
    return f"KB-{digest}"


def _normalized_constraints(name: str, eligibility: str, other: str) -> dict:
    text = f"{name} {eligibility} {other}"
    age = re.search(r"만\s*(\d+)\s*[~～-]\s*(\d+)세", text)
    age_min = int(age.group(1)) if age else (19 if "만 19세 이상" in text else None)
    age_max = int(age.group(2)) if age else None
    employment: list[str] = []
    if any(term in text for term in ("근로소득자", "직장인", "재직")):
        employment.append("employee")
    if any(term in text for term in ("사업소득자", "사업자")) and "사업자·사업소득" not in text:
        employment.append("business")
    if "공무원" in text:
        employment.append("public_official")
    if "연금소득자" in text:
        employment.append("pension")
    employee_months = None
    if "동일 직장 1년 이상" in text:
        employee_months = 12
    elif re.search(r"근로소득자는?\s*6개월 이상", text):
        employee_months = 6
    business_months = 12 if re.search(r"사업소득자는?\s*12개월 이상", text) else None
    spouse_limit = 7000.0 if "부부합산 연소득 7천만원 이하" in text else None
    max_homes = 0 if "무주택" in text else (1 if "1주택 이내" in text else None)
    return {
        "age_min": age_min,
        "age_max": age_max,
        "requires_korean_national": 1 if "내국인" in text else None,
        "employment_type_codes": ",".join(dict.fromkeys(employment)) or None,
        "eligible_marital_status_codes": (
            "married,prospective_newlywed" if "신혼부부" in text else None
        ),
        "min_employment_months_employee": employee_months,
        "min_employment_months_business": business_months,
        "requires_income_proof": 1 if any(term in text for term in ("소득 확인", "소득증빙")) else None,
        "requires_household_head": 1 if "세대주" in text else None,
        "allows_prospective_household_head": 1 if "예비세대주" in text or "세대주 인정자" in text else None,
        "max_home_count": max_homes,
        "spouse_income_limit_manwon": spouse_limit,
        "requires_contract_5pct": 1 if "계약금 5% 이상" in text else None,
        "requires_guarantee_review": 1 if any(term in text for term in ("보증 발급", "보증보험증권 발급")) else None,
        "requires_css_review": 1 if "CSS" in text else None,
    }


def normalize_workbook(path: Path = config.KB_LOAN_XLSX) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"KB 대출상품 Excel이 없습니다: {path}")
    catalog = pd.read_excel(path, sheet_name="상품목록_확인분").fillna("")
    details = pd.read_excel(path, sheet_name="상세정보_확인분").fillna("")
    disclosures = pd.read_excel(path, sheet_name="공시실_현행_확인분").fillna("")

    records: dict[str, dict] = {}
    for _, row in catalog.iterrows():
        name = _text(row.get("상품명"))
        if not name:
            continue
        records[name] = {
            "name": name,
            "category": _text(row.get("분류")) or "대출",
            "desc": _text(row.get("한줄설명")),
            "channel": _text(row.get("가입가능채널")),
            "loan_limit_text": _text(row.get("최고한도/한도표기")),
            "sale_status": _text(row.get("판매상태")),
            "data_level": _text(row.get("데이터수준")) or "catalog_verified",
            "source_url": _text(row.get("출처 URL")),
            "catalog_verified": 1,
            "current_disclosure_verified": 0,
        }

    for _, row in disclosures.iterrows():
        name = _text(row.get("상품명"))
        if not name:
            continue
        record = records.setdefault(name, {
            "name": name, "category": _text(row.get("공시실 분류")) or "대출",
            "desc": "KB국민은행 상품공시실 현행 목록에서 확인된 대출상품",
            "channel": "", "loan_limit_text": "", "sale_status": "현행 공시 확인",
            "data_level": "current_disclosure_verified", "source_url": "",
            "catalog_verified": 0, "current_disclosure_verified": 0,
        })
        record["current_disclosure_verified"] = 1
        record["disclosure_url"] = _text(row.get("공식 공시실 URL"))

    for _, row in details.iterrows():
        name = _text(row.get("상품명"))
        if not name:
            continue
        record = records.setdefault(name, {
            "name": name, "category": _text(row.get("분류")) or "대출",
            "desc": _text(row.get("상품특징/용도")), "channel": "",
            "loan_limit_text": "", "sale_status": "상세 페이지 확인",
            "data_level": "detail_verified", "source_url": "",
            "catalog_verified": 0, "current_disclosure_verified": 0,
        })
        detail_map = {
            "eligibility_text": "신청자격", "loan_period_text": "대출기간",
            "repayment_method": "상환방법", "rate_as_of": "금리 기준일",
            "base_rate_text": "기준금리/금리구조", "spread_rate_text": "가산금리",
            "preferential_rate_text": "우대금리", "preferential_rate_detail": "우대금리 상세",
            "early_repayment_fee_text": "중도상환수수료", "delinquency_rate_text": "연체이자/지연배상금",
            "required_documents": "필요서류", "validity_period_text": "공시/상품 유효기간",
            "department": "상품개발부서", "other_information": "기타 주요조건·제외사항",
        }
        for target, source in detail_map.items():
            record[target] = _text(row.get(source))
        record["detail_source_url"] = _text(row.get("상세 출처 URL"))
        record["channel"] = record.get("channel") or _text(row.get("신청/가입채널"))
        record["loan_limit_text"] = record.get("loan_limit_text") or _text(row.get("대출한도"))
        record["detail_loan_limit_text"] = _text(row.get("대출한도"))
        record["rate_min_pct"] = _number(row.get("최저금리"))
        record["rate_max_pct"] = _number(row.get("최고금리"))
        record["detail_verified"] = 1

    imported_at = datetime.now().astimezone().isoformat(timespec="seconds")
    output = []
    for name, source in records.items():
        detail_verified = int(source.get("detail_verified") or 0)
        eligibility = _text(source.get("eligibility_text"))
        constraints = _normalized_constraints(name, eligibility, _text(source.get("other_information")))
        limit = _limit_manwon(source.get("loan_limit_text") or source.get("detail_loan_limit_text"))
        min_rate = source.get("rate_min_pct")
        source_url = source.get("detail_source_url") or source.get("source_url") or source.get("disclosure_url") or ""
        row = {
            "program_id": _id(name), "name": name,
            "category": _text(source.get("category")) or "은행대출",
            "target": eligibility or "상품별 자격 및 KB국민은행 심사기준 충족 고객",
            "max_amount_manwon": limit, "rate_pct": min_rate,
            "income_limit_manwon": constraints["spouse_income_limit_manwon"],
            "source_url": source_url, "desc": _text(source.get("desc")),
            "policy_area": "금융", "product_kind": "대출", "region_scope": "전국",
            "eligible_regions": "전국", "support_content": _text(source.get("desc")),
            "application_period": "상시/상품 판매기간", "always_open": 1,
            "application_status": _text(source.get("sale_status")) or "공식 페이지 확인",
            "age_text": eligibility, "age_min": constraints["age_min"], "age_max": constraints["age_max"],
            "employment_status": constraints["employment_type_codes"],
            "additional_eligibility": eligibility,
            "application_procedure": _text(source.get("channel")),
            "application_site": source_url, "required_documents": _text(source.get("required_documents")),
            "other_information": _text(source.get("other_information")),
            "supervising_organization": "KB국민은행", "operating_organization": "KB국민은행",
            "reference_url_1": source_url, "tags": f"KB국민은행,{_text(source.get('category'))},대출",
            "last_modified_date": SOURCE_DATE, "rate_min_pct": min_rate,
            "rate_max_pct": source.get("rate_max_pct"),
            "source_type": "user_supplied_kb_loan_product_xlsx", "imported_at": imported_at,
            "provider": "KB국민은행", "channel": _text(source.get("channel")),
            "loan_limit_text": _text(source.get("loan_limit_text")),
            "loan_period_text": _text(source.get("loan_period_text")),
            "repayment_method": _text(source.get("repayment_method")),
            "rate_as_of": _text(source.get("rate_as_of")), "base_rate_text": _text(source.get("base_rate_text")),
            "spread_rate_text": _text(source.get("spread_rate_text")),
            "preferential_rate_text": _text(source.get("preferential_rate_text")),
            "preferential_rate_detail": _text(source.get("preferential_rate_detail")),
            "early_repayment_fee_text": _text(source.get("early_repayment_fee_text")),
            "delinquency_rate_text": _text(source.get("delinquency_rate_text")),
            "eligibility_text": eligibility, "validity_period_text": _text(source.get("validity_period_text")),
            "department": _text(source.get("department")),
            "data_level": "detail_verified" if detail_verified else _text(source.get("data_level")),
            "detail_verified": detail_verified, "catalog_verified": int(source.get("catalog_verified") or 0),
            "current_disclosure_verified": int(source.get("current_disclosure_verified") or 0),
            **constraints,
        }
        output.append(row)
    return pd.DataFrame(output).sort_values(["detail_verified", "category", "name"], ascending=[False, True, True])


def import_products(source: Path = config.KB_LOAN_XLSX,
                    output: Path = config.KB_LOAN_CSV,
                    rebuild_db: bool = True) -> pd.DataFrame:
    frame = normalize_workbook(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    if rebuild_db:
        from src.db.build_db import build_finance_db
        with sqlite3.connect(config.DB_PATH) as connection:
            build_finance_db(connection)
    return frame


if __name__ == "__main__":
    products = import_products()
    print(f"[KB] 정규화 상품: {len(products)}건")
    print(f"[KB] 상세 자격 확인: {int(products['detail_verified'].sum())}건")
    print(f"[KB] CSV: {config.KB_LOAN_CSV}")
    print(f"[KB] DB: {config.DB_PATH}")
