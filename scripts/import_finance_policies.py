"""청년정책 상세 페이지 붙여넣기 문서를 금융/주거정책 DB로 정규화한다.

실행:
    py -3 scripts/import_finance_policies.py <pasted-text.txt>

산출물:
    data/downloaded/finance_policies/source_youth_policies.txt
    data/downloaded/finance_policies/youth_housing_policies.csv
    data/generated/jeonse_helper.db::finance_programs
"""
from __future__ import annotations

import argparse
import re
import shutil
import sqlite3
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config


REGION_LABELS = {
    "전국", "서울", "경기", "인천", "부산", "대구", "대전", "광주", "울산",
    "세종", "충남", "충북", "전남", "전북", "경남", "경북", "강원", "제주",
}
FIELD_LABELS = {
    "정책번호", "정책분야", "지원내용", "사업 운영 기간", "사업 신청기간",
    "지원 규모(명)", "연령", "혼인여부", "거주지역", "소득", "학력", "전공",
    "취업상태", "특화분야", "추가사항", "참여제한 대상", "신청절차", "심사 및 발표",
    "신청 사이트", "제출 서류", "기타 정보", "주관 기관", "운영 기관",
    "참고사이트 1", "참고사이트 2",
}


def _field(lines: list[str], start: int, end: int, label: str) -> str:
    for i in range(start, end):
        if lines[i].strip() != label:
            continue
        values = []
        for j in range(i + 1, end):
            value = lines[j].strip()
            if value in FIELD_LABELS or value in {"신청자격", "신청방법", "기타", "정보 변경 내역"}:
                break
            if value and not value.startswith(f"{label} -"):
                values.append(value)
        return "\n".join(values).strip()
    return ""


def _korean_amounts_manwon(text: str) -> list[float]:
    values: list[float] = []
    for number in re.findall(r"(\d+(?:\.\d+)?)\s*억(?:원|원)?", text):
        values.append(float(number) * 10000)
    for number in re.findall(r"(\d+(?:\.\d+)?)\s*천\s*만?원", text):
        values.append(float(number) * 1000)
    for number in re.findall(r"(?<!천)(\d[\d,]*(?:\.\d+)?)\s*만원", text):
        values.append(float(number.replace(",", "")))
    return values


def _income_limit_manwon(income_text: str, support: str) -> float | None:
    """연소득 상한(만원)을 뽑는다. 별도 소득란이 비어 있으면 지원내용을 보완 사용한다."""
    source = income_text
    if not source or source.strip() in {"무관", "제한없음"}:
        snippets = []
        for line in support.splitlines():
            markers = [pos for marker in ("연소득", "소득기준")
                       if (pos := line.find(marker)) >= 0]
            if markers:
                snippets.append(line[min(markers):])
        source = "\n".join(snippets)
    values = _korean_amounts_manwon(source)
    return max(values) if values else None


def _support_amount_manwon(support: str, product_kind: str) -> float:
    """지원액/대출한도만 정규화하고 보증금·주택가격·월 납입액은 제외한다."""
    if "대출" in product_kind:
        limit_lines = [line for line in support.splitlines() if "한도" in line and "대출" not in line[:2]]
        amounts = _korean_amounts_manwon("\n".join(limit_lines))
        if amounts:
            return max(amounts)
    if product_kind == "지원":
        matches = re.findall(
            r"(?:지원(?:금|액)?[^\n]{0,30})?최대\s*(\d[\d,]*(?:\.\d+)?)\s*만원",
            support,
        )
        if matches:
            return max(float(value.replace(",", "")) for value in matches)
    return 0.0


def _parse_date_value(text: str) -> str | None:
    match = re.search(r"(20\d{2})년\s*(\d{1,2})월\s*(\d{1,2})일", text)
    if not match:
        return None
    return date(*map(int, match.groups())).isoformat()


def _application_dates(text: str) -> tuple[str | None, str | None, int]:
    if "상시" in text:
        return None, None, 1
    matches = re.findall(r"(20\d{2})년\s*(\d{1,2})월\s*(\d{1,2})일", text)
    parsed = [date(*map(int, item)).isoformat() for item in matches]
    return (parsed[0] if parsed else None,
            parsed[-1] if len(parsed) > 1 else (parsed[0] if parsed else None), 0)


def _age_bounds(text: str) -> tuple[int | None, int | None]:
    ages = [int(x) for x in re.findall(r"만\s*(\d{1,2})세", text)]
    if not ages:
        return None, None
    return min(ages), max(ages)


def _classify(name: str, support: str, tags: str) -> tuple[str, str]:
    combined = f"{name} {support} {tags}"
    if "청약통장" in combined:
        return "청약·연계대출", "청약,대출"
    if "보증료" in combined:
        return "보증료지원", "지원"
    if "생활관" in combined or "기숙사" in combined:
        return "기숙사", "주거공급"
    if "행복주택" in combined:
        return "공공임대주택", "주거공급"
    if "임대주택" in combined or "주택 공급" in combined:
        return "공공주택", "주거공급"
    return "주거지원", "지원"


def _rate_bounds(support: str) -> tuple[float | None, float | None, float | None]:
    range_match = re.search(
        r"금리\s*[:：]?\s*(\d+(?:\.\d+)?)%\s*~\s*(\d+(?:\.\d+)?)%", support)
    interest_match = re.search(r"이자율\s*[:：]?\s*최대\s*(\d+(?:\.\d+)?)%", support)
    if range_match:
        low, high = map(float, range_match.groups())
        overall_high = max(high, float(interest_match.group(1))) if interest_match else high
        return low, overall_high, low
    if interest_match:
        value = float(interest_match.group(1))
        return value, value, value
    return None, None, None


def _description(lines: list[str], policy_line: int) -> str:
    for i in range(policy_line - 1, max(-1, policy_line - 30), -1):
        if lines[i].startswith("#"):
            values = []
            for j in range(i + 1, policy_line):
                value = lines[j].strip()
                if value and value != "한 눈에 보는 정책 요약" and not value.startswith("한 눈에 보는 정책 요약 -"):
                    values.append(value)
            if values:
                return " ".join(values)[:1000]
    return ""


def parse_policies(text: str) -> pd.DataFrame:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    id_lines = [i for i, line in enumerate(lines) if line.strip() == "정책번호"]
    records: dict[str, dict] = {}
    for position, idx in enumerate(id_lines):
        if idx + 1 >= len(lines):
            continue
        policy_id = lines[idx + 1].strip()
        if not re.fullmatch(r"\d{20}", policy_id):
            continue
        end = id_lines[position + 1] if position + 1 < len(id_lines) else len(lines)

        title, region_scope, modified = "", "", None
        tags = []
        for j in range(idx - 1, max(-1, idx - 35), -1):
            value = lines[j].strip()
            if not title and value.startswith("최종 수정일"):
                title = lines[j - 1].strip() if j else ""
                modified = _parse_date_value(value)
            if not region_scope and value in REGION_LABELS:
                region_scope = value
            if value.startswith("#"):
                tags.append(value)

        support = _field(lines, idx + 2, end, "지원내용")
        application_period = _field(lines, idx + 2, end, "사업 신청기간")
        app_start, app_end, always_open = _application_dates(application_period)
        age_text = _field(lines, idx + 2, end, "연령")
        age_min, age_max = _age_bounds(age_text)
        income_text = _field(lines, idx + 2, end, "소득")
        income_limit = _income_limit_manwon(income_text, support)
        category, product_kind = _classify(title, support, " ".join(tags))
        rate_min, rate_max, representative_rate = _rate_bounds(support)
        max_amount = _support_amount_manwon(support, product_kind)
        site = _field(lines, idx + 2, end, "신청 사이트")
        reference1 = _field(lines, idx + 2, end, "참고사이트 1")
        source_url = site or reference1
        regions = _field(lines, idx + 2, end, "거주지역")
        restriction = _field(lines, idx + 2, end, "참여제한 대상")
        extra = _field(lines, idx + 2, end, "추가사항")
        target_parts = [x for x in (age_text, income_text, restriction) if x and x != "제한없음"]

        record = {
            # 기존 검색/GUI 호환 컬럼
            "program_id": policy_id,
            "name": title,
            "category": category,
            "target": " / ".join(target_parts)[:2000] or "상세 자격조건 확인 필요",
            "max_amount_manwon": max_amount,
            "rate_pct": representative_rate,
            "income_limit_manwon": income_limit,
            "source_url": source_url,
            "desc": _description(lines, idx),
            # 상세 정책 컬럼
            "policy_area": _field(lines, idx + 2, end, "정책분야") or "주거",
            "product_kind": product_kind,
            "region_scope": region_scope,
            "eligible_regions": regions,
            "support_content": support,
            "operation_period": _field(lines, idx + 2, end, "사업 운영 기간"),
            "application_period": application_period,
            "application_start_date": app_start,
            "application_end_date": app_end,
            "always_open": always_open,
            "application_status": "상시" if always_open else "기간지정",
            "support_scale": _field(lines, idx + 2, end, "지원 규모(명)"),
            "age_text": age_text,
            "age_min": age_min,
            "age_max": age_max,
            "marital_status": _field(lines, idx + 2, end, "혼인여부"),
            "income_text": income_text,
            "education": _field(lines, idx + 2, end, "학력"),
            "employment_status": _field(lines, idx + 2, end, "취업상태"),
            "specialization": _field(lines, idx + 2, end, "특화분야"),
            "additional_eligibility": extra,
            "participation_restrictions": restriction,
            "application_procedure": _field(lines, idx + 2, end, "신청절차"),
            "screening_announcement": _field(lines, idx + 2, end, "심사 및 발표"),
            "application_site": site,
            "required_documents": _field(lines, idx + 2, end, "제출 서류"),
            "other_information": _field(lines, idx + 2, end, "기타 정보"),
            "supervising_organization": _field(lines, idx + 2, end, "주관 기관"),
            "operating_organization": _field(lines, idx + 2, end, "운영 기관"),
            "reference_url_1": reference1,
            "reference_url_2": _field(lines, idx + 2, end, "참고사이트 2"),
            "tags": ",".join(reversed(tags)),
            "last_modified_date": modified,
            "rate_min_pct": rate_min,
            "rate_max_pct": rate_max,
            "source_type": "user_attached_youth_policy_portal_text",
            "imported_at": datetime.now().isoformat(timespec="seconds"),
        }
        # 같은 정책번호가 반복되면 더 뒤의 최신 복사본으로 자연스럽게 교체한다.
        records[policy_id] = record
    return pd.DataFrame(records.values()).sort_values(
        ["last_modified_date", "program_id"], ascending=[False, True]).reset_index(drop=True)


def import_policies(source: Path) -> tuple[pd.DataFrame, Path, Path]:
    text = source.read_text(encoding="utf-8")
    frame = parse_policies(text)
    if frame.empty:
        raise ValueError("첨부 문서에서 정책번호를 찾지 못했습니다")

    out_dir = config.DATA_RAW / "finance_policies"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_copy = out_dir / "source_youth_policies.txt"
    csv_path = out_dir / "youth_housing_policies.csv"
    if source.resolve() != raw_copy.resolve():
        shutil.copy2(source, raw_copy)
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 정부 정책만 테이블을 replace하면 사용자 제공 은행상품이 유실되므로 모든
    # 정규화 금융 소스를 다시 결합한다.
    from src.db.build_db import build_finance_db
    with sqlite3.connect(config.DB_PATH) as con:
        build_finance_db(con)
    return frame, raw_copy, csv_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="청년정책 상세 페이지 pasted-text.txt")
    args = parser.parse_args()
    frame, raw_copy, csv_path = import_policies(args.source)
    print(f"[finance-policy] 고유 정책 {len(frame)}건 적재")
    print(f"[finance-policy] 원문: {raw_copy}")
    print(f"[finance-policy] CSV: {csv_path}")
    print(f"[finance-policy] DB: {config.DB_PATH}::finance_programs")
    print(frame[["program_id", "region_scope", "name", "category",
                 "application_period"]].to_string(index=False))


if __name__ == "__main__":
    main()
