"""공식 공개 기준에 근거한 전세보증·최우선변제 사전검토."""
from __future__ import annotations

from datetime import date
from pathlib import Path
import sqlite3

from src import config
from src.db.build_db import build_guarantee_db


CAPITAL_REGIONS = {
    "서울", "서울특별시", "경기", "경기도", "인천", "인천광역시"
}


class GuaranteeProductTool:
    """가입 확정이 아닌 설명 가능한 사전검토 결과를 반환한다."""

    def __init__(self, db_path: Path = config.DB_PATH):
        self.db_path = db_path

    def _rows(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='guarantee_products'"
            ).fetchone()
            if not exists:
                build_guarantee_db(conn)
            return [
                dict(row) for row in conn.execute(
                    "SELECT * FROM guarantee_products ORDER BY product_id"
                )
            ]

    @staticmethod
    def _is_lease(prop: dict) -> bool:
        return str(
            prop.get("transaction_type") or prop.get("lease_type") or ""
        ) in {"전세", "월세"}

    def evaluate(
        self, prop: dict, user: dict, jeonse_ratio: dict | None = None
    ) -> dict:
        if not self._is_lease(prop):
            return {
                "available": False,
                "status": "not_applicable",
                "products": [],
            }
        deposit = max(0.0, float(prop.get("deposit_manwon") or 0))
        capital = str(prop.get("sido") or "") in CAPITAL_REGIONS
        ratio = jeonse_ratio or {}
        post = ((ratio.get("ratios") or {}).get("post_contract_ratio") or {})
        conservative = (
            (ratio.get("ratios") or {})
            .get("conservative_post_contract_ratio") or {}
        )
        p50 = post.get("p50")
        conservative_p90 = conservative.get("p90")
        results = []
        for row in self._rows():
            reasons: list[dict] = []
            status = "needs_official_review"
            limit = (
                row["capital_deposit_limit_manwon"]
                if capital else row["noncapital_deposit_limit_manwon"]
            )
            if limit is not None:
                passed = deposit <= float(limit)
                reasons.append({
                    "criterion": "보증금 지역 상한",
                    "passed": passed,
                    "observed": deposit,
                    "required": f"{float(limit):,.0f}만원 이하",
                })
                if not passed:
                    status = "basic_limit_failed"
            if row["value_limit_ratio"] is not None:
                value_passed = (
                    None if conservative_p90 is None
                    else float(conservative_p90) <= float(row["value_limit_ratio"])
                )
                reasons.append({
                    "criterion": "주택가액 대비 선순위채권·보증금",
                    "passed": value_passed,
                    "observed": conservative_p90,
                    "required": (
                        f"보수적 추정 {float(row['value_limit_ratio']) * 100:.0f}% 이하"
                    ),
                })
                if value_passed is False:
                    status = "basic_limit_failed"
            if row["requires_linked_loan_guarantee"]:
                linked = user.get("has_hf_jeonse_loan_guarantee")
                reasons.append({
                    "criterion": "HF 전세자금보증 연계",
                    "passed": linked,
                    "observed": linked,
                    "required": "동일 은행에서 HF 전세자금보증 이용",
                })
                if linked is False:
                    status = "basic_limit_failed"
            prelim_pass = (
                status != "basic_limit_failed"
                and all(item["passed"] is not False for item in reasons)
            )
            if prelim_pass and reasons and all(
                item["passed"] is True for item in reasons
            ):
                status = "precheck_passed_official_review_required"
            precheck_label = {
                "precheck_passed_official_review_required": "공개조건 예비 통과",
                "basic_limit_failed": "공개조건 미충족",
                "needs_official_review": "공식 심사 필요",
            }.get(status, "추가 확인")
            results.append({
                **row,
                "precheck_status": status,
                "precheck": precheck_label,
                "precheck_passed": prelim_pass,
                "estimated_covered_deposit_manwon": (
                    deposit if prelim_pass else 0.0
                ),
                "conditions": reasons,
                "detail": " · ".join(
                    f"{'충족' if item['passed'] is True else '미충족' if item['passed'] is False else '확인'} "
                    f"{item['criterion']}"
                    for item in reasons
                ) or "공식 상품 세부요건을 직접 확인해야 합니다.",
                "ratio_p50": p50,
                "conservative_ratio_p90": conservative_p90,
                "disclaimer": (
                    "사전검토이며 가입 확정이 아닙니다. 등기·권리침해·주택가격·"
                    "계약기간·신청시기와 기관 심사를 추가로 확인해야 합니다."
                ),
            })
        return {
            "available": True,
            "status": "precheck",
            "products": results,
            "uninsured_exposure_manwon": deposit,
            "insured_scenario_note": (
                "선택 상품의 공식 심사를 통과한다고 가정할 때 보증금 반환 위험의 "
                "일부 또는 전부를 보증기관에 이전하는 비교 시나리오입니다."
            ),
            "preferential_repayment": self.preferential_repayment(prop, user),
        }

    @staticmethod
    def preferential_repayment(prop: dict, user: dict) -> dict:
        """수원 현행 기준을 표시하되 법적 확정 표현을 금지한다."""
        region = f"{prop.get('sido') or ''} {prop.get('gugun') or ''}"
        deposit = max(0.0, float(prop.get("deposit_manwon") or 0))
        if "수원" not in region:
            return {
                "status": "region_rule_not_implemented",
                "legally_guaranteed": False,
            }
        small_tenant_limit = 14500.0
        statutory_ceiling = 4800.0
        mortgage_date = user.get("senior_mortgage_established_date")
        current_rule_applicable = None
        if mortgage_date:
            try:
                current_rule_applicable = (
                    date.fromisoformat(str(mortgage_date))
                    >= date(2023, 2, 21)
                )
            except ValueError:
                current_rule_applicable = None
        possession = user.get("move_in_registration_possible")
        eligible_amount = min(deposit, statutory_ceiling)
        return {
            "status": (
                "current_rule_precheck"
                if current_rule_applicable is True
                else "historical_lien_date_required"
            ),
            "region": "수원시(과밀억제권역)",
            "small_tenant_deposit_limit_manwon": small_tenant_limit,
            "statutory_ceiling_if_all_conditions_met_manwon": eligible_amount,
            "deposit_limit_passed": deposit <= small_tenant_limit,
            "move_in_and_registration_possible": possession,
            "senior_mortgage_established_date": mortgage_date,
            "current_2023_rule_applicable": current_rule_applicable,
            "legally_guaranteed": False,
            "warnings": [
                "근저당권 등 담보물권 취득일에 따라 적용 기준이 달라집니다.",
                "경매개시결정 등기 전에 주택 인도와 주민등록을 갖춰야 합니다.",
                "다가구 전체 소액임차인 최우선변제 합계는 주택가액의 1/2 한도에서 "
                "안분될 수 있어 4,800만원이 무조건 보장되는 것은 아닙니다.",
            ],
            "source_url": (
                "https://www.law.go.kr/LSW/lsInfoP.do?"
                "ancYnChk=0&chrClsCd=010202&efYd=20260102&"
                "joNo=000900&lsiSeq=280995&urlMode=lsInfoP"
            ),
        }
