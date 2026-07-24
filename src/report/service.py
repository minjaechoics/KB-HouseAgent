"""선택 매물에 대한 금융·자산·치안·생활·계약안전 리포트 조립."""
from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import pandas as pd

from src import config
from src.market_forecast import HousePriceForecaster
from src.report.budget import simulate
from src.report.lifestyle import estimate_monthly_lifestyle
from src.tools.convenience_tool import ConvenienceTool
from src.tools.finance_tool import FinanceTool
from src.tools.registry_tool import registry_check_guide
from src.tools.safety_tool import SafetyTool
from src.tools.naver_local_tool import NaverLocalSearchTool
from src.tools.map_tool import MapTool
from src.tools.ev_charger_tool import EVChargerTool
from src.market_data import RoneMarketTool
from src.fraud_risk.actual_model import (
    HF_PUBLISHED_COEFFICIENTS,
    build_actual_feature_frame,
)
from src.fraud_risk.infer import FraudRiskScorer


_RISK_FACTOR_LABELS = {
    "landlord_corporation": "법인 임대인",
    "landlord_multi_home": "다주택 임대인",
    "registered_rental_business": "등록 임대사업자",
    "tenant_youth": "청년 임차인",
    "house_officetel": "오피스텔",
    "house_row_multifamily": "연립·다세대·빌라",
    "house_detached_multihousehold": "단독·다가구",
    "debt_60_70": "부채비율 60~70%",
    "debt_70_80": "부채비율 70~80%",
    "debt_80_90": "부채비율 80~90%",
    "debt_90_plus": "부채비율 90% 이상",
    "log_deposit_won": "보증금 규모",
    "valuation_expert": "전문가 가격산정",
    "valuation_supplier": "공급자 가격산정",
    "monthly_rent_contract": "보증부 월세",
    "has_senior_claim": "선순위채권 존재",
    "has_jeonse_loan": "전세대출 이용",
    "loan_to_deposit": "전세대출금/보증금",
    "sale_price_index_decline": "매매가격지수 하락",
    "mortgage_rate_change_pctp": "주담대 금리 변화",
    "region_incheon": "인천 지역",
    "region_gyeonggi": "경기 지역",
}


def json_safe(value):
    """Replace non-finite numbers before the API response is serialized."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


class PropertyReportService:
    def __init__(self, db_path: Path = config.DB_PATH, llm=None, map_tool=None):
        self.db_path = db_path
        self.llm = llm
        self.finance = FinanceTool(db_path)
        self.local_search = NaverLocalSearchTool()
        self.convenience = ConvenienceTool(local_search=self.local_search)
        self.safety = SafetyTool(
            convenience_tool=self.convenience, local_search=self.local_search)
        self.forecaster = HousePriceForecaster(llm=llm)
        self.risk_scorer = FraudRiskScorer()
        self.map_tool = map_tool or MapTool()
        self.rone = RoneMarketTool(db_path)
        self.ev_chargers = EVChargerTool(db_path)

    def _risk_evidence(self, prop: dict) -> dict:
        """Build auditable, model-aligned facts before asking the LLM to explain."""
        if str(prop.get("transaction_type") or prop.get("lease_type") or "") != "전세":
            return {"available": False, "reason": "전세 매물만 분석합니다."}
        scored = self.risk_scorer.score(prop)
        stored = prop.get("fraud_score")
        score = float(stored) if stored is not None else float(scored["fraud_score"])
        threshold = float(scored["decision_threshold"])
        grade = "위험" if score >= threshold else (
            "주의" if score >= threshold / 2 else "낮음"
        )
        deposit = float(prop.get("deposit_manwon") or 0)
        market = float(prop.get("market_price_manwon") or 0)
        senior = float(prop.get("senior_deposit_sum_manwon") or 0) + float(
            prop.get("senior_mortgage_manwon") or 0
        )
        debt_pct = ((deposit + senior) / market * 100) if market > 0 else None
        drivers = []
        try:
            frame = build_actual_feature_frame(pd.DataFrame([prop])).iloc[0]
            for feature, coefficient in HF_PUBLISHED_COEFFICIENTS.items():
                value = float(frame.get(feature) or 0)
                if feature == "log_deposit_won" or abs(value) < 1e-12:
                    continue
                contribution = value * float(coefficient)
                drivers.append({
                    "feature": feature,
                    "label": _RISK_FACTOR_LABELS.get(feature, feature),
                    "direction": "risk" if contribution > 0 else "protective",
                    "value": round(value, 4),
                    "log_odds_contribution": round(contribution, 4),
                    "detail": (
                        "보증가입 심사를 통과한 외부 표본의 선택효과가 있어 안전요인으로 단정할 수 없습니다."
                        if feature == "has_senior_claim" else
                        "외부 HF 연구 공개계수에서 점수를 높이는 방향입니다."
                        if contribution > 0 else
                        "외부 HF 연구 공개계수에서 점수를 낮추는 방향입니다."
                    ),
                })
            drivers.sort(
                key=lambda item: abs(float(item["log_odds_contribution"])), reverse=True
            )
        except Exception:
            drivers = []
        if deposit > 0:
            drivers.insert(0, {
                "feature": "log_deposit_won", "label": "보증금 규모",
                "direction": "risk", "value_manwon": round(deposit, 1),
                "detail": "보증금의 로그 규모가 공개모형에 포함됩니다. 절편과 다른 요인을 함께 적용하므로 금액만으로 위험을 단정하지 않습니다.",
            })
        return {
            "available": True,
            "score": round(score, 6), "grade": grade,
            "decision_threshold": round(threshold, 6),
            "method": scored.get("method"),
            "label_source_status": scored.get("label_source_status"),
            "property_facts": {
                "region": f"{prop.get('sido') or ''} {prop.get('gugun') or ''}".strip(),
                "house_type": prop.get("house_type") or prop.get("property_type"),
                "deposit_manwon": round(deposit, 1),
                "market_price_manwon": round(market, 1) if market > 0 else None,
                "senior_claim_manwon": round(senior, 1),
                "hf_debt_ratio_pct": round(debt_pct, 1) if debt_pct is not None else None,
                "synthetic_listing": bool(prop.get("is_synthetic")),
            },
            "model_drivers": drivers[:6],
            "not_confirmed_in_model_input": [
                "임대인 다주택 여부", "실제 임차인의 전세대출", "계약기간 가격지수 변화",
                "계약기간 주담대 금리 변화",
            ],
            "separate_contract_checks": {
                "guarantee_eligible": prop.get("guarantee_eligible"),
                "trust_registration": prop.get("trust_registration"),
                "seizure_or_provisional_seizure": prop.get(
                    "seizure_or_provisional_seizure"
                ),
                "tax_arrears_checked": prop.get("tax_arrears_checked"),
            },
        }

    def _risk_explanation(self, prop: dict) -> dict:
        evidence = self._risk_evidence(prop)
        if not evidence.get("available"):
            return {"available": False, "strategy": "not_applicable",
                    "summary": evidence.get("reason")}
        grade = str(evidence.get("grade") or "확인 필요")
        factors = [{
            "tone": item.get("direction", "neutral"),
            "label": item.get("label", "모델 요인"),
            "detail": item.get("detail", ""),
        } for item in evidence.get("model_drivers", [])[:4]]
        fallback = {
            "available": True, "strategy": "deterministic_fallback",
            "headline": f"모델 기준 {grade} 구간입니다",
            "summary": (
                "HF 실제 보증사고 연구의 공개계수를 이 매물 조건에 적용한 참고 점수입니다. "
                "아래 요인은 점수의 방향을 설명하지만 인과관계나 계약 안전을 확정하지 않습니다."
            ),
            "factors": factors,
            "next_checks": ["HUG·HF 보증 가입 가능 여부 확인", "등기부의 신탁·압류·근저당 확인",
                            "임대인 납세증명서와 선순위보증금 확인"],
            "limitations": "미확인 입력은 0 또는 미상으로 처리되며 합성 매물은 실제 임대인 확인값이 아닙니다.",
            "model_evidence": evidence,
        }
        if not self.llm or not getattr(self.llm, "supports_agentic_calls", False):
            return fallback
        schema = {
            "type": "object", "additionalProperties": False,
            "properties": {
                "headline": {"type": "string"},
                "summary": {"type": "string"},
                "factors": {"type": "array", "maxItems": 4, "items": {
                    "type": "object", "additionalProperties": False,
                    "properties": {
                        "tone": {"type": "string", "enum": ["risk", "protective", "neutral"]},
                        "label": {"type": "string"},
                        "detail": {"type": "string"},
                    }, "required": ["tone", "label", "detail"],
                }},
                "next_checks": {"type": "array", "maxItems": 3,
                                "items": {"type": "string"}},
                "limitations": {"type": "string"},
            },
            "required": ["headline", "summary", "factors", "next_checks", "limitations"],
        }
        try:
            value = self.llm.analyze_json(
                operation="report.risk_explanation",
                system=(
                    "너는 전세 보증사고 추정 위험도 설명기다. 제공된 JSON 사실만 사용한다. "
                    "점수에 실제 반영된 model_drivers와 별도 계약확인 항목을 혼동하지 않는다. "
                    "fraud_score를 집주인이 사기범일 확률이나 손실 확정확률로 표현하지 않는다. "
                    "양의 계수는 모델상 점수를 높이는 연관 방향일 뿐 인과라고 말하지 않는다. "
                    "미확인 값을 안전하다고 해석하지 말고, 사용자가 먼저 확인할 행동을 짧게 쓴다."
                ),
                user=json.dumps(evidence, ensure_ascii=False),
                schema=schema, schema_name="housing_risk_explanation", max_tokens=900,
            )
            if not value:
                return fallback
            value["headline"] = str(value.get("headline") or "위험도 판단 근거")[:80]
            value["summary"] = str(value.get("summary") or "")[:600]
            value["limitations"] = str(value.get("limitations") or "")[:350]
            value["factors"] = [{
                "tone": item.get("tone") if item.get("tone") in
                        {"risk", "protective", "neutral"} else "neutral",
                "label": str(item.get("label") or "모델 요인")[:40],
                "detail": str(item.get("detail") or "")[:220],
            } for item in (value.get("factors") or [])[:4]]
            value["next_checks"] = [
                str(item)[:120] for item in (value.get("next_checks") or [])[:3]
            ]
            return {"available": True, "strategy": "llm_structured",
                    **value, "model_evidence": evidence}
        except Exception:
            return fallback

    def _emphasis(self, prop: dict, budget: dict, forecast: dict,
                  safety: dict, convenience: dict) -> dict:
        facts = {
            "transaction_type": prop.get("transaction_type"),
            "funding_gap_manwon": budget.get("funding", {}).get("funding_gap_manwon"),
            "monthly_housing_manwon": budget.get("cashflow", {}).get("monthly_housing_total_manwon"),
            "affordable": budget.get("funding", {}).get("affordable_under_guardrail"),
            "forecast_annual": forecast.get("annual_growth_rate"),
            "forecast_warning": forecast.get("warning"),
            "safety_grade": safety.get("grade"),
            "convenience_grade": convenience.get("grade"),
        }
        fallback = {
            "strategy": "deterministic",
            "items": [
                {"tone": "warning", "title": "자금 부족액",
                 "message": f"{facts['funding_gap_manwon'] or 0:,.0f}만원"},
                {"tone": "accent", "title": "월 주거비",
                 "message": f"{facts['monthly_housing_manwon'] or 0:,.1f}만원"},
            ],
        }
        if not self.llm or not getattr(self.llm, "supports_agentic_calls", False):
            return fallback
        schema = {
            "type": "object", "additionalProperties": False,
            "properties": {
                "items": {"type": "array", "maxItems": 4, "items": {
                    "type": "object", "additionalProperties": False,
                    "properties": {
                        "tone": {"type": "string", "enum": ["warning", "positive", "accent"]},
                        "title": {"type": "string"},
                        "message": {"type": "string"},
                    },
                    "required": ["tone", "title", "message"],
                }},
            },
            "required": ["items"],
        }
        try:
            value = self.llm.analyze_json(
                operation="report.emphasis",
                system=(
                    "이미 생성된 주택 의사결정 리포트의 핵심 강조 문구를 고른다. "
                    "제공된 사실만 사용하고 숫자를 새로 만들지 않는다. 위험·자금부족은 warning, "
                    "유리한 사실은 positive, 중립 핵심은 accent로 분류한다. 짧은 한국어로 쓴다."
                ),
                user=str(facts), schema=schema, schema_name="report_emphasis",
                max_tokens=500,
            )
            return {"strategy": "llm_post_generation", **(value or fallback)}
        except Exception:
            return fallback

    def _final_assessment(self, prop: dict, budget: dict, forecast: dict,
                          safety: dict, convenience: dict,
                          contract_safety: dict) -> dict:
        """Grounded cross-tab decision synthesis for the final product tab."""
        funding = budget.get("funding") or {}
        news = forecast.get("news") or {}
        score = 50
        score += 18 if funding.get("simulation_valid") else -28
        score += 8 if (safety.get("safety_score") or 0) >= 60 else 0
        score += 6 if (convenience.get("convenience_score") or 0) >= 50 else 0
        score += 8 if float(forecast.get("annual_growth_rate") or 0) > 0 else -5
        if prop.get("fraud_score") is not None:
            score -= min(25, round(float(prop.get("fraud_score") or 0) * 35))
        score = max(0, min(100, score))
        recommendation = ("recommend" if score >= 70 else
                          "conditional" if score >= 45 else "avoid")
        facts = {
            "transaction_type": prop.get("transaction_type"),
            "funding": funding,
            "market": {
                "annual_growth_rate": forecast.get("annual_growth_rate"),
                "price_history_available": (forecast.get("price_history") or {}).get("available"),
                "news_label": (forecast.get("market_assessment") or {}).get("label"),
                "relevant_news_count": news.get("relevant_count"),
            },
            "safety": {"score": safety.get("safety_score"), "grade": safety.get("grade")},
            "convenience": {"score": convenience.get("convenience_score"),
                            "grade": convenience.get("grade")},
            "contract": {
                "fraud_score": prop.get("fraud_score"),
                "guarantee_candidates": contract_safety.get("guarantee_candidates"),
                "trust_registration": prop.get("trust_registration"),
                "seizure": prop.get("seizure_or_provisional_seizure"),
            },
            "deterministic_score": score,
        }
        fallback = {
            "strategy": "deterministic_fallback", "score": score,
            "recommendation": recommendation,
            "headline": ("계약 전 조건 확인 후 고려할 수 있어요" if recommendation == "conditional"
                         else "현재 조건에서 검토 가치가 있어요" if recommendation == "recommend"
                         else "현재 조건에서는 다른 집도 함께 비교하세요"),
            "summary": str(funding.get("verdict_message") or "자금조달과 계약안전을 함께 확인하세요."),
            "reasons": [
                f"자금조달: {funding.get('verdict_title') or '확인 필요'}",
                f"시장: 연 전망 {float(forecast.get('annual_growth_rate') or 0) * 100:.1f}%",
                f"치안·편의: {safety.get('grade') or '미확인'} / {convenience.get('grade') or '미확인'}",
            ],
            "actions": ["등기부·보증 가입 가능 여부 확인", "은행의 실제 승인 한도 확인"],
            "component_scores": {
                "funding": 80 if funding.get("simulation_valid") else 20,
                "market": max(0, min(100, 50 + float(forecast.get("annual_growth_rate") or 0) * 500)),
                "safety": safety.get("safety_score"),
                "convenience": convenience.get("convenience_score"),
                "contract": max(0, 100 - float(prop.get("fraud_score") or 0) * 100),
            },
        }
        if not self.llm or not getattr(self.llm, "supports_agentic_calls", False):
            return fallback
        schema = {
            "type": "object", "additionalProperties": False,
            "properties": {
                "recommendation": {"type": "string", "enum": ["recommend", "conditional", "avoid"]},
                "headline": {"type": "string"}, "summary": {"type": "string"},
                "reasons": {"type": "array", "maxItems": 4, "items": {"type": "string"}},
                "actions": {"type": "array", "maxItems": 3, "items": {"type": "string"}},
            },
            "required": ["recommendation", "headline", "summary", "reasons", "actions"],
        }
        try:
            value = self.llm.analyze_json(
                operation="report.final_assessment",
                system=(
                    "너는 청년 주택 의사결정의 최종 검토자다. 제공된 JSON 사실만 사용한다. "
                    "자금조달이 불가능하면 추천하지 않는다. 예비 금융자격은 승인으로 단정하지 않는다. "
                    "시장 전망보다 계약 안전과 실제 자금조달을 우선한다. 숫자를 만들지 말고 "
                    "핵심 결론과 계약 전에 할 행동을 짧고 명확한 한국어로 쓴다."
                ), user=json.dumps(facts, ensure_ascii=False), schema=schema,
                schema_name="property_final_assessment", max_tokens=750)
            if not value:
                return fallback
            return {**fallback, **value, "strategy": "llm_structured", "score": score}
        except Exception:
            return fallback

    def property(self, property_id: str) -> dict | None:
        uri = self.db_path.resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        try:
            row = connection.execute(
                "SELECT * FROM properties WHERE property_id=?", (property_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            connection.close()

    @staticmethod
    def _public_property(prop: dict) -> dict:
        keys = (
            "property_id", "is_synthetic", "synthetic_notice", "sido", "gugun", "dong",
            "source_type", "source_provider", "source_url", "source_captured_at",
            "source_expires_at", "source_authorized", "last_verified_at",
            "road_address", "jibun_address", "lat", "lng", "transaction_type",
            "house_type", "property_type", "building_name", "sale_price_manwon",
            "asking_price_manwon", "deposit_manwon", "monthly_rent_manwon",
            "maintenance_fee_manwon", "area_m2", "room_count", "bathroom_count",
            "current_floor", "total_floors", "build_year", "direction", "parking_total",
            "elevator_count", "available_from_date", "fraud_score", "guarantee_eligible",
            "trust_registration", "seizure_or_provisional_seizure", "tax_arrears_checked",
            "tax_arrears_present", "landlord_information_presented", "mortgage_ltv_pct",
            "jeonse_ratio_pct", "mortgage_max_claim_manwon", "senior_rights_total_manwon",
            "registry_checked_at", "advertisement_title", "broker_office_name", "broker_phone",
        )
        return {key: prop.get(key) for key in keys}

    def build(self, user: dict, property_id: str, assumptions: dict | None = None) -> dict:
        prop = self.property(property_id)
        if prop is None:
            raise KeyError(property_id)
        lat, lng = float(prop["lat"]), float(prop["lng"])
        region = f"{prop.get('sido', '')} {prop.get('gugun', '')}".strip()
        programs = self.finance.search(
            user_income_manwon=user.get("monthly_income_manwon"),
            user_age=user.get("age"), region=region, finance_mode="eligibility", limit=50,
            user_profile=user,
        )
        regional_market = self.rone.market(prop)
        forecast = self.forecaster.forecast(prop, regional_market=regional_market)
        lifestyle_inputs = dict((assumptions or {}).get("lifestyle_inputs") or {})
        if not lifestyle_inputs.get("destinations") and user.get("workplace_or_school"):
            lifestyle_inputs["destinations"] = [{
                "category": "work", "label": "직장·학교",
                "query": user.get("workplace_or_school"), "visits_per_month": 20,
            }]
        lifestyle = estimate_monthly_lifestyle(
            user, prop, lifestyle_inputs, self.map_tool)
        assumptions = {**(assumptions or {}), "lifestyle": lifestyle}
        budget = simulate(user, prop, forecast, programs, assumptions)
        context = " ".join(str(prop.get(key) or "") for key in
                           ("sido", "gugun", "dong")).strip()
        # 선택 매물 리포트 하나에서 NAVER 장소검색은 최대 5회만 호출한다.
        self.local_search.begin_request(max_calls=5)
        safety = self.safety.assess(
            lat, lng, radius_m=300, context=context,
            exclude_cctv_anchor=bool(
                prop.get("is_synthetic")
                and "CCTV" in str(prop.get("region_coordinate_source") or "")
            ),
        )
        convenience = self.convenience.assess(
            lat, lng, radius_m=500, context=context)
        vehicle_powertrain = str(
            ((assumptions or {}).get("lifestyle_inputs") or {}).get("vehicle_powertrain")
            or "gasoline"
        )
        ev_chargers = (
            self.ev_chargers.nearby(lat, lng, radius_m=1500, limit=20)
            if vehicle_powertrain == "ev"
            else {"requested": False, "available": False, "stations": []}
        )
        is_synthetic = bool(prop.get("is_synthetic"))
        tax_status = "unverified"
        if not is_synthetic and prop.get("tax_arrears_checked"):
            tax_status = "arrears_present" if prop.get("tax_arrears_present") else "verified_clear"
        risk_explanation = self._risk_explanation(prop)
        is_lease = str(prop.get("transaction_type") or "") in {"전세", "월세"}
        deposit = float(prop.get("deposit_manwon") or 0)
        capital_limit = 70000 if str(prop.get("sido") or "") in {
            "서울", "서울특별시", "경기", "경기도", "인천", "인천광역시"
        } else 50000
        guarantee_candidates = []
        if is_lease:
            guarantee_candidates = [{
                "name": "HUG 전세보증금반환보증",
                "precheck": "금액기준 예비 적합" if deposit <= capital_limit else "보증금 한도 초과 가능",
                "detail": (f"이 지역 공식 보증금 상한 {capital_limit:,.0f}만원과 비교했습니다. "
                           "권리침해·주택가액·계약기간 등 추가 심사가 필요합니다."),
                "source_url": "https://www.khug.or.kr/hug/web/ig/dr/igdr000001.jsp",
            }, {
                "name": "HF 전세지킴보증", "precheck": "추가 심사 필요",
                "detail": "압류·가압류·채권양도 등 권리침해 여부와 HF 세부요건을 확인해야 합니다.",
                "source_url": "https://hf.go.kr/ko/sub02/sub02_05_06.do",
            }]
        contract_safety = {
            "fraud_score": prop.get("fraud_score"),
            "risk_explanation": risk_explanation,
            "guarantee_eligible": prop.get("guarantee_eligible"),
            "guarantee_candidates": guarantee_candidates,
            "trust_registration": prop.get("trust_registration"),
            "seizure_or_provisional_seizure": prop.get("seizure_or_provisional_seizure"),
            "tax_arrears": {
                "status": tax_status,
                "synthetic_field_ignored": is_synthetic,
                "message": (
                    "합성 매물의 체납 필드는 실제 임대인 확인값이 아니므로 위험도에 반영하지 않습니다."
                    if is_synthetic else
                    "임대인 동의 하의 납세증명서·공식 열람 결과만 반영합니다."
                ),
                "required_evidence": "임대인 납세증명서 또는 법정 절차에 따른 미납국세 열람",
            },
            "registry_guide": registry_check_guide(prop.get("road_address") or ""),
        }
        restricted_checks = {
            "sex_offender": {
                "status": "manual_identity_verification_required",
                "count": None,
                "official_url": "https://www.sexoffender.go.kr/",
                "message": (
                    "성범죄자 알림e는 실명인증 후 보호 목적으로만 열람해야 하므로 "
                    "주소·인원 정보를 자동 수집하거나 재게시하지 않습니다. 공식 앱에서 직접 확인하세요."
                ),
            },
            "landlord_tax_delinquency": contract_safety["tax_arrears"],
        }
        result = {
            "property": self._public_property(prop), "user_snapshot": {
                key: user.get(key) for key in (
                    "age", "monthly_income_manwon", "total_asset_manwon",
                    "monthly_living_cost_manwon", "employment_type", "employment_months",
                    "household_role", "home_ownership_count", "marital_status",
                    "spouse_annual_income_manwon", "minor_children_count",
                    "children_plans", "expected_inheritance_manwon",
                    "expected_inheritance_age", "workplace_or_school",
                    "is_korean_national",
                    "has_income_proof", "contract_deposit_paid_5pct",
                )
            },
            "forecast": forecast, "budget": budget,
            "regional_market": regional_market, "ev_chargers": ev_chargers,
            "finance_programs": programs, "safety": safety,
            "convenience": convenience, "contract_safety": contract_safety,
            "restricted_checks": restricted_checks,
            "provenance": {
                "price": (
                    "국토교통부 실거래 월간 집계 + GBDT 분위수 시나리오. "
                    "KB 시세는 별도 이용허락/제휴 피드가 설정된 경우에만 참조"
                ),
                "news": "NAVER 뉴스 검색 API(우선) 또는 공개 뉴스 RSS + LLM 가격영향 관련성·방향 판정",
                "safety": safety.get("sources"), "convenience": convenience.get("sources"),
                "naver_local_calls": self.local_search.calls,
                "naver_local_call_limit": self.local_search.max_calls,
                "privacy_policy": "개인식별 민감정보 자동수집·재게시 금지",
                "regional_market": "한국부동산원 R-ONE Open API",
                "ev_chargers": "V-World 활용모델 + 공공데이터포털 EvInfoServiceV2",
            },
        }
        result["ai_emphasis"] = self._emphasis(
            prop, budget, forecast, safety, convenience)
        result["final_assessment"] = self._final_assessment(
            prop, budget, forecast, safety, convenience, contract_safety)
        return json_safe(result)
