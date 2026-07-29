"""선택 매물에 대한 금융·자산·치안·생활·계약안전 리포트 조립."""
from __future__ import annotations

import json
import math
import os
import secrets
import sqlite3
import copy
import threading
import time
from datetime import date
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from src import config
from src.audit import DecisionAuditStore
from src.market_forecast import HousePriceForecaster
from src.simulation import simulate_probabilistic
from src.report.budget import simulate
from src.report.lifestyle import estimate_monthly_lifestyle
from src.tools.convenience_tool import ConvenienceTool
from src.tools.finance_tool import FinanceTool
from src.tools.registry_tool import registry_check_guide
from src.tools.safety_tool import SafetyTool
from src.tools.naver_local_tool import NaverLocalSearchTool
from src.tools.map_tool import MapTool
from src.tools.ev_charger_tool import EVChargerTool
from src.tools.guarantee_tool import GuaranteeProductTool
from src.market_data import RoneMarketTool
from src.fraud_risk.actual_model import (
    HF_PUBLISHED_COEFFICIENTS,
    build_actual_feature_frame,
)
from src.fraud_risk.infer import FraudRiskScorer
from src.owner_asset_ratio import OwnerAssetRatioIntegrationService
from src.senior_deposit import SeniorDepositIntegrationService
from src.jeonse_ratio import JeonseRatioIntegrationService


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
        self.local_search = NaverLocalSearchTool(
            timeout=max(
                0.5,
                float(os.environ.get(
                    "REPORT_LOCAL_SEARCH_TIMEOUT_SECONDS", "1.5"
                )),
            )
        )
        self.convenience = ConvenienceTool(local_search=self.local_search)
        self.safety = SafetyTool(
            convenience_tool=self.convenience, local_search=self.local_search)
        self.forecaster = HousePriceForecaster(llm=llm)
        self.risk_scorer = FraudRiskScorer()
        self.map_tool = map_tool or MapTool()
        self.rone = RoneMarketTool(db_path)
        self.ev_chargers = EVChargerTool(db_path)
        self.audit = DecisionAuditStore(db_path)
        self.senior_deposit = SeniorDepositIntegrationService()
        self.owner_asset_ratio = OwnerAssetRatioIntegrationService()
        self.jeonse_ratio = JeonseRatioIntegrationService()
        self.guarantees = GuaranteeProductTool(db_path)
        self._analysis_cache: dict[str, tuple[float, dict]] = {}
        self._analysis_cache_lock = threading.Lock()
        self._analysis_cache_ttl_seconds = int(
            os.environ.get("REPORT_ANALYSIS_CACHE_TTL_SECONDS", "3600")
        )
        self._risk_explanation_cache: dict[str, tuple[float, dict]] = {}
        self._metric_explanation_cache: dict[str, tuple[float, dict]] = {}

    def _base_analysis(self, prop: dict) -> tuple[dict, bool]:
        """Slow property-only lookups run concurrently and are cached."""
        cache_key = str(prop.get("property_id") or "")
        now = time.monotonic()
        with self._analysis_cache_lock:
            cached = self._analysis_cache.get(cache_key)
            if cached and now - cached[0] < self._analysis_cache_ttl_seconds:
                return copy.deepcopy(cached[1]), True

        def market_job() -> tuple[dict, dict, int]:
            started = time.perf_counter()
            regional = self.rone.market(prop)
            # 뉴스 LLM은 위험도 설명 LLM과 동시에 호출될 때 응답 지연을
            # 크게 키웠다. 상세 화면의 첫 응답은 근거 기반 키워드 판정을
            # 사용하고, 계약 위험 설명에만 LLM 호출을 남긴다.
            forecast = self.forecaster.forecast(
                prop, regional_market=regional, use_news_llm=False
            )
            return regional, forecast, round((time.perf_counter() - started) * 1000)

        def local_job() -> tuple[dict, dict, int, int, int]:
            started = time.perf_counter()
            # 서비스 수명 동안 같은 클라이언트/캐시를 재사용한다. 공공 CSV를
            # 우선 사용하고 외부 지역검색은 상세 리포트당 최대 2회만 허용한다.
            local_search = self.local_search
            convenience_tool = self.convenience
            safety_tool = self.safety
            local_search.begin_request(
                max_calls=int(os.environ.get(
                    "REPORT_LOCAL_SEARCH_MAX_CALLS", "2"
                ))
            )
            lat, lng = float(prop["lat"]), float(prop["lng"])
            context = " ".join(
                str(prop.get(key) or "") for key in ("sido", "gugun", "dong")
            ).strip()
            safety = safety_tool.assess(
                lat, lng, radius_m=300, context=context,
                exclude_cctv_anchor=bool(
                    prop.get("is_synthetic")
                    and "CCTV" in str(prop.get("region_coordinate_source") or "")
                ),
            )
            convenience = convenience_tool.assess(
                lat, lng, radius_m=500, context=context
            )
            return (
                safety, convenience, local_search.calls, local_search.max_calls,
                round((time.perf_counter() - started) * 1000),
            )

        def risk_job() -> tuple[dict, int]:
            started = time.perf_counter()
            # 다가구 전세는 아래의 통합 전세가율 계산 후 최신 근거로 다시
            # 설명하므로 이 단계에서는 중복 LLM 호출을 하지 않는다.
            value = self._risk_explanation(prop, use_llm=False)
            return value, round((time.perf_counter() - started) * 1000)

        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="property-report") as pool:
            market_future = pool.submit(market_job)
            local_future = pool.submit(local_job)
            risk_future = pool.submit(risk_job)
            regional_market, forecast, market_ms = market_future.result()
            safety, convenience, local_calls, local_limit, local_ms = local_future.result()
            risk_explanation, risk_ms = risk_future.result()
        value = {
            "regional_market": regional_market,
            "forecast": forecast,
            "safety": safety,
            "convenience": convenience,
            "risk_explanation": risk_explanation,
            "naver_local_calls": local_calls,
            "naver_local_call_limit": local_limit,
            "stage_elapsed_ms": {
                "market_and_news": market_ms,
                "local_places": local_ms,
                "risk_explanation": risk_ms,
            },
        }
        with self._analysis_cache_lock:
            self._analysis_cache[cache_key] = (now, copy.deepcopy(value))
            if len(self._analysis_cache) > 256:
                oldest = min(self._analysis_cache, key=lambda key: self._analysis_cache[key][0])
                self._analysis_cache.pop(oldest, None)
        return value, False

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

    @staticmethod
    def _owner_ratio_risk_evidence(
        prop: dict, owner_asset_ratio: dict, senior_deposit: dict
    ) -> dict:
        """Use one explicit formula for multifamily-jeonse contract risk."""
        estimate = owner_asset_ratio.get("estimate") or {}
        support = owner_asset_ratio.get("decision_support") or {}
        ratio = estimate.get("target_deposit_to_total_assets_ratio") or {}
        owner_assets = estimate.get("estimated_owner_total_assets") or {}
        senior_estimate = senior_deposit.get("estimate") or {}
        senior = senior_estimate.get("estimated_senior_deposit") or {}
        score = float(support.get("risk_score") or ratio.get("p50") or 0)
        grade = str(support.get("risk_grade") or (
            "위험" if score >= .8 else "주의" if score >= .6 else "낮음"
        ))
        target = float(prop.get("deposit_manwon") or 0)
        owner_p50 = float(owner_assets.get("p50") or 0)
        drivers = [{
            "feature": "target_deposit",
            "label": "선택 전세보증금",
            "direction": "risk",
            "value_manwon": round(target, 1),
            "detail": "위험비율 수식의 분자입니다.",
        }, {
            "feature": "estimated_owner_total_assets",
            "label": "추정 집주인 총자산",
            "direction": "protective",
            "value_manwon": round(owner_p50, 1),
            "detail": (
                "건물가치와 가계금융복지조사의 임대인 통계 prior를 "
                "결합한 분모이며 실제 임대인 조회값은 아닙니다."
            ),
        }, {
            "feature": "ratio_uncertainty",
            "label": "비율 불확실성",
            "direction": "neutral",
            "value": round(float(
                support.get("probability_ratio_over_0_8") or 0
            ), 4),
            "detail": "자산비율이 80%를 넘는 모의경로의 비중입니다.",
        }]
        if senior_deposit.get("available"):
            drivers.append({
                "feature": "senior_deposit_separate",
                "label": "기존 선순위 보증금",
                "direction": "neutral",
                "value_manwon": round(float(senior.get("p50") or 0) / 10_000, 1),
                "detail": (
                    "별도 계약 확인지표입니다. 사용자의 요청에 따라 "
                    "주 위험비율의 분자에는 합산하지 않았습니다."
                ),
            })
        return {
            "available": True,
            "score": round(score, 6),
            "grade": grade,
            "decision_threshold": float(
                support.get("danger_threshold") or .8
            ),
            "warning_threshold": float(
                support.get("warning_threshold") or .6
            ),
            "method": "target_jeonse_deposit/estimated_owner_total_assets",
            "samples": int(estimate.get("samples") or 0),
            "label_source_status": "actual_public_data_statistical_transfer",
            "formula": support.get("formula"),
            "property_facts": {
                "region": (
                    f"{prop.get('sido') or ''} "
                    f"{prop.get('gugun') or ''}"
                ).strip(),
                "house_type": prop.get("house_type"),
                "deposit_manwon": round(target, 1),
                "estimated_owner_total_assets_p50_manwon": round(
                    owner_p50, 1
                ),
                "owner_asset_ratio_pct": round(score * 100, 1),
                "probability_ratio_over_0_8": round(float(
                    support.get("probability_ratio_over_0_8") or 0
                ), 4),
                "senior_deposit_p50_manwon": (
                    round(float(senior.get("p50") or 0) / 10_000, 1)
                    if senior_deposit.get("available") else None
                ),
                "synthetic_listing": bool(prop.get("is_synthetic")),
            },
            "model_drivers": drivers,
            "not_confirmed_in_model_input": [
                "특정 임대인의 실제 금융자산",
                "특정 임대인의 다른 부동산 보유액",
                "계약일 현재 확정 선순위보증금",
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

    @staticmethod
    def _jeonse_ratio_risk_evidence(prop: dict, ratio_result: dict) -> dict:
        """계약 후 전세가율을 다가구 전세의 주 위험 지표로 사용한다."""
        ratios = ratio_result.get("ratios") or {}
        post = ratios.get("post_contract_ratio") or {}
        conservative = ratios.get("conservative_post_contract_ratio") or {}
        threshold = ratio_result.get("threshold_probabilities") or {}
        risk = ratio_result.get("risk") or {}
        score = float(risk.get("score") or post.get("p50") or 0)
        grade = str(risk.get("grade") or (
            "위험" if score >= .8 else "주의" if score >= .7
            else "안전" if score <= .6 else "관찰"
        ))
        deposit = float(prop.get("deposit_manwon") or 0)
        return {
            "available": True,
            "score": round(score, 6),
            "grade": grade,
            "decision_threshold": .8,
            "warning_threshold": .7,
            "method": (
                "(estimated_senior_deposit + selected_deposit)"
                "/estimated_market_value"
            ),
            "samples": int(ratio_result.get("sample_count") or 0),
            "label_source_status": "probabilistic_model_integration",
            "formula": (
                "(추정 선순위 임차보증금 + 선택 매물 보증금) "
                "/ 추정 건물 시장가치"
            ),
            "property_facts": {
                "region": (
                    f"{prop.get('sido') or ''} {prop.get('gugun') or ''}"
                ).strip(),
                "house_type": prop.get("house_type"),
                "deposit_manwon": round(deposit, 1),
                "post_contract_ratio_p10_pct": round(
                    float(post.get("p10") or 0) * 100, 1
                ),
                "post_contract_ratio_p50_pct": round(score * 100, 1),
                "post_contract_ratio_p90_pct": round(
                    float(post.get("p90") or 0) * 100, 1
                ),
                "conservative_ratio_p90_pct": round(
                    float(conservative.get("p90") or 0) * 100, 1
                ),
                "probability_over_80pct": float(
                    threshold.get("post_contract_over_0_8") or 0
                ),
                "probability_over_100pct": float(
                    threshold.get("post_contract_over_1_0") or 0
                ),
                "synthetic_listing": bool(prop.get("is_synthetic")),
            },
            "model_drivers": [
                {
                    "feature": "senior_deposit_distribution",
                    "label": "기존 선순위 임차보증금",
                    "direction": "risk",
                    "detail": "계약 후 내 보증금보다 먼저 회수될 수 있는 추정 권리입니다.",
                },
                {
                    "feature": "selected_deposit",
                    "label": "선택 매물 보증금",
                    "direction": "risk",
                    "value_manwon": round(deposit, 1),
                    "detail": "새 계약이 체결된 뒤 위험비율의 분자에 더합니다.",
                },
                {
                    "feature": "estimated_market_value",
                    "label": "추정 건물 시장가치",
                    "direction": "protective",
                    "detail": "개별 호실 가격이 아니라 다가구 건물 전체 시장가치 분포입니다.",
                },
                {
                    "feature": "uncertainty",
                    "label": "80% 초과 확률",
                    "direction": "neutral",
                    "value": float(
                        threshold.get("post_contract_over_0_8") or 0
                    ),
                    "detail": "2만 개 결합 경로 중 계약 후 전세가율이 80%를 넘는 비율입니다.",
                },
            ],
            "not_confirmed_in_model_input": [
                "등기부상 실제 근저당권·압류·신탁",
                "전체 임대차계약서 원문과 확정일자 순위",
                "임대인의 실제 체납·기타 자산",
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

    def _risk_explanation(
        self, prop: dict, evidence: dict | None = None,
        *, use_llm: bool = True,
    ) -> dict:
        evidence = evidence or self._risk_evidence(prop)
        if not evidence.get("available"):
            return {"available": False, "strategy": "not_applicable",
                    "summary": evidence.get("reason")}
        grade = str(evidence.get("grade") or "확인 필요")
        method = str(evidence.get("method") or "")
        jeonse_post_method = method.startswith(
            "(estimated_senior_deposit"
        )
        ratio_method = (
            method.startswith("target_jeonse_deposit/")
            or jeonse_post_method
        )
        factors = [{
            "tone": item.get("direction", "neutral"),
            "label": item.get("label", "모델 요인"),
            "detail": item.get("detail", ""),
        } for item in evidence.get("model_drivers", [])[:4]]
        fallback = {
            "available": True, "strategy": "deterministic_fallback",
            "headline": f"모델 기준 {grade} 구간입니다",
            "summary": (
                "기존 선순위 임차보증금과 내 보증금을 합친 뒤 추정 건물 "
                "시장가치로 나눈 계약 후 전세가율입니다. 분위수 결합 결과이므로 "
                "등기부와 실제 임대차계약 확인 전에는 확정값이 아닙니다."
                if jeonse_post_method else
                "선택 전세보증금을 추정 집주인 총자산으로 나눈 "
                "확률적 비율입니다. 실제 임대인 자산 조회값이 아니며 "
                "기존 선순위보증금은 별도 확인지표로 표시합니다."
                if ratio_method else
                "HF 실제 보증사고 연구의 공개계수를 이 매물 조건에 적용한 참고 점수입니다. "
                "아래 요인은 점수의 방향을 설명하지만 인과관계나 계약 안전을 확정하지 않습니다."
            ),
            "factors": factors,
            "next_checks": ["HUG·HF 보증 가입 가능 여부 확인", "등기부의 신탁·압류·근저당 확인",
                            "임대인 납세증명서와 선순위보증금 확인"],
            "limitations": "미확인 입력은 0 또는 미상으로 처리되며 합성 매물은 실제 임대인 확인값이 아닙니다.",
            "model_evidence": evidence,
        }
        if (
            not use_llm
            or not self.llm
            or not getattr(self.llm, "supports_agentic_calls", False)
        ):
            return fallback
        cache_key = json.dumps(
            evidence, ensure_ascii=False, sort_keys=True, default=str
        )
        now = time.monotonic()
        cache_lock = getattr(self, "_analysis_cache_lock", None)
        if cache_lock is not None:
            with cache_lock:
                cached = self._risk_explanation_cache.get(cache_key)
                if (
                    cached
                    and now - cached[0] < self._analysis_cache_ttl_seconds
                ):
                    value = copy.deepcopy(cached[1])
                    value["cache_hit"] = True
                    return value
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
                    "method가 (estimated_senior_deposit + selected_deposit)"
                    "/estimated_market_value이면 계약 후 전세가율 분포와 꼬리확률을 설명한다. "
                    "method가 target_jeonse_deposit/estimated_owner_total_assets이면 "
                    "선택 전세보증금/추정 집주인 총자산 비율만 주 위험판정으로 설명하고 "
                    "집주인 총자산은 특정인의 실제 조회값이 아님을 반드시 밝힌다. "
                    "선순위보증금은 별도 추정값이며 주 비율에 합산됐다고 말하지 않는다. "
                    "점수에 실제 반영된 model_drivers와 별도 계약확인 항목을 혼동하지 않는다. "
                    "fraud_score를 집주인이 사기범일 확률이나 손실 확정확률로 표현하지 않는다. "
                    "양의 계수는 모델상 점수를 높이는 연관 방향일 뿐 인과라고 말하지 않는다. "
                    "미확인 값을 안전하다고 해석하지 말고, 사용자가 먼저 확인할 행동을 짧게 쓴다."
                ),
                user=json.dumps(evidence, ensure_ascii=False),
                schema=schema, schema_name="housing_risk_explanation", max_tokens=600,
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
            result = {
                "available": True, "strategy": "llm_structured",
                **value, "model_evidence": evidence, "cache_hit": False,
            }
            if cache_lock is not None:
                with cache_lock:
                    self._risk_explanation_cache[cache_key] = (
                        now, copy.deepcopy(result)
                    )
                    if len(self._risk_explanation_cache) > 256:
                        oldest = min(
                            self._risk_explanation_cache,
                            key=lambda key: self._risk_explanation_cache[key][0],
                        )
                        self._risk_explanation_cache.pop(oldest, None)
            return result
        except Exception:
            fallback["cache_hit"] = False
            if cache_lock is not None:
                with cache_lock:
                    self._risk_explanation_cache[cache_key] = (
                        now, copy.deepcopy(fallback)
                    )
            return fallback

    def _emphasis(self, prop: dict, budget: dict, forecast: dict,
                  safety: dict, convenience: dict,
                  *, use_llm: bool = True) -> dict:
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
        if (
            not use_llm
            or not self.llm
            or not getattr(self.llm, "supports_agentic_calls", False)
        ):
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
                          contract_safety: dict,
                          *, use_llm: bool = True) -> dict:
        """Grounded cross-tab decision synthesis for the final product tab."""
        funding = budget.get("funding") or {}
        news = forecast.get("news") or {}
        score = 50
        score += 18 if funding.get("simulation_valid") else -28
        score += 8 if (safety.get("safety_score") or 0) >= 60 else 0
        score += 6 if (convenience.get("convenience_score") or 0) >= 50 else 0
        score += 8 if float(forecast.get("annual_growth_rate") or 0) > 0 else -5
        risk_evidence = (
            (contract_safety.get("risk_explanation") or {})
            .get("model_evidence") or {}
        )
        contract_risk_score = risk_evidence.get("score")
        if contract_risk_score is None:
            contract_risk_score = prop.get("fraud_score")
        contract_risk_score = (
            float(contract_risk_score)
            if contract_risk_score is not None else None
        )
        contract_risk_grade = str(
            risk_evidence.get("grade") or "확인 필요"
        )
        if contract_risk_score is not None:
            score -= min(30, round(contract_risk_score * 40))
        score = max(0, min(100, score))
        recommendation = ("recommend" if score >= 70 else
                          "conditional" if score >= 45 else "avoid")
        senior_deposit = contract_safety.get("senior_deposit") or {}
        senior_estimate = senior_deposit.get("estimate") or {}
        senior_support = senior_deposit.get("decision_support") or {}
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
                "fraud_score": contract_risk_score,
                "risk_grade": contract_risk_grade,
                "risk_formula": risk_evidence.get("formula"),
                "risk_method": risk_evidence.get("method"),
                "risk_property_facts": risk_evidence.get("property_facts"),
                "guarantee_candidates": contract_safety.get("guarantee_candidates"),
                "trust_registration": prop.get("trust_registration"),
                "seizure": prop.get("seizure_or_provisional_seizure"),
                "senior_deposit": {
                    "available": senior_deposit.get("available"),
                    "status": senior_deposit.get("status"),
                    "data_quality": senior_estimate.get("data_quality"),
                    "model_mode": senior_estimate.get("model_mode"),
                    "occupied_other_units": senior_estimate.get(
                        "occupied_other_units"),
                    "estimated_senior_deposit": senior_estimate.get(
                        "estimated_senior_deposit"),
                    "existing_deposit_conservative_p95_won": senior_support.get(
                        "existing_deposit_conservative_p95_won"),
                    "target_deposit_won": senior_support.get(
                        "target_deposit_won"),
                    "official_verification_required": senior_support.get(
                        "official_verification_required"),
                },
            },
            "deterministic_score": score,
        }
        senior_reason = None
        senior_action = None
        if senior_deposit.get("available"):
            conservative_p95 = float(
                senior_support.get(
                    "existing_deposit_conservative_p95_won") or 0
            ) / 100_000_000
            senior_reason = (
                f"기존 임차보증금 보수적 p95 추정은 약 "
                f"{conservative_p95:,.2f}억원이며 법적 확정값이 아닙니다."
            )
            senior_action = (
                "전입세대확인서·확정일자 현황·임대차계약 자료로 "
                "실제 선순위 보증금 총액 확인"
            )
        elif str(prop.get("transaction_type") or "") in {"전세", "월세"}:
            senior_reason = (
                "건축물대장 정확 주소 매칭이 없어 기존 임차보증금 총합은 "
                "추정하지 않았습니다."
            )
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
                (
                    "계약안전: "
                    f"{risk_evidence.get('formula') or '위험도'} "
                    f"{float(contract_risk_score or 0) * 100:.1f}% · "
                    f"{contract_risk_grade}"
                ),
            ] + ([senior_reason] if senior_reason else []),
            "actions": [
                "등기부·보증 가입 가능 여부 확인",
                "은행의 실제 승인 한도 확인",
            ] + ([senior_action] if senior_action else []),
            "component_scores": {
                "funding": 80 if funding.get("simulation_valid") else 20,
                "market": max(0, min(100, 50 + float(forecast.get("annual_growth_rate") or 0) * 500)),
                "safety": safety.get("safety_score"),
                "convenience": convenience.get("convenience_score"),
                "contract": max(
                    0, 100 - float(contract_risk_score or 0) * 100
                ),
            },
        }
        if contract_risk_grade == "위험":
            fallback["recommendation"] = "avoid"
            fallback["headline"] = "계약 위험 근거를 확인하기 전에는 추천하지 않아요"
            fallback["summary"] = (
                f"{risk_evidence.get('formula') or '계약 위험비율'}이 "
                f"{float(contract_risk_score or 0) * 100:.1f}%로 위험 구간입니다. "
                "보증 가입 가능성과 실제 임대인 자산·선순위 권리를 확인하세요."
            )
        if (
            not use_llm
            or not self.llm
            or not getattr(self.llm, "supports_agentic_calls", False)
        ):
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
                    "contract.risk_grade가 위험이면 계약안전이 우수하거나 "
                    "보증금 반환위험이 낮다고 말하지 말고 recommendation은 avoid로 한다. "
                    "contract.risk_formula와 risk_property_facts를 그대로 사용한다. "
                    "선순위 보증금 통계 추정은 법적 확정값으로 표현하지 말고 공식 확인 행동을 제시한다. "
                    "핵심 결론과 계약 전에 할 행동을 짧고 명확한 한국어로 쓴다."
                ), user=json.dumps(facts, ensure_ascii=False), schema=schema,
                schema_name="property_final_assessment", max_tokens=750)
            if not value:
                return fallback
            if contract_risk_grade == "위험":
                value["recommendation"] = "avoid"
                value["headline"] = fallback["headline"]
                value["summary"] = fallback["summary"]
                reasons = [
                    str(item) for item in (value.get("reasons") or [])
                    if "우수" not in str(item) and "낮" not in str(item)
                ]
                value["reasons"] = [
                    fallback["reasons"][3], *reasons
                ][:4]
            return {**fallback, **value, "strategy": "llm_structured", "score": score}
        except Exception:
            return fallback

    @staticmethod
    def _metric_facts(report: dict) -> list[dict]:
        """Collect the important numbers shown in the UI for one batch explanation."""
        facts: list[dict] = []

        def money(value) -> str:
            return f"{float(value or 0):,.0f}만원"

        def pct(value) -> str:
            return f"{float(value or 0) * 100:.1f}%"

        def score(value) -> str:
            return f"{float(value or 0):.0f}점"

        def add(metric_id: str, section: str, label: str, value,
                display: str, fallback: str, tone: str = "neutral") -> None:
            if value is None:
                return
            facts.append({
                "id": metric_id, "section": section, "label": label,
                "value": float(value), "display_value": display,
                "fallback_explanation": fallback, "tone": tone,
            })

        budget = report.get("budget") or {}
        funding = budget.get("funding") or {}
        add(
            "funding.required", "budget", "계약에 필요한 금액",
            funding.get("required_capital_manwon"),
            money(funding.get("required_capital_manwon")),
            "이 집을 계약하기 위해 처음 확보해야 하는 전체 금액입니다.",
        )
        add(
            "funding.cash", "budget", "현재 투입 가능 예산",
            funding.get("cash_used_manwon"),
            money(funding.get("cash_used_manwon")),
            "대출을 제외하고 지금 계약에 넣을 수 있는 자기자금입니다.",
            "positive",
        )
        initial_gap = funding.get("initial_cash_shortfall_manwon")
        if initial_gap is None:
            initial_gap = funding.get("funding_gap_manwon")
        add(
            "funding.initial_gap", "budget", "대출 전 부족액",
            initial_gap, money(initial_gap),
            (
                "추가 대출이나 별도 자금으로 채워야 계약할 수 있는 금액입니다."
                if float(initial_gap or 0) > 0
                else "현재 자기자금만으로 초기 계약금액을 충족한다는 뜻입니다."
            ),
            "warning" if float(initial_gap or 0) > 0 else "positive",
        )
        add(
            "funding.monthly_gap", "budget", "월 예산 초과액",
            funding.get("monthly_budget_shortfall_manwon"),
            money(funding.get("monthly_budget_shortfall_manwon")),
            "매달 감당 가능한 예산보다 주거비가 더 큰 부분입니다.",
            "warning" if float(
                funding.get("monthly_budget_shortfall_manwon") or 0
            ) > 0 else "positive",
        )

        forecast = report.get("forecast") or {}
        annual = forecast.get("annual_growth_rate")
        add(
            "market.annual_growth", "budget", "연간 집값 전망",
            annual, pct(annual),
            "과거 실거래 흐름으로 추정한 연간 가격 변화의 중심값입니다.",
            "positive" if float(annual or 0) > 0 else
            "warning" if float(annual or 0) < 0 else "neutral",
        )
        history = forecast.get("price_history") or {}
        add(
            "market.latest_price", "budget", "최근 실거래 월평균",
            history.get("latest_price_manwon"),
            money(history.get("latest_price_manwon")),
            "같은 지역·주택유형·거래유형에서 신고된 최근 월평균입니다.",
        )
        add(
            "market.period_change", "budget", "표시기간 실거래 변화",
            history.get("change_period"), pct(history.get("change_period")),
            "그래프 시작 시점과 최근 시점의 실거래 월평균 차이입니다.",
        )

        simulation = report.get("probabilistic_simulation") or {}
        base = simulation.get("base") or {}
        terminal = base.get("terminal_net_worth") or {}
        horizon = int(simulation.get("horizon_years") or 10)
        add(
            "assets.net_worth_p50", "assets",
            f"{horizon}년 후 순자산 중앙값", terminal.get("p50"),
            money(terminal.get("p50")),
            "미래 경로의 절반은 이보다 높고 절반은 낮은 중심 결과입니다.",
        )
        add(
            "assets.net_worth_p10", "assets", "순자산 하위 경로",
            terminal.get("p10"), money(terminal.get("p10")),
            "불리한 미래 경로에서 예상되는 보수적인 순자산 수준입니다.",
            "warning",
        )
        add(
            "assets.net_worth_p90", "assets", "순자산 상위 경로",
            terminal.get("p90"), money(terminal.get("p90")),
            "유리한 미래 경로에서 기대할 수 있는 상단 순자산 수준입니다.",
            "positive",
        )
        add(
            "assets.cash_depletion", "assets", "현금 고갈 확률",
            base.get("cash_depletion_probability"),
            pct(base.get("cash_depletion_probability")),
            "계산 기간 중 현금성 자산이 바닥난 미래 경로의 비중입니다.",
            "warning",
        )
        add(
            "assets.repayment_distress", "assets", "상환곤란 확률",
            base.get("repayment_distress_probability"),
            pct(base.get("repayment_distress_probability")),
            "대출 상환 부담이 연속해서 감당 한도를 넘은 경로의 비중입니다.",
            "warning",
        )
        add(
            "assets.cvar", "assets", "최악 경로 평균 변화",
            base.get("cvar_5_terminal_change_manwon"),
            money(base.get("cvar_5_terminal_change_manwon")),
            "가장 불리한 미래들을 모아 계산한 평균 자산 변화입니다.",
            "warning",
        )

        contract = report.get("contract_safety") or {}
        ratio = contract.get("jeonse_ratio") or {}
        ratios = ratio.get("ratios") or {}
        post = ratios.get("post_contract_ratio") or {}
        threshold = ratio.get("threshold_probabilities") or {}
        add(
            "contract.ratio_p50", "contract", "계약 후 전세가율 중앙값",
            post.get("p50"), pct(post.get("p50")),
            "선순위 보증금과 내 보증금을 건물가치로 나눈 중심 추정치입니다.",
            "warning" if float(post.get("p50") or 0) >= .6 else "positive",
        )
        add(
            "contract.ratio_p90", "contract", "계약 후 전세가율 상단",
            post.get("p90"), pct(post.get("p90")),
            "불리한 경우까지 고려한 높은 쪽 전세가율 추정치입니다.",
            "warning",
        )
        add(
            "contract.over_80", "contract", "위험 경계 초과 확률",
            threshold.get("post_contract_over_0_8"),
            pct(threshold.get("post_contract_over_0_8")),
            "보증금 부담이 위험 경계선을 넘는 미래 경로의 비중입니다.",
            "warning",
        )
        add(
            "contract.over_100", "contract", "건물가치 초과 확률",
            threshold.get("post_contract_over_1_0"),
            pct(threshold.get("post_contract_over_1_0")),
            "보증금 부담이 추정 건물가치보다 커지는 경로의 비중입니다.",
            "warning",
        )
        owner = contract.get("owner_asset_ratio") or {}
        owner_estimate = owner.get("estimate") or {}
        owner_assets = owner_estimate.get("estimated_owner_total_assets") or {}
        owner_ratio = owner_estimate.get(
            "target_deposit_to_total_assets_ratio") or {}
        add(
            "contract.owner_assets_p50", "contract", "추정 집주인 총자산",
            owner_assets.get("p50"), money(owner_assets.get("p50")),
            "특정 집주인 조회값이 아니라 통계 자료로 추정한 중심값입니다.",
        )
        add(
            "contract.deposit_asset_ratio", "contract",
            "전세금 대비 추정 총자산 비율",
            owner_ratio.get("p50"), pct(owner_ratio.get("p50")),
            "내 보증금이 추정 총자산에서 차지하는 비중의 중심값입니다.",
            "warning",
        )
        senior = contract.get("senior_deposit") or {}
        senior_support = senior.get("decision_support") or {}
        senior_p95 = senior_support.get(
            "existing_deposit_conservative_p95_won")
        add(
            "contract.senior_p95", "contract", "기존 보증금 보수 추정",
            (float(senior_p95) / 10_000 if senior_p95 is not None else None),
            money(float(senior_p95 or 0) / 10_000),
            "다른 임차인이 먼저 돌려받을 수 있는 보증금의 보수적 추정입니다.",
            "warning",
        )

        safety = report.get("safety") or {}
        add(
            "safety.score", "safety", "치안시설 점수",
            safety.get("safety_score"), score(safety.get("safety_score")),
            "주변 공공 치안시설의 거리와 개수를 합친 비교용 점수입니다.",
        )
        convenience = report.get("convenience") or {}
        add(
            "living.score", "living", "생활편의 점수",
            convenience.get("convenience_score"),
            score(convenience.get("convenience_score")),
            "주변 병원·약국·마트·음식점 등 접근성을 합친 비교용 점수입니다.",
        )
        final = report.get("final_assessment") or {}
        add(
            "final.score", "final", "최종 의사결정 점수",
            final.get("score"), score(final.get("score")),
            "자금·시장·치안·편의·계약안전을 함께 반영한 비교 점수입니다.",
        )
        return facts

    def explain_metrics(self, report: dict) -> dict:
        """Explain all visible key metrics with one grounded LLM call."""
        facts = self._metric_facts(report)
        fallback_items = [{
            "id": item["id"], "section": item["section"],
            "label": item["label"], "value": item["display_value"],
            "explanation": item["fallback_explanation"],
            "tone": item["tone"],
        } for item in facts]
        fallback = {
            "strategy": "deterministic_fallback",
            "items": fallback_items,
        }
        if not facts:
            return fallback
        if (
            not self.llm
            or not getattr(self.llm, "supports_agentic_calls", False)
        ):
            return fallback
        cache_key = json.dumps(
            facts, ensure_ascii=False, sort_keys=True, default=str
        )
        now = time.monotonic()
        with self._analysis_cache_lock:
            cached = self._metric_explanation_cache.get(cache_key)
            if cached and now - cached[0] < self._analysis_cache_ttl_seconds:
                value = copy.deepcopy(cached[1])
                value["cache_hit"] = True
                return value
        schema = {
            "type": "object", "additionalProperties": False,
            "properties": {
                "items": {
                    "type": "array", "maxItems": 32,
                    "items": {
                        "type": "object", "additionalProperties": False,
                        "properties": {
                            "id": {"type": "string"},
                            "explanation": {"type": "string"},
                            "tone": {
                                "type": "string",
                                "enum": ["positive", "warning", "neutral"],
                            },
                        },
                        "required": ["id", "explanation", "tone"],
                    },
                },
            },
            "required": ["items"],
        }
        try:
            value = self.llm.analyze_json(
                operation="report.metric_explanations",
                system=(
                    "너는 청년 주택 리포트의 숫자 해설자다. 입력 facts의 각 id를 "
                    "한 번씩 그대로 반환한다. 설명은 해당 숫자가 무엇을 뜻하고 "
                    "사용자의 결정에 어떤 영향을 주는지 쉬운 한국어 한 문장으로 쓴다. "
                    "안전 확인이나 예산 조정이 필요하면 문장 끝에 바로 할 행동을 붙인다. "
                    "값을 그대로 반복하지 말고 45자 안팎으로 간결하게 쓴다. "
                    "영문 약어, 모델명, 테이블명, 보정기법 이름은 절대 노출하지 않는다. "
                    "설명 문장에는 숫자·새 계산·새 사실을 절대 넣지 않는다. "
                    "예측값과 확률을 확정 사실로 표현하지 않는다. "
                    "집주인 자산은 실제 조회값이라고 말하지 않는다."
                ),
                user=json.dumps({"facts": facts}, ensure_ascii=False),
                schema=schema, schema_name="report_metric_explanations",
                max_tokens=1800,
            ) or {}
            by_id = {
                str(item.get("id")): item
                for item in value.get("items", [])
                if isinstance(item, dict)
            }
            items = []
            for fallback_item in fallback_items:
                generated = by_id.get(fallback_item["id"]) or {}
                explanation = str(
                    generated.get("explanation") or ""
                ).strip()[:140]
                # The displayed value already contains the number. Reject any
                # generated sentence that introduces another numeric claim.
                if not explanation or any(ch.isdigit() for ch in explanation):
                    explanation = fallback_item["explanation"]
                tone = str(generated.get("tone") or fallback_item["tone"])
                if tone not in {"positive", "warning", "neutral"}:
                    tone = fallback_item["tone"]
                items.append({
                    **fallback_item,
                    "explanation": explanation,
                    "tone": tone,
                })
            result = {
                "strategy": "llm_structured", "items": items,
                "cache_hit": False,
            }
            with self._analysis_cache_lock:
                self._metric_explanation_cache[cache_key] = (
                    now, copy.deepcopy(result)
                )
            return result
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
            "road_address", "jibun_address", "address_detail_public",
            "region_coordinate_source", "coordinate_distribution_method",
            "lat", "lng", "transaction_type",
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

    def build(self, user: dict, property_id: str, assumptions: dict | None = None,
              session_id: str | None = None) -> dict:
        analysis_started = time.perf_counter()
        assumptions = dict(assumptions or {})
        prop = self.property(property_id)
        if prop is None:
            raise KeyError(property_id)
        simulation_seed = int(
            assumptions.get("simulation_seed")
            if assumptions.get("simulation_seed") is not None
            else secrets.randbelow(2_147_483_647)
        )
        decision_run_id = self.audit.start_run(
            session_id=session_id, property_id=property_id,
            input_snapshot={"user": user, "property_id": property_id,
                            "assumptions": assumptions},
            simulation_seed=simulation_seed,
            model_versions={
                "forecast": getattr(self.forecaster.artifact, "version", None)
                            if not isinstance(self.forecaster.artifact, dict)
                            else self.forecaster.artifact.get("version"),
                "risk": "hf_actual_calibrated",
                "senior_deposit": "senior_deposit_mvp_v1_scenario_only",
                "owner_asset_ratio": "four_component_owner_asset_ratio_v1",
                "simulation": "vectorized_monthly_monte_carlo_v1",
            },
            data_version=str(prop.get("source_captured_at") or prop.get(
                "last_verified_at") or "workspace"),
        )
        lat, lng = float(prop["lat"]), float(prop["lng"])
        region = f"{prop.get('sido', '')} {prop.get('gugun', '')}".strip()
        programs = self.finance.search(
            user_income_manwon=user.get("monthly_income_manwon"),
            user_age=user.get("age"), region=region, finance_mode="eligibility", limit=50,
            user_profile=user,
        )
        finance_sql, finance_params = self.finance.build_query(
            user_income_manwon=user.get("monthly_income_manwon"),
            user_age=user.get("age"), region=region, finance_mode="eligibility",
            user_profile=user, limit=50,
        )
        self.audit.record_step(
            decision_run_id, stage="finance_eligibility", tool="sqlite",
            input_data={"region": region, "profile_fields": sorted(user)},
            output_data={"eligible_or_review_rows": len(programs)},
            sql_text=finance_sql, sql_parameters=finance_params,
            source_refs=["finance_programs"],
        )
        senior_reference_date = str(
            assumptions.get("reference_date") or date.today().isoformat()
        )
        # 두 확률 모델은 내부 수치 연산에서 CPU 스레드를 사용한다. 둘을
        # Python 스레드로 동시에 실행하면 작은 Lightsail 인스턴스에서는
        # 경합 때문에 각각이 2배 이상 느려졌다. 외부 I/O 중심인 기반 분석만
        # 백그라운드에서 돌리고 두 모델은 순차 실행한다.
        with ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="report-evidence"
        ) as pool:
            base_future = pool.submit(self._base_analysis, prop)
            stage_started = time.perf_counter()
            senior_deposit = self.senior_deposit.analyze_property(
                prop,
                reference_date=senior_reference_date,
                samples=int(assumptions.get("senior_deposit_samples", 1_000)),
                mode="scenario",
            )
            senior_deposit_elapsed_ms = round(
                (time.perf_counter() - stage_started) * 1000
            )
            stage_started = time.perf_counter()
            owner_asset_ratio = self.owner_asset_ratio.analyze_property(
                prop,
                reference_date=senior_reference_date,
                # 20,000 paths are still available through the dedicated
                # research endpoint. The interactive report uses 1,000 input
                # paths and reconstructs the integrated ratio with 20,000
                # paths below, retaining a stable displayed distribution.
                samples=int(assumptions.get("owner_asset_samples", 1_000)),
            )
            owner_asset_elapsed_ms = round(
                (time.perf_counter() - stage_started) * 1000
            )
            base_analysis, analysis_cache_hit = base_future.result()
        jeonse_ratio = self.jeonse_ratio.calculate(
            prop,
            senior_deposit,
            owner_asset_ratio,
            samples=int(assumptions.get("jeonse_ratio_samples", 20_000)),
            seed=simulation_seed,
            dependence=str(
                assumptions.get("jeonse_ratio_dependence") or "independence"
            ),
        )
        guarantee_review = self.guarantees.evaluate(
            prop, user, jeonse_ratio
        )
        regional_market = base_analysis["regional_market"]
        forecast = base_analysis["forecast"]
        self.audit.record_step(
            decision_run_id, stage="market_safety_evidence",
            tool="parallel_property_analysis",
            output_data={
                "cache_hit": analysis_cache_hit,
                "forecast_model": forecast.get("model_version"),
                "price_history_source": (forecast.get("price_history") or {}).get("source"),
                "stage_elapsed_ms": base_analysis.get("stage_elapsed_ms"),
            },
            source_refs=[
                "RTMS", "R-ONE", "NAVER local/news",
            ],
        )
        self.audit.record_step(
            decision_run_id,
            stage="senior_deposit_evidence",
            tool="building_hub_rtms_monte_carlo",
            input_data={
                "reference_date": senior_reference_date,
                "property_address": prop.get("road_address"),
            },
            output_data={
                "available": senior_deposit.get("available"),
                "status": senior_deposit.get("status"),
                "match": senior_deposit.get("match"),
                "data_quality": (
                    senior_deposit.get("estimate") or {}
                ).get("data_quality"),
                "conservative_p95_won": (
                    senior_deposit.get("decision_support") or {}
                ).get("existing_deposit_conservative_p95_won"),
                "risk_score_changed": (
                    senior_deposit.get("decision_support") or {}
                ).get("risk_score_changed"),
            },
            source_refs=["Building HUB", "RTMSDataSvcSHRent"],
        )
        self.audit.record_step(
            decision_run_id,
            stage="owner_asset_ratio_evidence",
            tool="building_hub_rtms_household_survey_monte_carlo",
            input_data={
                "reference_date": senior_reference_date,
                "property_address": prop.get("road_address"),
                "target_deposit_manwon": prop.get("deposit_manwon"),
            },
            output_data={
                "available": owner_asset_ratio.get("available"),
                "status": owner_asset_ratio.get("status"),
                "match": owner_asset_ratio.get("match"),
                "ratio_p50": (
                    owner_asset_ratio.get("decision_support") or {}
                ).get("ratio_p50"),
                "risk_grade": (
                    owner_asset_ratio.get("decision_support") or {}
                ).get("risk_grade"),
                "senior_deposit_added_to_ratio": (
                    owner_asset_ratio.get("decision_support") or {}
                ).get("senior_deposit_added_to_ratio"),
            },
            source_refs=[
                "Building HUB", "RTMSDataSvcSHRent",
                "RTMSDataSvcSHTrade", "가계금융복지조사",
            ],
        )
        self.audit.record_step(
            decision_run_id,
            stage="jeonse_ratio_integration",
            tool="quantile_inverse_cdf_gaussian_copula",
            input_data={
                "reference_date": senior_reference_date,
                "sample_count": int(
                    assumptions.get("jeonse_ratio_samples", 20_000)
                ),
                "dependence": str(
                    assumptions.get("jeonse_ratio_dependence")
                    or "independence"
                ),
            },
            output_data={
                "available": jeonse_ratio.get("available"),
                "status": jeonse_ratio.get("status"),
                "risk": jeonse_ratio.get("risk"),
                "data_quality": jeonse_ratio.get("data_quality"),
                "threshold_probabilities": (
                    jeonse_ratio.get("threshold_probabilities")
                ),
            },
            source_refs=[
                "senior_deposit_mvp_v1",
                "four_component_owner_asset_ratio_v1.property_value",
            ],
        )
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
        probabilistic = simulate_probabilistic(
            user, prop, forecast, budget, assumptions,
            paths=int(assumptions.get("monte_carlo_paths", 10_000)),
            seed=simulation_seed,
        )
        self.audit.record_step(
            decision_run_id, stage="probabilistic_asset_simulation",
            tool="numpy_monte_carlo",
            input_data={
                "paths": probabilistic["path_count"], "seed": simulation_seed,
                "assumptions": budget.get("assumptions"),
            },
            output_data={
                "ten_year_net_worth": probabilistic["base"]["ten_year_net_worth"],
                "cash_depletion_probability": probabilistic["base"][
                    "cash_depletion_probability"],
                "repayment_distress_probability": probabilistic["base"][
                    "repayment_distress_probability"],
                "cvar_5": probabilistic["base"]["cvar_5_terminal_change_manwon"],
            },
        )
        safety = base_analysis["safety"]
        convenience = base_analysis["convenience"]
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
        unverified_landlord_fields = (
            is_synthetic or not bool(prop.get("tax_arrears_checked"))
        )
        tax_status = "unverified"
        if not is_synthetic and prop.get("tax_arrears_checked"):
            tax_status = "arrears_present" if prop.get("tax_arrears_present") else "verified_clear"
        risk_explanation_started = time.perf_counter()
        if jeonse_ratio.get("available"):
            ratio_evidence = self._jeonse_ratio_risk_evidence(
                prop, jeonse_ratio
            )
            risk_explanation = self._risk_explanation(
                prop, evidence=ratio_evidence
            )
            prop["fraud_score"] = ratio_evidence["score"]
        elif owner_asset_ratio.get("available"):
            ratio_evidence = self._owner_ratio_risk_evidence(
                prop, owner_asset_ratio, senior_deposit
            )
            risk_explanation = self._risk_explanation(
                prop, evidence=ratio_evidence
            )
            # The report and final assessment must consume the exact same
            # contract-risk score.  The DB value remains untouched.
            prop["fraud_score"] = ratio_evidence["score"]
        else:
            risk_explanation = base_analysis["risk_explanation"]
            if (
                str(prop.get("transaction_type") or "") == "전세"
                and prop.get("fraud_score") is None
            ):
                prop["fraud_score"] = (
                    (risk_explanation.get("model_evidence") or {}).get("score")
                )
        risk_explanation_elapsed_ms = round(
            (time.perf_counter() - risk_explanation_started) * 1000
        )
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
        if guarantee_review.get("available"):
            guarantee_candidates = guarantee_review.get("products") or []
        contract_safety = {
            "fraud_score": prop.get("fraud_score"),
            "risk_explanation": risk_explanation,
            "guarantee_eligible": prop.get("guarantee_eligible"),
            "guarantee_candidates": guarantee_candidates,
            "trust_registration": prop.get("trust_registration"),
            "seizure_or_provisional_seizure": prop.get("seizure_or_provisional_seizure"),
            "tax_arrears": {
                "status": tax_status,
                "synthetic_field_ignored": unverified_landlord_fields,
                "message": (
                    "합성 매물의 체납 필드는 실제 임대인 확인값이 아니므로 위험도에 반영하지 않습니다."
                    if is_synthetic else
                    "임대인 공식 체납 확인값이 없어 위험도에 반영하지 않습니다."
                    if unverified_landlord_fields else
                    "임대인 동의 하의 납세증명서·공식 열람 결과만 반영합니다."
                ),
                "required_evidence": "임대인 납세증명서 또는 법정 절차에 따른 미납국세 열람",
            },
            "registry_guide": registry_check_guide(prop.get("road_address") or ""),
            "senior_deposit": senior_deposit,
            "owner_asset_ratio": owner_asset_ratio,
            "jeonse_ratio": jeonse_ratio,
            "guarantee_review": guarantee_review,
            "guarantee_candidates": guarantee_candidates,
            "preferential_repayment": guarantee_review.get(
                "preferential_repayment"
            ),
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
        life_stage = str(user.get("youth_life_stage") or "other")
        stage_defaults = {
            "student": ["학교 정문까지 이동시간", "가전·가구 옵션", "해충·방역 확인"],
            "early_career": ["직장 통근시간", "침실·주방 분리", "주차·채광"],
            "other": ["월 주거비", "계약 안전", "생활 편의"],
        }
        youth_profile = {
            "life_stage": life_stage,
            "evaluation_priorities": list(dict.fromkeys([
                *(user.get("housing_priorities") or []),
                *stage_defaults.get(life_stage, stage_defaults["other"]),
                (
                    "청결·하자 확인"
                    if str(prop.get("transaction_type") or "") in {"전세", "월세"}
                    else "자본차익·환금성"
                ),
            ])),
        }
        result = {
            "decision_run_id": decision_run_id,
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
                    "youth_life_stage", "housing_priorities",
                    "has_hf_jeonse_loan_guarantee",
                    "move_in_registration_possible", "fixed_date_possible",
                    "senior_mortgage_established_date",
                    "wants_guarantee_insurance",
                )
            },
            "forecast": forecast, "budget": budget,
            "probabilistic_simulation": probabilistic,
            "regional_market": regional_market, "ev_chargers": ev_chargers,
            "finance_programs": programs, "safety": safety,
            "convenience": convenience, "contract_safety": contract_safety,
            "youth_profile": youth_profile,
            "restricted_checks": restricted_checks,
            "provenance": {
                "price": (
                    "국토교통부 실거래 월간 집계 + GBDT 분위수 시나리오. "
                    "KB 시세는 별도 이용허락/제휴 피드가 설정된 경우에만 참조"
                ),
                "news": (
                    "NAVER 뉴스 검색 API(우선) 또는 공개 뉴스 RSS + "
                    "지역·주택 키워드 기반 빠른 관련성 판정"
                ),
                "safety": safety.get("sources"), "convenience": convenience.get("sources"),
                "naver_local_calls": base_analysis["naver_local_calls"],
                "naver_local_call_limit": base_analysis["naver_local_call_limit"],
                "privacy_policy": "개인식별 민감정보 자동수집·재게시 금지",
                "regional_market": "한국부동산원 R-ONE Open API",
                "ev_chargers": "V-World 활용모델 + 공공데이터포털 EvInfoServiceV2",
                "senior_deposit": (
                    "건축HUB 정확 도로명주소 매칭 + 기준일 이전 "
                    "국토교통부 단독·다가구 임대차 실거래 분포"
                ),
                "owner_asset_ratio": (
                    "건축HUB + 국토교통부 단독·다가구 전월세·매매 "
                    "실거래 + 가계금융복지조사 조건부 자산분포"
                ),
            },
        }
        # 위험도 설명은 위에서 LLM이 실제 계산 근거를 설명한다. 강조 문구와
        # 최종 점수까지 별도 LLM으로 다시 생성하면 동일 사실을 세 번 전송하고
        # 응답이 수십 초 늘어나므로 검증 가능한 결정론 합성을 사용한다.
        result["ai_emphasis"] = self._emphasis(
            prop, budget, forecast, safety, convenience, use_llm=False
        )
        result["final_assessment"] = self._final_assessment(
            prop, budget, forecast, safety, convenience, contract_safety,
            use_llm=False,
        )
        result["analysis_performance"] = {
            "cache_hit": analysis_cache_hit,
            "elapsed_ms": round((time.perf_counter() - analysis_started) * 1000),
            "slow_stages_parallelized": [
                "market_and_news", "local_places",
            ],
            "stage_elapsed_ms": {
                **(base_analysis.get("stage_elapsed_ms") or {}),
                "senior_deposit_model": senior_deposit_elapsed_ms,
                "owner_asset_model": owner_asset_elapsed_ms,
                "contract_risk_llm": risk_explanation_elapsed_ms,
            },
        }
        safe_result = json_safe(result)
        self.audit.record_step(
            decision_run_id, stage="llm_explanation", tool=type(self.llm).__name__,
            output_data={
                "emphasis_strategy": (result.get("ai_emphasis") or {}).get("strategy"),
                "final_recommendation": (result.get("final_assessment") or {}).get(
                    "recommendation"),
            },
        )
        self.audit.complete_run(
            decision_run_id,
            {
                "property_id": property_id,
                "funding": budget.get("funding"),
                "probabilistic_summary": {
                    "base": probabilistic["base"],
                    "stress_delta": probabilistic["stress_delta"],
                },
                "final_assessment": result.get("final_assessment"),
            },
            elapsed_ms=(time.perf_counter() - analysis_started) * 1000,
        )
        return safe_result
