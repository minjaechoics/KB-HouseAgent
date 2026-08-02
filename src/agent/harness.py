"""
(2)-2 AI Agent Harness — 상태 유지 2단계 대화 + Q&A 라우팅 + ATOM 추천.

대화 흐름(사용자 요구사항 반영):
  1) 사용자 발화 → Planner로 intent/slots/action 판단.
  2) action:
       - clarify : 되물음(뭔가 원하는데 슬롯 못 채움 / 무의도). '아무거나'는 clarify 아님.
       - confirm : 조건을 사용자에게 보여주고 "이대로 진행?" 확인 요청(바로 추천 X).
       - proceed : (아무거나 or Q&A) 즉시 실행.
  3) 확인 단계에서 yes → 추천 실행 / no → 초기화 / modify → 조건 재파싱 후 재확인.
  4) 추천 시 조건을 ATOM으로 분해 → 완전만족/1개누락/2개누락 그룹으로 나눠 표시.

상태(session)를 dict로 관리하여 CLI/서버 어디서든 재사용 가능.
모든 외부 의존(LLM/지도/실거래가/POI)은 mock 폴백이 있어 오프라인에서 완전 동작.
"""
from __future__ import annotations
import copy
import json
import math
import re
import sqlite3
import joblib
import pandas as pd

from src import config
from src.agent.llm import get_llm
from src.agent.planner import parse_confirmation, Plan
from src.agent.text2sql import Text2SQLPipeline
from src.agent import atoms as A
from src.preference.affordability import compute_affordability
from src.tools.map_tool import MapTool, SIDO_GUGUN_CENTROIDS
from src.tools.finance_tool import FinanceTool
from src.tools.property_db_tool import PropertyDBTool
from src.server.property_search import AtomicPropertySearch, atoms_from_slots, make_atom
from src.tools.advisory_tools import contract_checklist, cost_breakdown
from src.tools.external_tools import CATEGORY_KO
from src.tools.safety_tool import SafetyTool
from src.tools.convenience_tool import ConvenienceTool
from src.tools.registry_tool import registry_check_guide
from src.recommender import models as R
from src.optimization import optimize_housing_choices
from src.preferences import normalize_preferences
from src.report.budget import simulate as simulate_asset_budget
from src.simulation.monte_carlo import simulate_probabilistic

# 랜드마크 좌표(데모). 실서비스는 지오코딩 API로 대체.
LANDMARKS = {
    "ifc": (37.525, 126.9258), "여의도": (37.521, 126.924),
    "강남": (37.4979, 127.0276), "판교": (37.3948, 127.1112),
    "종로": (37.5729, 126.9793), "가산": (37.4816, 126.8827),
    "구로디지털": (37.4851, 126.9015), "을지로": (37.5660, 126.9910),
}

PREFERRED_SIDO_ALIASES = {
    "서울특별시": "서울", "서울시": "서울", "서울": "서울",
    "부산광역시": "부산", "부산시": "부산", "부산": "부산",
    "대구광역시": "대구", "대구시": "대구", "대구": "대구",
    "인천광역시": "인천", "인천시": "인천", "인천": "인천",
    "광주광역시": "광주", "광주시": "광주", "광주": "광주",
    "대전광역시": "대전", "대전시": "대전", "대전": "대전",
    "울산광역시": "울산", "울산시": "울산", "울산": "울산",
    "세종특별자치시": "세종", "세종시": "세종", "세종": "세종",
    "경기도": "경기", "경기": "경기", "강원특별자치도": "강원", "강원도": "강원", "강원": "강원",
    "충청북도": "충북", "충북": "충북", "충청남도": "충남", "충남": "충남",
    "전북특별자치도": "전북", "전라북도": "전북", "전북": "전북",
    "전라남도": "전남", "전남": "전남", "경상북도": "경북", "경북": "경북",
    "경상남도": "경남", "경남": "경남", "제주특별자치도": "제주", "제주도": "제주", "제주": "제주",
}


def _normalize_preferred_region(user: dict) -> dict:
    normalized = dict(user)
    raw = str(normalized.get("preferred_sido") or "").strip()
    if not raw:
        return normalized
    selected_alias = next(
        (alias for alias in sorted(PREFERRED_SIDO_ALIASES, key=len, reverse=True)
         if alias in raw),
        None,
    )
    if not selected_alias:
        return normalized
    normalized["preferred_sido"] = PREFERRED_SIDO_ALIASES[selected_alias]
    remainder = raw.replace(selected_alias, " ", 1)
    districts = re.findall(r"([가-힣]{1,8}(?:구|군|시))", remainder)
    if districts and not normalized.get("preferred_gugun"):
        normalized["preferred_gugun"] = districts[0]
    return normalized


def _classify_goal_finance(programs: list[dict], affordability) -> tuple[list[dict], dict]:
    """조회 정책을 전세보증금 증액 수단과 부대비용 절감 수단으로 구분한다."""
    direct_terms = (
        "전세자금", "전월세자금", "임차보증금", "전세보증금대출",
        "버팀목", "중소기업취업청년",
    )
    ancillary_terms = (
        "보증료", "반환보증", "이자 지원", "이자지원", "월세 지원", "월세지원",
    )
    direct, ancillary = [], []
    for original in programs:
        program = dict(original)
        haystack = " ".join(str(program.get(key) or "") for key in (
            "name", "category", "product_kind", "support_content", "desc", "target"
        )).lower()
        has_loan = "대출" in str(program.get("product_kind") or "") \
            or "대출" in str(program.get("category") or "")
        if has_loan and any(term in haystack for term in direct_terms):
            program["goal_role"] = "전세보증금 증액 후보"
            direct.append(program)
        elif any(term in haystack for term in ancillary_terms):
            program["goal_role"] = "보증료·부대비용 절감"
            ancillary.append(program)

    evaluated = []
    monthly_cap = float(affordability.max_monthly_housing_manwon)
    for program in direct:
        reported_limit = float(program.get("max_amount_manwon") or 0)
        if reported_limit <= 0:
            continue
        rate = program.get("rate_pct")
        if rate is not None and float(rate) > 0:
            # 전세자금대출을 이자만 납부하는 1차 상환능력 시뮬레이션.
            service_limit = monthly_cap * 12 / (float(rate) / 100)
            effective_limit = min(reported_limit, service_limit)
            capacity_verified = True
        else:
            # 금리 데이터가 없어 상환능력을 검증하지 못한다. 이 경우 신고된
            # 한도 자체를 0으로 무시하지 않고, 검증 필요 표시와 함께 그대로
            # 후보에 포함한다(최종 한도는 어차피 금융기관 심사에서 확정됨).
            effective_limit = reported_limit
            capacity_verified = False
        evaluated.append((effective_limit, program, capacity_verified))
    evaluated.sort(key=lambda item: item[0], reverse=True)
    selected = evaluated[0] if evaluated else None

    base_budget = float(affordability.recommended_jeonse_deposit_manwon)
    loan_limit = selected[0] if selected else 0.0
    capacity_verified = selected[2] if selected else None
    plan = {
        "base_jeonse_budget_manwon": round(base_budget, 1),
        "direct_loan_limit_manwon": round(loan_limit, 1),
        "estimated_max_deposit_manwon": round(base_budget + loan_limit, 1),
        "selected_program_id": selected[1].get("program_id") if selected else None,
        "selected_program_name": selected[1].get("name") if selected else None,
        "selected_program_repayment_capacity_verified": capacity_verified,
        "reviewed_program_count": len(programs),
        "direct_finance_count": len(direct),
        "ancillary_support_count": len(ancillary),
        "method": "자기자금 적정 전세예산 + 단일 직접 전세자금대출의 유효 한도",
        "assumptions": [
            "여러 대출 한도를 합산하지 않고 가장 큰 유효 한도 하나만 사용",
            "금리가 확인되는 상품은 표시 금리의 이자만 납부하는 1차 상환능력으로 한도를 검증",
            "금리 미확인 상품은 상환능력 검증 없이 신고된 한도를 그대로 사용(최종 심사 필요)",
            "실제 한도·보증비율·자격은 금융기관 및 보증기관 최종 심사 필요",
        ],
        "limitation": (
            None if selected else
            "현재 금융 DB에서 한도가 확인되는 직접 전세자금 상품을 찾지 못해 "
            "전세예산을 임의로 늘리지 않았습니다."
        ),
    }
    return direct + ancillary, plan


def _classify_transaction_finance(
    programs: list[dict], affordability, *, direct_terms: tuple[str, ...],
    base_budget_manwon: float,
) -> dict:
    """`_classify_goal_finance`를 전세 외 거래유형에도 쓸 수 있게 일반화한 버전.
    "가장 큰 유효 한도 대출 하나만 사용" 원칙은 동일하게 유지한다. 전세 목표
    대화(_handle_financed_jeonse_goal)는 하위 호환을 위해 기존 함수를 그대로 쓴다."""
    direct = []
    for original in programs:
        program = dict(original)
        haystack = " ".join(str(program.get(key) or "") for key in (
            "name", "category", "product_kind", "support_content", "desc", "target"
        )).lower()
        has_loan = "대출" in str(program.get("product_kind") or "") \
            or "대출" in str(program.get("category") or "")
        if has_loan and any(term in haystack for term in direct_terms):
            direct.append(program)

    evaluated = []
    monthly_cap = float(affordability.max_monthly_housing_manwon)
    for program in direct:
        reported_limit = float(program.get("max_amount_manwon") or 0)
        if reported_limit <= 0:
            continue
        rate = program.get("rate_pct")
        if rate is not None and float(rate) > 0:
            service_limit = monthly_cap * 12 / (float(rate) / 100)
            effective_limit = min(reported_limit, service_limit)
            capacity_verified = True
        else:
            effective_limit = reported_limit
            capacity_verified = False
        evaluated.append((effective_limit, program, capacity_verified))
    evaluated.sort(key=lambda item: item[0], reverse=True)
    selected = evaluated[0] if evaluated else None

    loan_limit = selected[0] if selected else 0.0
    return {
        "base_budget_manwon": round(base_budget_manwon, 1),
        "direct_loan_limit_manwon": round(loan_limit, 1),
        "estimated_max_budget_manwon": round(base_budget_manwon + loan_limit, 1),
        "selected_program_id": selected[1].get("program_id") if selected else None,
        "selected_program_name": selected[1].get("name") if selected else None,
        "selected_program_repayment_capacity_verified": (
            selected[2] if selected else None),
        "reviewed_program_count": len(programs),
        "direct_finance_count": len(direct),
        "limitation": (
            None if selected else
            "현재 금융 DB에서 한도가 확인되는 직접 대출 상품을 찾지 못해 "
            "예산을 임의로 늘리지 않았습니다."
        ),
    }


_PROPERTY_RECOMMENDING_INTENTS = {
    "recommend", "goal_financed_jeonse", "goal_best_affordable",
    "goal_alternative_areas",
}
_ADVISOR_REDIRECT_MESSAGE = (
    "이 채팅 상담은 매물을 직접 검색·추천하지 않고, 전세·월세 비교, 대출 자격, "
    "계약 위험, 시장 전망처럼 의사결정에 필요한 질문에 답하는 조언자 역할만 "
    "합니다. 조건에 맞는 매물은 지도 화면의 필터나 'AI 조건 추가'에서 찾아 "
    "주세요."
)


_LEASE_ARCHETYPE_LABELS = {
    "A": "자산 운용형", "B": "비용 절감 & 안정형", "C": "안전 최우선형",
}
_LEASE_ARCHETYPE_RECOMMENDATION = {"A": "월세", "B": "전세", "C": "반전세"}
_LEASE_ARCHETYPE_LEAD_SENTENCES = {
    "A": (
        "목돈을 투자에 활용할 계획이 있으시니, 순자산 중앙값만 보면 전세가 "
        "높게 나오더라도 계약 유연성과 보증금 회수 부담을 줄이기 위해 "
        "월세(또는 반전세)를 권합니다."),
    "B": (
        "특별한 투자처가 없고 거주기간도 충분히 확보하셨으니, 정책 대출과 "
        "전세보증보험을 활용해 월 주거비를 낮출 수 있는 전세를 권합니다."),
    "C": (
        "이 지역·매물은 보증금을 돌려받지 못할 위험이 낮지 않아, 보증금을 "
        "최소화한 반전세나 보증금이 낮은 월세로 원금 안전을 우선하시길 "
        "권합니다."),
}


def _lease_consult_message(archetype: str, comparison_summary: str) -> str:
    """전세/월세 상담 결론 문장을 코드가 먼저 확정한다. LLM 합성이 실패해도
    이 문장이 그대로 사용자에게 보이고(폴백), 합성이 성공해도 이 결론
    문장으로 시작하도록 지시해 LLM이 순자산 숫자만 보고 결론을 뒤집는
    것을 막는다."""
    return f"{_LEASE_ARCHETYPE_LEAD_SENTENCES[archetype]} (참고로 {comparison_summary})"


def _lease_consult_question(missing: list[str]) -> str:
    """전세/월세 상담 유형(자산운용형·비용절감형·안전최우선형)을 정하는 데
    필요한 정보가 없을 때 한 번만 묻는 결합 질문. LLM 합성 없이 그대로
    보여준다(질문 문구 자체가 흔들리면 안 되므로)."""
    parts = []
    if "investment" in missing:
        parts.append(
            "목돈을 주식·사업 등에 투자해 대출 금리보다 높은 수익을 "
            "기대할 수 있는 상황이신가요?")
    if "stay" in missing:
        parts.append("이 집에서 대략 몇 년 정도 거주하실 계획이세요?")
    return (
        "전세와 월세 중 어느 쪽이 유리한지 상황에 맞게 상담해 드리려면 "
        "몇 가지가 더 필요해요. " + " ".join(parts)
        + " 잘 모르시면 '모르겠다'고 답해 주셔도 괜찮아요."
    )


def _decide_lease_archetype(
    investment_edge: str, stay_years: float | None, risk_level: str,
) -> str:
    """전세/월세 상담 유형을 결정하는 규칙(LLM이 아니라 코드가 결정한다).

    보증금 반환 위험이 높으면 다른 조건과 무관하게 원금 보호를 우선한다(C).
    그 다음으로 대출금리보다 유리한 투자 기회가 있으면 유동성을 우선한다(A).
    그 외에는 거주기간이 2년 이상 확보됐을 때만 전세로 월 비용을 낮춘다(B).
    거주기간이 짧거나 불확실하면 계약 유연성을 위해 A로 둔다.
    """
    if risk_level == "high":
        return "C"
    if investment_edge == "yes":
        return "A"
    if stay_years is not None and stay_years >= 2:
        return "B"
    return "A"


def _normalize_gugun(text: str) -> str:
    """'수원시 팔달구'(properties/user 표기)와 '수원팔달구'(KHUG 원자료
    표기)처럼 같은 지역을 다르게 적은 두 관례를 맞춘다. 뒤에 글자가 더
    있는 '시'만 지우므로 '의정부시'처럼 시 단독 지역명은 그대로 둔다."""
    text = re.sub(r"\s+", "", str(text or ""))
    return re.sub(r"시(?=.)", "", text)


def _region_deposit_risk_percentile(sido: str | None, gugun: str | None) -> float | None:
    """(sido, gugun)의 KHUG 보증사고율이 전체 시군구 중 몇 번째 백분위인지
    반환한다. region_accident_stats는 실제 데이터로 이미 DB에 적재돼 있지만
    지금까지 어떤 실행 경로에서도 조회되지 않던 테이블이다. 표기 관례가
    달라 SQL로 직접 매칭하지 않고 정규화 후 파이썬에서 비교한다(행 수가
    252건뿐이라 매번 전체를 읽어도 비용이 작다)."""
    if not sido or not gugun:
        return None
    target_gugun = _normalize_gugun(gugun)
    try:
        uri = config.DB_PATH.resolve().as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=5) as connection:
            rows = connection.execute(
                "SELECT gugun, accident_rate_pct FROM region_accident_stats "
                "WHERE sido = ?", (sido,),
            ).fetchall()
    except sqlite3.Error:
        return None
    if not rows:
        return None
    target_rate = next(
        (rate for gugun_name, rate in rows
         if _normalize_gugun(gugun_name) == target_gugun), None)
    if target_rate is None:
        return None
    all_rates = [rate for _, rate in rows]
    return sum(1 for rate in all_rates if rate <= target_rate) / len(all_rates)


def _lease_deposit_risk_level(session: dict, user: dict) -> tuple[str, dict]:
    """보증금 미반환(깡통전세) 위험 수준을 'high'/'moderate'/'low'/'unknown'으로
    분류한다. 다가구 전세 매물을 선택한 상태면 그 매물의 전세가율 확률분포
    (이미 계산된 값)를 우선 쓰고, 아니면 KHUG 시군구 사고율 백분위로
    대체한다. 근거 없이 'high'를 만들지 않는다(둘 다 없으면 unknown)."""
    report = session.get("last_property_report") or {}
    prop = report.get("property") or {}
    jeonse_ratio = (report.get("contract_safety") or {}).get("jeonse_ratio") or {}
    if (prop.get("transaction_type") == "전세"
            and "다가구" in str(prop.get("house_type") or "")
            and jeonse_ratio.get("available")):
        thresholds = jeonse_ratio.get("threshold_probabilities") or {}
        over_1_0 = float(thresholds.get("post_contract_over_1_0") or 0)
        over_0_8 = float(thresholds.get("post_contract_over_0_8") or 0)
        level = "high" if (over_1_0 > .05 or over_0_8 > .2) else "low"
        return level, {
            "source": "property_jeonse_ratio",
            "post_contract_over_1_0": over_1_0,
            "post_contract_over_0_8": over_0_8,
        }
    percentile = _region_deposit_risk_percentile(
        user.get("preferred_sido"), user.get("preferred_gugun"))
    if percentile is None:
        return "unknown", {"source": "unavailable"}
    level = "high" if percentile >= .75 else "moderate" if percentile >= .4 else "low"
    return level, {
        "source": "region_accident_stats",
        "sido": user.get("preferred_sido"), "gugun": user.get("preferred_gugun"),
        "accident_rate_percentile": round(percentile, 4),
    }


class JeonseAgent:
    def __init__(self, recommender_name: str = "rule"):
        self.llm = get_llm()
        self.map_tool = MapTool()
        self.finance_tool = FinanceTool()
        self.db_tool = PropertyDBTool()
        self.property_search = AtomicPropertySearch(map_tool=self.map_tool)
        self.safety_tool = SafetyTool()
        self.convenience_tool = ConvenienceTool()
        # 매물 주변 시설 검색(qa_poi)은 카카오 대신 ConvenienceTool이 이미 쓰는
        # NAVER API HUB 지역검색 클라이언트를 공유한다(같은 호출 한도 추적기 재사용).
        self.poi_search = self.convenience_tool.local_search
        self.text2sql = Text2SQLPipeline(self.llm, self.db_tool, self.finance_tool)
        self.recommender_name = recommender_name
        self._reco = self._load_recommender(recommender_name)

    # ------------------------------------------------------------------
    def _load_recommender(self, name):
        if name == "content":
            return R.ContentBasedRecommender()
        if name in ("ltr_lgbm", "ltr_xgb"):
            path = config.MODELS_DIR / "recommender_model.joblib"
            if path.exists():
                b = joblib.load(path)
                ltr = R.LTRRecommender(b["backend"]); ltr.model = b["model"]
                return ltr
            print("  [warn] 저장된 LTR 없음 → rule 사용")
        return R.RuleBasedRecommender()

    def new_session(self, user: dict) -> dict:
        """대화 세션 상태 초기화."""
        user = _normalize_preferred_region(user)
        return {"user": user, "pending_slots": None, "pending_tool_calls": None,
                "pending_user_text": None, "pending_info": None,
                "pending_trace": None, "last_intent": None,
                "last_qa_args": {}, "stage": "idle",
                "lease_consult": {"investment_edge": None,
                                   "planned_stay_years": None,
                                   "asked_once": False}}

    def compute_affordable_budgets(self, user: dict) -> dict:
        """'구매가능' 토글용: 거래유형별(전세/매매/월세) 자기자금+최대 대출가능액 예산.

        여러 대출 한도를 합산하지 않고 거래유형별로 가장 큰 유효 한도 하나만
        쓰는 원칙은 _classify_goal_finance와 동일하다(_classify_transaction_finance).
        """
        aff = compute_affordability(user)
        region = " ".join(
            str(value) for value in
            (user.get("preferred_sido"), user.get("preferred_gugun")) if value
        ) or None
        programs = self.finance_tool.search(
            user_income_manwon=user.get("monthly_income_manwon"),
            user_age=user.get("age"), region=region, finance_mode="eligibility",
            user_profile=user, limit=50,
        )
        jeonse_terms = ("전세자금", "전월세자금", "임차보증금", "전세보증금대출",
                       "버팀목", "중소기업취업청년")
        sale_terms = ("담보대출", "주택구입자금", "주택자금대출", "부동산담보대출",
                      "보금자리론", "디딤돌")
        jeonse_plan = _classify_transaction_finance(
            programs, aff, direct_terms=jeonse_terms,
            base_budget_manwon=float(aff.recommended_jeonse_deposit_manwon))
        sale_plan = _classify_transaction_finance(
            programs, aff, direct_terms=sale_terms,
            base_budget_manwon=float(user.get("total_asset_manwon") or 0))
        return {
            "전세": {"deposit_manwon": jeonse_plan["estimated_max_budget_manwon"],
                    "financing_plan": jeonse_plan},
            "매매": {"sale_price_manwon": sale_plan["estimated_max_budget_manwon"],
                    "financing_plan": sale_plan},
            "월세": {"monthly_rent_manwon": float(aff.max_monthly_housing_manwon),
                    "deposit_manwon": float(aff.recommended_monthly_deposit_manwon),
                    "financing_plan": None},
        }

    # ------------------------------------------------------------------
    # 진입점: 한 턴 처리
    # ------------------------------------------------------------------
    def handle(self, session: dict, text: str,
               *, direct_recommend: bool = False,
               conversation_history: list[dict] | None = None,
               consult_only: bool = False) -> dict:
        """
        사용자 발화 한 턴 처리 → 응답 dict.
        응답 status: clarify | confirm | recommendation | qa | cancelled | error

        consult_only=True(상담 채널)면 매물을 직접 검색·추천하는 의도
        (recommend/goal_*)는 실행하지 않고 지도 검색으로 안내한다. 상담
        채널은 의사결정에 필요한 질문(qa_*)에만 답하는 조언자 역할이다.
        """
        # 지도 조건 추가는 별도 버튼 흐름에서 승인한다. 상담 채널의 직접 추천은
        # 이전 CLI식 "응" 확인 상태를 이어받지 않는다.
        if direct_recommend and session.get("stage") == "awaiting_confirmation":
            session["stage"] = "idle"
            session["pending_slots"] = None
            session["pending_info"] = None
            session["pending_trace"] = None

        # 확인 대기 중이면 확인 응답 우선 처리
        if not direct_recommend and session.get("stage") == "awaiting_confirmation":
            verdict = parse_confirmation(text)
            if verdict == "yes":
                trace = copy.deepcopy(session.get("pending_trace") or {
                    "input": session.get("pending_user_text") or text,
                    "planner": session.get("pending_plan_meta", {}),
                    "tools": [], "fallbacks": []})
                trace["confirmation"] = {"user_reply": text, "verdict": "yes"}
                return self._do_recommend(
                    session, session.get("pending_user_text") or text,
                    trace)
            if verdict == "no":
                session["stage"] = "idle"
                session["pending_slots"] = None
                return {"status": "cancelled",
                        "message": "알겠어요. 조건을 다시 말씀해 주세요."}
            # modify: 새 발화로 재파싱(기존 슬롯에 병합)
            plan = self.llm.plan(text, has_prior_region=bool(session.get("pending_slots")))
            merged = dict(session.get("pending_slots") or {})
            merged.update(plan.slots)
            plan.slots = merged
            return self._route(session, plan, text, consult_only=consult_only)

        # 신규 발화
        planning_text = text
        advisor_history = copy.deepcopy(conversation_history or [])
        context_used = None
        if not advisor_history and session.get("last_intent") and re.search(
                r"그중|그럼|그러면|그거|거기서|이중|그 상품|낮은|높은|더|만\s*(?:보여|찾)|"
                r"나는|내가|제\s*경우|받을\s*수", text):
            context_used = {
                "previous_intent": session.get("last_intent"),
                "previous_qa_args": session.get("last_qa_args") or {},
            }
            planning_text = (
                "이전 대화 문맥(JSON): "
                + json.dumps(context_used, ensure_ascii=False)
                + f"\n현재 사용자의 후속 요청: {text}"
            )
        plan = self.llm.plan(
            planning_text, conversation_history=advisor_history or None)
        # 이 표현은 적정예산을 무시하라는 명시적 요청이다. 플래너 스키마 밖의
        # 실행 제어 플래그로 보존해 ATOM 단계의 자동 예산 주입을 막는다.
        if direct_recommend and re.search(
                r"예산\s*(?:은\s*)?(?:상관\s*없|무관)|"
                r"예산\s*제한\s*없|가격\s*(?:은\s*)?상관\s*없", text):
            plan.slots["_ignore_budget"] = True
        if direct_recommend and re.search(
                r"(?:보증금|전세금).*(?:가장|제일|최저).*(?:낮|싸)|"
                r"(?:가장|제일|최저).*(?:낮|싼).*(?:보증금|전세금)|"
                r"보증금\s*(?:낮은|싼)\s*순", text):
            plan.intent = "recommend"
            plan.action = "proceed"
            plan.clarify_message = None
            plan.slots["sort_by"] = "price_asc"
            plan.slots["_deposit_sort"] = True
            if not any(
                    call.get("tool") == "property_search"
                    for call in plan.tool_calls):
                plan.tool_calls.append({"tool": "property_search", "args": {}})
        if advisor_history and plan.intent == "recommend":
            previous_slots = next((
                entry.get("slots") or {}
                for entry in reversed(advisor_history)
                if entry.get("role") == "assistant" and entry.get("slots")
            ), {})
            inherited = []
            for key in (
                "transaction_type", "lease_type", "property_type",
                "region_sido", "region_gugun",
            ):
                if plan.slots.get(key) is None and previous_slots.get(key) is not None:
                    plan.slots[key] = copy.deepcopy(previous_slots[key])
                    inherited.append(key)
            if inherited:
                plan.metadata = dict(plan.metadata or {})
                plan.metadata["inherited_context_slots"] = inherited
        if direct_recommend and plan.intent == "recommend" and plan.action == "confirm":
            plan.action = "proceed"
            plan.metadata = dict(plan.metadata or {})
            plan.metadata["advisor_direct_execution"] = True
        if context_used:
            plan.metadata = dict(plan.metadata or {})
            plan.metadata["conversation_context_used"] = context_used
        if advisor_history:
            plan.metadata = dict(plan.metadata or {})
            plan.metadata["advisor_history_turns_used"] = len(advisor_history)
        return self._route(
            session, plan, text, conversation_history=advisor_history,
            consult_only=consult_only)

    # ------------------------------------------------------------------
    def _route(self, session, plan: Plan, text: str,
               conversation_history: list[dict] | None = None,
               consult_only: bool = False) -> dict:
        trace = {"input": text, "planner": {
                    "intent": plan.intent,
                    "action": plan.action,
                    "slots": copy.deepcopy(plan.slots),
                    "qa_args": copy.deepcopy(plan.qa_args),
                    "tool_calls": copy.deepcopy(plan.tool_calls),
                    "reason": plan.reason,
                    "llm": plan.metadata or {"strategy": plan.reason},
                 },
                 "tools": [], "fallbacks": []}
        if conversation_history:
            # 최종 합성 단계까지만 전달하고 API trace에는 원문 대화를 노출하지 않는다.
            trace["_advisor_conversation_history"] = conversation_history
        if consult_only and plan.intent in _PROPERTY_RECOMMENDING_INTENTS:
            session["stage"] = "idle"
            return self._finalize(text, {
                "status": "qa", "qa_type": "advisor_redirect",
                "message": _ADVISOR_REDIRECT_MESSAGE,
            }, trace, synthesize=False)
        # 금융→예산→매물의 다단계 목표 처리
        if plan.intent == "goal_financed_jeonse":
            session["last_intent"] = plan.intent
            session["last_qa_args"] = copy.deepcopy(plan.qa_args)
            return self._handle_financed_jeonse_goal(session, plan, text, trace)
        if plan.intent in {"goal_best_affordable", "goal_alternative_areas"}:
            session["last_intent"] = plan.intent
            session["last_qa_args"] = copy.deepcopy(plan.qa_args)
            return self._handle_affordable_optimization_goal(
                session, plan, text, trace,
                alternative=plan.intent == "goal_alternative_areas",
            )

        # Q&A 의도 처리
        if plan.intent.startswith("qa_"):
            session["last_intent"] = plan.intent
            session["last_qa_args"] = copy.deepcopy(plan.qa_args)
            return self._handle_qa(session, plan, text, trace)

        if plan.action == "clarify":
            session["stage"] = "idle"
            return self._finalize(text, {"status": "clarify",
                    "message": plan.clarify_message, "reason": plan.reason}, trace,
                    synthesize=False)

        # proceed(아무거나) 또는 confirm(조건 확인)
        # 통근 도구 먼저 실행(지역 후보 확보)해서 확인 화면에 반영
        slots = dict(plan.slots)
        info = self._run_pretools(plan, session["user"], slots, trace)

        if plan.action == "proceed":
            # 아무거나 → 확인 없이 바로 추천
            session["pending_slots"] = slots
            session["pending_info"] = info
            session["pending_user_text"] = text
            session["pending_plan_meta"] = plan.metadata
            session["pending_trace"] = copy.deepcopy(trace)
            return self._do_recommend(session, text, trace)

        # confirm → 조건 요약 후 확인 요청
        session["stage"] = "awaiting_confirmation"
        session["pending_slots"] = slots
        session["pending_info"] = info
        session["pending_user_text"] = text
        session["pending_plan_meta"] = plan.metadata
        session["pending_trace"] = copy.deepcopy(trace)
        aff = compute_affordability(session["user"])
        return self._finalize(text, {
            "status": "confirm",
            "message": "아래 조건으로 찾아드릴까요? (맞으면 '응', 바꾸려면 수정 내용을 말씀해 주세요)",
            "confirmed_conditions": self._summarize_slots(slots, info, aff),
            "affordability": aff.model_dump(),
            "reason": plan.reason,
        }, trace, synthesize=False)

    # ------------------------------------------------------------------
    def _run_pretools(self, plan, user, slots, trace) -> dict:
        """추천 전 실행 도구(지도 통근 등). slots를 in-place 보강."""
        info = {"commute_regions": [], "workplace": None}
        for call in plan.tool_calls:
            if call["tool"] == "map_regions_within":
                wp = LANDMARKS.get(str(call["args"].get("landmark")).lower()) \
                    if call["args"].get("landmark") else None
                mins = call["args"].get("minutes", 30)
                if wp:
                    info["workplace"] = wp
                    try:
                        regs = self.map_tool.regions_within(wp, mins, "transit")
                        trace["tools"].append({"tool": "map_regions_within", "ok": True,
                                               "result_count": len(regs)})
                    except Exception as exc:
                        regs = []
                        trace["tools"].append({"tool": "map_regions_within", "ok": False,
                                               "error": self._error_text(exc)})
                        trace["fallbacks"].append("지도 지역 계산 실패: 지역 조건 없이 계속")
                    info["commute_regions"] = regs
                    if regs:
                        slots["region_gugun"] = [r["gugun"] for r in regs]
                        slots["region_sido"] = regs[0]["sido"]
                        # 통근시간 맵(ATOM predicate용)은 매물 조회 후 계산하므로 여기선 상한만
        # 지역 기본값(사용자 선호)
        if "region_sido" not in slots and user.get("preferred_sido"):
            slots["region_sido"] = user["preferred_sido"]
        if "region_gugun" not in slots and user.get("preferred_gugun"):
            slots["region_gugun"] = [user["preferred_gugun"]]
        return info

    def _summarize_slots(self, slots, info, aff) -> list[str]:
        aset = A.build_atoms(self._slots_for_atoms(slots, info, aff, {}))
        lines = aset.describe()
        # 매물별 계산이 필요한 조건은 여기서 텍스트로 안내
        if slots.get("min_safety_score") is not None:
            lines.append(f"치안 안전점수 ≥ {slots['min_safety_score']:.0f} (주변 300m 인프라)")
        if slots.get("min_convenience_score") is not None:
            lines.append(f"생활편의점수 ≥ {slots['min_convenience_score']:.0f} (주변 500m 시설)")
        if info.get("commute_regions"):
            regs = ", ".join(f"{r['gugun']}({r['minutes']:.0f}분)"
                             for r in info["commute_regions"][:5])
            lines.append(f"통근 후보 지역: {regs}")
        if not lines:
            lines = ["(특별한 조건 없음 — 전체에서 추천)"]
        return lines

    def _slots_for_atoms(self, slots, info, aff, commute_map,
                         safety_map=None, conv_map=None) -> dict:
        """추천 슬롯 → ATOM 빌더용 슬롯으로 변환(예산 상한 자동 주입)."""
        s = dict(slots)
        # 예산 상한 자동(사용자가 명시 안 했으면 적정예산으로)
        lease = s.get("lease_type")
        if "max_deposit_manwon" not in s and not s.get("_ignore_budget"):
            if lease == "전세":
                s["max_deposit_manwon"] = round(aff.recommended_jeonse_deposit_manwon * 1.3, 0)
            elif lease == "월세":
                s["max_deposit_manwon"] = round(aff.recommended_monthly_deposit_manwon * 2, 0)
            else:
                s["max_deposit_manwon"] = round(max(
                    aff.recommended_jeonse_deposit_manwon,
                    aff.recommended_monthly_deposit_manwon) * 1.3, 0)
        if (lease == "월세" and "max_monthly_rent_manwon" not in s
                and not s.get("_ignore_budget")):
            s["max_monthly_rent_manwon"] = round(aff.max_monthly_housing_manwon * 1.2, 0)
        if commute_map:
            s["_commute_minutes_by_id"] = commute_map
        if safety_map:
            s["_safety_score_by_id"] = safety_map
        if conv_map:
            s["_convenience_score_by_id"] = conv_map
        return s

    # ------------------------------------------------------------------
    _COMMUTE_SLOT_KEYS = (
        "max_commute_min", "workplace_landmark", "_workplace_landmark", "commute_mode")
    _NUMERIC_ATOM_FIELDS = {
        "deposit_manwon", "sale_price_manwon", "monthly_rent_manwon",
        "maintenance_fee_manwon", "area_m2", "building_age_years"}

    def _atomic_property_search(self, slots: dict, *, limit: int = 500,
                                rental_only: bool = False, relax_on_empty: bool = False,
                                sort_by: str = "recommended") -> tuple[list[dict], dict]:
        """Atomic·병렬·스케줄 기반 매물 조회(AtomicPropertySearch 재사용).

        통근시간은 여기서 hard 필터로 만들지 않는다 — ATOM 소프트 스코어링에서
        계속 처리하므로(_do_recommend의 commute_map), 그 결과를 바꾸지 않기 위해
        관련 슬롯을 조건 변환 전에 제거한다. source_text도 빈 문자열로 고정해
        atoms_from_slots의 정규식 기반 조건(반려동물·주차 등)이 새로 생기지 않게 한다.
        """
        clean_slots = {k: v for k, v in slots.items() if k not in self._COMMUTE_SLOT_KEYS}
        gugun = clean_slots.get("region_gugun")
        if gugun is not None and not isinstance(gugun, (list, tuple)):
            clean_slots["region_gugun"] = [gugun]

        atoms, _notes = atoms_from_slots(clean_slots, "", self.map_tool)
        if rental_only and not any(a["field"] == "transaction_type" for a in atoms):
            atoms.append(make_atom(
                field="transaction_type", operator="in", value=["전세", "월세"],
                label="전세/월세 매물만(매매 제외)", source="AI 대화"))

        active_atoms = list(atoms)
        result = self.property_search.search(active_atoms, limit=limit, sort_by=sort_by)
        relax_notes: list[str] = []
        if relax_on_empty and not result["properties"]:
            if any(a["field"] == "gugun" for a in active_atoms):
                active_atoms = [a for a in active_atoms if a["field"] != "gugun"]
                result = self.property_search.search(
                    active_atoms, limit=limit, sort_by=sort_by)
                relax_notes.append("시군구 조건 완화 후 재조회")
            # 거래유형(전세/월세/매매)이 명시된 경우에만 시/도까지 완화한다.
            # 거래유형이 없는 완전히 막연한 요청에서 시/도까지 지우면 서로 다른
            # 거래유형이 뒤섞여 비교 불가능한 후보가 섞이므로 완화하지 않는다.
            if (not result["properties"]
                    and any(a["field"] == "sido" for a in active_atoms)
                    and any(a["field"] == "transaction_type" for a in active_atoms)):
                active_atoms = [a for a in active_atoms if a["field"] != "sido"]
                result = self.property_search.search(
                    active_atoms, limit=limit, sort_by=sort_by)
                relax_notes.append("시도 조건 완화 후 재조회")
            if (not result["properties"]
                    and any(a["field"] in self._NUMERIC_ATOM_FIELDS for a in active_atoms)):
                active_atoms = [a for a in active_atoms
                               if a["field"] not in self._NUMERIC_ATOM_FIELDS]
                result = self.property_search.search(active_atoms, limit=limit, sort_by=sort_by)
                relax_notes.append("후보 확보를 위해 수치 필터를 완화하고 ATOM에서 재검증")

        rows = result["properties"]
        input_filters = {k: v for k, v in clean_slots.items() if not str(k).startswith("_")}
        if "region_sido" in input_filters:
            input_filters["sido"] = input_filters.pop("region_sido")
        if "region_gugun" in input_filters:
            input_filters["gugun"] = input_filters.pop("region_gugun")
        input_filters["limit"] = limit

        trace_dict = {
            "target": "properties",
            "strategy": "atomic_scheduled_search",
            "request_summary": "ATOM 분해 + PortfolioScheduler 병렬 조회로 매물 후보 검증",
            "input_filters": input_filters,
            "atoms": [{"field": a["field"], "operator": a["operator"],
                      "value": a["value"], "label": a["label"]} for a in active_atoms],
            "validation": "passed_atomic_scheduled_search",
            "row_count": len(rows),
            "final_sql": result["trace"].get("final_sql"),
            "parameters": result["trace"].get("final_parameters"),
            "fallback": False,
            "fallback_reason": None,
            "relaxation_notes": relax_notes,
            "atomic_search_trace": result["trace"],
        }
        return rows, trace_dict

    # ------------------------------------------------------------------
    def _do_recommend(self, session, user_text: str, trace: dict) -> dict:
        user = session["user"]
        slots = session.get("pending_slots") or {}
        info = session.get("pending_info") or {"commute_regions": [], "workplace": None}
        aff = compute_affordability(user)
        session["stage"] = "idle"

        # 1) ATOM 분해 + 병렬·스케줄 기반 매물 조회(AtomicPropertySearch 재사용)
        rental_only = bool(slots.get("_deposit_sort") and not (
            slots.get("transaction_type") or slots.get("lease_type")))
        # PropertyDBTool.build_query의 옛 기본 정렬(deposit_manwon ASC)과
        # 동일하게, risk_asc 요청이 없으면 저가순으로 후보를 가져온다.
        sort_by = "risk_asc" if slots.get("sort_by") == "risk_asc" else "price_asc"
        rows, prop_trace = self._atomic_property_search(
            slots, limit=500, rental_only=rental_only,
            relax_on_empty=True, sort_by=sort_by)
        trace["tools"].append({"tool": "property_text2sql", **prop_trace})
        trace["fallbacks"].extend(prop_trace.get("relaxation_notes", []))
        candidates = pd.DataFrame(rows)

        if slots.get("_deposit_sort") and "transaction_type" in candidates:
            candidates = candidates[candidates["transaction_type"] != "매매"].copy()
        if candidates.empty:
            return self._finalize(user_text, {"status": "no_result",
                    "message": "조건에 맞는 매물을 찾지 못했어요. 예산이나 지역을 넓혀보세요.",
                    "affordability": aff.model_dump()}, trace)

        # 2) 통근시간 맵 계산(ATOM predicate용)
        commute_map = {}
        if info.get("workplace") is not None and slots.get("max_commute_min"):
            wp = info["workplace"]
            for _, r in candidates.iterrows():
                try:
                    t = self.map_tool.travel_time((r["lat"], r["lng"]), wp, "transit")
                    commute_map[r["property_id"]] = t["minutes"]
                except Exception as exc:
                    trace["fallbacks"].append(
                        f"일부 통근시간 계산 실패: {self._error_text(exc)}")
                    break
            trace["tools"].append({"tool": "map_travel_time", "ok": bool(commute_map),
                                   "result_count": len(commute_map)})

        # 2b) 치안/편의 점수 맵(요청 시에만 계산 — 비용 절감)
        safety_map, conv_map = {}, {}
        need_safety = slots.get("min_safety_score") is not None
        need_conv = slots.get("min_convenience_score") is not None
        if need_safety or need_conv:
            # 후보가 많으면 상위 N만 계산(성능). ATOM 분류 전이므로 예산/안전 통과분에 한정.
            calc_pool = candidates.head(120)
            for _, r in calc_pool.iterrows():
                if need_safety:
                    safety_map[r["property_id"]] = \
                        self.safety_tool.assess(r["lat"], r["lng"])["safety_score"]
                if need_conv:
                    conv_map[r["property_id"]] = \
                        self.convenience_tool.assess(r["lat"], r["lng"])["convenience_score"]

        # 3) ATOM 분해 + 매물별 만족도
        atom_slots = self._slots_for_atoms(slots, info, aff, commute_map,
                                           safety_map, conv_map)
        aset = A.build_atoms(atom_slots)
        scored = A.score_by_atoms(candidates, aset)

        # 조건 완화 메모(위험도는 필터가 아니므로 임계값 완화가 없다)
        relaxed_notes = []
        if scored.empty:
            # 모든 매물이 hard 위반 → hard 완화(안전/예산만 유지) 폴백
            for soft_key in ("region_gugun", "region_sido", "max_commute_min"):
                atom_slots.pop(soft_key, None)
            aset = A.build_atoms(atom_slots)
            scored = A.score_by_atoms(candidates, aset)
        if scored.empty:
            return self._finalize(user_text, {"status": "no_result",
                    "message": "필수 조건(예산·지역)을 만족하는 매물이 없어요. 조건을 완화해 보세요.",
                    "affordability": aff.model_dump()}, trace)

        # 4) 그룹별(누락 0/1/2) 추천 순위화
        groups = A.group_by_missing(scored, max_missing=2)
        context = {"affordability": aff, "workplace": info.get("workplace"),
                   "sort_by": slots.get("sort_by", "recommended")}

        grouped_out = {}
        for k in sorted(groups):
            g = groups[k]
            ranked = self._reco.recommend(
                g, context, top_k=max(5, min(len(g), 500)))
            if slots.get("sort_by") == "price_asc":
                if slots.get("_deposit_sort"):
                    ranked = ranked.sort_values(
                        ["deposit_manwon", "monthly_rent_manwon", "score"],
                        ascending=[True, True, False], na_position="last",
                    )
                else:
                    ranked = ranked.assign(
                        _sort_price=ranked.apply(
                            lambda row: (
                                float(row.get("sale_price_manwon") or
                                      row.get("asking_price_manwon") or 0)
                                if row.get("transaction_type") == "매매"
                                else float(row.get("deposit_manwon") or 0)
                            ),
                            axis=1,
                        )
                    ).sort_values(
                        ["_sort_price", "score"],
                        ascending=[True, False], na_position="last",
                    )
                ranked = ranked.head(5)
            recs = []
            for _, r in ranked.iterrows():
                recs.append(self._format_rec(r))
            grouped_out[k] = recs

        return self._finalize(user_text, {
            "status": "recommendation",
            "affordability": aff.model_dump(),
            "atoms": aset.describe(),
            "commute_regions": info.get("commute_regions", []),
            "groups": grouped_out,   # {0: [완전만족], 1: [1개누락], 2: [2개누락]}
        }, trace)

    def _format_rec(self, r) -> dict:
        fs = r.get("fraud_score")
        score = r.get("score", 0)
        score = 0.0 if pd.isna(score) else float(score)
        def finite(value, default=0.0):
            try:
                number = float(value)
                return number if math.isfinite(number) else float(default)
            except (TypeError, ValueError):
                return float(default)
        return {
            "property_id": r["property_id"], "sido": r["sido"], "gugun": r["gugun"],
            "is_synthetic": bool(r.get("is_synthetic", True)),
            "synthetic_notice": r.get("synthetic_notice") or "합성 매물 데이터",
            "lease_type": r["lease_type"],
            "transaction_type": r.get("transaction_type"),
            "property_type": r.get("property_type") or r.get("house_type"),
            "deposit_manwon": finite(r.get("deposit_manwon")),
            "sale_price_manwon": finite(
                r.get("sale_price_manwon") or r.get("asking_price_manwon")),
            "monthly_rent_manwon": finite(r.get("monthly_rent_manwon")),
            "maintenance_fee_manwon": finite(r.get("maintenance_fee_manwon")),
            "fraud_score": (None if fs is None or pd.isna(fs)
                            else round(float(fs), 3)),
            "missing_conditions": r.get("missing_desc", []),
            "score": round(score, 3),
        }

    # ------------------------------------------------------------------
    # Q&A 라우팅
    # ------------------------------------------------------------------
    def _handle_financed_jeonse_goal(self, session, plan: Plan,
                                     text: str, trace: dict) -> dict:
        """금융 RAG 결과로 예산을 계산한 뒤 해당 예산의 최고가 전세를 추천한다."""
        user = session["user"]
        aff = compute_affordability(user)
        region = plan.slots.get("region_sido") or user.get("preferred_sido")
        gugun = plan.slots.get("region_gugun") or user.get("preferred_gugun")
        workflow = {
            "goal": "금융상품을 활용한 감당 가능한 전세보증금 최대화",
            "steps": [],
        }
        trace["workflow"] = workflow

        # 1) 사용자의 나이·연소득·지역을 적용해 금융 정책을 먼저 읽는다.
        programs, finance_trace = self.text2sql.search_finance(
            text, user, region=region, finance_mode="eligibility", limit=50,
        )
        trace["tools"].append({"tool": "finance_text2sql", **finance_trace})
        if finance_trace.get("fallback"):
            trace["fallbacks"].append("LLM 금융 SQL 대신 파라미터 쿼리 사용")
        workflow["steps"].append({
            "step": 1, "action": "eligible_finance_rag",
            "result": f"금융정책 {len(programs)}건 검토",
        })

        # 2) 검색 결과를 실제 보증금 증액 수단과 비용 절감 수단으로 구분하고
        #    상환 가능한 단일 대출 한도로 최대 전세예산을 계산한다.
        relevant_programs, financing_plan = _classify_goal_finance(programs, aff)
        budget = float(financing_plan["estimated_max_deposit_manwon"])
        trace["tools"].append({
            "tool": "financing_budget_calculator",
            "strategy": "single_best_direct_jeonse_loan",
            "input": {
                "base_budget_manwon": financing_plan["base_jeonse_budget_manwon"],
                "monthly_housing_cap_manwon": aff.max_monthly_housing_manwon,
                "reviewed_program_count": len(programs),
            },
            "result": financing_plan,
        })
        workflow["steps"].append({
            "step": 2, "action": "calculate_maximum_fundable_deposit",
            "result": f"추정 최대 전세보증금 {budget:,.0f}만원",
        })

        # 3) 계산된 예산을 WHERE 상한으로 넣고 전세 매물을 조회한다.
        atoms_slots = {
            "transaction_type": "전세", "lease_type": "전세",
            "max_deposit_manwon": budget,
        }
        if region:
            atoms_slots["region_sido"] = region
        if gugun:
            atoms_slots["region_gugun"] = gugun
        rows, property_trace = self._atomic_property_search(
            atoms_slots, limit=500, relax_on_empty=False, sort_by="recommended")
        trace["tools"].append({"tool": "property_text2sql", **property_trace})
        trace["fallbacks"].extend(property_trace.get("relaxation_notes", []))

        # SQL 표현과 무관하게 목표 함수(상한 내 보증금 최대)를 마지막에 재검증한다.
        candidates = [row for row in rows
                      if row.get("deposit_manwon") is not None
                      and float(row["deposit_manwon"]) <= budget]
        candidates.sort(key=lambda row: (
            -float(row.get("deposit_manwon") or 0),
            float(row.get("fraud_score")) if row.get("fraud_score") is not None else 1.0,
        ))
        recommendations = []
        for row in candidates[:5]:
            formatted = self._format_rec(row)
            formatted["remaining_budget_manwon"] = round(
                budget - formatted["deposit_manwon"], 1)
            recommendations.append(formatted)
        trace["tools"].append({
            "tool": "goal_ranker",
            "objective": "deposit_manwon 최대화, 동일 가격이면 fraud_score 최소화",
            "candidate_count": len(candidates),
            "selected_count": len(recommendations),
        })
        workflow["steps"].append({
            "step": 3, "action": "property_rag_and_goal_ranking",
            "result": f"예산 내 후보 {len(candidates)}건 중 {len(recommendations)}건 선별",
        })

        if financing_plan["selected_program_name"]:
            message = (
                f"'{financing_plan['selected_program_name']}' 등 직접 전세자금 후보를 반영한 "
                f"추정 예산 {budget:,.0f}만원 안에서 보증금이 가장 높은 전세 매물을 골랐습니다."
            )
            if financing_plan.get("selected_program_repayment_capacity_verified") is False:
                message += (
                    " 이 상품은 금리 정보가 DB에 없어 상환능력은 검증하지 못했고, "
                    "금융기관이 공시한 한도를 그대로 반영했습니다. 실제 한도는 심사 후 확정됩니다."
                )
        else:
            message = (
                "현재 금융 DB에는 보증금을 늘릴 수 있는 직접 전세자금 상품이 없어 "
                f"자기자금 기준 {budget:,.0f}만원 안에서 최고가 전세 후보를 골랐습니다."
            )
        return self._finalize(text, {
            "status": "recommendation",
            "recommendation_mode": "financed_jeonse_goal",
            "message": message,
            "affordability": aff.model_dump(),
            "financing_plan": financing_plan,
            "finance_programs": relevant_programs,
            "finance_screening": {
                "reviewed_count": len(programs),
                "relevant_count": len(relevant_programs),
                "excluded_count": len(programs) - len(relevant_programs),
                "exclusion_reason": "전세보증금 증액 또는 전세 관련 비용 절감과 직접 관련 없음",
            },
            "groups": {0: recommendations},
            "commute_regions": [],
        }, trace)

    @staticmethod
    def _reference_point(session, user) -> tuple[float, float, str, int]:
        """선택 매물이 있으면 그 좌표/주소(반경 1km), 없으면 사용자 선호지역
        중심좌표(반경 5km, 텍스트 검색이라 동 단위 정밀도가 없어 더 넓게 잡음)로 폴백."""
        report = session.get("last_property_report") or {}
        prop = report.get("property") or {}
        if prop.get("lat") is not None and prop.get("lng") is not None:
            context = " ".join(
                str(part) for part in
                (prop.get("sido"), prop.get("gugun"), prop.get("dong")) if part
            )
            return float(prop["lat"]), float(prop["lng"]), context, 1000
        sido = user.get("preferred_sido") or "경기"
        gugun = user.get("preferred_gugun") or "수원시 팔달구"
        lat, lng = SIDO_GUGUN_CENTROIDS.get((sido, gugun), (37.4784, 126.9516))
        return lat, lng, f"{sido} {gugun}".strip(), 5000

    def _handle_qa(self, session, plan: Plan, text: str, trace: dict) -> dict:
        user = session["user"]
        intent = plan.intent
        args = plan.qa_args

        if intent == "qa_finance":
            progs, sql_trace = self.text2sql.search_finance(
                text, user, category=args.get("category"),
                max_rate_pct=args.get("max_rate_pct"),
                product_kind=args.get("product_kind"),
                region=plan.slots.get("region_sido"),
                finance_mode=args.get("finance_mode", "catalog"), limit=10)
            trace["tools"].append({"tool": "finance_text2sql", **sql_trace})
            if sql_trace.get("fallback"):
                trace["fallbacks"].append("LLM 금융 SQL 대신 파라미터 쿼리 사용")
            finance_mode = args.get("finance_mode", "catalog")
            message = ("입력하신 프로필로 1차 조회한 청년 주거금융 후보예요."
                       if finance_mode == "eligibility" else
                       "현재 금융서비스 DB에 저장된 청년 주거금융 제도예요.")
            return self._finalize(text, {"status": "qa", "qa_type": "finance",
                    "finance_mode": finance_mode, "message": message,
                    "programs": progs}, trace)

        if intent == "qa_affordability":
            aff = compute_affordability(user)
            return self._finalize(text, {"status": "qa", "qa_type": "affordability",
                    "message": "소득·자산 기준 적정 주거 예산이에요.",
                    "affordability": aff.model_dump()}, trace)

        if intent == "qa_contract":
            cc = contract_checklist(args.get("lease_type", "전세"), is_multi_family=True)
            report = session.get("last_property_report") or {}
            contract = report.get("contract_safety")
            if contract:
                trace["tools"].append({
                    "tool": "selected_property_contract_safety",
                    "property_id": (report.get("property") or {}).get("property_id"),
                    "evidence": [
                        "fraud_risk", "senior_deposit",
                        "owner_asset_ratio", "guarantee_review",
                    ],
                })
            return self._finalize(text, {"status": "qa", "qa_type": "contract",
                    "message": (
                        "선택한 매물의 위험모델 결과와 계약 체크리스트를 함께 확인했습니다."
                        if contract else
                        "선택한 매물이 없어 일반 계약 체크리스트를 안내합니다."
                    ),
                    "checklist": cc, "contract_safety": contract,
                    "property": report.get("property") if contract else None}, trace)

        if intent == "qa_lease_compare":
            consult = session.setdefault("lease_consult", {
                "investment_edge": None, "planned_stay_years": None,
                "asked_once": False,
            })
            if args.get("investment_edge") in ("yes", "no"):
                consult["investment_edge"] = args["investment_edge"]
            if args.get("planned_stay_years") is not None:
                try:
                    consult["planned_stay_years"] = max(
                        0.5, min(float(args["planned_stay_years"]), 30))
                except (TypeError, ValueError):
                    pass

            preference = normalize_preferences(user.get("preferences"))
            investment_edge = consult["investment_edge"]
            investment_edge_is_default = False
            if investment_edge is None and preference.get("mode") in ("growth", "stable"):
                investment_edge = "yes" if preference["mode"] == "growth" else "no"
                investment_edge_is_default = True
            risk_level, risk_evidence = _lease_deposit_risk_level(session, user)

            # 위험이 높으면 무엇을 답하든 결론이 C로 고정되고, 투자 기회가
            # 있으면 거주기간과 무관하게 A로 고정된다 — 이런 경우까지 굳이
            # 되묻지 않는다. 거주기간은 "투자 기회 없음(또는 미확인) + 위험
            # 낮음"일 때만 B/A를 가르는 데 실제로 필요하다.
            missing = []
            if risk_level != "high":
                if investment_edge is None:
                    missing.append("investment")
                if investment_edge != "yes" and consult["planned_stay_years"] is None:
                    missing.append("stay")

            if missing and not consult["asked_once"]:
                consult["asked_once"] = True
                return self._finalize(text, {
                    "status": "qa", "qa_type": "lease_compare",
                    "needs_clarification": True,
                    "message": _lease_consult_question(missing),
                }, trace, synthesize=False)

            if investment_edge is None:
                investment_edge = "no"
            archetype = _decide_lease_archetype(
                investment_edge, consult["planned_stay_years"], risk_level)
            comparison = self._lease_monte_carlo_comparison(
                session, trace, stay_years=consult["planned_stay_years"])
            trace["tools"].append({
                "tool": "lease_consulting_archetype",
                "archetype": archetype,
                "investment_edge": investment_edge,
                "investment_edge_is_default": investment_edge_is_default,
                "planned_stay_years": consult["planned_stay_years"],
                "risk_level": risk_level,
            })
            return self._finalize(text, {
                "status": "qa", "qa_type": "lease_compare",
                "message": _lease_consult_message(archetype, comparison["summary"]),
                "lease_monte_carlo": comparison,
                "lease_consult": {
                    "archetype": archetype,
                    "archetype_label": _LEASE_ARCHETYPE_LABELS[archetype],
                    "recommended_transaction": _LEASE_ARCHETYPE_RECOMMENDATION[archetype],
                    "investment_edge": investment_edge,
                    "investment_edge_is_default": investment_edge_is_default,
                    "planned_stay_years": consult["planned_stay_years"],
                    "risk_level": risk_level,
                    "risk_evidence": risk_evidence,
                },
            }, trace)

        if intent == "qa_cost":
            # 마지막 추천 매물이 있으면 그것, 없으면 적정예산 기준 예시
            aff = compute_affordability(user)
            cb = cost_breakdown(
                deposit_manwon=aff.recommended_monthly_deposit_manwon,
                monthly_rent_manwon=aff.recommended_monthly_rent_manwon,
                maintenance_fee_manwon=7, onetime_fee_manwon=60)
            return self._finalize(text, {"status": "qa", "qa_type": "cost", "breakdown": cb,
                    "note": "특정 매물 기준으로 계산하려면 매물을 선택해 주세요."}, trace)

        if intent == "qa_poi":
            lat, lng, context, radius_m = self._reference_point(session, user)
            category = args.get("category", "subway")
            keyword = CATEGORY_KO.get(category, category)
            naver_result = self.poi_search.search(lat, lng, context, keyword, radius_m)
            r = {
                "category": category,
                "count": naver_result.get("count"),
                "nearest_m": (naver_result["places"][0]["distance_m"]
                             if naver_result.get("places") else None),
                "places": naver_result.get("places", []),
                "source": naver_result.get("source"),
            }
            trace["tools"].append({
                "tool": "poi_search", "provider": "naver_api_hub_local",
                "ok": bool(naver_result.get("available")),
            })
            return self._finalize(text, {"status": "qa", "qa_type": "poi", "result": r,
                    "note": "특정 매물 주변으로 조회하려면 매물을 선택해 주세요."}, trace)

        if intent == "qa_market":
            report = session.get("last_property_report") or {}
            if report.get("property") and report.get("forecast"):
                forecast = report["forecast"]
                market = {
                    "property": report["property"],
                    "annual_growth_rate": forecast.get("annual_growth_rate"),
                    "annual_low": forecast.get("annual_low"),
                    "annual_high": forecast.get("annual_high"),
                    "direction": (
                        "상승" if float(forecast.get("annual_growth_rate") or 0) > 0
                        else "하락" if float(forecast.get("annual_growth_rate") or 0) < 0
                        else "보합"
                    ),
                    "price_history": forecast.get("price_history"),
                    "news": forecast.get("news"),
                    "market_assessment": forecast.get("market_assessment"),
                    "model_version": forecast.get("model_version"),
                }
                trace["tools"].append({
                    "tool": "selected_property_market_forecast",
                    "property_id": report["property"].get("property_id"),
                    "time_series_available": bool(
                        (forecast.get("price_history") or {}).get("available")),
                })
                return self._finalize(text, {
                    "status": "qa", "qa_type": "market",
                    "message": "선택한 매물 지역의 실거래 시계열과 뉴스 조정 전망입니다.",
                    "market_outlook": market,
                }, trace)
            return self._finalize(text, {"status": "qa", "qa_type": "market",
                    "message": (
                        "어느 동네를 뜻하는지 확인하려면 지도에서 매물 하나를 먼저 선택해 "
                        "주세요. 선택 후 실거래 시계열과 관련 뉴스로 전망하겠습니다."
                    ),
                    "needs_property_selection": True}, trace)

        if intent == "qa_buy_or_wait":
            analysis = self._buy_or_wait(session)
            trace["tools"].append({
                "tool": "buy_now_vs_wait",
                "status": analysis.get("status"),
                "horizons_years": [1, 2],
            })
            return self._finalize(text, {
                "status": "qa", "qa_type": "buy_or_wait",
                "message": analysis["message"],
                "buy_or_wait": analysis,
            }, trace)

        if intent == "qa_registry":
            return self._finalize(text, {"status": "qa", "qa_type": "registry",
                    "guide": registry_check_guide(text)}, trace)

        if intent == "qa_safety":
            lat, lng, context, _ = self._reference_point(session, user)
            r = self.safety_tool.assess(lat, lng, context=context)
            trace["tools"].append({"tool": "safety_assess", "ok": True})
            return self._finalize(text, {"status": "qa", "qa_type": "safety", "result": r,
                    "note": "특정 매물 주변으로 조회하려면 매물을 선택해 주세요."}, trace)

        if intent == "qa_convenience":
            lat, lng, context, _ = self._reference_point(session, user)
            r = self.convenience_tool.assess(lat, lng, context=context)
            trace["tools"].append({"tool": "convenience_assess", "ok": True})
            return self._finalize(text, {"status": "qa", "qa_type": "convenience", "result": r,
                    "note": "특정 매물 주변으로 조회하려면 매물을 선택해 주세요."}, trace)

        return self._finalize(text, {"status": "qa", "qa_type": "unknown",
                "message": "그 부분은 아직 도와드리기 어려워요. 매물 추천/계약/시세/금융 관련으로 물어봐 주세요."}, trace)

    @staticmethod
    def _diverse_property_pool(rows: list[dict], limit: int = 60) -> list[dict]:
        """한 동의 매물만 먼저 잘리지 않도록 동별 round-robin 후보를 만든다."""
        buckets: dict[str, list[dict]] = {}
        for row in rows:
            buckets.setdefault(str(row.get("dong") or "기타"), []).append(row)
        result: list[dict] = []
        while len(result) < limit and any(buckets.values()):
            for dong in sorted(buckets):
                if buckets[dong] and len(result) < limit:
                    result.append(buckets[dong].pop(0))
        return result

    def _handle_affordable_optimization_goal(
        self, session: dict, plan: Plan, text: str, trace: dict,
        *, alternative: bool,
    ) -> dict:
        """금융상품→매물 교집합→MILP/Pareto 대표점의 실제 상담 경로."""
        user = session["user"]
        region = plan.slots.get("region_sido") or user.get("preferred_sido")
        gugun = plan.slots.get("region_gugun") or user.get("preferred_gugun")
        # The LLM can return a broad/common administrative name such as
        # "경기도/수원시", while this prototype DB scope is stored as
        # "경기/수원시 팔달구". Narrow a containing parent scope to the
        # session's real DB scope before exact-match Text-to-SQL is executed.
        sido_aliases = {
            "서울특별시": "서울", "부산광역시": "부산", "대구광역시": "대구",
            "인천광역시": "인천", "광주광역시": "광주", "대전광역시": "대전",
            "울산광역시": "울산", "세종특별자치시": "세종",
            "경기도": "경기", "강원특별자치도": "강원",
            "충청북도": "충북", "충청남도": "충남",
            "전북특별자치도": "전북", "전라남도": "전남",
            "경상북도": "경북", "경상남도": "경남",
            "제주특별자치도": "제주",
        }
        region = sido_aliases.get(str(region), region)
        preferred_sido = sido_aliases.get(
            str(user.get("preferred_sido")), user.get("preferred_sido"))
        if preferred_sido and region == preferred_sido:
            region = preferred_sido
        requested_gugun = (
            list(gugun) if isinstance(gugun, (list, tuple, set))
            else ([gugun] if gugun else [])
        )
        preferred_gugun = str(user.get("preferred_gugun") or "").strip()
        def district_terms(value) -> set[str]:
            terms = set()
            for token in str(value or "").split():
                terms.add(token)
                if token.endswith(("시", "군", "구")) and len(token) > 1:
                    terms.add(token[:-1])
            return terms

        preferred_terms = district_terms(preferred_gugun)
        if preferred_gugun and any(
            preferred_gugun == str(value).strip()
            or preferred_gugun.startswith(f"{str(value).strip()} ")
            or bool(preferred_terms & district_terms(value))
            for value in requested_gugun
        ):
            gugun = [preferred_gugun]
        atoms_slots: dict = {}
        transaction = plan.slots.get("transaction_type")
        initial_transactions = [
            value for value in (user.get("transaction_types") or [])
            if value in {"매매", "전세", "월세"}
        ]
        if not transaction and len(initial_transactions) == 1:
            transaction = initial_transactions[0]
        if transaction:
            atoms_slots.update(
                transaction_type=transaction, lease_type=transaction)
        if region:
            atoms_slots["region_sido"] = region
        if gugun:
            atoms_slots["region_gugun"] = gugun
        current_intersection = list(
            (session.get("map_ui") or {}).get("last_search_properties") or [])
        if current_intersection:
            rows = current_intersection
            property_trace = {
                "strategy": "current_map_intersection",
                "row_count": len(rows),
                "validation": "already_validated_initial_and_ai_intersection",
                "fallback": False,
            }
        else:
            rows, property_trace = self._atomic_property_search(
                atoms_slots, limit=500, relax_on_empty=False, sort_by="recommended")
        if initial_transactions and not plan.slots.get("transaction_type"):
            rows = [
                row for row in rows
                if row.get("transaction_type") in initial_transactions]
        initial_house_types = set(user.get("house_types") or [])
        if initial_house_types:
            rows = [
                row for row in rows
                if (row.get("house_type") in initial_house_types
                    or row.get("property_type") in initial_house_types)]
        for field in (
            "max_deposit_manwon", "max_sale_price_manwon",
            "max_monthly_rent_manwon", "max_maintenance_manwon",
        ):
            limit_value = user.get(field)
            if limit_value is None:
                continue
            property_field = {
                "max_deposit_manwon": "deposit_manwon",
                "max_sale_price_manwon": "sale_price_manwon",
                "max_monthly_rent_manwon": "monthly_rent_manwon",
                "max_maintenance_manwon": "maintenance_fee_manwon",
            }[field]
            rows = [
                row for row in rows
                if float(row.get(property_field) or 0) <= float(limit_value)]
        trace["tools"].append({"tool": "property_text2sql", **property_trace})

        excluded_dong = None
        if alternative:
            current = (
                (session.get("last_property_report") or {}).get("property")
                or (session.get("last_recommended_properties") or [None])[0]
                or {}
            )
            excluded_dong = current.get("dong")
            if excluded_dong:
                rows = [row for row in rows if row.get("dong") != excluded_dong]
        pool = self._diverse_property_pool(rows, limit=60)

        finance_region = " ".join(str(value) for value in (region, gugun)
                                  if value)
        programs = self.finance_tool.search(
            user_income_manwon=user.get("monthly_income_manwon"),
            user_age=user.get("age"), region=finance_region or None,
            finance_mode="eligibility", user_profile=user, limit=50,
        )
        trace["tools"].append({
            "tool": "finance_search", "finance_mode": "eligibility",
            "row_count": len(programs),
        })
        optimization = optimize_housing_choices(
            pool, programs, user,
            normalize_preferences(user.get("preferences")),
            {"horizon_years": 10},
        )
        trace["tools"].append({
            "tool": "pareto_milp_optimizer",
            "property_count": len(pool), "finance_count": len(programs),
            "status": optimization.get("status"),
            "candidate_count": optimization.get("candidates_evaluated", 0),
            "pareto_count": optimization.get("pareto_candidate_count", 0),
        })
        trace["workflow"] = {
            "goal": (
                "같은 예산의 대안 동네 추천" if alternative
                else "대출 포함 감당 가능한 최적 매물 추천"
            ),
            "steps": [
                "eligible_finance_rag", "property_intersection_max_60",
                "property_finance_loan_grid", "pareto_front", "milp_representatives",
            ],
        }
        if optimization.get("status") != "ok":
            return self._finalize(text, {
                "status": "no_result",
                "message": optimization.get("message"),
                "optimization": optimization,
                "searched_property_count": len(pool),
            }, trace)

        by_id = {str(row.get("property_id")): row for row in pool}
        recommendations = []
        for representative in optimization.get("representatives") or []:
            row = by_id.get(str(representative.get("property_id")))
            if not row:
                continue
            item = self._format_rec(row)
            item.update({
                "recommendation_profile": representative.get("profile"),
                "selection_reason": representative.get("selection_reason"),
                "finance_program_id": representative.get("finance_program_id"),
                "finance_program_name": representative.get("finance_program_name"),
                "loan_amount_manwon": representative.get("loan_amount_manwon"),
                "monthly_housing_cost_manwon": representative.get(
                    "monthly_housing_cost_manwon"),
                "utility_score": representative.get("utility_score"),
                "hard_constraints": representative.get("hard_constraints"),
            })
            recommendations.append(item)
        session["last_recommended_properties"] = [
            by_id[str(item["property_id"])] for item in recommendations
            if str(item["property_id"]) in by_id
        ]
        return self._finalize(text, {
            "status": "recommendation",
            "recommendation_mode": (
                "alternative_areas_pareto" if alternative
                else "best_affordable_pareto"
            ),
            "message": (
                f"현재 동({excluded_dong})을 제외하고 같은 자금·상환 제약을 만족하는 "
                "대안 후보를 골랐습니다."
                if alternative and excluded_dong else
                "자기자금·예비 금융자격·월 상환액을 함께 적용해 목적별 최적 후보를 골랐습니다."
            ),
            "optimization": optimization,
            "groups": {0: recommendations},
            "excluded_dong": excluded_dong,
            "searched_property_count": len(pool),
            "finance_program_count": len(programs),
        }, trace)

    def _lease_monte_carlo_comparison(
        self, session: dict, trace: dict, stay_years: float | None = None,
    ) -> dict:
        user = session["user"]
        horizon_years = max(1, min(round(stay_years), 30)) if stay_years else 10
        affordability = compute_affordability(user)
        region = " ".join(str(value) for value in (
            user.get("preferred_sido"), user.get("preferred_gugun")) if value)
        programs = self.finance_tool.search(
            user_income_manwon=user.get("monthly_income_manwon"),
            user_age=user.get("age"), region=region or None,
            finance_mode="eligibility", user_profile=user, limit=50,
        )
        selected = (session.get("last_property_report") or {}).get("property") or {}
        selected_transaction = selected.get("transaction_type")
        representative = {
            "전세": {
                "property_id": "budget-representative-jeonse",
                "transaction_type": "전세",
                "deposit_manwon": affordability.recommended_jeonse_deposit_manwon,
                "monthly_rent_manwon": 0.0,
                "maintenance_fee_manwon": 7.0,
            },
            "월세": {
                "property_id": "budget-representative-monthly",
                "transaction_type": "월세",
                "deposit_manwon": affordability.recommended_monthly_deposit_manwon,
                "monthly_rent_manwon": affordability.recommended_monthly_rent_manwon,
                "maintenance_fee_manwon": 7.0,
            },
        }
        if selected_transaction in representative:
            representative[selected_transaction] = {
                **selected, "transaction_type": selected_transaction,
            }
        forecast = {
            "annual_growth_rate": 0.0, "annual_low": -0.02,
            "annual_high": 0.02,
        }
        scenarios: dict[str, dict] = {}
        seed = 20260729
        for transaction, prop in representative.items():
            budget = simulate_asset_budget(
                user, prop, forecast, programs, {"horizon_years": horizon_years})
            probabilistic = simulate_probabilistic(
                user, prop, forecast, budget, {"horizon_years": horizon_years},
                paths=3000, seed=seed,
            )
            base = probabilistic["base"]
            scenarios[transaction] = {
                "property_basis": (
                    "selected_property"
                    if selected_transaction == transaction
                    else "affordability_representative"
                ),
                "deposit_manwon": prop.get("deposit_manwon"),
                "monthly_rent_manwon": prop.get("monthly_rent_manwon"),
                "selected_finance_program": (
                    budget.get("funding") or {}).get("chosen_program_name"),
                "terminal_net_worth": base["terminal_net_worth"],
                "ten_year_net_worth": base["ten_year_net_worth"],
                "cash_depletion_probability": base[
                    "cash_depletion_probability"],
                "repayment_distress_probability": base[
                    "repayment_distress_probability"],
                "cvar_5_terminal_change_manwon": base[
                    "cvar_5_terminal_change_manwon"],
                "rate_plus_2pp": probabilistic["rate_plus_2pp"],
            }
        winner = max(
            scenarios,
            key=lambda key: float(
                scenarios[key]["terminal_net_worth"]["p50"]))
        gap = (
            float(scenarios[winner]["terminal_net_worth"]["p50"])
            - float(scenarios["월세" if winner == "전세" else "전세"][
                "terminal_net_worth"]["p50"])
        )
        trace["tools"].append({
            "tool": "lease_monte_carlo",
            "model": "vectorized_monthly_monte_carlo_v1",
            "path_count_per_option": 3000, "seed": seed,
            "finance_program_count": len(programs),
            "selected_property_used": bool(
                selected_transaction in representative),
        })
        return {
            "preferred": winner,
            "p50_gap_manwon": round(gap, 1),
            "summary": (
                f"현재 입력과 {horizon_years}년·각 3,000개 경로 기준 중앙값은 "
                f"{winner}가 약 {gap:,.0f}만원 우세합니다."
            ),
            "scenarios": scenarios,
            "path_count_per_option": 3000,
            "horizon_years": horizon_years,
            "basis": (
                "선택 매물과 적정예산 대표 시나리오 비교"
                if selected_transaction in representative
                else "특정 매물이 아닌 사용자 적정예산 대표 시나리오 비교"
            ),
            "disclaimer": (
                "P10·P50·P90과 스트레스 결과는 입력 가정에 따른 모형 분포이며 "
                "실제 수익이나 대출승인을 보장하지 않습니다."
            ),
        }

    def _buy_or_wait(self, session: dict) -> dict:
        report = session.get("last_property_report") or {}
        prop = report.get("property") or {}
        forecast = report.get("forecast") or {}
        if not prop or prop.get("transaction_type") != "매매" or not forecast:
            return {
                "status": "needs_purchase_property",
                "message": (
                    "비교할 매매 매물을 먼저 선택해 주세요. 선택한 집의 실거래 "
                    "시계열·가격 전망과 기다리는 동안의 주거비를 함께 계산하겠습니다."
                ),
            }
        price = float(
            prop.get("sale_price_manwon")
            or prop.get("asking_price_manwon") or 0)
        growth = float(forecast.get("annual_growth_rate") or 0)
        low = float(
            forecast.get("annual_low")
            if forecast.get("annual_low") is not None else growth)
        high = float(
            forecast.get("annual_high")
            if forecast.get("annual_high") is not None else growth)
        affordability = compute_affordability(session["user"])
        wait_monthly = float(
            affordability.recommended_monthly_rent_manwon)
        horizons = []
        for years in (1, 2):
            projected = price * (1 + growth) ** years
            low_price = price * (1 + low) ** years
            high_price = price * (1 + high) ** years
            wait_cost = wait_monthly * 12 * years
            total_difference = projected - price + wait_cost
            horizons.append({
                "years": years,
                "projected_price_manwon": round(projected, 1),
                "price_interval_manwon": {
                    "low": round(min(low_price, high_price), 1),
                    "high": round(max(low_price, high_price), 1),
                },
                "estimated_wait_housing_cost_manwon": round(wait_cost, 1),
                "extra_required_vs_buy_now_manwon": round(
                    total_difference, 1),
            })
        two_year = horizons[-1]
        recommendation = (
            "buy_now" if two_year["extra_required_vs_buy_now_manwon"] > 0
            else "wait"
        )
        return {
            "status": "ok", "recommendation": recommendation,
            "message": (
                "기준 전망과 대기 중 주거비를 합치면 지금 매수가 상대적으로 유리합니다."
                if recommendation == "buy_now" else
                "기준 전망에서는 1~2년 대기가 상대적으로 유리하지만 예측구간을 확인해야 합니다."
            ),
            "property_id": prop.get("property_id"),
            "current_price_manwon": round(price, 1),
            "forecast": {
                "annual_growth_rate": growth,
                "annual_low": low, "annual_high": high,
                "price_history": forecast.get("price_history"),
                "news": forecast.get("news"),
            },
            "wait_cost_assumption": (
                "사용자 적정 월세 × 대기 개월 수; 이사비·취득세 변화는 제외"
            ),
            "horizons": horizons,
        }

    def _finalize(self, user_text: str, result: dict, trace: dict,
                  synthesize: bool = True) -> dict:
        """최종 응답 합성. LLM 실패 시 기존 구조화 응답이 그대로 폴백이다."""
        conversation_history = trace.pop(
            "_advisor_conversation_history", [])
        result["agent_trace"] = trace
        if synthesize and self.llm.supports_agentic_calls:
            answer = self.llm.synthesize(
                user_text, result,
                conversation_history=conversation_history or None)
            if answer:
                result["answer"] = answer
                trace["synthesis"] = {"strategy": "llm_grounded", "ok": True,
                                      "attempts": list(self.llm.last_trace)}
            else:
                trace["synthesis"] = {"strategy": "template", "ok": False}
                trace["fallbacks"].append("최종 문장 합성 실패: 구조화 응답 사용")
        if conversation_history:
            trace["conversation_memory"] = {
                "history_entries_used": len(conversation_history),
                "history_exposed_in_trace": False,
            }
        return result

    @staticmethod
    def _error_text(exc: Exception) -> str:
        return f"{type(exc).__name__}: {exc}"[:500]
