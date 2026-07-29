"""
플래너 — 자연어 → 구조화된 계획(plan).

세 가지 판단을 한다:
  1) intent   : 무엇을 원하는가? (recommend / qa_* / chitchat / vague)
  2) slots    : 추천용 조건 슬롯 추출
  3) action   : proceed(검색) / clarify(되물음) / confirm(확인요청) 중 무엇을 할지

핵심 규칙(사용자 요구사항 반영):
  - "그냥 아무거나" → 조건 없이 검색 진행(clarify 아님). 진짜 아무거나 추천.
  - 뭔가 원하는데 슬롯을 못 채움(모호) → clarify(구체적 정보 요구).
  - 조건이 하나라도 잡히면 → 바로 추천 말고 confirm(조건 확인 후 진행).

실제 LLM(Qwen)로 교체 시 이 로직을 프롬프트로 옮기면 된다. 현재 MockLLM은
결정론적 규칙으로 같은 인터페이스를 구현해 오프라인 테스트를 보장한다.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional



# 랜드마크(통근) — harness와 공유
LANDMARKS = ["ifc", "여의도", "강남", "판교", "종로", "가산", "구로디지털", "을지로"]


@dataclass
class Plan:
    intent: str = "recommend"           # recommend | qa_contract | qa_lease_compare
                                        # | qa_cost | qa_poi | qa_market | qa_registry
                                        # | qa_finance | qa_affordability | vague | chitchat
    slots: dict = field(default_factory=dict)
    tool_calls: list = field(default_factory=list)   # 검색 전 실행할 도구(map 등)
    action: str = "proceed"             # proceed | clarify | confirm
    clarify_message: Optional[str] = None
    reason: str = ""
    qa_args: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)       # 모델/재시도/폴백 감사 정보


# 자연어에서 "아무거나/상관없음" 신호
_ANY_SIGNAL = re.compile(r"아무거나|아무데나|상관\s*없|무관|알아서|딱히\s*없|추천\s*해\s*줘$|골라\s*줘")
# 뭔가 원하지만 불명확 신호(질문성이지만 조건 추출 실패용)
_WANT_SIGNAL = re.compile(r"좋은|괜찮은|살고\s*싶|구하고\s*싶|찾고\s*싶|원해|필요")


class Planner:
    """규칙 기반 플래너(MockLLM 내부에서 사용). 실 LLM으로 교체 가능."""

    def plan(self, text: str, has_prior_region: bool = False) -> Plan:
        t = text.strip()
        low = t.lower()
        p = Plan()

        # ---------- 0) 금융→예산→매물 추천 복합 목표 ----------
        financed_goal = self._detect_financed_jeonse_goal(t)
        if financed_goal:
            return financed_goal
        alternative_goal = self._detect_alternative_area_goal(t)
        if alternative_goal:
            return alternative_goal
        affordable_goal = self._detect_best_affordable_goal(t)
        if affordable_goal:
            return affordable_goal

        # ---------- 0-1) Q&A 의도 우선 감지 ----------
        qa = self._detect_qa(t, low)
        if qa:
            p.intent, p.qa_args = qa
            p.action = "proceed"
            p.reason = f"Q&A 의도({p.intent})"
            return p

        # ---------- 1) 추천용 슬롯 추출 ----------
        slots = {}

        # 거래유형(매매도 DB에 존재하므로 임대와 동등하게 정규화)
        if re.search(r"매매|매수|사고\s*싶|구매", t):
            slots["transaction_type"] = "매매"
            slots["lease_type"] = "매매"
        elif re.search(r"전세", t):
            slots["transaction_type"] = "전세"
            slots["lease_type"] = "전세"
        elif re.search(r"월세", t):
            slots["transaction_type"] = "월세"
            slots["lease_type"] = "월세"

        # 공인중개사 매물 유형
        property_patterns = {
            "아파트": r"아파트", "오피스텔": r"오피스텔",
            "다가구": r"다가구", "다세대": r"다세대|빌라",
            "연립": r"연립", "단독": r"단독(?:주택)?",
        }
        for property_type, pattern in property_patterns.items():
            if re.search(pattern, t):
                slots["property_type"] = property_type
                break

        # 지역(시도)
        for sido in ["서울", "경기", "인천", "부산", "대구", "대전", "광주", "울산", "세종"]:
            if sido in t:
                slots["region_sido"] = sido
                break
        # 구/군 직접 언급
        gu = re.findall(r"([가-힣]{1,4}(?:구|군|시))(?![도])", t)
        gu = [g for g in gu if g not in ("경기도", "충청도")]
        if gu:
            slots["region_gugun"] = gu

        # 예산(보증금/월세 상한): "5천만원", "3억", "월세 60"
        dep = self._parse_deposit(t)
        if dep is not None:
            if slots.get("lease_type") == "매매":
                slots["max_sale_price_manwon"] = dep
            else:
                slots["max_deposit_manwon"] = dep
        rent = self._parse_monthly_rent(t)
        if rent is not None:
            slots["max_monthly_rent_manwon"] = rent

        # 관리비 상한
        m = re.search(r"관리비\s*(\d+)\s*만", t)
        if m:
            slots["max_maintenance_manwon"] = float(m.group(1))

        # 면적("투룸", "20평", "40제곱")
        area = self._parse_area(t)
        if area is not None:
            slots["min_area_m2"] = area

        # 연식("신축", "10년 이내")
        if re.search(r"신축|새\s*집|새것", t):
            slots["max_building_age"] = 5
        else:
            m = re.search(r"(\d+)\s*년\s*(?:이내|이하|미만)", t)
            if m:
                slots["max_building_age"] = float(m.group(1))

        # 안전/위험 선호
        if re.search(r"안전|안심|사기\s*없|위험(?:도|이)?\s*(?:낮|적|없)", t):
            # 위험도는 후보를 제거하는 조건이 아니라 결과 정렬 기준이다.
            slots["sort_by"] = "risk_asc"

        # 치안 선호(주변 안전인프라)
        if re.search(r"치안|우범\s*아|밤에\s*안전|방범", t):
            slots["min_safety_score"] = 50.0

        # 생활편의 선호
        if re.search(r"편의\s*시설|생활\s*편의|살기\s*(?:편|좋)|인프라\s*좋", t):
            slots["min_convenience_score"] = 50.0

        # 통근/랜드마크
        commute_min, landmark = self._parse_commute(t, low)
        if landmark or commute_min:
            p.tool_calls.append({"tool": "map_regions_within",
                                 "args": {"landmark": landmark,
                                          "minutes": commute_min or 30}})
            slots["max_commute_min"] = commute_min or 30
            slots["_workplace_landmark"] = landmark
            p.reason += f"통근 감지(랜드마크={landmark}, {commute_min or 30}분). "

        p.slots = slots
        if slots:
            p.tool_calls.append({"tool": "property_search", "args": {}})

        # ---------- 2) action 결정 ----------
        has_any_signal = bool(_ANY_SIGNAL.search(t))
        has_want = bool(_WANT_SIGNAL.search(t))
        has_slots = len(slots) > 0 or len(p.tool_calls) > 0 or has_prior_region

        if has_any_signal and not has_slots:
            # "그냥 아무거나" → 조건 없이 진행(진짜 아무거나)
            p.action = "proceed"
            p.reason += "무조건 추천(아무거나) 요청. "
        elif not has_slots:
            # 조건이 하나도 안 잡힘
            if has_want:
                # 뭔가 원하는데 불명확 → 되물음
                p.action = "clarify"
                p.clarify_message = (
                    "원하시는 조건을 조금만 더 알려주세요. 예: 지역(예: 서울 관악구), "
                    "전세/월세, 예산(보증금·월세), 통근 목적지, 안전 우선 여부 중 아무거나요."
                )
                p.reason += "선호는 있으나 슬롯 추출 실패 → 되물음. "
            else:
                # 잡담/무의도
                p.intent = "vague"
                p.action = "clarify"
                p.clarify_message = (
                    "어떤 집을 찾아드릴까요? 지역·전세/월세·예산·통근지 중 하나라도 "
                    "말씀해 주시면 바로 찾아드릴게요. (그냥 '아무거나'라고 하셔도 돼요.)"
                )
                p.reason += "의도 불명확 → 되물음. "
        else:
            # 조건이 잡힘 → 바로 추천 말고 확인 요청
            p.action = "confirm"
            p.reason += "조건 추출됨 → 사용자 확인 후 추천. "

        return p

    def _detect_financed_jeonse_goal(self, text: str) -> Optional[Plan]:
        """금융 조회 자체가 아니라 금융을 수단으로 전세 추천까지 원하는 목표."""
        has_finance = bool(re.search(r"금융|대출|지원\s*(?:상품|제도|책)?|보증금\s*지원", text))
        has_max_goal = bool(re.search(
            r"최대한|최고|가장|비싼|더\s*(?:비싼|좋은)|예산\s*(?:을\s*)?늘|"
            r"한도\s*(?:를\s*)?최대|가능한\s*큰", text))
        wants_action = bool(re.search(r"추천|찾아|골라|방안|살고\s*싶", text))
        if not ("전세" in text and has_finance and has_max_goal and wants_action):
            return None

        slots = {"transaction_type": "전세", "lease_type": "전세"}
        for sido in [
            "서울", "경기", "인천", "부산", "대구", "대전", "광주", "울산", "세종",
            "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
        ]:
            if sido in text:
                slots["region_sido"] = sido
                break
        districts = re.findall(r"([가-힣]{1,8}(?:구|군|시))(?![도])", text)
        if districts:
            slots["region_gugun"] = districts
        return Plan(
            intent="goal_financed_jeonse",
            slots=slots,
            tool_calls=[{"tool": "finance_search", "args": {}},
                        {"tool": "property_search", "args": {}}],
            action="proceed",
            reason="금융 활용 전세예산 최대화 및 매물 추천 복합 목표",
            qa_args={"finance_mode": "eligibility"},
        )

    def _goal_region_slots(self, text: str) -> dict:
        slots: dict = {}
        for sido in [
            "서울", "경기", "인천", "부산", "대구", "대전", "광주", "울산", "세종",
            "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
        ]:
            if sido in text:
                slots["region_sido"] = sido
                break
        districts = re.findall(r"([가-힣]{1,8}(?:구|군|시))(?![도])", text)
        if districts:
            slots["region_gugun"] = districts
        elif "수원" in text:
            # Prototype scope alias; the harness narrows this parent city to
            # the session's concrete DB district (for example 수원시 팔달구).
            slots["region_gugun"] = ["수원"]
        return slots

    def _detect_best_affordable_goal(self, text: str) -> Optional[Plan]:
        """대출을 포함한 감당 가능 매물의 다목적 최적 추천."""
        has_budget = bool(re.search(
            r"내\s*예산|가진\s*예산|보유\s*자산|감당\s*가능|살\s*수\s*있는|"
            r"대출\s*포함|자금\s*조달", text))
        has_best = bool(re.search(
            r"제일\s*좋|가장\s*좋|최적|베스트|추천|골라|찾아", text))
        has_home = bool(re.search(r"집|주택|매물|아파트|오피스텔", text))
        if not (has_budget and has_best and has_home):
            return None
        slots = self._goal_region_slots(text)
        if re.search(r"매매|매수|구매|내\s*집", text):
            slots.update(transaction_type="매매", lease_type="매매")
        return Plan(
            intent="goal_best_affordable",
            slots=slots,
            tool_calls=[
                {"tool": "finance_search", "args": {}},
                {"tool": "property_search", "args": {}},
            ],
            action="proceed",
            reason="자기자금·대출·상환제약을 적용한 Pareto 최적 매물 추천 목표",
            qa_args={"finance_mode": "eligibility"},
        )

    def _detect_alternative_area_goal(self, text: str) -> Optional[Plan]:
        """현재 지역/매물 대신 같은 예산에서 가능한 대안 지역 추천."""
        alternative = bool(re.search(
            r"여기\s*말고|다른\s*(?:동네|지역)|대안\s*(?:동네|지역)|"
            r"근처\s*다른|옆\s*동네", text))
        budget = bool(re.search(r"예산|가격|감당|대출|살\s*수", text))
        if not (alternative and budget):
            return None
        return Plan(
            intent="goal_alternative_areas",
            slots=self._goal_region_slots(text),
            tool_calls=[
                {"tool": "finance_search", "args": {}},
                {"tool": "property_search", "args": {}},
            ],
            action="proceed",
            reason="동일 자금·상환 제약을 적용한 대안 지역 Pareto 추천 목표",
            qa_args={"finance_mode": "eligibility"},
        )

    # ------------------------------------------------------------------
    def _detect_qa(self, t: str, low: str):
        """Q&A 의도 감지 → (intent, args) 또는 None."""
        # 매물 검색 신호가 있으면 QA가 아니라 추천(치안/편의는 조건 atom으로 처리)
        is_search = bool(re.search(r"전세|월세|추천|매물|집\s*(?:구|찾|추천)|살\s*곳|이내|이하", t))

        if not is_search and re.search(
                r"치안|안전\s*한\s*(?:동네|지역)|밤에\s*안전|우범|범죄|"
                r"cctv|CCTV|비상벨|파출소|방범", t):
            return ("qa_safety", {})
        if not is_search and re.search(
                r"편의\s*시설|생활\s*(?:편의|인프라)|주변\s*(?:에\s*)?(?:뭐|시설)|"
                r"인프라|살기\s*(?:편|좋)", t):
            return ("qa_convenience", {})
        if re.search(
                r"특약|계약서|체크리스트|계약\s*할\s*때|계약해도\s*안전|"
                r"확정일자|전입신고", t):
            lt = "전세" if "전세" in t else ("월세" if "월세" in t else "전세")
            return ("qa_contract", {"lease_type": lt})
        if re.search(r"전세.*월세.*(비교|유리|나아|나은|좋)|월세.*전세.*(비교|유리|좋)|"
                     r"전세랑\s*월세|전세가\s*(?:나아|좋)|월세가\s*(?:나아|좋)", t):
            return ("qa_lease_compare", {})
        if re.search(
                r"지금\s*(?:사|매수|구매).*(?:기다|나을)|"
                r"(?:1|2|1\s*[~～-]\s*2)\s*년\s*(?:뒤|후|기다)|"
                r"기다리.*(?:사|매수|구매)", t):
            return ("qa_buy_or_wait", {})
        if re.search(
                r"실\s*부담|부담액|월\s*얼마|한\s*달에\s*얼마|"
                r"총\s*(?:비용|주거비)|얼마나\s*드", t):
            return ("qa_cost", {})
        if re.search(r"지하철|역\s*(?:가까|근처|있)|편의점|병원|마트|카페|헬스장|주변\s*(?:시설|편의)", t):
            cat = ("subway" if "지하철" in t or "역" in t else
                   "convenience" if "편의점" in t else
                   "hospital" if "병원" in t else
                   "mart" if "마트" in t else
                   "cafe" if "카페" in t else "subway")
            return ("qa_poi", {"category": cat})
        if re.search(
                r"시세|적정\s*가|비싼\s*거\s*아|바가지|실거래|"
                r"가격.*적정|적정.*가격|"
                r"집값.*(?:오를|내릴|전망|상승|하락)|"
                r"가격.*(?:오를|내릴|전망|상승|하락)|"
                r"(?:오를까|내릴까).*(?:집값|가격)", t):
            return ("qa_market", {})
        if re.search(r"등기부|등기\s*확인|근저당\s*확인|신탁등기", t):
            return ("qa_registry", {})
        if re.search(r"대출|금융\s*(?:지원|상품|제도)|주거\s*금융|버팀목|"
                     r"월세\s*지원|지원\s*(?:제도|책)|보증금\s*지원", t):
            personal = bool(re.search(
                r"나한테|내가|제\s*(?:소득|조건)|받을\s*수|자격|해당|가능할까|가능해", t))
            args = {"finance_mode": "eligibility" if personal else "catalog"}
            rate = re.search(r"금리\s*(\d+(?:\.\d+)?)\s*%?\s*(?:미만|아래|이하)", t)
            if rate:
                args["max_rate_pct"] = float(rate.group(1))
            if "대출" in t:
                args["product_kind"] = "대출"
            elif re.search(r"지원금|보조금|월세\s*지원|현금\s*지원", t):
                args["product_kind"] = "지원"
            elif "청약" in t:
                args["product_kind"] = "청약"
            return ("qa_finance", args)
        if re.search(r"얼마짜리|내\s*소득|내\s*형편|얼마까지|감당|적정\s*(?:예산|보증금|주거비)", t):
            return ("qa_affordability", {})
        return None

    # ------------------------------------------------------------------
    def _parse_deposit(self, t: str) -> Optional[float]:
        """보증금 상한(만원)을 파싱. '3억', '5천만원', '보증금 2000'."""
        # N억(+N천)
        m = re.search(r"(\d+)\s*억(?:\s*(\d+)\s*천)?", t)
        if m:
            eok = int(m.group(1)) * 10000
            cheon = int(m.group(2)) * 1000 if m.group(2) else 0
            return float(eok + cheon)
        # N천만원
        m = re.search(r"(\d+)\s*천\s*만?\s*원?", t)
        if m and ("보증" in t or "전세" in t or "이내" in t or "이하" in t):
            return float(int(m.group(1)) * 1000)
        # 보증금 N (만원 단위 숫자)
        m = re.search(r"보증금\s*(\d{3,5})\s*만?", t)
        if m:
            return float(m.group(1))
        return None

    def _parse_monthly_rent(self, t: str) -> Optional[float]:
        m = re.search(r"월세\s*(\d{1,3})\s*만?", t)
        if m:
            return float(m.group(1))
        m = re.search(r"월\s*(\d{1,3})\s*만?\s*원?\s*(?:이하|이내|까지)", t)
        if m:
            return float(m.group(1))
        return None

    def _parse_area(self, t: str) -> Optional[float]:
        if re.search(r"투룸|２룸|2룸", t):
            return 30.0
        if re.search(r"쓰리룸|３룸|3룸", t):
            return 45.0
        m = re.search(r"(\d+)\s*평", t)
        if m:
            return float(m.group(1)) * 3.3
        m = re.search(r"(\d+)\s*(?:제곱|㎡|m2)", t)
        if m:
            return float(m.group(1))
        return None

    def _parse_commute(self, t: str, low: str):
        m = re.search(r"(\d+)\s*분", t)
        commute_min = int(m.group(1)) if m else None
        landmark = None
        for lm in LANDMARKS:
            if lm.lower() in low:
                landmark = lm
                break
        if not landmark and re.search(r"회사|직장|출근|퇴근|통근", t):
            commute_min = commute_min or 30
        return commute_min, landmark


# ----------------------------------------------------------------------
# 확인 응답 파싱: 사용자가 "응/맞아" 또는 수정("월세로")을 보냄
# ----------------------------------------------------------------------
_YES = re.compile(r"^\s*(응|네|예|어|그래|맞아|맞아요|좋아|좋아요|ok|okay|오케이|진행|해\s*줘|그렇게|ㅇㅇ|yes|y)\s*[.!]*\s*$",
                  re.IGNORECASE)
_NO = re.compile(r"^\s*(아니|아니요|취소|그만|no|n|다시)\s*[.!]*\s*$", re.IGNORECASE)


def parse_confirmation(text: str) -> str:
    """'yes' | 'no' | 'modify' 반환."""
    if _YES.match(text):
        return "yes"
    if _NO.match(text):
        return "no"
    return "modify"   # 그 외는 조건 수정으로 간주(재파싱)
