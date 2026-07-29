"""지역 주택 뉴스 수집과 LLM 기반 가격영향 판정.

NAVER 공식 뉴스 검색 API의 제목·요약·원문 링크를 수집하고, 구조화 출력 LLM이
지역 주택가격과 인과 경로가 있는 기사만 남긴다. 뉴스는 예측을 ±1.5%p 안에서
보정하는 보조 신호이며 원문 전문을 복제하지 않는다.
"""
from __future__ import annotations

import html
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import requests

LEGACY_NEWS_URL = "https://openapi.naver.com/v1/search/news.json"
HUB_NEWS_URL = "https://naverapihub.apigw.ntruss.com/search/v1/news"
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"

NEWS_JUDGE_SYSTEM = """당신은 한국 주거시장 리서치 심사역이다.
후보 뉴스의 제목·검색 요약만 근거로 사용한다. 지정 지역의 주택가격에 영향을 줄
구체적 경로(공급, 수요, 교통, 일자리, 금리·대출, 정비사업, 규제, 재해)가 있는
기사만 relevant=true로 판정하라. 단순 사건사고, 광고, 다른 지역 기사, 같은 단어만
포함된 기사는 제외하라. 방향과 영향도는 과장하지 말고 불확실하면 neutral 또는
mixed로 둔다. 종합평가는 시계열 수치와 선별 기사 양쪽을 사용하되, 뉴스가 없으면
수치만 근거로 삼았다고 명시한다. kb_match_type이 regional_reference이면 선택한
합성 매물 자체가 아니라 지역 비교 단지의 참고 수치로만 취급한다. 제공되지 않은
사실은 만들지 않는다."""

NEWS_JUDGE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "articles": {
            "type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "index": {"type": "integer"},
                    "relevant": {"type": "boolean"},
                    "direction": {"type": "string", "enum": [
                        "positive", "negative", "neutral", "mixed"]},
                    "impact_score": {"type": "number", "minimum": -1, "maximum": 1},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string"},
                },
                "required": ["index", "relevant", "direction", "impact_score",
                             "confidence", "reason"],
            },
        },
        "overall": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "label": {"type": "string", "enum": [
                    "positive", "negative", "neutral", "mixed"]},
                "score": {"type": "number", "minimum": -1, "maximum": 1},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "summary": {"type": "string"},
                "positive_drivers": {"type": "array", "items": {"type": "string"}},
                "negative_drivers": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["label", "score", "confidence", "summary",
                         "positive_drivers", "negative_drivers"],
        },
    },
    "required": ["articles", "overall"],
}

POSITIVE = {"상승", "회복", "반등", "호재", "착공", "개통", "증가", "완화",
            "재건축", "재개발", "금리인하", "규제완화", "기업유치"}
NEGATIVE = {"하락", "침체", "급락", "미분양", "공급과잉", "부실", "위축",
            "금리인상", "규제강화", "역전세", "전세사기", "경매"}
MARKET_TERMS = {"아파트", "주택", "부동산", "집값", "매매", "전세", "분양",
                "재건축", "재개발", "정비사업", "교통", "철도", "금리", "대출"}


def _plain(value: str) -> str:
    return re.sub(r"<[^>]+>", "", html.unescape(str(value or ""))).strip()


def _age_days(pub_date: str | None) -> float:
    try:
        published = parsedate_to_datetime(pub_date or "")
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) -
                         published.astimezone(timezone.utc)).total_seconds() / 86400)
    except (TypeError, ValueError, OverflowError):
        return 30.0


class NewsSignalTool:
    def __init__(self, timeout: float = 4.0, llm=None, session=None):
        hub_id = os.environ.get("NAVER_API_HUB_CLIENT_ID", "").strip()
        hub_secret = os.environ.get("NAVER_API_HUB_CLIENT_SECRET", "").strip()
        legacy_id = os.environ.get("NAVER_NEWS_CLIENT_ID", "").strip()
        legacy_secret = os.environ.get("NAVER_NEWS_CLIENT_SECRET", "").strip()
        self.client_id = hub_id or legacy_id
        self.client_secret = hub_secret or legacy_secret
        self.api_kind = "api_hub" if hub_id and hub_secret else "legacy"
        self.url = os.environ.get(
            "NAVER_NEWS_SEARCH_URL",
            HUB_NEWS_URL if self.api_kind == "api_hub" else LEGACY_NEWS_URL,
        ).strip()
        self.timeout = timeout
        self.llm = llm
        self.session = session or requests
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}

    @property
    def configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def _neutral(self, reason: str, market_context: dict | None = None) -> dict[str, Any]:
        context = market_context or {}
        base = float(context.get("time_series_annual_growth_rate") or 0)
        label = "positive" if base > 0.015 else "negative" if base < -0.015 else "neutral"
        return {
            "configured": self.configured, "source": "neutral_fallback",
            "reason": reason, "sentiment_score": 0.0, "applied": False,
            "annual_adjustment_pct_point": 0.0, "headlines": [],
            "relevant_headlines": [], "candidate_count": 0, "relevant_count": 0,
            "judge_strategy": "unavailable", "judge_model": None,
            "overall_assessment": {
                "label": label, "score": 0.0, "confidence": 0.2,
                "summary": "뉴스 판단을 사용할 수 없어 시계열 수치만 유지했습니다.",
                "positive_drivers": [], "negative_drivers": [],
            },
            "warning": "뉴스 신호는 보조 지표이며 가격 예측의 사실 보증이 아닙니다.",
        }

    def _headers(self) -> dict[str, str]:
        if self.api_kind == "api_hub":
            return {"X-NCP-APIGW-API-KEY-ID": self.client_id,
                    "X-NCP-APIGW-API-KEY": self.client_secret}
        return {"X-Naver-Client-Id": self.client_id,
                "X-Naver-Client-Secret": self.client_secret}

    def _collect(self, query: str) -> tuple[list[dict], str, str | None]:
        """NAVER를 우선 사용하고 실패/미설정 시 공개 뉴스 RSS로 대체한다."""
        naver_error = None
        if self.configured:
            try:
                response = self.session.get(
                    self.url, headers=self._headers(),
                    # 리포트 한 번에 30개 요약을 LLM에 넘기면 모바일 첫 요청이
                    # 30초 이상 걸렸다. 최신 8개면 지역 호재·악재 판정에 충분하고
                    # 토큰·지연을 크게 줄일 수 있다.
                    params={"query": query, "display": 8, "sort": "date"},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json().get("items", [])[:8], \
                    f"naver_news_search_{self.api_kind}", None
            except Exception as exc:
                naver_error = f"{type(exc).__name__}: {exc}"[:200]
        response = self.session.get(
            GOOGLE_NEWS_RSS_URL,
            params={"q": query, "hl": "ko", "gl": "KR", "ceid": "KR:ko"},
            headers={"User-Agent": "Mozilla/5.0 (compatible; JeonseHelper/1.0)"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)
        items = []
        for node in root.findall(".//item")[:8]:
            items.append({
                "title": node.findtext("title") or "",
                "description": node.findtext("description") or "",
                "pubDate": node.findtext("pubDate") or "",
                "originallink": node.findtext("link") or "",
            })
        return items, "google_news_rss", naver_error

    @staticmethod
    def _fallback_judgement(items: list[dict], region_terms: list[str],
                            market_context: dict) -> dict:
        decisions = []
        scores = []
        for index, item in enumerate(items):
            document = f"{item['title']} {item['description']}"
            relevant = (any(term and term in document for term in region_terms)
                        and any(term in document for term in MARKET_TERMS))
            pos = sum(term in document for term in POSITIVE)
            neg = sum(term in document for term in NEGATIVE)
            score = 0.0 if pos == neg else (pos - neg) / max(pos + neg, 1)
            decisions.append({"index": index, "relevant": relevant,
                              "direction": "positive" if score > 0 else
                              "negative" if score < 0 else "neutral",
                              "impact_score": score, "confidence": 0.25,
                              "reason": "LLM 미사용 시 지역·주택 키워드 안전 폴백"})
            if relevant:
                scores.append(score)
        mean = sum(scores) / len(scores) if scores else 0.0
        base = float(market_context.get("time_series_annual_growth_rate") or 0)
        combined = max(-1.0, min(1.0, mean * 0.5 + base * 5))
        label = "positive" if combined > 0.15 else "negative" if combined < -0.15 else "neutral"
        return {"articles": decisions, "overall": {
            "label": label, "score": combined, "confidence": 0.25,
            "summary": "LLM을 사용할 수 없어 제한적인 키워드 폴백으로 판정했습니다.",
            "positive_drivers": [], "negative_drivers": [],
        }}

    def assess(self, sido: str, gugun: str, house_type: str,
               *, building_name: str = "", market_context: dict | None = None,
               use_llm: bool = True) -> dict[str, Any]:
        market_context = market_context or {}
        query = f"{sido} {gugun} 주택 부동산 개발 교통".strip()
        cache_key = json.dumps([query, house_type, building_name, market_context, use_llm],
                               ensure_ascii=False, sort_keys=True, default=str)
        cached = self._cache.get(cache_key)
        if cached and time.time() - cached[0] < 1800:
            return cached[1]
        try:
            raw_items, news_source, source_fallback_reason = self._collect(query)
            items = [{
                "title": _plain(x.get("title")),
                "description": _plain(x.get("description"))[:500],
                "published_at": x.get("pubDate"),
                "link": x.get("originallink") or x.get("link"),
            } for x in raw_items if _plain(x.get("title"))]
            payload = {
                "target": {"sido": sido, "gugun": gugun,
                           "house_type": house_type, "building_name": building_name},
                "quantitative_context": market_context,
                "candidate_articles": [
                    {"index": i, "title": x["title"],
                     "description": x["description"], "published_at": x["published_at"]}
                    for i, x in enumerate(items)
                ],
            }
            judgement = None
            strategy = "keyword_fallback"
            judge_error = None
            if use_llm and self.llm and hasattr(self.llm, "analyze_json"):
                try:
                    judgement = self.llm.analyze_json(
                        operation="llm.market_news_impact",
                        system=NEWS_JUDGE_SYSTEM,
                        user=json.dumps(payload, ensure_ascii=False, default=str),
                        schema=NEWS_JUDGE_SCHEMA,
                        schema_name="market_news_impact",
                        max_tokens=900,
                    )
                    if judgement:
                        strategy = "llm_structured"
                except Exception as exc:
                    judge_error = f"{type(exc).__name__}: {exc}"[:250]
            if not judgement:
                judgement = self._fallback_judgement(
                    items, [str(sido), str(gugun)], market_context)
                if not use_llm:
                    judgement["overall"]["summary"] = (
                        "빠른 기사 분류를 먼저 표시하고 AI 정밀 판정을 준비하고 있습니다."
                    )

            decisions = {int(x.get("index")): x for x in judgement.get("articles", [])
                         if isinstance(x, dict) and str(x.get("index", "")).isdigit()}
            relevant = []
            weighted_sum = weight_total = 0.0
            for index, item in enumerate(items):
                decision = decisions.get(index)
                if not decision or not decision.get("relevant"):
                    continue
                score = max(-1.0, min(1.0, float(decision.get("impact_score") or 0)))
                confidence = max(0.0, min(1.0, float(decision.get("confidence") or 0)))
                recency = 0.5 ** (_age_days(item.get("published_at")) / 45.0)
                weight = max(0.05, confidence) * recency
                weighted_sum += score * weight
                weight_total += weight
                relevant.append({**item,
                                 "direction": decision.get("direction") or "neutral",
                                 "impact_score": round(score, 3),
                                 "confidence": round(confidence, 3),
                                 "reason": str(decision.get("reason") or "")[:300]})
            score = max(-1.0, min(1.0, weighted_sum / weight_total)) if weight_total else 0.0
            overall = judgement.get("overall") or {}
            overall = {
                "label": overall.get("label") or "neutral",
                "score": round(float(overall.get("score") or 0), 3),
                "confidence": round(float(overall.get("confidence") or 0), 3),
                "summary": str(overall.get("summary") or "")[:600],
                "positive_drivers": list(overall.get("positive_drivers") or [])[:5],
                "negative_drivers": list(overall.get("negative_drivers") or [])[:5],
            }
            result = {
                "configured": True, "source": news_source,
                "query": query, "sentiment_score": round(score, 3),
                "annual_adjustment_pct_point": round(score * 1.5, 3),
                "applied": bool(relevant),
                "method": (
                    "NAVER 뉴스 검색 + LLM 관련성·방향 구조화 판정 + 45일 반감기"
                    if strategy == "llm_structured"
                    else "NAVER 뉴스 검색 + 지역·주택 키워드 즉시 판정 + 45일 반감기"
                ),
                "candidate_count": len(items), "relevant_count": len(relevant),
                "article_count": len(items), "headlines": relevant,
                "relevant_headlines": relevant, "overall_assessment": overall,
                "judge_strategy": strategy,
                "judge_model": getattr(self.llm, "model", None) if strategy == "llm_structured" else None,
                "judge_error": judge_error,
                "source_fallback_reason": source_fallback_reason,
                "warning": ((("LLM 판정 실패로 지역·주택 키워드 폴백을 사용했습니다. "
                              if judge_error else
                              "빠른 상세 분석에서는 키워드 판정을 우선 사용했습니다. "
                              if not use_llm else "")) +
                            "제목·검색 요약 기반 보조 판단입니다. 기사 원문과 실제 사업 진행 여부를 확인하세요."),
            }
            self._cache[cache_key] = (time.time(), result)
            return result
        except Exception as exc:
            return self._neutral(f"{type(exc).__name__}로 뉴스·LLM 판단 사용 불가", market_context)
