from src.market_forecast.kb_complex import (
    CHART_URL, SIGUNGU_URL, SIDO_URL, TABLE_URL, KBLandComplexPriceTool,
)
from src.market_forecast.news import NewsSignalTool


class Response:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self.data


class KBSession:
    def get(self, url, params=None, **kwargs):
        if url == SIDO_URL:
            data = [{"법정동코드": "4100000000", "시도명": "경기도", "시도명1": "경기"}]
        elif url == SIGUNGU_URL:
            data = [{"법정동코드": "4111700000", "시군구명": "수원시 영통구",
                     "하위시군구존재여부": "0"}]
        elif url == TABLE_URL:
            data = {"데이터목록": [{
                "단지기본일련번호": 11, "면적일련번호": 22,
                "지역명": "영통구 원천동", "데이터정보": {
                    "I2010": {"단지명": "테스트단지"},
                    "I2020": {"전용면적": "59.90"},
                    "I2060": {"지수": "5.50"},
                },
            }]}
        elif url == CHART_URL:
            assert params["데이터셋코드"] == "I2060"
            data = {"날짜정보": ["20250101", "20260101"],
                    "데이터정보": [{"지수": ["5.00", "5.50"]}]}
        else:
            raise AssertionError(url)
        return Response({"dataBody": {"data": data}})


def test_kb_history_marks_synthetic_listing_as_regional_reference():
    result = KBLandComplexPriceTool(session=KBSession()).history({
        "house_type": "아파트", "sido": "경기도", "gugun": "수원영통구",
        "area_m2": 60, "sale_price_manwon": 54000,
        "building_name": "합성 영통구 아파트 1", "is_synthetic": 1,
    })

    assert result["available"] is True
    assert result["match_type"] == "regional_reference"
    assert result["complex_name"] == "테스트단지"
    assert result["latest_price_manwon"] == 55000
    assert len(result["series"]) == 2
    assert "선택 매물 자체" in result["warning"]


class NewsSession:
    def get(self, *args, **kwargs):
        return Response({"items": [
            {"title": "수원 영통구 광역철도 착공", "description": "주택 접근성 개선 기대",
             "pubDate": "Mon, 20 Jul 2026 09:00:00 +0900", "originallink": "https://a.test/1"},
            {"title": "수원 축구팀 경기 승리", "description": "스포츠 소식",
             "pubDate": "Mon, 20 Jul 2026 09:00:00 +0900", "originallink": "https://a.test/2"},
        ]})


class NewsLLM:
    model = "test-model"

    def analyze_json(self, **kwargs):
        assert kwargs["operation"] == "llm.market_news_impact"
        return {
            "articles": [
                {"index": 0, "relevant": True, "direction": "positive",
                 "impact_score": 0.7, "confidence": 0.8,
                 "reason": "교통 접근성 개선은 주택 수요에 영향을 줄 수 있음"},
                {"index": 1, "relevant": False, "direction": "neutral",
                 "impact_score": 0, "confidence": 0.95,
                 "reason": "주택가격과 관련 없는 스포츠 기사"},
            ],
            "overall": {"label": "positive", "score": 0.5, "confidence": 0.7,
                        "summary": "시계열과 교통 개선 기사를 함께 보면 제한적으로 긍정적입니다.",
                        "positive_drivers": ["광역철도"], "negative_drivers": []},
        }


def test_news_llm_only_exposes_price_relevant_articles(monkeypatch):
    monkeypatch.setenv("NAVER_NEWS_CLIENT_ID", "id")
    monkeypatch.setenv("NAVER_NEWS_CLIENT_SECRET", "secret")
    tool = NewsSignalTool(llm=NewsLLM(), session=NewsSession())
    result = tool.assess(
        "경기도", "수원영통구", "아파트",
        market_context={"time_series_annual_growth_rate": 0.02},
    )

    assert result["judge_strategy"] == "llm_structured"
    assert result["candidate_count"] == 2
    assert result["relevant_count"] == 1
    assert result["relevant_headlines"][0]["link"] == "https://a.test/1"
    assert all("축구" not in row["title"] for row in result["relevant_headlines"])
    assert result["overall_assessment"]["label"] == "positive"
    assert result["annual_adjustment_pct_point"] > 0
