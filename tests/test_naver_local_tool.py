"""NAVER 지역검색의 반경 재검증·호출한도 테스트."""
from src.tools.naver_local_tool import NaverLocalSearchTool


class _Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {"items": [
            {"title": "<b>가까운 파출소</b>", "category": "공공기관>경찰서",
             "roadAddress": "테스트로 1", "mapx": "127.0005", "mapy": "37.0005"},
            {"title": "먼 경찰서", "category": "공공기관>경찰서",
             "roadAddress": "테스트로 999", "mapx": "127.2", "mapy": "37.2"},
        ]}


def test_local_search_filters_by_distance_and_never_invents_counts(monkeypatch):
    monkeypatch.setenv("NAVER_API_HUB_CLIENT_ID", "test-id")
    monkeypatch.setenv("NAVER_API_HUB_CLIENT_SECRET", "test-secret")
    monkeypatch.setattr("src.tools.naver_local_tool.requests.get",
                        lambda *args, **kwargs: _Response())
    tool = NaverLocalSearchTool()
    tool.begin_request(max_calls=1)
    result = tool.search(37.0, 127.0, "테스트시", "파출소", 300)
    blocked = tool.search(37.0, 127.0, "테스트시", "소방서", 300)

    assert result["count"] == 1
    assert result["places"][0]["name"] == "가까운 파출소"
    assert blocked["count"] is None
    assert blocked["source"] == "unavailable"
