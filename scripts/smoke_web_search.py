"""운영 컨테이너에서 OpenAI 웹 검색 도구의 최소 동작을 확인한다."""
from src.server.app import _agent


result = _agent.llm.web_search(
    "대한민국 아주대학교 공식 도로명주소", "배포 웹 검색 도구 점검")
assert result and result.get("sources") and result.get("map_query"), result
geocode = _agent.map_tool.geocode(result["map_query"])
assert geocode.get("ok"), geocode
print({
    "map_query": result["map_query"],
    "source_count": len(result["sources"]),
    "geocode_source": geocode.get("source"),
    "model": _agent.llm.model,
})
