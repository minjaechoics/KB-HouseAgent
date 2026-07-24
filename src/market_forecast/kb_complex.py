"""KB부동산 단지 시세 어댑터.

KB 이용약관은 별도 허락 없는 정보 도용(웹 크롤링 포함)을 금지한다. 따라서
운영 기본값은 네트워크 호출 금지이며, KB와의 서면 허가·제휴 계약 범위를
확인한 운영자가 ``KB_DATA_LICENSED=true``를 설정한 경우에만 조회한다.
"""
from __future__ import annotations

import math
import os
import re
import time
from datetime import datetime
from typing import Any

import requests


SOURCE_URL = "https://data.kbland.kr/kbstats/investment-table"
SIDO_URL = "https://data-api.kbland.kr/bfmavm/map/siDoAreaNameList"
SIGUNGU_URL = "https://data-api.kbland.kr/bfmavm/map/siGunGuAreaNameList"
TABLE_URL = "https://api.kbland.kr/land-extra/price/v1/api/invstTblAptSearch"
CHART_URL = "https://api.kbland.kr/land-extra/price/v1/api/invstTblAptChartSearch"


def _key(value: Any, *, remove_city: bool = False) -> str:
    text = re.sub(r"\s+", "", str(value or ""))
    for suffix in ("특별자치도", "특별자치시", "특별시", "광역시", "도"):
        text = text.replace(suffix, "")
    if remove_city:
        text = text.replace("시", "")
    return text


def _number(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


class KBLandComplexPriceTool:
    def __init__(self, timeout: float = 6.0, session=None, cache_seconds: int = 1800):
        self.timeout = timeout
        self.session = session or requests.Session()
        # 주입 세션은 네트워크가 없는 단위 테스트 fixture로만 사용한다.
        self.licensed = (session is not None or
                         os.environ.get("KB_DATA_LICENSED", "").lower()
                         in {"1", "true", "yes"})
        self.cache_seconds = cache_seconds
        self._cache: dict[str, tuple[float, Any]] = {}
        self.headers = {
            "Referer": "https://data.kbland.kr/",
            "Origin": "https://data.kbland.kr",
            "User-Agent": "Mozilla/5.0 (compatible; JeonseHelper/1.0)",
            "Accept": "application/json",
        }

    def _get(self, url: str, params: dict | None = None) -> dict:
        cache_key = f"{url}|{sorted((params or {}).items())}"
        cached = self._cache.get(cache_key)
        if cached and time.time() - cached[0] < self.cache_seconds:
            return cached[1]
        response = self.session.get(
            url, params=params, headers=self.headers, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        self._cache[cache_key] = (time.time(), data)
        return data

    @staticmethod
    def _body_data(payload: dict) -> Any:
        return ((payload.get("dataBody") or {}).get("data"))

    def _region_code(self, sido: str, gugun: str) -> tuple[str | None, str | None]:
        provinces = self._body_data(self._get(SIDO_URL)) or []
        target_sido = _key(sido)
        province = next((row for row in provinces if target_sido in {
            _key(row.get("시도명")), _key(row.get("시도명1"))}), None)
        if not province:
            return None, None
        province_code = str(province.get("법정동코드") or "")
        if not gugun or _key(gugun, remove_city=True) in {"", "세종"}:
            return province_code, str(province.get("시도명1") or sido)
        districts = self._body_data(self._get(
            SIGUNGU_URL, {"법정동코드": province_code})) or []
        target = _key(gugun, remove_city=True)
        exact = [row for row in districts
                 if _key(row.get("시군구명"), remove_city=True) == target]
        candidates = exact or [row for row in districts if
                               target in _key(row.get("시군구명"), remove_city=True)
                               or _key(row.get("시군구명"), remove_city=True) in target]
        if not candidates:
            return None, None
        candidates.sort(key=lambda row: (
            str(row.get("하위시군구존재여부")) != "0",
            len(str(row.get("시군구명") or "")),
        ))
        row = candidates[0]
        return str(row.get("법정동코드") or province_code), str(row.get("시군구명") or gugun)

    @staticmethod
    def _unavailable(reason: str) -> dict[str, Any]:
        return {
            "available": False,
            "source": "KB부동산 데이터허브 공개 투자테이블",
            "source_url": SOURCE_URL,
            "reason": reason,
            "series": [],
            "warning": "KB 조회 실패를 추정값이나 모의 데이터로 대체하지 않았습니다.",
        }

    def history(self, prop: dict) -> dict[str, Any]:
        if not self.licensed:
            return self._unavailable(
                "KB 이용약관상 별도 허락 없는 웹 크롤링이 금지되어 자동 조회를 중지했습니다. "
                "서면 이용허락 또는 제휴 API/내보내기 파일이 필요합니다.")
        house_type = str(prop.get("house_type") or prop.get("property_type") or "")
        if house_type != "아파트":
            return self._unavailable("KB 투자테이블의 단지 그래프는 아파트만 지원합니다.")
        try:
            region_code, region_name = self._region_code(
                str(prop.get("sido") or ""), str(prop.get("gugun") or ""))
            if not region_code:
                return self._unavailable("매물 지역을 KB 지역코드로 찾지 못했습니다.")
            table = self._body_data(self._get(TABLE_URL, {
                "기간코드": "6", "지역리스트": region_code,
            })) or {}
            rows = table.get("데이터목록") or []
            if not rows:
                return self._unavailable(f"{region_name or region_code}의 KB 단지 시세가 없습니다.")

            area = _number(prop.get("area_m2")) or 60.0
            asking = (_number(prop.get("sale_price_manwon"))
                      or _number(prop.get("asking_price_manwon")))
            building = _key(prop.get("building_name"))
            synthetic = bool(prop.get("is_synthetic")) or building.startswith("합성")

            def parsed(row: dict) -> dict:
                info = row.get("데이터정보") or {}
                area_info = info.get("I2020") or {}
                return {
                    "raw": row,
                    "name": str((info.get("I2010") or {}).get("단지명") or ""),
                    "area": _number(area_info.get("전용면적")),
                    "price_manwon": ((_number((info.get("I2060") or {}).get("지수")) or 0) * 10000),
                }

            candidates = [parsed(row) for row in rows]
            candidates = [row for row in candidates if row["name"] and row["area"]]
            exact = [] if synthetic else [row for row in candidates
                                           if _key(row["name"]) == building]
            pool = exact or candidates

            def distance(row: dict) -> float:
                area_distance = abs(math.log(max(row["area"], 1) / max(area, 1)))
                if asking and row["price_manwon"]:
                    return area_distance + 0.25 * abs(math.log(row["price_manwon"] / asking))
                return area_distance

            chosen = min(pool, key=distance)
            raw = chosen["raw"]
            chart = self._body_data(self._get(CHART_URL, {
                "기간코드": "24", "데이터셋코드": "I2060",
                "단지기본일련번호": raw.get("단지기본일련번호"),
                "면적일련번호": raw.get("면적일련번호"),
            })) or {}
            dates = chart.get("날짜정보") or []
            data_rows = chart.get("데이터정보") or []
            values = (data_rows[0].get("지수") or []) if data_rows else []
            series = []
            for date, value in zip(dates, values):
                price_eok = _number(value)
                if price_eok is None:
                    continue
                try:
                    iso_date = datetime.strptime(str(date), "%Y%m%d").date().isoformat()
                except ValueError:
                    continue
                series.append({"date": iso_date,
                               "price_manwon": round(price_eok * 10000, 1)})
            if len(series) < 2:
                return self._unavailable("선택한 KB 단지·면적의 가격 이력이 부족합니다.")

            def change(period: int) -> float | None:
                if len(series) <= period or not series[-period - 1]["price_manwon"]:
                    return None
                return round(series[-1]["price_manwon"] /
                             series[-period - 1]["price_manwon"] - 1, 4)

            match_type = "exact_complex" if exact else "regional_reference"
            warning = (
                "선택 매물과 동일한 실제 KB 단지·면적의 공개 시세입니다."
                if exact else
                "이 매물은 분석용 합성 데이터입니다. 표시 그래프는 같은 시·군·구에서 "
                "전용면적과 가격이 가까운 실제 KB 비교 단지이며 선택 매물 자체의 이력이 아닙니다."
            )
            return {
                "available": True,
                "source": "KB부동산 데이터허브 공개 투자테이블",
                "source_url": SOURCE_URL,
                "match_type": match_type,
                "complex_name": chosen["name"],
                "region_name": str(raw.get("지역명") or region_name or ""),
                "exclusive_area_m2": chosen["area"],
                "latest_price_manwon": series[-1]["price_manwon"],
                "as_of": series[-1]["date"],
                "change_6m": change(26),
                "change_1y": change(52),
                "series": series,
                "unit": "만원",
                "warning": warning,
            }
        except Exception as exc:
            return self._unavailable(f"{type(exc).__name__}로 KB 데이터 조회에 실패했습니다.")
