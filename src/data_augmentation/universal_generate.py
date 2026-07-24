"""Multi-source synthetic Korean real-estate listing generator.

The module normalizes the downloaded MOLIT RTMS and Gyeonggi portal tables to
one internal anchor schema, then creates fully synthetic broker listings for
six residential property types and sale/jeonse/monthly-rent transactions.
Every output row records whether it came from a directly supported source
combination or from a documented cross-source proxy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.data_sources import DATASETS
from src.data_augmentation.property_schema import (
    BROKER_LISTING_COLUMNS,
    LEGACY_REQUIRED_COLUMNS,
    missing_schema_columns,
)
from src.data_augmentation.region_stats import load_region_accident_stats
from src.data_augmentation.region_catalog import (
    EXPECTED_SIDOS,
    SIDO_ALIASES,
    load_nationwide_coordinate_anchors,
    load_nationwide_region_catalog,
)


GENERATOR_VERSION = "multi-source-public-address-map-matched-stratified-v6"
CURRENT_YEAR = 2026
HOUSE_TYPES = ("아파트", "오피스텔", "단독주택", "다가구주택", "다세대주택", "연립주택")
TRANSACTION_TYPES = ("매매", "전세", "월세")
DEFAULT_HOUSE_WEIGHTS = {
    "아파트": 0.32,
    "오피스텔": 0.24,
    "단독주택": 0.12,
    "다가구주택": 0.18,
    "다세대주택": 0.09,
    "연립주택": 0.05,
}
DEFAULT_TRANSACTION_WEIGHTS = {"매매": 1 / 3, "전세": 1 / 3, "월세": 1 / 3}

# 합성 좌표는 실제 주소가 아닌 지도 분석용이다. 시군구 대표점 하나에 매물이
# 몰리지 않도록 도시 규모에 맞춘 타원 안에서 저불일치 수열로 고르게 분산한다.
REGION_COORDINATE_RADII = {
    "서울": (0.022, 0.028), "부산": (0.030, 0.038),
    "대구": (0.030, 0.038), "인천": (0.035, 0.045),
    "광주": (0.028, 0.035), "대전": (0.030, 0.038),
    "울산": (0.036, 0.046), "세종": (0.035, 0.042),
    "경기": (0.045, 0.060), "강원": (0.050, 0.068),
    "충북": (0.045, 0.060), "충남": (0.047, 0.063),
    "전북": (0.046, 0.062), "전남": (0.050, 0.070),
    "경북": (0.052, 0.070), "경남": (0.048, 0.066),
    "제주": (0.050, 0.075),
}

SEOUL_DISTRICT_CENTERS = {
    "종로구": (37.5735, 126.9790), "중구": (37.5641, 126.9979),
    "용산구": (37.5326, 126.9905), "성동구": (37.5633, 127.0371),
    "광진구": (37.5384, 127.0822), "동대문구": (37.5744, 127.0396),
    "중랑구": (37.6063, 127.0927), "성북구": (37.5894, 127.0167),
    "강북구": (37.6398, 127.0255), "도봉구": (37.6688, 127.0471),
    "노원구": (37.6542, 127.0568), "은평구": (37.6027, 126.9291),
    "서대문구": (37.5791, 126.9368), "마포구": (37.5663, 126.9014),
    "양천구": (37.5170, 126.8666), "강서구": (37.5509, 126.8495),
    "구로구": (37.4955, 126.8874), "금천구": (37.4569, 126.8955),
    "영등포구": (37.5264, 126.8963), "동작구": (37.5124, 126.9393),
    "관악구": (37.4784, 126.9516), "서초구": (37.4836, 127.0327),
    "강남구": (37.5173, 127.0473), "송파구": (37.5146, 127.1059),
    "강동구": (37.5301, 127.1238),
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def lognormal_from_mean_std(mean: float, std: float, size, rng) -> np.ndarray:
    mean = max(float(mean), 1e-6)
    sigma2 = np.log1p((float(std) ** 2) / (mean ** 2))
    mu = np.log(mean) - sigma2 / 2
    return rng.lognormal(mu, np.sqrt(sigma2), size=size)


def _read_csv(path: Path | str, **kwargs) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "cp949", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error or UnicodeDecodeError("utf-8", b"", 0, 1, "cannot decode")


def _num(values) -> pd.Series:
    return pd.to_numeric(pd.Series(values).astype(str).str.replace(",", "", regex=False), errors="coerce")


def _month_ordinal(values) -> np.ndarray:
    x = _num(values).fillna(0).astype(int).to_numpy()
    return (x // 100) * 12 + (x % 100) - 1


def _round_to(values, step: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.maximum(np.round(values / step) * step, 0)


def _valid_trade(df: pd.DataFrame) -> pd.Series:
    cancelled = df.get("cdealType", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
    return ~cancelled.isin(["O", "1"])


def _standard_frame(size: int) -> pd.DataFrame:
    return pd.DataFrame(index=np.arange(size), columns=[
        "source_dataset", "source_index", "source_transaction_type", "sale_subtype",
        "house_type", "sido", "gugun", "dong", "lawd_cd", "deal_ym", "deal_day",
        "build_year", "area_m2", "floor", "building_name", "sale_price_manwon",
        "deposit_manwon", "monthly_rent_manwon", "land_area_m2",
        "building_total_area_m2", "contract_type", "contract_term", "use_rr_right",
        "dealing_type", "source_supported",
    ])


def _district_map(*tables: pd.DataFrame) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for df in tables:
        if "lawd_cd" not in df or "sggNm" not in df:
            continue
        pairs = pd.DataFrame({"code": _num(df["lawd_cd"]), "name": df["sggNm"]}).dropna()
        mapping.update({int(r.code): str(r.name) for r in pairs.itertuples(index=False)})
    return mapping


def _normalize_sh_rent(raw: pd.DataFrame) -> pd.DataFrame:
    out = _standard_frame(len(raw))
    dep, monthly, area = _num(raw["deposit"]), _num(raw["monthlyRent"]), _num(raw["totalFloorAr"])
    out["source_dataset"] = "rtms_sh_rent"
    out["source_index"] = raw.index
    out["source_transaction_type"] = np.where(monthly.eq(0), "전세", "월세")
    out["sale_subtype"] = "해당없음"
    out["house_type"] = raw["houseType"].map({"단독": "단독주택", "다가구": "다가구주택"})
    out["sido"], out["gugun"], out["dong"] = "서울", raw["sggNm"], raw["umdNm"]
    out["lawd_cd"], out["deal_ym"], out["deal_day"] = _num(raw["lawd_cd"]), _num(raw["deal_ym"]), _num(raw["dealDay"])
    out["build_year"], out["area_m2"] = _num(raw["buildYear"]), area
    out["deposit_manwon"], out["monthly_rent_manwon"] = dep, monthly
    out["contract_type"] = raw.get("contractType", "신규")
    out["contract_term"] = raw.get("contractTerm", "협의")
    out["use_rr_right"] = raw.get("useRRRight", "해당없음")
    out["dealing_type"], out["source_supported"] = "임대차 실거래", True
    return out[out["house_type"].notna() & area.between(5, 500) & dep.ge(0) & monthly.ge(0) & (dep + monthly > 0)].reset_index(drop=True)


def _normalize_sh_trade(raw: pd.DataFrame) -> pd.DataFrame:
    out = _standard_frame(len(raw))
    price, gross, land = _num(raw["dealAmount"]), _num(raw["totalFloorAr"]), _num(raw["plottageAr"])
    out["source_dataset"] = "rtms_sh_trade"
    out["source_index"] = raw.index
    out["source_transaction_type"], out["sale_subtype"] = "매매", "일반매매"
    out["house_type"] = raw["houseType"].map({"단독": "단독주택", "다가구": "다가구주택"})
    out["sido"], out["gugun"], out["dong"] = "서울", raw["sggNm"], raw["umdNm"]
    out["lawd_cd"], out["deal_ym"], out["deal_day"] = _num(raw["lawd_cd"]), _num(raw["deal_ym"]), _num(raw["dealDay"])
    out["build_year"], out["area_m2"] = _num(raw["buildYear"]), gross
    out["sale_price_manwon"], out["land_area_m2"], out["building_total_area_m2"] = price, land, gross
    out["dealing_type"] = raw.get("dealingGbn", "중개거래")
    out["source_supported"] = True
    return out[_valid_trade(raw).to_numpy() & out["house_type"].notna() & price.gt(0) & gross.between(5, 2000) & land.between(5, 5000)].reset_index(drop=True)


def _normalize_apt_trade(raw: pd.DataFrame, code_map: dict[int, str]) -> pd.DataFrame:
    out = _standard_frame(len(raw))
    price, area = _num(raw["dealAmount"]), _num(raw["excluUseAr"])
    code = _num(raw["lawd_cd"])
    out["source_dataset"] = "rtms_apt_trade"
    out["source_index"] = raw.index
    out["source_transaction_type"], out["sale_subtype"], out["house_type"] = "매매", "일반매매", "아파트"
    out["sido"], out["gugun"], out["dong"] = "서울", code.map(code_map), raw["umdNm"]
    out["lawd_cd"], out["deal_ym"], out["deal_day"] = code, _num(raw["deal_ym"]), _num(raw["dealDay"])
    out["build_year"], out["area_m2"], out["floor"] = _num(raw["buildYear"]), area, _num(raw["floor"])
    out["building_name"], out["sale_price_manwon"] = raw["aptNm"], price
    out["dealing_type"] = raw.get("dealingGbn", "중개거래")
    out["source_supported"] = True
    return out[_valid_trade(raw).to_numpy() & price.gt(0) & area.between(10, 350) & out["gugun"].notna()].reset_index(drop=True)


def _normalize_silv_trade(raw: pd.DataFrame) -> pd.DataFrame:
    out = _standard_frame(len(raw))
    price, area = _num(raw["dealAmount"]), _num(raw["excluUseAr"])
    out["source_dataset"] = "rtms_silv_trade"
    out["source_index"] = raw.index
    out["source_transaction_type"], out["sale_subtype"], out["house_type"] = "매매", "분양권전매", "아파트"
    out["sido"], out["gugun"], out["dong"] = "서울", raw["sggNm"], raw["umdNm"]
    out["lawd_cd"], out["deal_ym"], out["deal_day"] = _num(raw["lawd_cd"]), _num(raw["deal_ym"]), _num(raw["dealDay"])
    out["build_year"] = np.clip((_num(raw["dealYear"]).fillna(CURRENT_YEAR) + 1), CURRENT_YEAR, CURRENT_YEAR + 3)
    out["area_m2"], out["floor"], out["building_name"] = area, _num(raw["floor"]), raw["aptNm"]
    out["sale_price_manwon"] = price
    out["dealing_type"] = raw.get("dealingGbn", "중개거래")
    out["source_supported"] = True
    return out[_valid_trade(raw).to_numpy() & price.gt(0) & area.between(10, 350)].reset_index(drop=True)


def _normalize_offi_rent(raw: pd.DataFrame) -> pd.DataFrame:
    out = _standard_frame(len(raw))
    dep, monthly, area = _num(raw["deposit"]), _num(raw["monthlyRent"]), _num(raw["excluUseAr"])
    out["source_dataset"] = "rtms_offi_rent"
    out["source_index"] = raw.index
    out["source_transaction_type"] = np.where(monthly.eq(0), "전세", "월세")
    out["sale_subtype"], out["house_type"] = "해당없음", "오피스텔"
    out["sido"], out["gugun"], out["dong"] = "서울", raw["sggNm"], raw["umdNm"]
    out["lawd_cd"], out["deal_ym"], out["deal_day"] = _num(raw["lawd_cd"]), _num(raw["deal_ym"]), _num(raw["dealDay"])
    out["build_year"], out["area_m2"], out["floor"] = _num(raw["buildYear"]), area, _num(raw["floor"])
    out["building_name"], out["deposit_manwon"], out["monthly_rent_manwon"] = raw["offiNm"], dep, monthly
    out["contract_type"], out["contract_term"], out["use_rr_right"] = raw.get("contractType", "신규"), raw.get("contractTerm", "협의"), raw.get("useRRRight", "해당없음")
    out["dealing_type"], out["source_supported"] = "임대차 실거래", True
    return out[area.between(5, 300) & dep.ge(0) & monthly.ge(0) & (dep + monthly > 0)].reset_index(drop=True)


def _normalize_offi_trade(raw: pd.DataFrame) -> pd.DataFrame:
    out = _standard_frame(len(raw))
    price, area = _num(raw["dealAmount"]), _num(raw["excluUseAr"])
    out["source_dataset"] = "rtms_offi_trade"
    out["source_index"] = raw.index
    out["source_transaction_type"], out["sale_subtype"], out["house_type"] = "매매", "일반매매", "오피스텔"
    out["sido"], out["gugun"], out["dong"] = "서울", raw["sggNm"], raw["umdNm"]
    out["lawd_cd"], out["deal_ym"], out["deal_day"] = _num(raw["lawd_cd"]), _num(raw["deal_ym"]), _num(raw["dealDay"])
    out["build_year"], out["area_m2"], out["floor"] = _num(raw["buildYear"]), area, _num(raw["floor"])
    out["building_name"], out["sale_price_manwon"] = raw["offiNm"], price
    out["dealing_type"] = raw.get("dealingGbn", "중개거래")
    out["source_supported"] = True
    return out[_valid_trade(raw).to_numpy() & price.gt(0) & area.between(5, 300)].reset_index(drop=True)


def _normalize_gyeonggi_offi_rent(raw: pd.DataFrame) -> pd.DataFrame:
    out = _standard_frame(len(raw))
    dep, monthly, area = _num(raw["보증금액"]), _num(raw["월세금액"]), _num(raw["전용면적"])
    legal = _num(raw["읍면동코드"])
    city = raw["시군구명"].astype(str).str.replace("경기도", "", regex=False).str.strip()
    out["source_dataset"] = "gyeonggi_real_estate_portal_officetel_jeonse"
    out["source_index"] = raw.index
    out["source_transaction_type"] = np.where(monthly.eq(0), "전세", "월세")
    out["sale_subtype"], out["house_type"] = "해당없음", "오피스텔"
    out["sido"], out["gugun"], out["dong"] = "경기", city, raw["읍면동리명"]
    out["lawd_cd"] = (legal // 100000).where(legal.notna(), _num(raw["시군구코드"]))
    contract_date = _num(raw["계약일"]).fillna(20240101).astype(int).astype(str).str.zfill(8)
    out["deal_ym"], out["deal_day"] = _num(contract_date.str[:6]), _num(contract_date.str[6:8])
    out["build_year"], out["area_m2"], out["floor"] = _num(raw["건축년도"]), area, _num(raw["층수"])
    out["building_name"], out["deposit_manwon"], out["monthly_rent_manwon"] = raw["오피스텔단지명"], dep, monthly
    out["contract_type"] = raw.get("신규여부", "신규").fillna(raw.get("갱신여부", "확인필요"))
    out["contract_term"], out["use_rr_right"] = raw.get("계약기간", "협의"), "확인필요"
    out["dealing_type"], out["source_supported"] = "경기도 확정일자", True
    return out[area.between(5, 300) & dep.ge(0) & monthly.ge(0) & (dep + monthly > 0) & city.ne("")].reset_index(drop=True)


def load_source_pools() -> dict[str, pd.DataFrame]:
    """Load, clean and normalize every downloaded transaction source used."""
    sh_rent_raw = _read_csv(DATASETS.rtms_sh_rent)
    sh_trade_raw = _read_csv(DATASETS.rtms_sh_trade)
    offi_rent_raw = _read_csv(DATASETS.rtms_officetel_rent)
    offi_trade_raw = _read_csv(DATASETS.rtms_officetel_trade)
    code_map = _district_map(sh_rent_raw, sh_trade_raw, offi_rent_raw, offi_trade_raw)
    pools = {
        "sh_rent": _normalize_sh_rent(sh_rent_raw),
        "sh_trade": _normalize_sh_trade(sh_trade_raw),
        "apt_trade": _normalize_apt_trade(_read_csv(DATASETS.rtms_apt_trade), code_map),
        "silv_trade": _normalize_silv_trade(_read_csv(DATASETS.rtms_silv_trade)),
        "offi_rent": _normalize_offi_rent(offi_rent_raw),
        "offi_trade": _normalize_offi_trade(offi_trade_raw),
        "gyeonggi_offi_rent": _normalize_gyeonggi_offi_rent(_read_csv(DATASETS.gyeonggi_officetel_jeonse)),
    }
    for name, frame in pools.items():
        if frame.empty:
            raise ValueError(f"usable source pool is empty: {name}")
    return pools


def load_latest_apt_jeonse_ratios() -> dict[str, float]:
    """Read the latest downloaded KB apartment sale-to-jeonse ratios.

    Keys include province/city/district labels found in any ``지역`` hierarchy
    column. Values are fractions (for example, 55.2% becomes ``0.552``).
    """
    df = _read_csv(DATASETS.kb_apt_average_sale_to_jeonse_price_ratio_monthly)
    value_columns = [c for c in df.columns if "년" in str(c) and "월" in str(c)]
    if not value_columns:
        return {}
    latest = value_columns[-1]
    values = _num(df[latest])
    region_columns = [c for c in df.columns if str(c).startswith("지역")]
    result: dict[str, float] = {}
    for idx, value in values.items():
        if not np.isfinite(value) or not 20 <= value <= 100:
            continue
        for column in region_columns:
            key = str(df.at[idx, column]).strip()
            if key and key not in {"nan", "지역", "전국"}:
                result[key] = float(value) / 100.0
    return result


def load_sido_price_factors() -> dict[str, float]:
    """KB 시도별 아파트 평균 전세가격(만원/㎡)을 서울 대비 가격계수로 변환한다."""
    df = _read_csv(DATASETS.kb_apt_jeonse_average_price_by_region_monthly)
    value_columns = [c for c in df.columns if "년" in str(c) and "월" in str(c)]
    if not value_columns:
        return {sido: 1.0 for sido in EXPECTED_SIDOS}
    latest = value_columns[-1]
    values = _num(df[latest])
    result: dict[str, float] = {}
    region_columns = [c for c in df.columns if str(c).startswith("지역")]
    for sido in EXPECTED_SIDOS:
        mask = pd.Series(False, index=df.index)
        for column in region_columns:
            mask |= df[column].astype(str).str.strip().eq(sido)
        candidates = values[mask & values.gt(0)]
        if len(candidates):
            result[sido] = float(candidates.iloc[0])
    seoul = result.get("서울") or max(result.values(), default=1.0)
    median = float(np.median(list(result.values()))) if result else seoul
    return {
        sido: float(np.clip(result.get(sido, median) / seoul, 0.22, 1.15))
        for sido in EXPECTED_SIDOS
    }


def _balanced_region_assignment(
    house_types: np.ndarray,
    transaction_types: np.ndarray,
    catalog: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """각 주택유형×거래유형 층 안에서 시군구를 최대한 균등 배정한다."""
    assigned = np.empty(len(transaction_types), dtype=int)
    region_indexes = np.arange(len(catalog))
    strata = pd.DataFrame({"house": house_types, "transaction": transaction_types})
    for row_indexes in strata.groupby(["house", "transaction"], sort=False).indices.values():
        row_indexes = np.asarray(row_indexes, dtype=int)
        full, remainder = divmod(len(row_indexes), len(catalog))
        choices = np.tile(region_indexes, full)
        if remainder:
            choices = np.concatenate([
                choices,
                rng.choice(region_indexes, size=remainder, replace=False),
            ])
        rng.shuffle(choices)
        assigned[row_indexes] = choices
    return catalog.iloc[assigned].reset_index(drop=True)


def _balanced_price_diversity_factors(
    house_types: np.ndarray,
    transaction_types: np.ndarray,
    assigned_regions: pd.DataFrame,
    rng: np.random.Generator,
) -> np.ndarray:
    """같은 지역·유형 안에도 저가~고가 가격대가 섞이도록 층화 계수를 만든다."""
    factors = np.ones(len(house_types), dtype=float)
    bands = np.array([0.74, 0.88, 1.00, 1.16, 1.36])
    strata = pd.DataFrame({
        "sido": assigned_regions["sido"].astype(str).to_numpy(),
        "gugun": assigned_regions["gugun"].astype(str).to_numpy(),
        "house": house_types,
        "transaction": transaction_types,
    })
    for row_indexes in strata.groupby(
        ["sido", "gugun", "house", "transaction"], sort=False
    ).indices.values():
        row_indexes = np.asarray(row_indexes, dtype=int)
        offset = int(rng.integers(0, len(bands)))
        choices = bands[(np.arange(len(row_indexes)) + offset) % len(bands)].copy()
        rng.shuffle(choices)
        factors[row_indexes] = choices
    factors *= rng.lognormal(0, 0.025, len(factors))
    return np.clip(factors, 0.68, 1.48)


def _spread_region_coordinates(
    assigned_regions: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map candidates to public road-address anchors within each district."""
    lat = assigned_regions["lat"].astype(float).to_numpy().copy()
    lng = assigned_regions["lng"].astype(float).to_numpy().copy()
    sources = assigned_regions["coordinate_source"].astype(str).to_numpy().copy()
    road_addresses = np.full(len(assigned_regions), "", dtype=object)
    jibun_addresses = np.full(len(assigned_regions), "", dtype=object)
    anchors = load_nationwide_coordinate_anchors()
    anchor_groups = {
        (str(sido), str(gugun)): part[
            ["lat", "lng", "road_address", "jibun_address"]
        ].to_numpy(object)
        for (sido, gugun), part in anchors.groupby(["sido", "gugun"], sort=False)
    } if len(anchors) else {}
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    groups = assigned_regions.groupby(["sido", "gugun"], sort=False).indices
    for (sido, gugun), row_indexes in groups.items():
        row_indexes = np.asarray(row_indexes, dtype=int)
        pool = anchor_groups.get((str(sido), str(gugun)))
        if pool is not None and len(pool):
            selected = pool[rng.integers(0, len(pool), size=len(row_indexes))]
            lat[row_indexes] = selected[:, 0].astype(float)
            lng[row_indexes] = selected[:, 1].astype(float)
            road_addresses[row_indexes] = selected[:, 2].astype(str)
            jibun_addresses[row_indexes] = selected[:, 3].astype(str)
            sources[row_indexes] = "공공 CCTV 도로명주소 기준 위치 앵커"
            continue
        lat_radius, lng_radius = REGION_COORDINATE_RADII.get(
            str(sido), (0.040, 0.052)
        )
        sequence = np.arange(len(row_indexes), dtype=float)
        radius = np.sqrt((sequence + 0.5) / max(len(row_indexes), 1))
        theta = float(rng.uniform(0, 2 * np.pi)) + sequence * golden_angle
        rng.shuffle(row_indexes)
        lat[row_indexes] += lat_radius * radius * np.sin(theta)
        lng[row_indexes] += lng_radius * radius * np.cos(theta)
        sources[row_indexes] = np.char.add(
            sources[row_indexes].astype(str), "+golden-angle 균등분산"
        )
    return (
        np.clip(lat, 32.5, 39.5),
        np.clip(lng, 124.0, 132.0),
        sources,
        road_addresses,
        jibun_addresses,
    )


def _assign_legal_dongs(
    assigned_regions: pd.DataFrame, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign real legal-dong names/codes within the selected district."""
    path = config.DATA_RAW / "real_estate" / "national_legal_dong_codes_20260630.csv"
    names = np.full(len(assigned_regions), "행정동 미확인", dtype=object)
    codes = np.full(len(assigned_regions), "", dtype=object)
    if not path.exists():
        return names, codes
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    frame = frame[frame["읍면동명"].ne("")].copy()
    frame["sido"] = frame["시도명"].map(SIDO_ALIASES).fillna(frame["시도명"])

    def norm(value: str) -> str:
        return str(value).replace(" ", "").replace("특별자치", "").replace("시", "")

    frame["gugun_norm"] = frame["시군구명"].map(norm)
    pools = {
        (str(sido), str(gugun)): part[["읍면동명", "법정동코드"]].drop_duplicates().to_numpy(object)
        for (sido, gugun), part in frame.groupby(["sido", "gugun_norm"], sort=False)
    }
    groups = assigned_regions.groupby(["sido", "gugun"], sort=False).indices
    for (sido, gugun), row_indexes in groups.items():
        indexes = np.asarray(row_indexes, dtype=int)
        pool = pools.get((str(sido), norm(str(gugun))))
        if pool is None or not len(pool):
            continue
        selected = pool[rng.integers(0, len(pool), size=len(indexes))]
        names[indexes] = selected[:, 0]
        codes[indexes] = selected[:, 1]
    return names, codes


def _recency_weights(deal_ym: pd.Series, half_life_months: float) -> np.ndarray:
    ordinals = _month_ordinal(deal_ym)
    age = np.maximum(ordinals.max() - ordinals, 0)
    weights = np.exp(-np.log(2) * age / max(float(half_life_months), 1.0))
    return weights / weights.sum()


def _allocate_combinations(n: int, house_weights: dict[str, float], txn_weights: dict[str, float]) -> list[tuple[str, str, int]]:
    combos = [(h, t, float(hw) * float(txn_weights[t])) for h, hw in house_weights.items() for t in txn_weights]
    raw = np.array([n * x[2] for x in combos])
    counts = np.floor(raw).astype(int)
    if n >= len(combos):
        counts = np.maximum(counts, 1)
    delta = n - int(counts.sum())
    if delta > 0:
        order = np.argsort(-(raw - np.floor(raw)))
        for i in order[:delta]:
            counts[i] += 1
    elif delta < 0:
        for i in np.argsort(raw - np.floor(raw)):
            removable = min(counts[i] - 1, -delta)
            if removable > 0:
                counts[i] -= removable
                delta += removable
            if delta == 0:
                break
    return [(h, t, int(c)) for (h, t, _), c in zip(combos, counts) if c > 0]


def _sample(pool: pd.DataFrame, n: int, rng: np.random.Generator, half_life: float) -> pd.DataFrame:
    idx = rng.choice(len(pool), size=n, replace=True, p=_recency_weights(pool["deal_ym"], half_life))
    return pool.iloc[idx].reset_index(drop=True)


def _select_anchors(house_type: str, txn: str, n: int, pools: dict[str, pd.DataFrame], rng, half_life: float) -> pd.DataFrame:
    if house_type == "아파트":
        sale_pool = pd.concat([pools["apt_trade"], pools["silv_trade"].sample(frac=0.35, random_state=17)], ignore_index=True)
        anchor = _sample(sale_pool, n, rng, half_life)
        direct = txn == "매매"
    elif house_type == "오피스텔":
        if txn == "매매":
            anchor, direct = _sample(pools["offi_trade"], n, rng, half_life), True
        else:
            rent_pool = pd.concat([pools["offi_rent"], pools["gyeonggi_offi_rent"]], ignore_index=True)
            rent_pool = rent_pool[rent_pool["source_transaction_type"] == txn]
            anchor, direct = _sample(rent_pool, n, rng, half_life), True
    elif house_type in ("단독주택", "다가구주택"):
        key = "sh_trade" if txn == "매매" else "sh_rent"
        pool = pools[key]
        pool = pool[(pool["house_type"] == house_type) & (pool["source_transaction_type"] == txn)]
        anchor, direct = _sample(pool, n, rng, half_life), True
    else:
        # No directly downloaded row-house/multi-household file is present.
        # Apartment unit transactions provide the unit-level structural and
        # spatial anchor; type-specific scale factors are applied later.
        anchor, direct = _sample(pools["apt_trade"], n, rng, half_life), False
    anchor = anchor.copy()
    anchor["target_house_type"] = house_type
    anchor["target_transaction_type"] = txn
    anchor["source_supported"] = direct
    return anchor


def _stable_source_hash(row) -> str:
    payload = f"{row.source_dataset}|{row.source_index}|{row.lawd_cd}|{row.deal_ym}|{row.area_m2}"
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=10).hexdigest()


def _region_center(sido: str, gugun: str) -> tuple[float, float]:
    if sido == "서울" and gugun in SEOUL_DISTRICT_CENTERS:
        return SEOUL_DISTRICT_CENTERS[gugun]
    digest = hashlib.blake2b(f"{sido}|{gugun}".encode("utf-8"), digest_size=4).digest()
    a, b = int.from_bytes(digest[:2], "big"), int.from_bytes(digest[2:], "big")
    if sido == "경기":
        return 37.42 + (a / 65535 - 0.5) * 0.80, 127.18 + (b / 65535 - 0.5) * 1.10
    return 37.5665, 126.9780


def _region_hazard_for(sido: np.ndarray, gugun: np.ndarray) -> np.ndarray:
    try:
        stats = load_region_accident_stats()
    except Exception:
        return np.full(len(gugun), 0.02)
    stats = stats.copy()
    mapping = stats.groupby(["sido", "gugun"])["accident_rate_pct"].median().to_dict()
    fallback = float(stats["accident_rate_pct"].median()) if not stats.empty else 2.0
    values = []
    for s, g in zip(sido, gugun):
        matches = [v for (ss, gg), v in mapping.items() if str(s) in str(ss) and str(g) == str(gg)]
        values.append(float(matches[0] if matches else fallback) / 100.0)
    return np.asarray(values)


def _parse_weight_spec(spec: str | None, defaults: dict[str, float]) -> dict[str, float]:
    if not spec:
        return defaults.copy()
    result: dict[str, float] = {}
    for item in spec.split(","):
        key, sep, value = item.strip().partition("=")
        if not sep or key not in defaults:
            raise ValueError(f"invalid weight item: {item!r}; allowed={list(defaults)}")
        result[key] = float(value)
    if not result or any(v < 0 for v in result.values()) or sum(result.values()) <= 0:
        raise ValueError("weights must contain non-negative values with a positive sum")
    total = sum(result.values())
    return {k: v / total for k, v in result.items() if v > 0}


def _hedonic_features(frame: pd.DataFrame, area=None, build_year=None, house_type=None) -> pd.DataFrame:
    """Build the small, auditable feature set used by the price model."""
    areas = _num(frame["area_m2"]).fillna(50).to_numpy(float) if area is None else np.asarray(area, dtype=float)
    years = _num(frame["build_year"]).fillna(2000).to_numpy(float) if build_year is None else np.asarray(build_year, dtype=float)
    types = frame["house_type"].fillna("아파트").astype(str).to_numpy() if house_type is None else np.asarray(house_type, dtype=str)
    return pd.DataFrame({
        "log_area": np.log(np.maximum(areas, 1)),
        "building_age": np.maximum(CURRENT_YEAR - years, 0),
        "floor": _num(frame["floor"]).fillna(-1).to_numpy(float),
        "deal_month": _month_ordinal(frame["deal_ym"]),
        "gugun": frame["gugun"].fillna("미상").astype(str).to_numpy(),
        "house_type": types,
    })


def fit_hedonic_price_model(pools: dict[str, pd.DataFrame], random_state: int = 42) -> dict:
    """Fit a time-holdout gradient-boosted hedonic price-per-area model.

    The latest roughly 15% of transactions in each source house type are held
    out. Their log residuals become the empirical innovation distribution used
    during synthesis, preventing deterministic predictions from collapsing the
    price variance.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import r2_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    pieces = []
    limits = {"apt_trade": 60000, "offi_trade": 30000, "sh_trade": 20000}
    for key in ("apt_trade", "offi_trade", "sh_trade"):
        part = pools[key].copy()
        if len(part) > limits[key]:
            part = part.sample(n=limits[key], random_state=random_state)
        pieces.append(part)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The behavior of DataFrame concatenation", category=FutureWarning)
        sales = pd.concat(pieces, ignore_index=True)
    price = _num(sales["sale_price_manwon"]).to_numpy(float)
    area = _num(sales["area_m2"]).to_numpy(float)
    valid = np.isfinite(price) & np.isfinite(area) & (price > 0) & (area > 0)
    sales = sales.loc[valid].reset_index(drop=True)
    price, area = price[valid], area[valid]
    y = np.log(price / area)
    features = _hedonic_features(sales)

    ordinals = features["deal_month"].to_numpy()
    holdout = np.zeros(len(sales), dtype=bool)
    for property_type in sales["house_type"].unique():
        mask = sales["house_type"].to_numpy() == property_type
        cutoff = np.quantile(ordinals[mask], 0.85)
        holdout |= mask & (ordinals >= cutoff)
    if holdout.mean() > 0.30 or holdout.sum() < 1000:
        order = np.argsort(ordinals)
        holdout[:] = False
        holdout[order[int(len(order) * 0.85):]] = True

    categorical = ["gugun", "house_type"]
    numeric = ["log_area", "building_age", "floor", "deal_month"]
    preprocess = ColumnTransformer([
        ("numeric", "passthrough", numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=30, sparse_output=False), categorical),
    ])
    model = Pipeline([
        ("preprocess", preprocess),
        ("regressor", HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.07, max_iter=140,
            max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=0.15,
            random_state=random_state,
        )),
    ])
    model.fit(features.loc[~holdout], y[~holdout])
    pred = model.predict(features.loc[holdout])
    residual = y[holdout] - pred
    true_price = np.exp(y[holdout])
    pred_price = np.exp(pred)
    metrics = {
        "holdout_rows": int(holdout.sum()),
        "mdape_pct": float(np.median(np.abs(pred_price - true_price) / true_price) * 100),
        "r2": float(r2_score(y[holdout], pred)),
    }
    residuals: dict[str, np.ndarray] = {"__all__": residual}
    holdout_types = sales.loc[holdout, "house_type"].astype(str).to_numpy()
    for property_type in np.unique(holdout_types):
        values = residual[holdout_types == property_type]
        residuals[property_type] = values if len(values) >= 30 else residual
    return {"model": model, "residuals": residuals, "metrics": metrics, "training_rows": int((~holdout).sum())}


def _predict_hedonic_market(
    fitted: dict,
    anchor: pd.DataFrame,
    area: np.ndarray,
    build_year: np.ndarray,
    target_house_type: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    base_type = np.where(np.isin(target_house_type, ["다세대주택", "연립주택"]), "아파트", target_house_type)
    features = _hedonic_features(anchor, area=area, build_year=build_year, house_type=base_type)
    log_ppm2 = fitted["model"].predict(features)
    innovations = np.empty(len(anchor))
    for property_type in np.unique(base_type):
        mask = base_type == property_type
        pool = fitted["residuals"].get(str(property_type), fitted["residuals"]["__all__"])
        innovations[mask] = rng.choice(pool, size=int(mask.sum()), replace=True)
    # Winsorization prevents a single holdout outlier from creating an absurd
    # listing while preserving the central 99% residual distribution.
    innovations = np.clip(innovations, np.quantile(fitted["residuals"]["__all__"], 0.005), np.quantile(fitted["residuals"]["__all__"], 0.995))
    return np.exp(log_ppm2 + innovations) * area


def generate_properties(
    n: int,
    rng: np.random.Generator,
    rent_path: Path | str | None = None,
    trade_path: Path | str | None = None,
    recency_half_life_months: float = 18.0,
    house_weights: dict[str, float] | None = None,
    transaction_weights: dict[str, float] | None = None,
    price_model: str = "hedonic",
) -> pd.DataFrame:
    """Generate exactly ``n`` diverse sale/jeonse/monthly-rent listings.

    ``rent_path`` and ``trade_path`` remain accepted for backward API
    compatibility.  The v2 generator intentionally uses the canonical paths in
    :mod:`src.data_sources` so that all downloaded source types participate.
    """
    del rent_path, trade_path
    if n <= 0:
        raise ValueError("n_properties must be a positive integer")
    hweights = house_weights or DEFAULT_HOUSE_WEIGHTS
    tweights = transaction_weights or DEFAULT_TRANSACTION_WEIGHTS
    pools = load_source_pools()
    if price_model not in {"hedonic", "none"}:
        raise ValueError("price_model must be 'hedonic' or 'none'")
    fitted_price = fit_hedonic_price_model(pools, int(rng.integers(0, 2**31 - 1))) if price_model == "hedonic" else None
    parts = [
        _select_anchors(h, t, count, pools, rng, recency_half_life_months)
        for h, t, count in _allocate_combinations(n, hweights, tweights)
    ]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*behavior of DataFrame concatenation.*",
            category=FutureWarning,
        )
        anchor = pd.concat(parts, ignore_index=True)
    anchor = anchor.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)

    house = anchor["target_house_type"].astype(str).to_numpy()
    txn = anchor["target_transaction_type"].astype(str).to_numpy()
    source_sido = anchor["sido"].fillna("서울").astype(str).to_numpy()
    region_catalog = load_nationwide_region_catalog()
    assigned_regions = _balanced_region_assignment(house, txn, region_catalog, rng)
    sido = assigned_regions["sido"].astype(str).to_numpy()
    gugun = assigned_regions["gugun"].astype(str).to_numpy()
    dong, real_legal_dong_code = _assign_legal_dongs(assigned_regions, rng)
    region_id = assigned_regions["region_id"].astype(int).to_numpy()
    price_factors = load_sido_price_factors()
    source_factor = np.array([price_factors.get(value, 1.0) for value in source_sido])
    target_factor = np.array([price_factors.get(value, 1.0) for value in sido])
    regional_price_factor = np.clip(target_factor / np.maximum(source_factor, 0.05), 0.20, 1.50)
    price_diversity_factor = _balanced_price_diversity_factors(
        house, txn, assigned_regions, rng
    )
    price_adjustment_factor = np.clip(
        regional_price_factor * price_diversity_factor, 0.16, 2.10
    )
    is_sale, is_jeonse, is_monthly = txn == "매매", txn == "전세", txn == "월세"
    direct = anchor["source_supported"].astype(bool).to_numpy()
    source_area = _num(anchor["area_m2"]).fillna(50).to_numpy(float)
    sigma = np.where(direct, 0.025, 0.060)
    area = source_area * rng.lognormal(0, sigma)
    area = np.where(house == "오피스텔", np.clip(area, 10, 180), area)
    area = np.where(np.isin(house, ["다세대주택", "연립주택"]), np.clip(area, 15, 250), area)
    area = np.clip(area, 10, 1500)

    build_year = _num(anchor["build_year"]).to_numpy(float)
    build_year = np.where(np.isfinite(build_year), build_year, np.where(house == "아파트", 2006, 2001))
    build_year = np.clip(np.rint(build_year + rng.choice([-1, 0, 0, 0, 1], n)), 1900, CURRENT_YEAR + 3).astype(int)

    source_sale_original = _num(anchor["sale_price_manwon"]).fillna(0).to_numpy(float)
    source_dep_original = _num(anchor["deposit_manwon"]).fillna(0).to_numpy(float)
    source_monthly_original = _num(anchor["monthly_rent_manwon"]).fillna(0).to_numpy(float)
    source_sale = source_sale_original * price_adjustment_factor
    source_dep = source_dep_original * price_adjustment_factor
    source_monthly = source_monthly_original * price_adjustment_factor
    proxy_factor = np.where(house == "다세대주택", 0.58, np.where(house == "연립주택", 0.66, 1.0))
    empirical_market = source_sale * (area / np.maximum(source_area, 1)) ** 0.82 * proxy_factor * rng.lognormal(0, 0.045, n)
    if fitted_price is not None:
        ml_market = (_predict_hedonic_market(
            fitted_price, anchor, area, build_year, house, rng
        ) * proxy_factor * price_adjustment_factor)
        # Direct sale anchors retain most of their local empirical value. The
        # model supplies a smooth hedonic correction; proxy/lease rows use the
        # model as their price anchor.
        blended_sale = np.exp(0.70 * np.log(np.maximum(empirical_market, 1)) + 0.30 * np.log(np.maximum(ml_market, 1)))
        market_unit = np.where(source_sale > 0, blended_sale, ml_market)
    else:
        ml_market = np.zeros(n)
        market_unit = np.where(source_sale > 0, empirical_market, 0)

    conversion_rate = np.clip(rng.normal(0.055, 0.009, n), 0.03, 0.09)
    kb_ratio_map = load_latest_apt_jeonse_ratios()
    apt_ratio = np.array([kb_ratio_map.get(g, kb_ratio_map.get(s, 0.55)) for s, g in zip(sido, gugun)])
    ratio_mean = np.select(
        [house == "아파트", house == "오피스텔", house == "단독주택", house == "다가구주택", house == "다세대주택", house == "연립주택"],
        [apt_ratio, 0.64, 0.50, 0.58, 0.64, 0.62], default=0.58,
    )
    jeonse_ratio = np.clip(ratio_mean + rng.normal(0, 0.07, n), 0.32, 0.85)

    exact_lease = (~is_sale) & (source_dep + source_monthly > 0)
    equivalent_source = source_dep + source_monthly * 12 / conversion_rate
    inferred_market = equivalent_source / np.maximum(jeonse_ratio, 0.2)
    if fitted_price is not None:
        seoul_exact = exact_lease & (sido == "서울")
        market_unit = np.where(seoul_exact, np.exp(0.65 * np.log(np.maximum(inferred_market, 1)) + 0.35 * np.log(np.maximum(ml_market, 1))), market_unit)
    market_unit = np.where((market_unit <= 0) & exact_lease, inferred_market, market_unit)
    market_unit = np.maximum(market_unit, np.where(np.isin(house, ["아파트", "오피스텔"]), 3000, 5000))

    deposit = np.zeros(n)
    monthly = np.zeros(n)
    exact_dep = source_dep * rng.lognormal(0, 0.04, n)
    exact_month = source_monthly + (source_dep - exact_dep) * conversion_rate / 12
    exact_month = np.maximum(exact_month * rng.lognormal(0, 0.03, n), 0)
    deposit[exact_lease] = exact_dep[exact_lease]
    monthly[exact_lease] = exact_month[exact_lease]
    derived_lease = (~is_sale) & ~exact_lease
    equivalent = market_unit * jeonse_ratio
    deposit[derived_lease & is_jeonse] = equivalent[derived_lease & is_jeonse] * rng.lognormal(0, 0.035, (derived_lease & is_jeonse).sum())
    monthly_share = np.clip(rng.beta(2.0, 5.0, n), 0.04, 0.65)
    deposit[derived_lease & is_monthly] = equivalent[derived_lease & is_monthly] * monthly_share[derived_lease & is_monthly]
    monthly[derived_lease & is_monthly] = (equivalent[derived_lease & is_monthly] - deposit[derived_lease & is_monthly]) * conversion_rate[derived_lease & is_monthly] / 12
    monthly[is_jeonse | is_sale] = 0
    deposit[is_sale] = 0
    deposit = _round_to(deposit, 10)
    monthly = _round_to(monthly, 1)
    deposit[is_jeonse] = np.maximum(deposit[is_jeonse], 100)
    deposit[is_monthly] = np.maximum(deposit[is_monthly], 10)
    monthly[is_monthly] = np.maximum(monthly[is_monthly], 1)

    units = np.select(
        [house == "아파트", house == "오피스텔", house == "단독주택", house == "다가구주택", house == "다세대주택", house == "연립주택"],
        [rng.integers(60, 1201, n), rng.integers(20, 501, n), rng.choice([1, 1, 1, 2, 3], n), rng.integers(3, 31, n), rng.integers(6, 51, n), rng.integers(4, 31, n)],
        default=1,
    ).astype(int)
    multi_building = house == "다가구주택"
    market_price = market_unit.copy()
    market_price[multi_building & ~is_sale] *= units[multi_building & ~is_sale]
    market_price = np.maximum(_round_to(market_price, 100), np.where(multi_building, deposit * 1.05 * units, deposit * 1.05))
    sale_price = np.where(is_sale, market_price, 0)

    source_land = _num(anchor["land_area_m2"]).to_numpy(float)
    source_gross = _num(anchor["building_total_area_m2"]).to_numpy(float)
    building_total_area = np.where(np.isfinite(source_gross), source_gross * rng.lognormal(0, 0.03, n), area * units * rng.uniform(1.12, 1.38, n))
    building_total_area = np.maximum(building_total_area, area)
    land_area = np.where(np.isfinite(source_land), source_land * rng.lognormal(0, 0.035, n), building_total_area / rng.uniform(120, 340, n) * 100)
    land_area = np.clip(land_area, 10, 10000)

    floor_anchor = _num(anchor["floor"]).to_numpy(float)
    total_floors = np.select(
        [house == "아파트", house == "오피스텔", house == "단독주택", house == "다가구주택", house == "다세대주택", house == "연립주택"],
        [rng.integers(5, 50, n), rng.integers(5, 35, n), rng.integers(1, 4, n), rng.integers(2, 7, n), rng.integers(3, 8, n), rng.integers(3, 7, n)], default=5,
    ).astype(int)
    total_floors = np.maximum(total_floors, np.nan_to_num(floor_anchor, nan=1).astype(int))
    current_floor = np.where(np.isfinite(floor_anchor), np.clip(np.rint(floor_anchor), 1, total_floors), [rng.integers(1, x + 1) for x in total_floors]).astype(int)
    basement = ((rng.random(n) < np.where(build_year < 2000, 0.25, 0.08)) & np.isin(house, ["단독주택", "다가구주택", "다세대주택", "연립주택"])).astype(int)
    rooms = np.clip(np.rint(area / np.where(house == "오피스텔", rng.uniform(18, 28, n), rng.uniform(16, 25, n))), 1, 12).astype(int)
    bathrooms = np.clip(np.rint(area / 55), 1, 5).astype(int)
    complex_buildings = np.where(np.isin(house, ["아파트", "오피스텔"]), rng.integers(1, 21, n), 1)

    building_name = np.array([
        f"분석후보 {g} {h} {i + 1:05d}"
        for i, (g, h) in enumerate(zip(gugun, house))
    ])
    building_use = np.array(house, dtype=object)
    structures = np.where(np.isin(house, ["아파트", "오피스텔", "다세대주택", "연립주택"]), "철근콘크리트구조", rng.choice(["철근콘크리트구조", "벽돌구조", "일반목구조", "블록구조"], n, p=[0.58, 0.27, 0.09, 0.06]))

    lat, lng, coordinate_source, map_road_address, map_jibun_address = _spread_region_coordinates(
        assigned_regions, rng
    )
    map_road_address = np.asarray([
        "" if str(value).strip().lower() in {"", "nan", "none"} else str(value).strip()
        for value in map_road_address
    ], dtype=object)
    map_jibun_address = np.asarray([
        "" if str(value).strip().lower() in {"", "nan", "none"} else str(value).strip()
        for value in map_jibun_address
    ], dtype=object)
    map_reference_address = np.where(
        map_road_address != "", map_road_address,
        np.where(
            map_jibun_address != "", map_jibun_address,
            [f"{s} {g} {d}" for s, g, d in zip(sido, gugun, dong)],
        ),
    )

    parking_ratio = np.select([house == "아파트", house == "오피스텔", house == "단독주택", house == "다가구주택"], [1.05, 0.70, 0.8, 0.55], default=0.65)
    parking_total = np.maximum(0, np.rint(units * np.clip(rng.normal(parking_ratio, 0.18), 0, 2))).astype(int)
    elevator_count = np.where(total_floors >= 5, np.maximum(1, np.rint(units / 120)), np.where((total_floors >= 4) & (rng.random(n) < 0.55), 1, 0)).astype(int)
    maintenance = np.maximum(0, area * np.where(house == "아파트", 0.12, np.where(house == "오피스텔", 0.16, 0.07)) + elevator_count * 0.8 + rng.normal(0, 2, n))
    maintenance[rng.random(n) < np.where(np.isin(house, ["단독주택", "다가구주택"]), 0.25, 0.04)] = 0
    maintenance = np.round(maintenance, 1)

    occupancy = np.clip(rng.beta(4, 1.8, n), 0.2, 1.0)
    occupied_other = np.minimum(np.rint((units - 1) * occupancy).astype(int), units - 1)
    senior_count = np.where(multi_building & ~is_sale, [rng.integers(0, x + 1) for x in occupied_other], 0).astype(int)
    my_rank = np.where(multi_building & ~is_sale, senior_count + 1, 1)
    equivalent_deposit = deposit + monthly * 12 / conversion_rate
    senior_deposit = np.where(multi_building, senior_count * equivalent_deposit * rng.lognormal(0, 0.16, n), 0)
    senior_deposit = np.minimum(_round_to(senior_deposit, 10), market_price * 0.72)
    has_mortgage = rng.random(n) < np.where(house == "다가구주택", 0.58, 0.43)
    mortgage_ltv = np.where(has_mortgage, 0.04 + 0.54 * rng.beta(2.0, 3.8, n), 0)
    senior_mortgage = _round_to(market_price * mortgage_ltv, 100)
    max_claim = _round_to(senior_mortgage * rng.uniform(1.05, 1.30, n), 100)

    transaction_value = np.where(is_sale, sale_price, np.where(deposit + monthly * 100 < 5000, deposit + monthly * 70, deposit + monthly * 100))
    broker_fee_rate = np.where(is_sale, np.where(transaction_value < 5000, 0.60, np.where(transaction_value < 20000, 0.50, 0.40)), np.where(transaction_value < 5000, 0.50, np.where(transaction_value < 10000, 0.40, 0.30)))
    broker_fee = np.round(transaction_value * broker_fee_rate / 100, 1)
    broker_vat = np.round(broker_fee * 0.1, 1)
    actual_expense = rng.choice([0, 3, 5, 10], n, p=[0.25, 0.30, 0.35, 0.10])
    onetime = broker_fee + broker_vat + actual_expense

    ref_dates = pd.to_datetime(
        _num(anchor["deal_ym"]).fillna(202601).astype(int).astype(str)
        + _num(anchor["deal_day"]).fillna(1).astype(int).clip(1, 28).astype(str).str.zfill(2),
        format="%Y%m%d", errors="coerce",
    ).fillna(pd.Timestamp("2026-01-01"))
    created_dates = ref_dates + pd.to_timedelta(rng.integers(1, 31, n), unit="D")
    updated_dates = created_dates + pd.to_timedelta(rng.integers(0, 15, n), unit="D")
    available_dates = updated_dates + pd.to_timedelta(rng.integers(0, 61, n), unit="D")
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    source_hash = np.array([_stable_source_hash(r) for r in anchor.itertuples(index=False)])
    generation_method = np.where(
        direct,
        "실거래 유형기반 부트스트랩+전국지역 가격보정",
        "교차자료 유형프록시+전국지역 가격보정",
    )
    price_method = np.where(
        fitted_price is not None,
        np.where(is_sale & direct, "실거래가+헤도닉 GBDT 혼합·홀드아웃 잔차", np.where(exact_lease, "임대 환산가치+헤도닉 GBDT·잔차", "헤도닉 GBDT+잔차×유형별 전세가율")),
        np.where(is_sale & direct, "동일유형 실거래가 면적·잡음 보정", np.where(exact_lease, "실제 임대료 환산가치 기반", "매매가×유형별 전세가율·전월세전환율")),
    )
    price_method = np.char.add(price_method.astype(str), "+KB 시도별 가격계수")
    model_metrics = fitted_price["metrics"] if fitted_price is not None else {"mdape_pct": np.nan, "r2": np.nan}
    privacy_distance = np.round(np.abs(np.log(area / np.maximum(source_area, 1))) + np.where(source_sale > 0, np.abs(np.log(np.maximum(market_unit, 1) / np.maximum(source_sale, 1))), 0.05), 5)

    direction = rng.choice(["남향", "남동향", "남서향", "동향", "서향", "북향"], n, p=[0.31, 0.18, 0.16, 0.14, 0.11, 0.10])
    illegal = rng.random(n) < np.clip((CURRENT_YEAR - np.minimum(build_year, CURRENT_YEAR)) / 650 + 0.012, 0.01, 0.09)
    ledger_discrepancy = rng.random(n) < 0.035
    trust = rng.random(n) < 0.018
    tax_present = rng.random(n) < 0.025
    private_rental = (~is_sale) & (rng.random(n) < 0.075)
    rental_guarantee = private_rental & (rng.random(n) < 0.88)
    debt_ratio = (deposit + senior_deposit + senior_mortgage) / np.maximum(market_price * config.AUCTION_RECOVERY_RATIO, 1)
    guarantee = is_jeonse & ~illegal & ~trust & ~tax_present & (debt_ratio <= 1.0)
    furnished = rng.random(n) < np.where(area <= 35, 0.62, 0.22)
    aircon = np.where(rng.random(n) < 0.84, np.maximum(1, np.rint(area / 48)), 0).astype(int)
    office_no = rng.integers(1, 9999, n)
    # 공식 법정동코드로 오인하지 않도록 지역목록 순번을 SYN 코드로만 사용한다.
    lawd = 90000 + region_id
    lot_main = rng.integers(1, 999, n)
    lot_sub = rng.integers(0, 100, n)
    road_main = rng.integers(1, 300, n)
    building_dong = np.where(np.isin(house, ["아파트", "오피스텔"]), rng.integers(101, 110, n).astype(str), "단일동")

    acquisition_rate = np.where(is_sale, np.where(sale_price <= 60000, 1.0, np.where(sale_price <= 90000, 2.0, 3.0)), 0.0)
    loan_available = np.where(is_sale, rng.choice(["가능", "불가", "확인필요"], n, p=[0.72, 0.05, 0.23]), np.where(is_jeonse, rng.choice(["가능", "불가", "확인필요"], n, p=[0.62, 0.08, 0.30]), "해당없음"))
    total_complex_buildings = complex_buildings.astype(int)
    photo_count = rng.integers(3, 31, n)

    data = {
        "property_id": [f"P{i:08d}" for i in range(n)], "listing_id": [f"L-SYN-{i:08d}" for i in range(n)],
        "is_synthetic": True, "synthetic_notice": "분석용 합성 후보이며 실제 거래 매물이 아닙니다. 지도 위치는 공공 도로명주소 기준점입니다.",
        "source_type": "MOLIT_RTMS+GYEONGGI_PORTAL+HUG_REGION+KB_REGIONAL_PRICE", "generator_version": GENERATOR_VERSION,
        "generated_at": generated_at, "source_dataset": anchor["source_dataset"].to_numpy(), "source_record_hash": source_hash,
        "source_transaction_type": anchor["source_transaction_type"].to_numpy(),
        "generation_method": np.char.add(generation_method.astype(str), "+가격대층화+시군구내 좌표균등분산"),
        "price_estimation_method": price_method, "privacy_distance_score": privacy_distance,
        "region_assignment_method": "HUG 252개 시군구×주택유형×거래유형 균형 층화",
        "regional_price_factor": np.round(regional_price_factor, 5),
        "price_diversity_factor": np.round(price_diversity_factor, 5),
        "region_coordinate_source": coordinate_source,
        "coordinate_distribution_method": "시군구 중심 타원 내 golden-angle 저불일치 균등분산",
        "price_model_name": "HistGradientBoosting+holdout-residual-bootstrap" if fitted_price is not None else "none",
        "price_model_holdout_mdape_pct": round(float(model_metrics["mdape_pct"]), 4) if fitted_price is not None else np.nan,
        "price_model_holdout_r2": round(float(model_metrics["r2"]), 6) if fitted_price is not None else np.nan,
        "base_rent_record_hash": np.where(anchor["source_transaction_type"].isin(["전세", "월세"]), source_hash, ""),
        "base_trade_record_hash": np.where(anchor["source_transaction_type"].eq("매매"), source_hash, ""),
        "reference_rent_deal_ym": np.where(anchor["source_transaction_type"].isin(["전세", "월세"]), _num(anchor["deal_ym"]).fillna(0), 0),
        "reference_trade_deal_ym": np.where(anchor["source_transaction_type"].eq("매매"), _num(anchor["deal_ym"]).fillna(0), 0),
        "listing_status": "거래가능(합성)", "listing_created_at": created_dates.dt.strftime("%Y-%m-%d"),
        "listing_updated_at": updated_dates.dt.strftime("%Y-%m-%d"),
        "sido": sido, "gugun": gugun, "dong": dong,
        "legal_dong_code": real_legal_dong_code,
        "road_address": map_reference_address,
        "jibun_address": map_jibun_address,
        "address_detail_public": "분석 기준 위치(실제 매물 상세주소 아님)", "lat": np.round(lat, 6), "lng": np.round(lng, 6),
        "lot_main_no": lot_main, "lot_sub_no": lot_sub, "road_name": "공공주소 기준", "building_main_no": road_main, "building_sub_no": 0,
        "land_category": rng.choice(["대", "전", "답"], n, p=[0.96, 0.025, 0.015]), "land_area_m2": np.round(land_area, 2),
        "land_share_m2": np.round(land_area / np.maximum(units, 1), 2),
        "zoning": rng.choice(["제1종일반주거지역", "제2종일반주거지역", "제3종일반주거지역", "준주거지역"], n, p=[0.18, 0.55, 0.22, 0.05]),
        "land_use_status": "주거용 건부지", "road_access": rng.choice(["세로(가)", "세로(불)", "소로한면", "중로한면"], n, p=[0.35, 0.18, 0.38, 0.09]),
        "land_transaction_permit_zone": rng.random(n) < 0.08,
        "transaction_type": txn, "lease_type": txn, "asking_price_manwon": np.where(is_sale, sale_price, deposit),
        "sale_price_manwon": sale_price, "sale_subtype": np.where(is_sale, anchor["sale_subtype"].fillna("일반매매"), "해당없음"),
        "dealing_type": anchor["dealing_type"].fillna("중개거래"), "deposit_manwon": deposit, "monthly_rent_manwon": monthly,
        "maintenance_fee_manwon": maintenance, "maintenance_fee_items": np.where(maintenance > 0, "공용전기|청소|수도|승강기", "별도 관리비 없음"),
        "maintenance_fee_other": "사용량에 따른 전기·가스 별도", "price_negotiable": rng.random(n) < 0.34,
        "rent_conversion_rate_pct": np.where(is_sale, 0, np.round(conversion_rate * 100, 2)), "move_in_negotiable": rng.random(n) < 0.56,
        "contract_type": np.where(is_sale, "매매계약", anchor["contract_type"].fillna("신규")),
        "contract_term": np.where(is_sale, "소유권이전일 협의", anchor["contract_term"].fillna("협의")),
        "use_renewal_right": np.where(is_sale, "해당없음", anchor["use_rr_right"].fillna("확인필요")),
        "available_from_date": available_dates.dt.strftime("%Y-%m-%d"),
        "occupancy_status": rng.choice(["공실", "임차인 거주", "소유자 거주"], n, p=[0.31, 0.48, 0.21]),
        "onetime_fee_manwon": np.round(onetime, 1), "broker_fee_rate_pct": broker_fee_rate,
        "broker_fee_manwon": broker_fee, "broker_fee_vat_manwon": broker_vat, "actual_expense_manwon": actual_expense,
        "property_type": "주거용 건축물", "house_type": house, "building_name": building_name, "building_use": building_use,
        "building_dong": building_dong, "unit_number_public": "실제 동·호수 없음(분석용 후보)",
        "unit_type": np.where(np.isin(house, ["단독주택", "다가구주택"]), "건물/호실", "구분소유 호실"),
        "building_structure": structures, "approval_date": [f"{min(y, CURRENT_YEAR):04d}-06-30" for y in build_year], "build_year": build_year,
        "building_age_years": np.maximum(CURRENT_YEAR - build_year, 0), "building_total_area_m2": np.round(building_total_area, 2),
        "building_coverage_ratio_pct": np.round(np.clip(building_total_area / np.maximum(total_floors * land_area, 1) * 100, 15, 85), 1),
        "floor_area_ratio_pct": np.round(np.clip(building_total_area / np.maximum(land_area, 1) * 100, 30, 1000), 1),
        "current_floor": current_floor, "total_floors": total_floors, "basement_floors": basement,
        "area_m2": np.round(area, 2), "exclusive_area_m2": np.round(area, 2), "supply_area_m2": np.round(area * rng.uniform(1.08, 1.30, n), 2),
        "contract_area_m2": np.round(area * rng.uniform(1.12, 1.42, n), 2), "room_count": rooms, "bathroom_count": bathrooms,
        "direction": direction, "direction_basis": "주실 창문 기준", "entrance_type": rng.choice(["계단식", "복도식", "독립출입"], n, p=[0.48, 0.27, 0.25]),
        "building_total_units": units, "building_total_households": units, "total_complex_buildings": total_complex_buildings,
        "balcony_expansion": np.isin(house, ["아파트", "다세대주택", "연립주택"]) & (rng.random(n) < 0.48),
        "duplex": rng.random(n) < 0.025, "terrace": rng.random(n) < 0.08, "yard": np.isin(house, ["단독주택", "다가구주택"]) & (rng.random(n) < 0.38),
        "rooftop_access": np.isin(house, ["단독주택", "다가구주택", "다세대주택", "연립주택"]) & (rng.random(n) < 0.35),
        "ceiling_height_m": np.round(np.clip(rng.normal(2.35, 0.12, n), 2.1, 3.2), 2),
        "parking_total": parking_total, "parking_per_household": np.round(parking_total / np.maximum(units, 1), 2),
        "parking_method": np.where(parking_total > 0, rng.choice(["자주식", "기계식", "혼합"], n, p=[0.80, 0.12, 0.08]), "주차 불가"),
        "elevator_count": elevator_count, "heating_method": rng.choice(["개별난방", "중앙난방", "지역난방"], n, p=[0.78, 0.07, 0.15]),
        "heating_fuel": rng.choice(["도시가스", "전기", "열병합"], n, p=[0.79, 0.06, 0.15]),
        "cooling_facility": np.where(aircon > 0, "벽걸이/스탠드 에어컨", "없음"), "aircon_count": aircon,
        "built_in_appliances": np.where(furnished, "에어컨|냉장고|세탁기|가스레인지", np.where(aircon > 0, "에어컨", "없음")),
        "furnished": furnished, "pet_allowed": rng.choice(["가능", "불가", "협의"], n, p=[0.22, 0.45, 0.33]), "loan_available": loan_available,
        "water_supply": "상수도", "electricity_supply": "정상", "gas_supply": "도시가스", "drainage": "하수도",
        "fire_safety_facility": np.where(total_floors >= 5, "소화기|감지기|완강기|스프링클러", "소화기|단독경보형감지기"),
        "security_facility": rng.choice(["공동현관|CCTV", "CCTV", "도어락", "없음"], n, p=[0.38, 0.28, 0.28, 0.06]),
        "accessibility_facility": np.where(elevator_count > 0, "승강기", "없음"),
        "wall_crack": rng.random(n) < np.clip((CURRENT_YEAR - np.minimum(build_year, CURRENT_YEAR)) / 350, 0.01, 0.18),
        "water_leak": rng.random(n) < np.clip((CURRENT_YEAR - np.minimum(build_year, CURRENT_YEAR)) / 500, 0.005, 0.12),
        "wallpaper_condition": rng.choice(["양호", "보통", "수리필요"], n, p=[0.64, 0.31, 0.05]),
        "noise_level": rng.choice(["낮음", "보통", "높음"], n, p=[0.37, 0.51, 0.12]), "floor_condition": rng.choice(["양호", "보통", "수리필요"], n, p=[0.69, 0.27, 0.04]),
        "vibration_level": rng.choice(["없음", "보통", "있음"], n, p=[0.81, 0.16, 0.03]),
        "sunlight_level": np.where(np.isin(direction, ["남향", "남동향", "남서향"]), "양호", rng.choice(["보통", "부족"], n, p=[0.76, 0.24])),
        "renovation_status": rng.choice(["없음", "부분수리", "전체수리"], n, p=[0.58, 0.33, 0.09]), "illegal_building": illegal,
        "ledger_discrepancy": ledger_discrepancy, "violation_details": np.where(illegal | ledger_discrepancy, "대장 및 현황 추가 확인 필요(합성)", "해당없음"),
        "ownership_type": rng.choice(["개인", "공동소유", "법인"], n, p=[0.88, 0.09, 0.03]), "owner_relation": rng.choice(["소유자 직접 의뢰", "적법한 대리인 의뢰"], n, p=[0.93, 0.07]),
        "trust_registration": trust, "seizure_or_provisional_seizure": rng.random(n) < 0.018, "easement": rng.random(n) < 0.006,
        "leasehold_registration": rng.random(n) < 0.012, "tenant_right_registration": rng.random(n) < 0.008,
        "tax_arrears_checked": True, "tax_arrears_present": tax_present, "landlord_information_presented": ~is_sale,
        "resident_household_certificate_checked": ~is_sale, "small_deposit_priority_protection_explained": ~is_sale,
        "private_rental_housing": private_rental, "rental_deposit_guarantee_joined": rental_guarantee,
        "rental_deposit_guarantee_details": np.where(private_rental, np.where(rental_guarantee, "임대보증금 보증 가입(합성)", "가입 여부 추가 확인 필요(합성)"), "민간임대주택 해당없음"),
        "market_price_manwon": market_price, "official_land_price_manwon_m2": np.round((market_price * rng.uniform(0.20, 0.46, n)) / np.maximum(land_area, 1), 1),
        "official_building_price_manwon": np.round(market_price * rng.uniform(0.42, 0.72, n), 1),
        "registered_owner_type": rng.choice(["개인 단독", "개인 공동", "법인"], n, p=[0.86, 0.11, 0.03]),
        "mortgage_max_claim_manwon": max_claim, "senior_rights_total_manwon": np.round(max_claim + senior_deposit, 1),
        "registry_checked_at": updated_dates.dt.strftime("%Y-%m-%d"), "building_ledger_checked_at": updated_dates.dt.strftime("%Y-%m-%d"),
        "deposit_return_guarantee_provider": np.where(guarantee, rng.choice(["HUG", "HF", "SGI"], n, p=[0.58, 0.30, 0.12]), "해당없음/확인필요"),
        "my_priority_rank": my_rank, "senior_tenant_count": senior_count, "senior_deposit_sum_manwon": np.round(senior_deposit, 1),
        "senior_mortgage_manwon": senior_mortgage, "mortgage_ltv_pct": np.round(senior_mortgage / np.maximum(market_price, 1) * 100, 2),
        "jeonse_ratio_pct": np.where(is_jeonse, np.round(deposit / np.maximum(np.where(multi_building, market_price / units, market_price), 1) * 100, 2), 0),
        "guarantee_eligible": guarantee, "guarantee_ineligible_reason": np.where(guarantee, "해당없음", np.where(~is_jeonse, "전세 대상 아님", "권리·부채·건축물 조건 확인 필요")),
        "acquisition_tax_type": np.where(is_sale, "주택 취득세(합성 추정)", "임차 거래 해당없음"), "estimated_acquisition_tax_rate_pct": acquisition_rate,
        "fraud_label": np.nan, "fraud_score": np.nan,
        "subway_walk_minutes": np.clip(np.rint(rng.lognormal(np.log(9), 0.55, n)), 1, 45).astype(int), "bus_stop_walk_minutes": np.clip(np.rint(rng.lognormal(np.log(4), 0.45, n)), 1, 25).astype(int),
        "school_walk_minutes": np.clip(np.rint(rng.lognormal(np.log(8), 0.50, n)), 1, 40).astype(int), "mart_walk_minutes": np.clip(np.rint(rng.lognormal(np.log(6), 0.50, n)), 1, 35).astype(int),
        "hospital_walk_minutes": np.clip(np.rint(rng.lognormal(np.log(12), 0.60, n)), 1, 60).astype(int), "park_walk_minutes": np.clip(np.rint(rng.lognormal(np.log(10), 0.55, n)), 1, 45).astype(int),
        "noise_source": rng.choice(["없음", "생활도로", "간선도로", "상업시설"], n, p=[0.46, 0.28, 0.17, 0.09]), "odor_source": rng.choice(["없음", "음식점", "하수시설", "기타"], n, p=[0.87, 0.08, 0.03, 0.02]),
        "flood_risk_level": rng.choice(["낮음", "보통", "높음"], n, p=[0.76, 0.20, 0.04]), "nonpreferred_facility": rng.choice(["없음", "고압선", "철도", "유흥시설", "공장"], n, p=[0.86, 0.025, 0.045, 0.05, 0.02]),
        "advertisement_medium": rng.choice(["중개사 홈페이지", "부동산 포털", "현장 안내"], n, p=[0.20, 0.72, 0.08]),
        "advertisement_title": [f"[분석후보] {g} {h} {t} {a:.0f}㎡" for g, h, t, a in zip(gugun, house, txn, area)],
        "advertisement_description": "실거래 분포 기반 연구용 합성 레코드. 실제 계약에 사용할 수 없습니다.",
        "broker_office_name": [f"합성 {g} {x:04d} 공인중개사사무소" for g, x in zip(gugun, office_no)],
        "broker_registration_no": [f"SYN-{c:05d}-{x:04d}" for c, x in zip(lawd, office_no)],
        "broker_representative_name": [f"합성대표{x:04d}" for x in office_no], "broker_agent_name": [f"합성중개사{x:04d}" for x in office_no],
        "broker_office_address": [f"{s} {g} 합성중개로 {x % 100 + 1}" for s, g, x in zip(sido, gugun, office_no)],
        "broker_phone": [f"000-0000-{x:04d}" for x in office_no], "broker_guarantee_type": rng.choice(["공제", "보증보험", "공탁"], n, p=[0.78, 0.20, 0.02]),
        "broker_guarantee_amount_manwon": rng.choice([10000, 20000, 30000], n, p=[0.55, 0.35, 0.10]), "broker_guarantee_period": "생성일 기준 유효로 가정(합성)",
        "joint_brokerage": rng.random(n) < 0.12, "advertisement_confirmed_at": updated_dates.dt.strftime("%Y-%m-%d"),
        "photo_count": photo_count, "video_present": rng.random(n) < 0.18, "virtual_tour_present": rng.random(n) < 0.09,
        "viewing_available": True, "viewing_method": rng.choice(["예약 방문", "즉시 방문", "임차인 협의"], n, p=[0.54, 0.24, 0.22]), "exclusive_listing": rng.random(n) < 0.28,
        "evidence_title_deed": rng.random(n) < 0.12, "evidence_registry": True, "evidence_land_ledger": True, "evidence_building_ledger": True,
        "evidence_cadastral_map": rng.random(n) < 0.88, "evidence_land_use_plan": True, "evidence_owner_request": True,
        "explanation_completed": True, "explanation_notes": "합성 데이터의 확인·설명 항목이며 실제 서류 확인을 대체하지 않습니다.",
        "_region_hazard": _region_hazard_for(sido, gugun),
    }
    df = pd.DataFrame(data)
    missing = missing_schema_columns(df.columns)
    if missing:
        raise AssertionError(f"generator omitted broker schema columns: {missing}")
    return df


def assign_fraud_labels(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Create a structural synthetic loss/fraud label for jeonse rows only."""
    df = df.copy()
    is_jeonse = df["lease_type"].to_numpy() == "전세"
    market = df["market_price_manwon"].to_numpy(float)
    deposit = df["deposit_manwon"].to_numpy(float)
    senior_dep = df["senior_deposit_sum_manwon"].to_numpy(float)
    senior_mtg = df["senior_mortgage_manwon"].to_numpy(float)
    units = df["building_total_units"].to_numpy(float)
    rank = df["my_priority_rank"].to_numpy(float)
    multi = df["house_type"].to_numpy() == "다가구주택"
    unit_value = np.where(multi, market / np.maximum(units, 1), market)
    recover = market * config.AUCTION_RECOVERY_RATIO
    debt_ratio = (deposit + senior_dep + senior_mtg) / np.maximum(recover, 1)
    rank_frac = (rank - 1) / np.maximum(units - 1, 1)
    ratio = deposit / np.maximum(unit_value, 1)
    hazard = np.clip(df.get("_region_hazard", pd.Series(0.02, index=df.index)).to_numpy(float), 0.001, 0.5)
    hazard_logit = np.log(hazard / (1 - hazard))
    hazard_centered = hazard_logit - np.nanmean(hazard_logit)
    document_risk = sum(df.get(c, False).astype(float).to_numpy() for c in ("illegal_building", "trust_registration", "tax_arrears_present"))
    logit = -4.25 + 3.2 * np.clip(debt_ratio - 0.82, 0, None) + rank_frac + 1.4 * np.clip(ratio - 0.78, 0, None) + 0.7 * hazard_centered + 0.75 * document_risk + rng.normal(0, 0.45, len(df))
    label = (rng.random(len(df)) < sigmoid(logit)).astype(int)
    df["fraud_label"] = np.where(is_jeonse, label, np.nan)
    df["fraud_score"] = np.nan
    df = df.drop(columns=["_region_hazard"], errors="ignore")
    extra = [c for c in df.columns if c not in BROKER_LISTING_COLUMNS]
    return df[BROKER_LISTING_COLUMNS + extra]


def generate_users(n: int, rng: np.random.Generator) -> pd.DataFrame:
    if n <= 0:
        return pd.DataFrame(columns=["user_id", "age", "monthly_income_manwon", "total_asset_manwon", "monthly_living_cost_manwon", "income_decile", "preferred_sido", "preferred_gugun", "nl_preference", "workplace_lat", "workplace_lng"])
    z = rng.normal(0, 1, n)
    income = np.clip(np.exp(np.log(230) + 0.45 * z + rng.normal(0, 0.25, n)), 90, 900)
    asset = np.clip(np.exp(np.log(3000) + 0.8 * z + rng.normal(0, 0.7, n)), 100, 40000)
    living = np.clip(income * rng.uniform(0.30, 0.55, n), 40, 400)
    return pd.DataFrame({
        "user_id": [f"U{i:06d}" for i in range(n)], "age": rng.integers(config.YOUTH_AGE_MIN, config.YOUTH_AGE_MAX + 1, n),
        "monthly_income_manwon": np.round(income, 1), "total_asset_manwon": np.round(asset, 1), "monthly_living_cost_manwon": np.round(living, 1),
        "income_decile": np.digitize(income, np.array(config.INCOME_DECILE_BOUNDARIES_MAN)) + 1,
        "preferred_sido": rng.choice(EXPECTED_SIDOS, n), "preferred_gugun": None,
        "nl_preference": "", "workplace_lat": np.nan, "workplace_lng": np.nan,
    })


def validate_generated_properties(df: pd.DataFrame, expected_n: int | None = None) -> list[str]:
    errors: list[str] = []
    if expected_n is not None and len(df) != expected_n:
        errors.append(f"row count {len(df)} != {expected_n}")
    missing = missing_schema_columns(df.columns)
    if missing:
        errors.append(f"missing schema columns: {missing}")
    missing_legacy = [c for c in LEGACY_REQUIRED_COLUMNS if c not in df.columns]
    if missing_legacy:
        errors.append(f"missing legacy columns: {missing_legacy}")
    if not df["transaction_type"].isin(TRANSACTION_TYPES).all() or not df["lease_type"].isin(TRANSACTION_TYPES).all():
        errors.append("invalid transaction type")
    if not df["house_type"].isin(HOUSE_TYPES).all():
        errors.append("invalid house type")
    if (df["market_price_manwon"] <= 0).any() or (df["deposit_manwon"] < 0).any() or (df["monthly_rent_manwon"] < 0).any():
        errors.append("invalid money value")
    if (df.loc[df["transaction_type"] == "매매", ["deposit_manwon", "monthly_rent_manwon"]] != 0).any().any():
        errors.append("sale row has rental amount")
    if (df.loc[df["transaction_type"] == "전세", "monthly_rent_manwon"] != 0).any():
        errors.append("jeonse row has monthly rent")
    if (df.loc[df["transaction_type"] == "월세", "monthly_rent_manwon"] <= 0).any():
        errors.append("monthly-rent row has non-positive monthly rent")
    if (df["area_m2"] <= 0).any() or (df["building_total_units"] < 1).any():
        errors.append("invalid area or unit count")
    if (df["my_priority_rank"] > df["building_total_units"]).any():
        errors.append("priority rank exceeds unit count")
    if not df["property_id"].is_unique or not df["listing_id"].is_unique:
        errors.append("duplicate generated identifiers")
    if not df["is_synthetic"].astype(bool).all():
        errors.append("synthetic marker missing")
    if not df["lat"].between(32.5, 39.5).all() or not df["lng"].between(124.0, 132.0).all():
        errors.append("coordinate outside Korea analysis bounds")
    if len(df) >= 252 * len(TRANSACTION_TYPES):
        expected_regions = set(map(
            tuple,
            load_nationwide_region_catalog()[["sido", "gugun"]].astype(str).to_numpy(),
        ))
        actual_regions = set(map(tuple, df[["sido", "gugun"]].astype(str).to_numpy()))
        missing_regions = expected_regions - actual_regions
        if missing_regions:
            errors.append(f"nationwide district coverage missing: {len(missing_regions)}")
        region_txn = df.groupby(["sido", "gugun", "transaction_type"]).size()
        missing_region_txn = sum(
            (sido, gugun, transaction_type) not in region_txn.index
            for sido, gugun in expected_regions
            for transaction_type in TRANSACTION_TYPES
        )
        if missing_region_txn:
            errors.append(f"district-transaction coverage missing: {missing_region_txn}")
        if len(df) >= len(expected_regions) * len(TRANSACTION_TYPES) * len(HOUSE_TYPES):
            region_house_txn = df.groupby(
                ["sido", "gugun", "house_type", "transaction_type"]
            ).size()
            missing_full_combinations = sum(
                (sido, gugun, house_type, transaction_type) not in region_house_txn.index
                for sido, gugun in expected_regions
                for house_type in HOUSE_TYPES
                for transaction_type in TRANSACTION_TYPES
            )
            if missing_full_combinations:
                errors.append(
                    f"district-house-transaction coverage missing: {missing_full_combinations}"
                )
    return errors


def build_quality_report(df: pd.DataFrame, *_args, **_kwargs) -> dict:
    coverage = pd.crosstab(df["house_type"], df["transaction_type"])
    region_transaction = df.groupby(
        ["sido", "gugun", "transaction_type"]
    ).size().unstack(fill_value=0)
    region_house_transaction = df.groupby(
        ["sido", "gugun", "house_type", "transaction_type"]
    ).size()
    coordinate_spread = df.groupby(["sido", "gugun"]).agg(
        lat_span=("lat", lambda values: float(values.max() - values.min())),
        lng_span=("lng", lambda values: float(values.max() - values.min())),
    )
    numeric = {}
    for col in ("area_m2", "sale_price_manwon", "deposit_manwon", "monthly_rent_manwon", "market_price_manwon"):
        s = pd.to_numeric(df[col], errors="coerce")
        positive = s[s > 0]
        numeric[col] = {"min": round(float(positive.min()), 2) if len(positive) else 0, "median": round(float(positive.median()), 2) if len(positive) else 0, "p95": round(float(positive.quantile(0.95)), 2) if len(positive) else 0, "max": round(float(positive.max()), 2) if len(positive) else 0}
    return {
        "generator_version": GENERATOR_VERSION, "generated_rows": int(len(df)),
        "schema_columns": int(len(df.columns)),
        "canonical_schema_columns": int(len(BROKER_LISTING_COLUMNS)),
        "validation_errors": validate_generated_properties(df, len(df)),
        "house_type_counts": {str(k): int(v) for k, v in df["house_type"].value_counts().items()},
        "transaction_type_counts": {str(k): int(v) for k, v in df["transaction_type"].value_counts().items()},
        "region_counts": {str(k): int(v) for k, v in df["sido"].value_counts().items()},
        "district_count": int(df[["sido", "gugun"]].drop_duplicates().shape[0]),
        "district_transaction_min_count": int(region_transaction.min().min()),
        "district_transaction_max_count": int(region_transaction.max().max()),
        "districts_with_all_transaction_types": int((region_transaction > 0).all(axis=1).sum()),
        "district_house_transaction_cells": int(len(region_house_transaction)),
        "expected_district_house_transaction_cells": int(
            df[["sido", "gugun"]].drop_duplicates().shape[0]
            * len(HOUSE_TYPES) * len(TRANSACTION_TYPES)
        ),
        "district_house_transaction_min_count": int(region_house_transaction.min()),
        "district_house_transaction_max_count": int(region_house_transaction.max()),
        "coordinate_spread_median": {
            "lat_span": round(float(coordinate_spread["lat_span"].median()), 6),
            "lng_span": round(float(coordinate_spread["lng_span"].median()), 6),
        },
        "coordinate_source_counts": {
            str(k): int(v)
            for k, v in df["region_coordinate_source"].value_counts().items()
        },
        "real_map_anchor_ratio": round(float(
            df["region_coordinate_source"].astype(str).str.contains(
                "주소 기준", regex=False
            ).mean()
        ), 6),
        "price_diversity_factor_quantiles": {
            str(q): round(float(df["price_diversity_factor"].quantile(q)), 4)
            for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "coverage_matrix": {str(i): {str(k): int(v) for k, v in row.items()} for i, row in coverage.to_dict(orient="index").items()},
        "generation_method_counts": {str(k): int(v) for k, v in df["generation_method"].value_counts().items()},
        "source_dataset_counts": {str(k): int(v) for k, v in df["source_dataset"].value_counts().items()},
        "price_model": {
            "name": str(df["price_model_name"].iloc[0]),
            "holdout_mdape_pct": float(df["price_model_holdout_mdape_pct"].iloc[0]) if df["price_model_holdout_mdape_pct"].notna().any() else None,
            "holdout_r2": float(df["price_model_holdout_r2"].iloc[0]) if df["price_model_holdout_r2"].notna().any() else None,
        },
        "numeric_summary": numeric,
        "privacy_checks": {
            "all_rows_disclose_synthetic_status": bool(
                df["is_synthetic"].astype(bool).all()
                and df["synthetic_notice"].astype(str).str.contains("실제 거래 매물이 아닙니다", regex=False).all()
            ),
            "map_reference_addresses_present": int(df["road_address"].astype(str).ne("").sum()),
            "all_broker_ids_marked_synthetic": bool(df["broker_registration_no"].astype(str).str.startswith("SYN-").all()),
            "exact_zero_distance_rows": int((pd.to_numeric(df["privacy_distance_score"], errors="coerce") == 0).sum()),
        },
        "fraud_summary": {
            "jeonse_rows": int((df["transaction_type"] == "전세").sum()),
            "synthetic_fraud_rate": round(float(df.loc[df["transaction_type"] == "전세", "fraud_label"].mean()), 5),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="다운로드 실거래 기반 전체 주택유형 합성 매물 생성기")
    parser.add_argument("--n-properties", "--n_properties", "--count", dest="n_properties", type=int, default=5000)
    parser.add_argument("--n-users", "--n_users", dest="n_users", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=config.GLOBAL_SEED)
    parser.add_argument("--output", type=Path, default=config.DATA_GEN / "properties.csv")
    parser.add_argument("--users-output", type=Path, default=config.DATA_GEN / "users.csv")
    parser.add_argument("--quality-report", type=Path, default=config.DATA_GEN / "property_generation_quality.json")
    parser.add_argument("--recency-half-life-months", type=float, default=18.0)
    parser.add_argument("--house-weights", help="예: 아파트=0.4,오피스텔=0.2,단독주택=0.1,다가구주택=0.15,다세대주택=0.1,연립주택=0.05")
    parser.add_argument("--transaction-weights", help="예: 매매=0.3,전세=0.4,월세=0.3")
    parser.add_argument("--price-model", choices=["hedonic", "none"], default="hedonic", help="hedonic=GBDT+홀드아웃 잔차 부트스트랩, none=경험분포만 사용")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    house_weights = _parse_weight_spec(args.house_weights, DEFAULT_HOUSE_WEIGHTS)
    transaction_weights = _parse_weight_spec(args.transaction_weights, DEFAULT_TRANSACTION_WEIGHTS)
    print(f"[gen] 전체 주택유형 합성 매물 {args.n_properties:,}건 생성 중...")
    props = generate_properties(args.n_properties, rng, recency_half_life_months=args.recency_half_life_months, house_weights=house_weights, transaction_weights=transaction_weights, price_model=args.price_model)
    props = assign_fraud_labels(props, rng)
    errors = validate_generated_properties(props, args.n_properties)
    if errors:
        raise RuntimeError("generated data validation failed: " + "; ".join(errors))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    props.to_csv(args.output, index=False, encoding="utf-8-sig")
    if args.n_users > 0:
        users = generate_users(args.n_users, rng)
        args.users_output.parent.mkdir(parents=True, exist_ok=True)
        users.to_csv(args.users_output, index=False, encoding="utf-8-sig")
        print(f"[gen] 사용자: {args.users_output} ({len(users):,}명)")
    report = build_quality_report(props)
    args.quality_report.parent.mkdir(parents=True, exist_ok=True)
    args.quality_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[gen] 매물: {args.output} ({len(props):,}건, {len(props.columns)}개 컬럼)")
    print(f"[gen] 품질 보고서: {args.quality_report}")
    print(json.dumps({"house_type_counts": report["house_type_counts"], "transaction_type_counts": report["transaction_type_counts"], "region_counts": report["region_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
