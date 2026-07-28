"""
조건 ATOM 시스템.

사용자 요구를 원자적 조건(ConditionAtom)들로 분해하고,
각 매물이 몇 개의 ATOM을 만족하는지 계산한다.

핵심 아이디어(사용자 요구사항):
  - 모든 조건을 만족하는 매물만 보여주지 말 것.
  - 조건 1개 누락 / 2개 누락 매물도 '타협 옵션'으로 별도 표시.
  - 사용자가 어떤 조건을 포기하면 되는지 명확히 보이게.

각 ATOM은:
  - key         : 식별자 (예: "lease_type", "max_deposit", "region", "commute", "safety")
  - description : 사람이 읽을 설명 (예: "전세", "보증금 5000만원 이하")
  - predicate   : 매물 row -> bool (만족 여부)
  - hard        : True면 절대 양보 불가(누락 그룹에서도 제외). 예: 예산 상한 초과는 위험.
  - weight      : 중요도(정렬 tie-break용)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
import pandas as pd


def _number(value):
    """PDF/외부 피드의 None·NaN 숫자를 비교 가능한 값으로 정규화한다."""
    try:
        number = float(value)
        return number if pd.notna(number) else None
    except (TypeError, ValueError):
        return None


def _lte(value, maximum) -> bool:
    number = _number(value)
    return number is not None and number <= float(maximum)


def _gte(value, minimum) -> bool:
    number = _number(value)
    return number is not None and number >= float(minimum)


@dataclass
class ConditionAtom:
    key: str
    description: str
    predicate: Callable[[pd.Series], bool]
    hard: bool = False
    weight: float = 1.0


@dataclass
class AtomSet:
    """분해된 조건 원자들의 모음."""
    atoms: list[ConditionAtom] = field(default_factory=list)

    def add(self, atom: ConditionAtom):
        self.atoms.append(atom)

    def hard_atoms(self):
        return [a for a in self.atoms if a.hard]

    def soft_atoms(self):
        return [a for a in self.atoms if not a.hard]

    def __len__(self):
        return len(self.atoms)

    def describe(self) -> list[str]:
        return [("[필수] " if a.hard else "") + a.description for a in self.atoms]


# ----------------------------------------------------------------------
# 슬롯 dict -> AtomSet 분해
# ----------------------------------------------------------------------
def build_atoms(slots: dict) -> AtomSet:
    """
    확인된 조건 슬롯을 ConditionAtom 리스트로 분해.

    지원 슬롯:
      lease_type, region_sido, region_gugun(list), max_deposit_manwon,
      max_monthly_rent_manwon, max_commute_min(+workplace),
      max_maintenance_manwon, min_area_m2, max_building_age
    """
    aset = AtomSet()

    if lt := slots.get("lease_type"):
        aset.add(ConditionAtom(
            "lease_type", f"거래유형 = {lt}",
            lambda r, lt=lt: r["lease_type"] == lt,
            hard=False, weight=2.0,
        ))

    if pt := slots.get("property_type"):
        aset.add(ConditionAtom(
            "property_type", f"주택유형 = {pt}",
            lambda r, pt=pt: (str(r.get("property_type", "")) == pt
                              or pt in str(r.get("house_type", ""))),
            hard=False, weight=1.5,
        ))

    if sido := slots.get("region_sido"):
        aset.add(ConditionAtom(
            "region_sido", f"광역시도 = {sido}",
            lambda r, s=sido: r["sido"] == s,
            hard=False, weight=1.5,
        ))

    gugun = slots.get("region_gugun")
    if gugun:
        gset = set(gugun) if isinstance(gugun, (list, tuple)) else {gugun}
        label = ", ".join(sorted(gset)) if len(gset) <= 4 else f"{len(gset)}개 구/군"
        aset.add(ConditionAtom(
            "region_gugun", f"지역(구/군) ∈ {{{label}}}",
            lambda r, gs=gset: r["gugun"] in gs,
            hard=False, weight=1.5,
        ))

    if (v := slots.get("max_deposit_manwon")) is not None:
        # 예산 초과는 hard: 감당 못 하면 추천 의미 없음
        aset.add(ConditionAtom(
            "max_deposit", f"보증금 ≤ {v:,.0f}만원",
            lambda r, v=v: _lte(r.get("deposit_manwon"), v),
            hard=slots.get("deposit_is_hard", True), weight=3.0,
        ))

    if (v := slots.get("max_sale_price_manwon")) is not None:
        aset.add(ConditionAtom(
            "max_sale_price", f"매매가 ≤ {v:,.0f}만원",
            lambda r, v=v: (float(r.get("sale_price_manwon", 0) or
                                  r.get("asking_price_manwon", 0) or 0) <= v),
            hard=slots.get("sale_price_is_hard", True), weight=3.0,
        ))

    if (v := slots.get("max_monthly_rent_manwon")) is not None:
        aset.add(ConditionAtom(
            "max_monthly_rent", f"월세 ≤ {v:,.0f}만원",
            lambda r, v=v: _lte(r.get("monthly_rent_manwon"), v),
            hard=False, weight=2.0,
        ))

    if (v := slots.get("max_maintenance_manwon")) is not None:
        aset.add(ConditionAtom(
            "max_maintenance", f"관리비 ≤ {v:,.0f}만원",
            lambda r, v=v: _lte(r.get("maintenance_fee_manwon"), v),
            hard=False, weight=1.0,
        ))

    if (v := slots.get("min_area_m2")) is not None:
        aset.add(ConditionAtom(
            "min_area", f"전용면적 ≥ {v:.0f}㎡",
            lambda r, v=v: _gte(r.get("area_m2"), v),
            hard=False, weight=1.0,
        ))

    if (v := slots.get("max_building_age")) is not None:
        aset.add(ConditionAtom(
            "max_age", f"건물연식 ≤ {v:.0f}년",
            lambda r, v=v: _lte(r.get("building_age_years"), v),
            hard=False, weight=1.0,
        ))

    # 통근: workplace 좌표 + 상한 분이 있으면 매물별 계산이 필요 → predicate에 주입
    max_commute = slots.get("max_commute_min")
    commute_map = slots.get("_commute_minutes_by_id")  # {property_id: minutes}
    if max_commute is not None and commute_map is not None:
        aset.add(ConditionAtom(
            "commute", f"통근시간 ≤ {max_commute:.0f}분",
            lambda r, cm=commute_map, mx=max_commute:
                (cm.get(r["property_id"], 9999) <= mx),
            hard=False, weight=2.0,
        ))

    # 치안: 매물별 안전점수 맵(주변 300m 인프라)
    min_safety = slots.get("min_safety_score")
    safety_map = slots.get("_safety_score_by_id")
    if min_safety is not None and safety_map is not None:
        aset.add(ConditionAtom(
            "safety_area", f"치안 안전점수 ≥ {min_safety:.0f}",
            lambda r, sm=safety_map, mn=min_safety:
                (sm.get(r["property_id"]) is not None
                 and sm.get(r["property_id"]) >= mn),
            hard=False, weight=2.0,
        ))

    # 생활편의: 매물별 편의점수 맵(주변 500m)
    min_conv = slots.get("min_convenience_score")
    conv_map = slots.get("_convenience_score_by_id")
    if min_conv is not None and conv_map is not None:
        aset.add(ConditionAtom(
            "convenience", f"생활편의점수 ≥ {min_conv:.0f}",
            lambda r, cm=conv_map, mn=min_conv:
                (cm.get(r["property_id"]) is not None
                 and cm.get(r["property_id"]) >= mn),
            hard=False, weight=1.5,
        ))

    return aset


# ----------------------------------------------------------------------
# 매물별 ATOM 만족도 계산 + 누락 그룹 분류
# ----------------------------------------------------------------------
def score_by_atoms(candidates: pd.DataFrame, aset: AtomSet) -> pd.DataFrame:
    """
    각 매물에 대해:
      - satisfied_atoms : 만족한 ATOM key 리스트
      - missing_atoms   : 누락한 (soft) ATOM key 리스트
      - hard_ok         : 모든 hard ATOM 만족 여부
      - n_missing_soft  : 누락한 soft 조건 수
    반환: 원본 + 위 컬럼들이 추가된 DataFrame (hard 위반 매물은 제외).
    """
    if len(aset) == 0:
        # 조건이 하나도 없으면(=아무거나) 전부 통과, 누락 0
        df = candidates.copy()
        df["satisfied_atoms"] = [[] for _ in range(len(df))]
        df["missing_atoms"] = [[] for _ in range(len(df))]
        df["missing_desc"] = [[] for _ in range(len(df))]
        df["hard_ok"] = True
        df["n_missing_soft"] = 0
        return df

    hard = aset.hard_atoms()
    soft = aset.soft_atoms()
    rows = []
    for _, r in candidates.iterrows():
        hard_ok = all(a.predicate(r) for a in hard)
        if not hard_ok:
            continue  # 필수 조건 위반 매물은 아예 제외
        sat, miss, miss_desc = [], [], []
        for a in soft:
            if a.predicate(r):
                sat.append(a.key)
            else:
                miss.append(a.key)
                miss_desc.append(a.description)
        rr = r.to_dict()
        rr["satisfied_atoms"] = sat
        rr["missing_atoms"] = miss
        rr["missing_desc"] = miss_desc
        rr["hard_ok"] = True
        rr["n_missing_soft"] = len(miss)
        rows.append(rr)

    if not rows:
        return pd.DataFrame(columns=list(candidates.columns) +
                            ["satisfied_atoms", "missing_atoms", "missing_desc",
                             "hard_ok", "n_missing_soft"])
    return pd.DataFrame(rows)


def group_by_missing(scored: pd.DataFrame, max_missing: int = 2) -> dict[int, pd.DataFrame]:
    """
    누락 soft 조건 개수별로 그룹화.
      0 : 모든 조건 만족(완벽)
      1 : 조건 1개 양보
      2 : 조건 2개 양보 ...
    max_missing 초과는 버림.
    """
    groups = {}
    for k in range(max_missing + 1):
        g = scored[scored["n_missing_soft"] == k]
        if len(g) > 0:
            groups[k] = g
    return groups
