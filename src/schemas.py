"""
공통 데이터 스키마.

합성 데이터, 위험도 모델, 추천 모델, Agent 도구가 모두 같은 필드명을 공유하도록
여기서 단일 정의한다. (컬럼명이 흩어지면 유지보수가 지옥이 된다.)
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class PropertyRecord(BaseModel):
    """다가구주택 매물 1건 (주택조사 DB의 한 row)."""
    property_id: str
    sido: str                      # 광역시도 (예: 서울)
    gugun: str                     # 시군구 (예: 관악구)
    lat: float
    lng: float
    lease_type: str                # "전세" | "월세"

    # 금액(만원 단위)
    deposit_manwon: float          # 보증금(전세금 또는 월세보증금)
    monthly_rent_manwon: float     # 월세(전세면 0)
    maintenance_fee_manwon: float  # 월 관리비
    onetime_fee_manwon: float      # 일회성 비용(중개비 등)

    # 다가구주택 위험 관련 물건 속성
    market_price_manwon: float     # 건물 시세(감정가 근사)
    building_total_units: int      # 건물 전체 세대수
    my_priority_rank: int          # 내 전입 순위(1=최선순위)
    senior_deposit_sum_manwon: float   # 선순위 보증금 총합
    senior_mortgage_manwon: float      # 선순위 근저당 채권최고액
    building_age_years: int        # 건물 연식
    area_m2: float                 # 전용면적

    # 라벨/파생(전세만 유효)
    fraud_label: Optional[int] = None      # 1=전세사기 위험 실현, 0=정상 (합성 GT)
    fraud_score: Optional[float] = None    # 모델 추론 위험도 0~1 (DB 업데이트용)


class UserProfile(BaseModel):
    """서비스 사용자(청년) 재무/선호 프로필."""
    user_id: str
    age: int
    monthly_income_manwon: float        # 월 세전 소득
    total_asset_manwon: float           # 총 자산(가용 현금성)
    monthly_living_cost_manwon: float   # 월 생활비(주거 제외)
    income_decile: int = Field(ge=1, le=10)   # 소득분위 1~10

    # 선호
    preferred_sido: Optional[str] = None
    preferred_gugun: Optional[str] = None
    nl_preference: str = ""             # 자연어 선호 입력
    workplace_lat: Optional[float] = None
    workplace_lng: Optional[float] = None


class AffordabilityResult(BaseModel):
    """(2)-1 적정 주거비 정적 공식 결과."""
    max_monthly_housing_manwon: float   # 감당 가능한 월 주거비 상한(RIR 기반)
    recommended_jeonse_deposit_manwon: float
    recommended_monthly_deposit_manwon: float
    recommended_monthly_rent_manwon: float
    rir_at_recommended: float
    notes: list[str] = []
