"""데이터 증강 결과가 현실적인 범위인지 확인하는 스모크 테스트."""
import numpy as np
import pandas as pd
from src import config
from src.data_augmentation.generate import (
    generate_properties, assign_fraud_labels, generate_users,
    validate_generated_properties, HOUSE_TYPES, TRANSACTION_TYPES,
)
from src.data_augmentation.property_schema import BROKER_LISTING_COLUMNS


def test_property_realism():
    rng = np.random.default_rng(0)
    df = assign_fraud_labels(generate_properties(3000, rng), rng)
    j = df[df.lease_type == "전세"]
    # 전세사기 양성률은 한 자릿수~십수 % (현실적 imbalance)
    rate = j.fraud_label.mean()
    assert 0.02 < rate < 0.25, f"unrealistic fraud rate {rate}"

    # 부채비율이 높을수록 사기율이 높아야 한다 (인과 신호 존재)
    recover = j.market_price_manwon * config.AUCTION_RECOVERY_RATIO
    debt = (j.deposit_manwon + j.senior_deposit_sum_manwon
            + j.senior_mortgage_manwon) / recover.clip(lower=1)
    hi = j[debt > 1.0].fraud_label.mean()
    lo = j[debt <= 1.0].fraud_label.mean()
    assert hi > lo, "danger-zone units must have higher fraud rate"

    # 보증금/시세가 양수, 결측 없음
    assert (df.loc[df.transaction_type.isin(["전세", "월세"]), "deposit_manwon"] > 0).all()
    assert (df.loc[df.transaction_type == "매매", "deposit_manwon"] == 0).all()
    assert df.market_price_manwon.notna().all()

    # 공인중개사 확인·설명/표시·광고 통합 스키마와 기존 Agent 계약을 모두 충족
    assert not validate_generated_properties(df, expected_n=3000)
    assert set(BROKER_LISTING_COLUMNS).issubset(df.columns)
    assert df.is_synthetic.all()
    assert df.property_id.is_unique
    assert (df.my_priority_rank <= df.building_total_units).all()
    assert (df.loc[df.lease_type == "전세", "monthly_rent_manwon"] == 0).all()
    assert (df.loc[df.transaction_type == "매매", ["deposit_manwon", "monthly_rent_manwon"]] == 0).all().all()
    assert set(HOUSE_TYPES) == set(df.house_type.unique())
    assert set(TRANSACTION_TYPES) == set(df.transaction_type.unique())
    coverage = df.groupby(["house_type", "transaction_type"]).size()
    assert len(coverage) == len(HOUSE_TYPES) * len(TRANSACTION_TYPES)
    assert df.sido.nunique() == 17
    assert df[["sido", "gugun"]].drop_duplicates().shape[0] == 252
    region_txn = df.groupby(["sido", "gugun", "transaction_type"]).size().unstack(fill_value=0)
    assert len(region_txn) == 252 and (region_txn > 0).all().all()
    # 하나의 실제 공공 주소 기준점에는 서로 다른 주택형·거래유형의 분석 후보가
    # 함께 존재할 수 있다. 좌표 자체를 임의로 흔들지 않고 실제 주소와 정확히
    # 일치시키므로 전 행 좌표 고유성 대신 충분한 공간 다양성을 검증한다.
    unique_coordinate_ratio = df[["lat", "lng"]].drop_duplicates().shape[0] / len(df)
    assert unique_coordinate_ratio > 0.90
    assert not df["road_address"].astype(str).str.contains("합성", regex=False).any()
    assert df["synthetic_notice"].astype(str).str.contains(
        "실제 거래 매물이 아닙니다", regex=False
    ).all()
    assert df["region_coordinate_source"].str.contains(
        "CCTV", regex=False
    ).mean() > 0.80
    txn_counts = df.transaction_type.value_counts()
    assert txn_counts.max() - txn_counts.min() <= 2
    daejeon_yuseong = df[(df.sido == "대전") & (df.gugun == "유성구")]
    assert set(daejeon_yuseong.transaction_type) == set(TRANSACTION_TYPES)


def test_user_realism():
    rng = np.random.default_rng(0)
    u = generate_users(2000, rng)
    assert u.income_decile.between(1, 10).all()
    assert (u.age.between(config.YOUTH_AGE_MIN, config.YOUTH_AGE_MAX)).all()
    # 소득과 자산은 양의 상관
    assert u[["monthly_income_manwon", "total_asset_manwon"]].corr().iloc[0, 1] > 0.2


if __name__ == "__main__":
    test_property_realism()
    test_user_realism()
    print("OK: data augmentation tests passed")
