"""
(4) 추천 — 합성 선호(클릭) 라벨 생성기.

실제 사용자 로그가 모이기 전, LTR(Learning-to-Rank) 모델을 학습/실험하기 위한
합성 클릭 라벨을 만든다. "사용자가 어떤 매물을 클릭/선호했는가"를 아래 선호함수로
확률적으로 생성한다.

선호 점수(효용) = 예산적합도 + 안전성 + 통근 + 관리비 + 소음/연식 등의 가중합.
  - 이 효용을 소프트맥스로 확률화 → 세션마다 후보 중 클릭을 샘플.
  - 가중치는 config가 아니라 여기서 명시(실험 대상). 사용자별로 약간의 무작위성 부여.

주의: 이 라벨은 '합성'이며, 실서비스에서는 실제 클릭 로그로 대체해야 한다.
      (사용자 요구사항: 실제 사용자 데이터 수집 전까지 합성으로 실험)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from src import config
from src.preference.affordability import compute_affordability
from src.fraud_risk.infer import FraudRiskScorer
from src.tools.map_tool import MapTool, SIDO_GUGUN_CENTROIDS


# 세션당 노출 후보 수
CANDIDATES_PER_SESSION = 15


def _utility_features(prop_row: pd.Series, user: dict, budget, workplace, map_tool):
    """매물-사용자 쌍의 효용 계산용 피처."""
    # 예산 적합도: 적정 보증금 대비 초과 정도(초과할수록 penalty)
    if prop_row["lease_type"] == "매매":
        budget_ref = max(
            budget.recommended_jeonse_deposit_manwon / config.PURCHASE_EQUITY_RATIO,
            1,
        )
        over = (prop_row["sale_price_manwon"] - budget_ref) / budget_ref
    elif prop_row["lease_type"] == "전세":
        budget_ref = max(budget.recommended_jeonse_deposit_manwon, 1)
        over = (prop_row["deposit_manwon"] - budget_ref) / budget_ref
    else:
        budget_ref = max(budget.max_monthly_housing_manwon, 1)
        monthly = prop_row["monthly_rent_manwon"] + prop_row["maintenance_fee_manwon"]
        over = (monthly - budget_ref) / budget_ref
    budget_fit = -np.clip(over, -1, 3)  # 예산 이내면 +, 초과하면 -

    # 안전성: fraud_score 낮을수록 좋음(월세는 0 취급)
    fs = prop_row["fraud_score"]
    safety = -(0.0 if pd.isna(fs) else fs)

    # 통근: 직장 좌표가 있으면 이동시간(분) penalty
    if workplace is not None:
        t = map_tool.travel_time((prop_row["lat"], prop_row["lng"]), workplace, "transit")
        commute = -t["minutes"] / 30.0  # 30분 기준 정규화
    else:
        commute = 0.0

    # 관리비 penalty
    maint = -prop_row["maintenance_fee_manwon"] / 10.0
    # 연식 penalty
    age = -prop_row["building_age_years"] / 30.0
    return np.array([budget_fit, safety, commute, maint, age])


# 기본 선호 가중치(실험 대상). [예산, 안전, 통근, 관리비, 연식]
DEFAULT_PREF_WEIGHTS = np.array([2.5, 3.0, 1.5, 0.8, 0.5])


def generate_click_labels(
    n_sessions: int = 2000,
    seed: int = config.GLOBAL_SEED,
    pref_weights: np.ndarray = DEFAULT_PREF_WEIGHTS,
) -> pd.DataFrame:
    """
    n_sessions개 세션 생성. 각 세션 = 한 사용자에게 노출된 후보 매물 그룹 + 클릭 라벨.
    return: DataFrame(session_id, user_id, property_id, 피처..., clicked)
    """
    rng = np.random.default_rng(seed)
    props = pd.read_csv(config.DATA_GEN / "properties.csv")
    jeonse_mask = props["lease_type"].eq("전세")
    if jeonse_mask.any():
        props.loc[jeonse_mask, "fraud_score"] = FraudRiskScorer().score_batch(
            props.loc[jeonse_mask]
        )
    users = pd.read_csv(config.DATA_GEN / "users.csv")
    map_tool = MapTool()
    centroids = list(SIDO_GUGUN_CENTROIDS.values())

    rows = []
    for sid in range(n_sessions):
        user = users.iloc[rng.integers(len(users))].to_dict()
        budget = compute_affordability(dict(
            user_id=user["user_id"], age=int(user["age"]),
            monthly_income_manwon=user["monthly_income_manwon"],
            total_asset_manwon=user["total_asset_manwon"],
            monthly_living_cost_manwon=user["monthly_living_cost_manwon"],
            income_decile=int(user["income_decile"]),
        ))
        # 일부 사용자는 직장(통근 선호) 보유
        workplace = centroids[rng.integers(len(centroids))] if rng.random() < 0.6 else None

        # 후보 노출: 랜덤 샘플(실서비스는 1차 필터 결과)
        cand = props.sample(CANDIDATES_PER_SESSION, random_state=int(rng.integers(1e9)))
        feats = np.array([
            _utility_features(r, user, budget, workplace, map_tool)
            for _, r in cand.iterrows()
        ])
        # 사용자별 가중치 지터
        w = pref_weights * rng.uniform(0.8, 1.2, size=len(pref_weights))
        utility = feats @ w + rng.normal(0, 0.5, len(cand))
        # 소프트맥스 확률로 1~2개 클릭 샘플
        p = np.exp(utility - utility.max())
        p = p / p.sum()
        n_click = rng.integers(1, 3)
        clicked_idx = rng.choice(len(cand), size=min(n_click, len(cand)),
                                 replace=False, p=p)
        clicked = np.zeros(len(cand), dtype=int)
        clicked[clicked_idx] = 1

        for j, (_, r) in enumerate(cand.iterrows()):
            rows.append(dict(
                session_id=sid, user_id=user["user_id"], property_id=r["property_id"],
                budget_fit=feats[j][0], safety=feats[j][1], commute=feats[j][2],
                maint=feats[j][3], age=feats[j][4],
                deposit_manwon=r["deposit_manwon"],
                lease_type=r["lease_type"],
                fraud_score=(np.nan if pd.isna(r["fraud_score"]) else float(r["fraud_score"])),
                clicked=int(clicked[j]),
            ))
    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = generate_click_labels(n_sessions=1500)
    out = config.DATA_GEN / "click_labels.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[click_labels] {len(df)} rows, {df.session_id.nunique()} sessions → {out}")
    print(f"  클릭률: {df.clicked.mean():.3%}")
    # 선호함수 검증: 클릭 매물은 예산적합/통근이 좋고, 전세 중 위험도가 낮아야 함
    print(f"  [예산적합] 클릭 {df[df.clicked==1].budget_fit.mean():.3f} vs 비클릭 {df[df.clicked==0].budget_fit.mean():.3f}")
    print(f"  [통근]     클릭 {df[df.clicked==1].commute.mean():.3f} vs 비클릭 {df[df.clicked==0].commute.mean():.3f}")
    j = df[df.lease_type == "전세"]
    print(f"  [전세 위험] 클릭 {j[j.clicked==1].fraud_score.mean():.3f} vs 비클릭 {j[j.clicked==0].fraud_score.mean():.3f}")
