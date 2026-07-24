"""
(4) 추천 모듈 — 여러 후보 구조(사용자가 실험으로 선택).

구현된 추천기 4종(모두 동일 인터페이스 recommend(candidates_df, context) -> ranked_df):

  1) RuleBasedRecommender   : 해석 가능한 가중합 스코어(예산·안전·통근·관리비).
  2) LTRRecommender         : Learning-to-Rank (LightGBM lambdarank / XGBoost rank).
  3) ContentBasedRecommender: 사용자 선호벡터 ↔ 매물벡터 코사인 유사도.
  4) HybridRecommender      : 위험 하드필터 → LTR 랭킹 → 통근 재랭킹.

context(dict) 필수 키:
  affordability: AffordabilityResult
  workplace: (lat,lng) or None
  sort_by: recommended 또는 risk_asc 등 결과 정렬 의도
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from src import config
from src.tools.map_tool import MapTool


# ----------------------------------------------------------------------
# 공통: 매물-사용자 피처 (추천용)
# ----------------------------------------------------------------------
def compute_reco_features(cand: pd.DataFrame, context: dict,
                          map_tool: MapTool | None = None) -> pd.DataFrame:
    df = cand.copy()
    aff = context["affordability"]
    workplace = context.get("workplace")
    map_tool = map_tool or MapTool()

    # 예산 적합도
    def _budget_fit(r):
        if r["lease_type"] == "매매":
            ref = max(
                aff.recommended_jeonse_deposit_manwon / config.PURCHASE_EQUITY_RATIO,
                1,
            )
            over = (r.get("sale_price_manwon", 0) - ref) / ref
        elif r["lease_type"] == "전세":
            ref = max(aff.recommended_jeonse_deposit_manwon, 1)
            over = (r["deposit_manwon"] - ref) / ref
        else:
            ref = max(aff.max_monthly_housing_manwon, 1)
            monthly = r.get("monthly_rent_manwon", 0) + r.get("maintenance_fee_manwon", 0)
            over = (monthly - ref) / ref
        return -np.clip(over, -1, 3)
    df["f_budget"] = df.apply(_budget_fit, axis=1)

    # 안전성
    df["f_safety"] = -pd.to_numeric(df["fraud_score"], errors="coerce").fillna(0.0)

    # 통근
    if workplace is not None:
        df["f_commute"] = df.apply(
            lambda r: -map_tool.travel_time((r["lat"], r["lng"]), workplace, "transit")["minutes"] / 30.0,
            axis=1,
        )
    else:
        df["f_commute"] = 0.0

    df["f_maint"] = -df.get("maintenance_fee_manwon", pd.Series(0, index=df.index)).fillna(0) / 10.0
    return df


FEATURE_COLS = ["f_budget", "f_safety", "f_commute", "f_maint"]


# ----------------------------------------------------------------------
# 1) 규칙 기반
# ----------------------------------------------------------------------
@dataclass
class RuleBasedRecommender:
    weights: tuple = (2.5, 3.0, 1.5, 0.8)   # 예산, 안전, 통근, 관리비

    def recommend(self, cand: pd.DataFrame, context: dict, top_k: int = 10):
        df = compute_reco_features(cand, context)
        w = np.array(self.weights)
        df["score"] = df[FEATURE_COLS].to_numpy() @ w
        return df.sort_values("score", ascending=False).head(top_k)


# ----------------------------------------------------------------------
# 2) Learning-to-Rank
# ----------------------------------------------------------------------
class LTRRecommender:
    def __init__(self, backend: str = "lightgbm"):
        self.backend = backend
        self.model = None

    def fit(self, click_df: pd.DataFrame):
        """click_labels.csv 로 학습. 그룹=session_id."""
        feats = ["budget_fit", "safety", "commute", "maint", "age"]
        X = click_df[feats].fillna(0).to_numpy()
        y = click_df["clicked"].to_numpy()
        group = click_df.groupby("session_id").size().to_numpy()

        if self.backend == "lightgbm":
            import lightgbm as lgb
            self.model = lgb.LGBMRanker(
                objective="lambdarank", n_estimators=300, num_leaves=31,
                learning_rate=0.05, random_state=42, verbose=-1,
            )
            self.model.fit(X, y, group=group)
        elif self.backend == "xgboost":
            import xgboost as xgb
            self.model = xgb.XGBRanker(
                objective="rank:pairwise", n_estimators=300, max_depth=5,
                learning_rate=0.05, random_state=42,
            )
            self.model.fit(X, y, group=group)
        else:
            raise ValueError(f"unknown backend {self.backend}")
        self._feats = feats
        return self

    def recommend(self, cand: pd.DataFrame, context: dict, top_k: int = 10):
        df = compute_reco_features(cand, context)
        # 학습 피처명(budget_fit 등)에 맞춰 매핑
        Xmap = pd.DataFrame({
            "budget_fit": df["f_budget"], "safety": df["f_safety"],
            "commute": df["f_commute"], "maint": df["f_maint"],
            "age": -df.get("building_age_years", pd.Series(0, index=df.index)).fillna(0) / 30.0,
        })
        df["score"] = self.model.predict(Xmap.to_numpy())
        return df.sort_values("score", ascending=False).head(top_k)


# ----------------------------------------------------------------------
# 3) 콘텐츠 기반 (코사인 유사도)
# ----------------------------------------------------------------------
class ContentBasedRecommender:
    def recommend(self, cand: pd.DataFrame, context: dict, top_k: int = 10):
        df = compute_reco_features(cand, context)
        M = df[FEATURE_COLS].to_numpy()
        # 이상적 프로필: 모든 피처 최대(예산적합↑, 안전↑, 통근↑, 관리비↑)
        ideal = M.max(axis=0)
        num = M @ ideal
        den = (np.linalg.norm(M, axis=1) * np.linalg.norm(ideal) + 1e-9)
        df["score"] = num / den
        return df.sort_values("score", ascending=False).head(top_k)


# ----------------------------------------------------------------------
# 4) 하이브리드
# ----------------------------------------------------------------------
class HybridRecommender:
    def __init__(self, ltr: LTRRecommender):
        self.ltr = ltr

    def recommend(self, cand: pd.DataFrame, context: dict, top_k: int = 10):
        # (a) LTR 랭킹 — 위험도는 후보 제거에 사용하지 않는다.
        ranked = self.ltr.recommend(cand, context, top_k=top_k * 2)
        # (b) 통근 재랭킹(동점 tie-break): 최종 점수에 통근 가중 소폭 추가
        ranked = ranked.copy()
        ranked["score"] = ranked["score"] + 0.3 * ranked["f_commute"]
        if context.get("sort_by") == "risk_asc" and "fraud_score" in ranked:
            return ranked.sort_values(
                ["fraud_score", "score"], ascending=[True, False], na_position="last"
            ).head(top_k)
        return ranked.sort_values("score", ascending=False).head(top_k)


REGISTRY = {
    "rule": RuleBasedRecommender,
    "ltr_lgbm": lambda: LTRRecommender("lightgbm"),
    "ltr_xgb": lambda: LTRRecommender("xgboost"),
    "content": ContentBasedRecommender,
    # hybrid는 학습된 ltr이 필요해 train 스크립트에서 조립
}
