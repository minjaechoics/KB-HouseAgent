"""
(4) 추천 — 학습/평가 실험 스크립트.

실행:
    # 클릭라벨 없으면 먼저 생성
    python -m src.recommender.click_labels
    # 추천기 실험(전 후보 비교)
    python -m src.recommender.train --experiment
    # 특정 LTR 저장
    python -m src.recommender.train --model ltr_lgbm --save

평가지표(세션 홀드아웃):
    NDCG@k, MAP@k  — 랭킹 품질(클릭을 상위에 두는가).
    Safety@k       — 상위 k개 중 위험(fraud_score>0.5) 매물 비율(낮을수록 좋음, 안전성).
"""
from __future__ import annotations
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

from src import config
from src.recommender import models as R

warnings.filterwarnings("ignore")

FEATS = ["budget_fit", "safety", "commute", "maint", "age"]


def load_clicks() -> pd.DataFrame:
    path = config.DATA_GEN / "click_labels.csv"
    if not path.exists():
        raise FileNotFoundError("먼저 `python -m src.recommender.click_labels` 실행")
    return pd.read_csv(path)


def ndcg_at_k(relevance: np.ndarray, k: int) -> float:
    rel = relevance[:k]
    if rel.sum() == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(rel) + 2))
    dcg = (rel * discounts).sum()
    ideal = np.sort(relevance)[::-1][:k]
    idcg = (ideal * (1.0 / np.log2(np.arange(2, len(ideal) + 2)))).sum()
    return dcg / idcg if idcg > 0 else 0.0


def map_at_k(relevance: np.ndarray, k: int) -> float:
    rel = relevance[:k]
    hits, score = 0, 0.0
    for i, r in enumerate(rel):
        if r > 0:
            hits += 1
            score += hits / (i + 1)
    denom = min(relevance.sum(), k)
    return score / denom if denom > 0 else 0.0


def evaluate_ltr(model, test_df, k=5):
    """세션별로 예측 정렬 후 NDCG/MAP 계산."""
    ndcgs, maps = [], []
    for sid, grp in test_df.groupby("session_id"):
        X = grp[FEATS].fillna(0).to_numpy()
        scores = model.predict(X)
        order = np.argsort(scores)[::-1]
        rel = grp["clicked"].to_numpy()[order]
        ndcgs.append(ndcg_at_k(rel, k))
        maps.append(map_at_k(rel, k))
    return float(np.mean(ndcgs)), float(np.mean(maps))


def evaluate_scorer(score_fn, test_df, k=5):
    """규칙/콘텐츠 기반: score_fn(grp)->scores 배열."""
    ndcgs, maps = [], []
    for sid, grp in test_df.groupby("session_id"):
        scores = score_fn(grp)
        order = np.argsort(scores)[::-1]
        rel = grp["clicked"].to_numpy()[order]
        ndcgs.append(ndcg_at_k(rel, k))
        maps.append(map_at_k(rel, k))
    return float(np.mean(ndcgs)), float(np.mean(maps))


def run_experiment(k=5):
    df = load_clicks()
    sessions = df.session_id.unique()
    tr_s, te_s = train_test_split(sessions, test_size=0.25, random_state=42)
    tr = df[df.session_id.isin(tr_s)]
    te = df[df.session_id.isin(te_s)]

    results = []

    # --- 규칙 기반 (학습 불필요): 피처 가중합 ---
    rule_w = np.array([2.5, 3.0, 1.5, 0.8, 0.5])
    def rule_score(grp):
        return grp[FEATS].fillna(0).to_numpy() @ rule_w
    n, m = evaluate_scorer(rule_score, te, k)
    results.append(dict(model="rule", ndcg=n, map=m))

    # --- 콘텐츠 기반: 이상프로필 코사인 ---
    def content_score(grp):
        M = grp[FEATS].fillna(0).to_numpy()
        ideal = M.max(axis=0)
        num = M @ ideal
        den = np.linalg.norm(M, axis=1) * np.linalg.norm(ideal) + 1e-9
        return num / den
    n, m = evaluate_scorer(content_score, te, k)
    results.append(dict(model="content", ndcg=n, map=m))

    # --- LTR: LightGBM / XGBoost ---
    for backend in ("lightgbm", "xgboost"):
        try:
            ltr = R.LTRRecommender(backend).fit(tr)
            n, m = evaluate_ltr(ltr.model, te, k)
            results.append(dict(model=f"ltr_{backend}", ndcg=n, map=m))
        except ImportError:
            print(f"  [skip] {backend} 미설치")

    # --- 랜덤 baseline ---
    def rand_score(grp):
        return np.random.default_rng(0).random(len(grp))
    n, m = evaluate_scorer(rand_score, te, k)
    results.append(dict(model="random_baseline", ndcg=n, map=m))

    res = pd.DataFrame(results).sort_values("ndcg", ascending=False)
    print(f"\n=== 추천기 실험 (NDCG@{k}, MAP@{k}) ===")
    print(res.to_string(index=False))
    best = res.iloc[0]
    print(f"\n[best] {best['model']}  NDCG@{k}={best['ndcg']:.3f}")
    out = config.MODELS_DIR / "recommender_experiment_results.csv"
    res.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[experiment] 저장: {out}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", action="store_true")
    ap.add_argument("--model", default="ltr_lgbm",
                    choices=["ltr_lgbm", "ltr_xgb"])
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    if args.experiment:
        run_experiment(args.k)
    else:
        df = load_clicks()
        backend = "lightgbm" if args.model == "ltr_lgbm" else "xgboost"
        ltr = R.LTRRecommender(backend).fit(df)
        if args.save:
            out = config.MODELS_DIR / "recommender_model.joblib"
            joblib.dump(dict(model=ltr.model, backend=backend, feats=FEATS), out)
            print(f"[saved] {out}")


if __name__ == "__main__":
    main()
