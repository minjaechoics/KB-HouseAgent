"""
(1) 전세사기 위험도 — 훈련 모듈.

실행 예:
    # 전체 모델 x 전체 피처세트 실험(권장 첫 실행)
    python -m src.fraud_risk.train --experiment

    # 특정 조합만 학습해 저장(추론에 쓸 최종 모델)
    python -m src.fraud_risk.train --model xgboost --feature_set core_plus --save

불균형 대응:
    --smote  (imbalanced-learn 설치 시) 학습셋에 SMOTE 오버샘플링 적용.

평가지표:
    사기예측은 소수 양성이 중요 → ROC-AUC 와 PR-AUC(평균정밀도)를 함께 본다.
    임계 0.5 기준 precision/recall/F1 및 confusion matrix도 출력.
"""
from __future__ import annotations
import argparse
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix,
)
import joblib

from src import config
from src.fraud_risk import features as F
from src.fraud_risk import models as M

warnings.filterwarnings("ignore")


def load_data() -> pd.DataFrame:
    path = config.DATA_GEN / "properties.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 없음. 먼저 `python -m src.data_augmentation.generate` 실행."
        )
    return pd.read_csv(path)


def _maybe_smote(X, y, use_smote):
    if not use_smote:
        return X, y
    try:
        from imblearn.over_sampling import SMOTE
        Xr, yr = SMOTE(random_state=42).fit_resample(X, y)
        return Xr, yr
    except ImportError:
        print("  [warn] imbalanced-learn 미설치 → SMOTE 건너뜀")
        return X, y


def evaluate(model, X_te, y_te) -> dict:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_te)[:, 1]
    else:
        proba = model.decision_function(X_te)
    pred = (proba >= 0.5).astype(int)
    return dict(
        roc_auc=float(roc_auc_score(y_te, proba)),
        pr_auc=float(average_precision_score(y_te, proba)),
        f1=float(f1_score(y_te, pred, zero_division=0)),
        precision=float(precision_score(y_te, pred, zero_division=0)),
        recall=float(recall_score(y_te, pred, zero_division=0)),
        confusion=confusion_matrix(y_te, pred).tolist(),
    )


def train_one(df, model_name, feature_set, use_smote=False, save=False):
    X, y, cols = F.build_xy(df, feature_set)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    X_tr, y_tr = _maybe_smote(X_tr, y_tr, use_smote)

    est = M.build_estimator(model_name, y_tr)
    est.fit(X_tr, y_tr)
    metrics = evaluate(est, X_te, y_te)

    if save:
        config.MODELS_DIR.mkdir(exist_ok=True)
        bundle = dict(model=est, feature_set=feature_set, feature_names=cols,
                      model_name=model_name)
        out = config.MODELS_DIR / "fraud_risk_model.joblib"
        joblib.dump(bundle, out)
        print(f"  [saved] {out}")
    return metrics, cols


def run_experiment(df, use_smote=False):
    models = M.available_models()
    fsets = list(F.FEATURE_SETS)
    print(f"[experiment] models={models}")
    print(f"[experiment] feature_sets={fsets}  smote={use_smote}\n")
    rows = []
    for fs in fsets:
        for m in models:
            metrics, _ = train_one(df, m, fs, use_smote=use_smote, save=False)
            rows.append(dict(model=m, feature_set=fs, **{
                k: v for k, v in metrics.items() if k != "confusion"
            }))
    res = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False))
    best = res.iloc[0]
    print(f"\n[best by PR-AUC] model={best['model']} "
          f"feature_set={best['feature_set']} "
          f"PR-AUC={best['pr_auc']:.3f} ROC-AUC={best['roc_auc']:.3f}")
    out = config.MODELS_DIR / "fraud_experiment_results.csv"
    res.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[experiment] 결과 저장: {out}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", action="store_true",
                    help="모든 모델 x 피처세트 조합 실험")
    ap.add_argument("--model", default="xgboost", choices=list(M._REGISTRY))
    ap.add_argument("--feature_set", default="core_plus", choices=list(F.FEATURE_SETS))
    ap.add_argument("--smote", action="store_true")
    ap.add_argument("--save", action="store_true", help="단일 학습 결과를 추론용으로 저장")
    args = ap.parse_args()

    df = load_data()
    if args.experiment:
        run_experiment(df, use_smote=args.smote)
    else:
        metrics, cols = train_one(
            df, args.model, args.feature_set,
            use_smote=args.smote, save=args.save,
        )
        print(f"[train] model={args.model} feature_set={args.feature_set}")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
