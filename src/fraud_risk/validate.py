"""
(1) 전세사기 위험 모델 — 검증(Validation) 모듈.

★ 핵심 질문: "전세사기라는 라벨이 있어야 검증할 수 있는 것 아닌가?" → 맞다.
검증에는 반드시 '정답 라벨(y)'이 필요하다. 우리는 두 단계로 접근한다.

────────────────────────────────────────────────────────────────────
[단계 1] 지금(합성 라벨)의 검증 — '방법론이 맞는지'를 검증
────────────────────────────────────────────────────────────────────
현재 fraud_label은 (0) 데이터 증강에서 '구조적 위험식 + 실제 지역 사고율 +
노이즈'의 로지스틱 결합으로 생성한 합성 정답이다. 이 라벨로 하는 검증은
"모델 자체가 위험 신호를 학습·재현할 수 있는가"라는 파이프라인 검증이다.
  - 교차검증(Stratified K-Fold)으로 PR-AUC/ROC-AUC의 안정성 측정.
  - 보정(calibration): 예측확률이 실제 빈도와 일치하는지(신뢰도).
  - 주의: 합성 라벨 성능은 '상한'이 아니라 '방법 타당성' 지표다.
    데이터 생성기가 만든 규칙을 모델이 되찾는지를 보는 것.

────────────────────────────────────────────────────────────────────
[단계 2] 실서비스(실측 라벨)의 검증 — '실제 성능'을 검증
────────────────────────────────────────────────────────────────────
실제 검증에 쓸 수 있는 '진짜 라벨' 후보(획득 가능):
  (a) HUG/HF 전세보증 '사고'(보증이행) 발생 여부 — 실제 반환채무 불이행.
      → 우리가 이미 쓰는 KHUG 사고현황이 지역집계. 매물 단위 사고 데이터를
        확보하면 그대로 y가 된다.
  (b) 법원 경매/판례에서 임차인 배당 손실 발생 여부.
  (c) 정부 전세사기피해자 결정(전세사기특별법) 명단 매칭.
  (d) 계약 후 추적: 만기 시 보증금 미반환/지연 발생 여부(라벨 지연).
검증 방식:
  - 시점 분할(temporal split): 과거 계약으로 학습, 이후 계약으로 검증
    (미래 예측 성능 = 실제 서비스 성능). 랜덤 분할보다 현실적.
  - 지역/시기 OOD 검증: 학습에 없던 지역·기간에서 성능 확인.
  - 임계값은 '피해 예방(recall) vs 과잉경보(precision)' 트레이드오프로 결정.

실행:
    python -m src.fraud_risk.validate                 # 합성 라벨 CV + 보정
    python -m src.fraud_risk.validate --real labels.csv  # 실측 라벨 파일로 검증
    python -m src.fraud_risk.validate --temporal contract_date  # 시점분할
"""
from __future__ import annotations
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve

from src import config
from src.fraud_risk import features as F
from src.fraud_risk import models as M

warnings.filterwarnings("ignore")


def load_props(real_label_csv: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(config.DATA_GEN / "properties.csv")
    if real_label_csv:
        # 실측 라벨 병합: property_id + true_fraud 컬럼을 가진 CSV
        real = pd.read_csv(real_label_csv)
        df = df.merge(real[["property_id", "true_fraud"]], on="property_id", how="inner")
        df["fraud_label"] = df["true_fraud"]
        print(f"[validate] 실측 라벨 {len(df)}건 병합 (양성률 {df.fraud_label.mean():.3%})")
    return df


def cross_validate(df, model_name="logistic", feature_set="core_plus", k=5):
    """Stratified K-Fold 교차검증 → PR-AUC/ROC-AUC 평균±표준편차 + 보정."""
    X, y, cols = F.build_xy(df, feature_set)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    pr_aucs, roc_aucs, briers = [], [], []
    all_proba, all_y = [], []

    for tr, te in skf.split(X, y):
        est = M.build_estimator(model_name, y[tr])
        est.fit(X[tr], y[tr])
        proba = est.predict_proba(X[te])[:, 1]
        pr_aucs.append(average_precision_score(y[te], proba))
        roc_aucs.append(roc_auc_score(y[te], proba))
        briers.append(brier_score_loss(y[te], proba))
        all_proba.append(proba); all_y.append(y[te])

    proba = np.concatenate(all_proba); yy = np.concatenate(all_y)
    print(f"\n=== 교차검증 ({model_name} / {feature_set} / {k}-fold) ===")
    print(f"  PR-AUC : {np.mean(pr_aucs):.3f} ± {np.std(pr_aucs):.3f}  "
          f"(baseline=양성률 {y.mean():.3f})")
    print(f"  ROC-AUC: {np.mean(roc_aucs):.3f} ± {np.std(roc_aucs):.3f}")
    print(f"  Brier  : {np.mean(briers):.3f}  (낮을수록 확률보정 좋음)")

    # 보정 곡선(예측확률 vs 실제빈도)
    frac_pos, mean_pred = calibration_curve(yy, proba, n_bins=8, strategy="quantile")
    print("  [보정] 예측확률→실제빈도 (이상적이면 y≈x):")
    for mp, fp in zip(mean_pred, frac_pos):
        bar = "█" * int(fp * 20)
        print(f"    예측 {mp:.2f} → 실제 {fp:.2f} {bar}")
    return dict(pr_auc=float(np.mean(pr_aucs)), roc_auc=float(np.mean(roc_aucs)),
                brier=float(np.mean(briers)))


def temporal_validate(df, date_col, model_name="logistic", feature_set="core_plus"):
    """시점 분할 검증: 과거로 학습 → 이후로 검증(실서비스 성능 근사)."""
    if date_col not in df.columns:
        print(f"[validate] '{date_col}' 컬럼이 없어 시점분할 생략. "
              f"(실서비스에선 계약일 컬럼 추가)")
        return None
    df = df.sort_values(date_col)
    split = int(len(df) * 0.7)
    tr_df, te_df = df.iloc[:split], df.iloc[split:]
    Xtr, ytr, _ = F.build_xy(tr_df, feature_set)
    Xte, yte, _ = F.build_xy(te_df, feature_set)
    est = M.build_estimator(model_name, ytr); est.fit(Xtr, ytr)
    proba = est.predict_proba(Xte)[:, 1]
    print(f"\n=== 시점분할 검증 (학습 {len(tr_df)} → 검증 {len(te_df)}) ===")
    print(f"  PR-AUC : {average_precision_score(yte, proba):.3f}")
    print(f"  ROC-AUC: {roc_auc_score(yte, proba):.3f}")


def threshold_sweep(df, model_name="logistic", feature_set="core_plus"):
    """임계값별 precision/recall — 피해예방(recall) 우선 지점 탐색."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score, f1_score
    X, y, _ = F.build_xy(df, feature_set)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    est = M.build_estimator(model_name, ytr); est.fit(Xtr, ytr)
    proba = est.predict_proba(Xte)[:, 1]
    print(f"\n=== 임계값 스윕 (피해예방=recall 우선 관점) ===")
    print(f"  {'임계':>5} {'정밀도':>7} {'재현율':>7} {'F1':>6}")
    for th in [0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        pred = (proba >= th).astype(int)
        p = precision_score(yte, pred, zero_division=0)
        r = recall_score(yte, pred, zero_division=0)
        f = f1_score(yte, pred, zero_division=0)
        print(f"  {th:>5.2f} {p:>7.3f} {r:>7.3f} {f:>6.3f}")
    print("  → 전세사기는 '놓치면 큰 피해'이므로 recall을 높이는 낮은 임계가 흔히 선호됨.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="logistic")
    ap.add_argument("--feature_set", default="core_plus")
    ap.add_argument("--real", default=None, help="실측 라벨 CSV(property_id,true_fraud)")
    ap.add_argument("--temporal", default=None, help="시점분할 기준 날짜 컬럼")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    df = load_props(args.real)
    cross_validate(df, args.model, args.feature_set, args.k)
    threshold_sweep(df, args.model, args.feature_set)
    if args.temporal:
        temporal_validate(df, args.temporal, args.model, args.feature_set)

    if not args.real:
        print("\n[안내] 지금은 '합성 라벨' 기준 검증입니다(방법론 타당성 확인).")
        print("       실제 성능 검증은 --real 로 실측 사고/피해 라벨을 넣어 실행하세요.")
        print("       실측 라벨 후보: HUG 보증사고, 법원 경매 배당손실, 전세사기피해자 결정 명단.")


if __name__ == "__main__":
    main()
