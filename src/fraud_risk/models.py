"""
(1) 전세사기 위험도 — 모델 후보군(Model Zoo).

논문/실무에서 사기·부도 예측에 쓰이는 대표 모델들을 모두 구현하여
동일 인터페이스로 실험 가능하게 한다.

포함 모델과 선택 근거:
  - logistic       : 로지스틱 회귀. 신용/사기 스코어링의 표준 baseline, 해석성 최고.
  - random_forest  : 랜덤포레스트. 비선형·상호작용 포착, robust.
  - grad_boost     : 사이킷런 GradientBoosting. 부스팅 baseline.
  - xgboost        : XGBoost. 테이블데이터 사기예측 SOTA급, 불균형 대응(scale_pos_weight).
  - lightgbm       : LightGBM. 대용량/고속, 범주형 친화.
  - mlp            : 다층퍼셉트론(sklearn). 신경망 baseline.
  - (옵션) tabnet  : 딥테이블 모델. torch/pytorch-tabnet 필요. 미설치 시 건너뜀.

불균형 라벨(사기=소수) 대응:
  - class_weight / scale_pos_weight 자동 설정
  - 학습 스크립트에서 SMOTE 오버샘플링 옵션 제공(train.py)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _pos_weight(y: np.ndarray) -> float:
    pos = max(int(y.sum()), 1)
    neg = len(y) - pos
    return neg / pos


@dataclass
class ModelSpec:
    name: str
    builder: Callable[[np.ndarray], object]   # y -> fitted-ready estimator
    needs_scaling: bool = False               # 파이프라인에서 스케일 필요 여부


def _logistic(y):
    return LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)


def _rf(y):
    return RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=5,
        class_weight="balanced_subsample", n_jobs=-1, random_state=42,
    )


def _gb(y):
    return GradientBoostingClassifier(random_state=42)


def _xgb(y):
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=_pos_weight(y),
        eval_metric="aucpr", tree_method="hist", n_jobs=-1, random_state=42,
    )


def _lgbm(y):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=600, num_leaves=31, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=_pos_weight(y), n_jobs=-1, random_state=42, verbose=-1,
    )


def _mlp(y):
    return MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu",
        alpha=1e-3, max_iter=500, random_state=42,
    )


# 등록 테이블 -----------------------------------------------------------
_REGISTRY: dict[str, ModelSpec] = {
    "logistic":      ModelSpec("logistic", _logistic, needs_scaling=True),
    "random_forest": ModelSpec("random_forest", _rf, needs_scaling=False),
    "grad_boost":    ModelSpec("grad_boost", _gb, needs_scaling=False),
    "xgboost":       ModelSpec("xgboost", _xgb, needs_scaling=False),
    "lightgbm":      ModelSpec("lightgbm", _lgbm, needs_scaling=False),
    "mlp":           ModelSpec("mlp", _mlp, needs_scaling=True),
}


def available_models() -> list[str]:
    """설치 여부를 확인해 실제 사용 가능한 모델만 반환."""
    ok = []
    for name in _REGISTRY:
        try:
            if name == "xgboost":
                import xgboost  # noqa
            if name == "lightgbm":
                import lightgbm  # noqa
            ok.append(name)
        except ImportError:
            continue
    return ok


def build_estimator(name: str, y: np.ndarray):
    """
    이름으로 estimator(또는 스케일 파이프라인) 생성.
    스케일이 필요한 모델은 StandardScaler와 묶은 Pipeline 반환.
    """
    if name not in _REGISTRY:
        raise KeyError(f"unknown model '{name}'. choices={list(_REGISTRY)}")
    spec = _REGISTRY[name]
    est = spec.builder(y)
    if spec.needs_scaling:
        return Pipeline([("scaler", StandardScaler()), ("clf", est)])
    return est
