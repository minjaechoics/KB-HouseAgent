"""Train, calibrate, evaluate, and deploy an actual-label guarantee model.

Example:
  python -m src.fraud_risk.train_actual --labels actual.csv \
    --provenance actual.provenance.json --apply
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.fraud_risk.actual_model import ACTUAL_FEATURE_NAMES, build_actual_feature_frame
from src.fraud_risk.calibration import (
    choose_cost_threshold,
    expected_calibration_error,
    fit_calibrator_with_temporal_selection,
)
from src.fraud_risk.real_labels import load_actual_contract_labels


def temporal_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """60/20/20 chronological split with no random future leakage."""
    dates = pd.to_datetime(df["guarantee_issue_date"])
    unique = np.array(sorted(dates.dt.normalize().unique()))
    if len(unique) < 5:
        raise ValueError("at least five distinct issue dates are required for temporal validation")
    train_end = unique[max(0, int(len(unique) * 0.60) - 1)]
    calibration_end = unique[max(1, int(len(unique) * 0.80) - 1)]
    train = (dates <= train_end).to_numpy()
    calibration = ((dates > train_end) & (dates <= calibration_end)).to_numpy()
    test = (dates > calibration_end).to_numpy()
    for name, mask in (("train", train), ("calibration", calibration), ("test", test)):
        labels = df.loc[mask, "accident_label"]
        if len(labels) < 20 or labels.nunique() != 2:
            raise ValueError(f"{name} period needs >=20 rows and both classes")
    metadata = {
        "train_end": pd.Timestamp(train_end).date().isoformat(),
        "calibration_end": pd.Timestamp(calibration_end).date().isoformat(),
        "test_end": dates.max().date().isoformat(),
        "rows": {"train": int(train.sum()), "calibration": int(calibration.sum()), "test": int(test.sum())},
    }
    return train, calibration, test, metadata


def _metrics(y, probability, threshold: float) -> dict:
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    pred = probability >= threshold
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability)),
        "ece_10": expected_calibration_error(y, probability, bins=10),
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "confusion": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "observed_rate": float(y.mean()),
        "predicted_rate": float(probability.mean()),
    }


def train_actual_model(
    labels_path: Path,
    provenance_path: Path,
    *,
    false_negative_cost: float = 20.0,
    false_positive_cost: float = 1.0,
) -> tuple[dict, dict]:
    df, audit, provenance = load_actual_contract_labels(labels_path, provenance_path)
    X = build_actual_feature_frame(df)
    y = df["accident_label"].to_numpy(dtype=int)
    train_mask, calibration_mask, test_mask, split_meta = temporal_split(df)

    estimator = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(max_iter=2_000, C=1.0, solver="lbfgs")),
    ])
    estimator.fit(X.loc[train_mask], y[train_mask])
    uncalibrated = estimator.predict_proba(X.loc[calibration_mask])[:, 1]
    calibrator, calibration_meta = fit_calibrator_with_temporal_selection(
        uncalibrated, y[calibration_mask]
    )
    calibrated = calibrator.transform(uncalibrated)
    decision = choose_cost_threshold(
        y[calibration_mask], calibrated,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    )

    test_raw = estimator.predict_proba(X.loc[test_mask])[:, 1]
    test_probability = calibrator.transform(test_raw)
    metrics = _metrics(y[test_mask], test_probability, decision.threshold)
    trained_at = datetime.now(timezone.utc).isoformat()
    bundle = {
        "bundle_version": 2,
        "kind": "sklearn_actual_contract_labels",
        "model": estimator,
        "calibrator": calibrator,
        "calibration_method": calibration_meta["method"],
        "feature_names": ACTUAL_FEATURE_NAMES,
        "decision_threshold": decision.threshold,
        "trained_at": trained_at,
    }
    metadata = {
        "schema_version": 2,
        "kind": "sklearn_actual_contract_labels",
        "label_source_status": "actual_contract_level_verified",
        "model_file": "fraud_risk_model.joblib",
        "trained_at": trained_at,
        "provider": provenance["provider"],
        "label_definition": provenance.get("label_definition"),
        "audit": asdict(audit),
        "split": split_meta,
        "calibration": calibration_meta,
        "cost_policy": {
            "false_negative_cost": false_negative_cost,
            "false_positive_cost": false_positive_cost,
            "decision_threshold": decision.threshold,
            "calibration_expected_cost_per_row": decision.expected_cost,
            "calibration_false_negatives": decision.false_negatives,
            "calibration_false_positives": decision.false_positives,
        },
        "test_metrics": metrics,
        "feature_names": ACTUAL_FEATURE_NAMES,
        "limitations": [
            "보증가입 표본에 조건부인 사고확률이며 전체 임대차 계약의 무조건부 확률이 아님",
            "새 관찰기간·지역·상품에 적용하기 전 드리프트와 재보정 검증 필요",
        ],
    }
    return bundle, metadata


def save_model(bundle: dict, metadata: dict) -> tuple[Path, Path]:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODELS_DIR / "fraud_risk_model.joblib"
    metadata_path = config.MODELS_DIR / "fraud_risk_model.json"
    joblib.dump(bundle, model_path)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return model_path, metadata_path


def apply_to_database() -> int:
    """Re-score only jeonse rows in the serving SQLite database."""
    from src.fraud_risk.infer import FraudRiskScorer

    if not config.DB_PATH.exists():
        raise FileNotFoundError(config.DB_PATH)
    conn = sqlite3.connect(config.DB_PATH)
    rows = pd.read_sql_query("SELECT * FROM properties WHERE lease_type = '전세'", conn)
    scorer = FraudRiskScorer()
    scores = scorer.score_batch(rows)
    conn.executemany(
        "UPDATE properties SET fraud_score=? WHERE property_id=?",
        [(float(score), property_id) for score, property_id in zip(scores, rows["property_id"])],
    )
    conn.commit()
    conn.close()
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--false-negative-cost", type=float, default=20.0)
    parser.add_argument("--false-positive-cost", type=float, default=1.0)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    bundle, metadata = train_actual_model(
        args.labels, args.provenance,
        false_negative_cost=args.false_negative_cost,
        false_positive_cost=args.false_positive_cost,
    )
    model_path, metadata_path = save_model(bundle, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"[saved] {model_path}\n[saved] {metadata_path}")
    if args.apply:
        print(f"[applied] {apply_to_database():,} jeonse rows")


if __name__ == "__main__":
    main()

