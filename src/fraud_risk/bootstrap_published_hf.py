"""Deploy the actual-label HF published model with transparent prior transfer."""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src import config
from src.fraud_risk.actual_model import (
    ACTUAL_FEATURE_NAMES,
    HF_PAPER_DOI,
    HF_PUBLISHED_COEFFICIENTS,
    HF_PUBLISHED_INTERCEPT,
    HF_SAMPLE_ACCIDENT_RATE,
    cost_ratio_threshold,
    published_hf_logits,
    solve_prior_logit_shift,
    stable_sigmoid,
)


def create_metadata(
    properties_path: Path | None,
    target_rate: float,
    false_negative_cost: float,
    false_positive_cost: float,
    database_path: Path | None = None,
) -> dict:
    if database_path is not None:
        connection = sqlite3.connect(database_path)
        try:
            jeonse = pd.read_sql_query(
                "SELECT * FROM properties WHERE lease_type = '전세'", connection
            )
        finally:
            connection.close()
        portfolio_source = str(database_path)
    elif properties_path is not None:
        properties = pd.read_csv(properties_path)
        jeonse = properties.loc[properties["lease_type"] == "전세"].copy()
        portfolio_source = str(properties_path)
    else:
        raise ValueError("properties_path or database_path is required")
    if jeonse.empty:
        raise ValueError("no jeonse listings found")
    logits = published_hf_logits(jeonse)
    shift = solve_prior_logit_shift(logits, target_rate)
    probability = stable_sigmoid(logits + shift)
    threshold = cost_ratio_threshold(false_negative_cost, false_positive_cost)
    return {
        "schema_version": 2,
        "kind": "published_hf_actual_label_transfer",
        "label_source_status": "actual_labels_external_published_coefficients",
        "trained_at": "published-2025",
        "deployed_at": datetime.now(timezone.utc).isoformat(),
        "source": HF_PAPER_DOI,
        "source_sample": {
            "contracts": 453_122,
            "accidents": 22_601,
            "issue_period": "2019-01 through 2023-12",
            "outcomes_observed_through": "2024-12",
            "scope": "Seoul, Gyeonggi, Incheon guarantee contracts",
        },
        "feature_names": ACTUAL_FEATURE_NAMES,
        "coefficients": HF_PUBLISHED_COEFFICIENTS,
        "published_intercept": HF_PUBLISHED_INTERCEPT,
        "prior_calibration": {
            "method": "logit_intercept_shift_to_reference_incidence",
            "target_rate": target_rate,
            "reference_rate": HF_SAMPLE_ACCIDENT_RATE,
            "logit_shift": shift,
            "portfolio_rows": len(jeonse),
            "portfolio_source": portfolio_source,
            "portfolio_mean_after": float(probability.mean()),
            "note": "발급연도 통제계수가 논문 표에서 생략되어 절편만 현재 포트폴리오에 정렬",
        },
        "cost_policy": {
            "false_negative_cost": false_negative_cost,
            "false_positive_cost": false_positive_cost,
            "decision_threshold": threshold,
            "method": "C_FP / (C_FP + C_FN), calibrated-probability Bayes rule",
        },
        "limitations": [
            "로컬 원시 계약 라벨로 재학습한 모델이 아니라 실제 HF 라벨로 추정된 공개 계수의 이식 모델",
            "HUG 공개 개별 대위변제 파일은 합성자료이므로 학습에서 제외",
            "전국·비보증 매물로의 외삽이며 개별 계약의 법률·등기 실사 또는 보증가입 심사를 대체하지 않음",
            "현재 매물에 없는 임대인 다주택·임차인 연령·실제 전세대출 변수는 0/미상으로 처리",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--properties", type=Path, default=config.DATA_GEN / "properties.csv")
    parser.add_argument(
        "--database", type=Path,
        help="보정 모집단을 서비스 SQLite에서 직접 읽는다. 지정 시 --properties보다 우선한다.",
    )
    parser.add_argument("--target-rate", type=float, default=HF_SAMPLE_ACCIDENT_RATE)
    parser.add_argument("--false-negative-cost", type=float, default=20.0)
    parser.add_argument("--false-positive-cost", type=float, default=1.0)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    metadata = create_metadata(
        args.properties, args.target_rate, args.false_negative_cost, args.false_positive_cost,
        database_path=args.database,
    )
    output = config.MODELS_DIR / "fraud_risk_model.json"
    output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"[saved] {output}")
    if args.apply:
        from src.fraud_risk.train_actual import apply_to_database
        print(f"[applied] {apply_to_database():,} jeonse rows")


if __name__ == "__main__":
    main()
