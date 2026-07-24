"""Runtime scoring for calibrated guarantee-accident probability models."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Union

import joblib
import numpy as np
import pandas as pd

from src import config
from src.fraud_risk import features as legacy_features
from src.fraud_risk.actual_model import (
    PublishedHFModel,
    build_actual_feature_frame,
    cost_ratio_threshold,
)
from src.schemas import PropertyRecord

MODEL_PATH = config.MODELS_DIR / "fraud_risk_model.joblib"
METADATA_PATH = config.MODELS_DIR / "fraud_risk_model.json"
DEFAULT_DECISION_THRESHOLD = cost_ratio_threshold(20.0, 1.0)


@lru_cache(maxsize=4)
def _read_metadata(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def get_decision_threshold(metadata_path: Path = METADATA_PATH) -> float:
    metadata = _read_metadata(str(metadata_path.resolve()))
    return float(metadata.get("cost_policy", {}).get("decision_threshold", DEFAULT_DECISION_THRESHOLD))


def _grade(score: float, threshold: float) -> str:
    if score >= threshold:
        return "위험"
    if score >= threshold / 2:
        return "주의"
    return "낮음"


def rule_based_score(row: pd.Series) -> float:
    """Emergency fallback only; not presented as a calibrated accident probability."""
    debt_ratio = float(row["debt_ratio"])
    return float(1.0 / (1.0 + np.exp(-6.0 * (debt_ratio - 0.95))))


class FraudRiskScorer:
    def __init__(self, model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.metadata = _read_metadata(str(self.metadata_path.resolve()))
        self.bundle = None
        self.kind = self.metadata.get("kind")
        self.decision_threshold = float(
            self.metadata.get("cost_policy", {}).get(
                "decision_threshold", DEFAULT_DECISION_THRESHOLD
            )
        )
        if self.kind == "sklearn_actual_contract_labels":
            self.bundle = joblib.load(self.model_path)
            self.decision_threshold = float(
                self.bundle.get("decision_threshold", self.decision_threshold)
            )
        elif self.kind == "published_hf_actual_label_transfer":
            self.published_model = PublishedHFModel(
                prior_logit_shift=float(self.metadata["prior_calibration"]["logit_shift"]),
                decision_threshold=self.decision_threshold,
            )
        elif self.model_path.exists():
            # Backward compatibility with the original synthetic-label bundle.
            self.bundle = joblib.load(self.model_path)
            self.kind = "legacy_synthetic_or_unspecified"

    @staticmethod
    def _to_frame(prop: Union[dict, PropertyRecord]) -> pd.DataFrame:
        if isinstance(prop, PropertyRecord):
            prop = prop.model_dump()
        return pd.DataFrame([prop])

    def _predict(self, df: pd.DataFrame) -> tuple[np.ndarray, str]:
        if self.kind == "published_hf_actual_label_transfer":
            return self.published_model.predict_proba(df), "hf_actual_labels/published_logit+prior_calibration"
        if self.kind == "sklearn_actual_contract_labels":
            X = build_actual_feature_frame(df)
            raw = self.bundle["model"].predict_proba(X)[:, 1]
            return self.bundle["calibrator"].transform(raw), (
                f"actual_contract_labels/logistic+{self.bundle['calibration_method']}"
            )
        if self.bundle is not None:
            engineered = legacy_features.engineer(df)
            columns = self.bundle["feature_names"]
            probability = self.bundle["model"].predict_proba(
                engineered[columns].astype(float).to_numpy()
            )[:, 1]
            return probability, (
                f"legacy:{self.bundle.get('model_name', 'unknown')}/"
                f"{self.bundle.get('feature_set', 'unknown')}"
            )
        engineered = legacy_features.engineer(df)
        return engineered.apply(rule_based_score, axis=1).to_numpy(), "emergency_rule/not_calibrated"

    def score(self, prop: Union[dict, PropertyRecord]) -> dict:
        df = self._to_frame(prop)
        probability, method = self._predict(df)
        score = float(probability[0])
        legacy = legacy_features.engineer(df).iloc[0]
        return {
            "fraud_score": round(score, 6),
            "grade": _grade(score, self.decision_threshold),
            "decision_threshold": round(self.decision_threshold, 6),
            "debt_ratio": round(float(legacy["debt_ratio"]), 3),
            "senior_ratio": round(float(legacy["senior_ratio"]), 3),
            "recovery_cushion": round(float(legacy["recovery_cushion"]), 3),
            "method": method,
            "label_source_status": self.metadata.get("label_source_status", "unverified"),
        }

    def score_batch(self, df: pd.DataFrame) -> np.ndarray:
        probability, _ = self._predict(df)
        return np.asarray(probability, dtype=float)

