from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import pandas as pd

from src.owner_asset_ratio.models import QuantileModel
from src.owner_asset_ratio.pipeline import (
    DEPOSIT_CATEGORICAL,
    DEPOSIT_NUMERIC,
    OwnerAssetRatioPipeline,
    _temporal_spatial_split,
)
from .simulation import infer_senior_deposit_distribution


SENIOR_DEPOSIT_QUANTILES = (.05, .1, .25, .5, .75, .9, .95)


class SeniorDepositPipeline:
    model_version = "senior_deposit_mvp_v1"

    def __init__(
        self, unit_model, deposit_model, metadata: dict,
        market_reference: pd.DataFrame | None = None,
    ):
        self.unit_model = unit_model
        self.deposit_model = deposit_model
        self.metadata = metadata
        self.market_reference = market_reference
        self.seniority_model = None

    @classmethod
    def fit_from_actual_data(
        cls,
        *,
        owner_pipeline: OwnerAssetRatioPipeline,
        leases: pd.DataFrame,
        seed: int = 20260728,
    ) -> "SeniorDepositPipeline":
        if owner_pipeline.metadata.get("data_kind") != "actual":
            raise ValueError("an actual owner pipeline is required")
        train, validation, test, spatial_holdout = _temporal_spatial_split(
            leases, "contract_year_month", "legal_dong")
        deposit_model = QuantileModel(
            DEPOSIT_NUMERIC,
            DEPOSIT_CATEGORICAL,
            quantiles=SENIOR_DEPOSIT_QUANTILES,
            random_state=seed,
        ).fit(train, "deposit", validation)
        metadata = {
            "model_version": cls.model_version,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "data_kind": "actual",
            "model_mode": "scenario_only",
            "seed": seed,
            "occupancy_priors": {
                "low": {"alpha": 7.0, "beta": 3.0},
                "baseline": {"alpha": 18.0, "beta": 2.0},
                "high": {"alpha": 38.0, "beta": 2.0},
            },
            "baseline_senior_probability": .9,
            "within_building_sigma": .12,
            "occupancy_model_trained": False,
            "seniority_model_trained": False,
            "final_calibrator_trained": False,
            "label_counts": {
                "occupancy": 0,
                "seniority": 0,
                "building_total": 0,
            },
            "splits": {
                "lease": [len(train), len(validation), len(test)],
                "lease_spatial_holdout": spatial_holdout,
                "unit_model": owner_pipeline.metadata.get(
                    "splits", {}).get("building"),
            },
            "validation": {
                "deposit": deposit_model.validation_,
                "occupancy": "not_evaluated_no_actual_labels",
                "seniority": "not_evaluated_no_actual_labels",
                "final_calibrator": "not_evaluated_no_actual_labels",
            },
            "test": {
                "deposit": deposit_model.evaluate(test, "deposit"),
                "occupancy": "not_evaluated_no_actual_labels",
                "seniority": "not_evaluated_no_actual_labels",
                "final_senior_deposit": (
                    "not_evaluated_no_building_level_actual_labels"),
            },
            "provenance": {
                "building_registry": "actual",
                "lease_rtms": "actual",
                "occupancy_labels": "unavailable",
                "seniority_labels": "unavailable",
                "building_total_labels": "unavailable",
                "partial_rtms_lot_exact_join": False,
            },
            "limitations": [
                "current occupancy is a declared Beta-Binomial scenario prior",
                "legal seniority is not observed and is not classified",
                "conservative p90/p95 is the representative safety output",
                "the result does not replace official tenant and fixed-date records",
            ],
            "market_reference_rows": len(leases),
            "market_reference_max_month": int(pd.to_numeric(
                leases["contract_year_month"], errors="coerce").max()),
        }
        market_columns = [
            "contract_year_month",
            "legal_dong",
            "rental_area",
            "legal_dong_3m_deposit_median",
            "legal_dong_12m_deposit_median",
            "legal_dong_12m_deposit_growth",
            "transaction_count_3m",
            "transaction_count_12m",
        ]
        market_reference = leases[
            [column for column in market_columns if column in leases]
        ].copy()
        return cls(
            owner_pipeline.unit_model,
            deposit_model,
            metadata,
            market_reference=market_reference,
        )

    def predict_seniority_probability(self, frame, reference_date) -> float:
        if self.seniority_model is None:
            raise RuntimeError("seniority model is not trained")
        prediction = self.seniority_model.predict_proba(frame)[:, 1]
        return float(prediction.mean())

    def infer(self, building, **kwargs) -> dict:
        return infer_senior_deposit_distribution(self, building, **kwargs)

    def evaluation_summary(self) -> dict:
        return {
            "data_kind": self.metadata.get("data_kind"),
            "model_mode": self.metadata.get("model_mode"),
            "splits": self.metadata.get("splits"),
            "validation": self.metadata.get("validation"),
            "test": self.metadata.get("test"),
            "label_counts": self.metadata.get("label_counts"),
            "legal_seniority_metrics_reported": False,
            "reason": "no verified move-in/fixed-date seniority labels",
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        path.with_suffix(".metadata.json").write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(
        cls, path: str | Path, *, allow_synthetic: bool = False,
    ) -> "SeniorDepositPipeline":
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError("artifact is not a SeniorDepositPipeline")
        if (model.metadata.get("data_kind") != "actual"
                and not allow_synthetic):
            raise RuntimeError("synthetic senior-deposit artifact is not allowed")
        return model
