from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .calibration import choose_k_scale, population_quantile_loss
from .models import OwnerAssetPrior, QuantileModel, UnitCountModel
from .schemas import BuildingEstimateInput
from .simulation import infer_ratio_distribution


UNIT_NUMERIC = (
    "total_floor_area", "land_area", "ground_floors",
    "parking_count", "building_age",
)
UNIT_CATEGORICAL = ("structure_code", "legal_dong_code", "main_use_code")
DEPOSIT_NUMERIC = (
    "rental_area", "building_age", "monthly_rent",
    "legal_dong_3m_deposit_median", "legal_dong_12m_deposit_median",
    "legal_dong_12m_deposit_growth", "transaction_count_3m",
    "transaction_count_12m",
)
DEPOSIT_CATEGORICAL = ("legal_dong", "contract_type", "housing_type")
VALUE_NUMERIC = (
    "land_area", "total_floor_area", "residential_floor_area", "building_age",
    "ground_floors", "parking_count", "official_house_price",
    "official_land_price", "nearby_sale_price_per_land_area",
    "nearby_sale_price_per_floor_area",
)
VALUE_CATEGORICAL = ("structure_code", "main_use_code", "legal_dong")


def _deposit_baselines(train: pd.DataFrame,
                       validation: pd.DataFrame) -> dict:
    grouped = train.groupby(
        ["legal_dong", "contract_type"], dropna=False)["deposit"].median()
    global_median = float(pd.to_numeric(
        train["deposit"], errors="coerce").median())
    prediction = np.array([
        grouped.get((row.legal_dong, row.contract_type), global_median)
        for row in validation[["legal_dong", "contract_type"]].itertuples(
            index=False)
    ], dtype=float)
    truth = pd.to_numeric(
        validation["deposit"], errors="coerce").to_numpy(float)
    return {
        "legal_dong_contract_median": {
            "median_absolute_error": float(
                np.nanmedian(np.abs(truth - prediction))),
            "median_ape": float(np.nanmedian(
                np.abs(truth - prediction) / np.maximum(truth, 1e-6))),
        }
    }


def _value_baselines(train: pd.DataFrame,
                     validation: pd.DataFrame) -> dict:
    truth = pd.to_numeric(
        validation["sale_price"], errors="coerce").to_numpy(float)
    official_train = pd.to_numeric(
        train["official_house_price"]
        if "official_house_price" in train
        else pd.Series(np.nan, index=train.index),
        errors="coerce")
    sale_train = pd.to_numeric(train["sale_price"], errors="coerce")
    valid = official_train.gt(0) & sale_train.gt(0)
    multiplier = float(np.median(
        sale_train[valid] / official_train[valid])) if valid.any() else np.nan
    official_validation = pd.to_numeric(
        validation["official_house_price"]
        if "official_house_price" in validation
        else pd.Series(np.nan, index=validation.index),
        errors="coerce").to_numpy(float)
    official_prediction = official_validation * multiplier

    land_train = pd.to_numeric(train["land_area"], errors="coerce")
    per_land = sale_train / land_train.where(land_train.gt(0))
    dong_rate = (
        train.assign(_rate=per_land)
        .groupby("legal_dong")["_rate"].median())
    fallback_rate = float(per_land.median())
    land_validation = pd.to_numeric(
        validation["land_area"], errors="coerce").to_numpy(float)
    comparable_prediction = np.array([
        dong_rate.get(dong, fallback_rate) * area
        for dong, area in zip(validation["legal_dong"], land_validation)
    ])

    def metrics(prediction):
        mask = np.isfinite(truth) & np.isfinite(prediction) & (truth > 0)
        return {
            "rows": int(mask.sum()),
            "median_absolute_error": float(np.median(
                np.abs(truth[mask] - prediction[mask]))),
            "median_ape": float(np.median(
                np.abs(truth[mask] - prediction[mask]) / truth[mask])),
        } if mask.any() else {"rows": 0}

    return {
        "official_price_multiplier": {
            "multiplier": multiplier, **metrics(official_prediction)},
        "legal_dong_land_area_median": metrics(comparable_prediction),
    }


def _temporal_split(frame: pd.DataFrame, month_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    month = pd.to_numeric(frame[month_col], errors="coerce")
    train = frame[month <= 202412]
    validation = frame[(month >= 202501) & (month <= 202512)]
    test = frame[month >= 202601]
    if min(len(train), len(validation), len(test)) < 30:
        ordered = frame.assign(_month=month).sort_values("_month")
        first, second = int(len(ordered) * .7), int(len(ordered) * .85)
        train = ordered.iloc[:first].drop(columns="_month")
        validation = ordered.iloc[first:second].drop(columns="_month")
        test = ordered.iloc[second:].drop(columns="_month")
    return train.copy(), validation.copy(), test.copy()


def _temporal_spatial_split(
    frame: pd.DataFrame,
    month_col: str,
    group_col: str,
    *,
    spatial_fraction: float = .2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    groups = sorted(
        str(value) for value in frame[group_col].dropna().unique())
    holdout_count = max(1, int(round(len(groups) * spatial_fraction)))
    spatial_groups = groups[-holdout_count:] if len(groups) > 1 else []
    spatial_mask = frame[group_col].astype(str).isin(spatial_groups)
    development = frame[~spatial_mask].copy()
    train, validation, temporal_test = _temporal_split(
        development, month_col)
    test = pd.concat(
        [temporal_test, frame[spatial_mask]], ignore_index=True)
    return train, validation, test, spatial_groups


class OwnerAssetRatioPipeline:
    model_version = "four_component_owner_asset_ratio_v1"

    def __init__(self, unit_model: UnitCountModel,
                 deposit_model: QuantileModel,
                 value_model: QuantileModel,
                 owner_prior: OwnerAssetPrior,
                 metadata: dict):
        self.unit_model = unit_model
        self.deposit_model = deposit_model
        self.value_model = value_model
        self.owner_prior = owner_prior
        self.metadata = metadata

    @classmethod
    def fit(cls, buildings: pd.DataFrame, leases: pd.DataFrame,
            sales: pd.DataFrame, survey: pd.DataFrame, *,
            data_kind: str, seed: int = 20260728) -> "OwnerAssetRatioPipeline":
        if data_kind not in {"actual", "synthetic_smoke_only"}:
            raise ValueError("data_kind must explicitly be actual or synthetic_smoke_only")

        known = buildings[
            pd.to_numeric(
                buildings["registered_units_observed"], errors="coerce")
            .between(1, 100)
        ].copy()
        building_month = (
            "snapshot_month" if "snapshot_month" in known else None)
        if building_month:
            b_train, b_val, b_test, b_holdout = _temporal_spatial_split(
                known, building_month, "legal_dong_code")
        else:
            groups = sorted(
                str(value) for value in
                known["legal_dong_code"].dropna().unique())
            group_count = max(1, int(round(len(groups) * .2)))
            if len(groups) >= 3:
                test_groups = groups[-group_count:]
                val_groups = groups[-2 * group_count:-group_count]
                train_groups = [
                    value for value in groups
                    if value not in set(test_groups + val_groups)]
                b_train = known[
                    known["legal_dong_code"].astype(str).isin(train_groups)]
                b_val = known[
                    known["legal_dong_code"].astype(str).isin(val_groups)]
                b_test = known[
                    known["legal_dong_code"].astype(str).isin(test_groups)]
                b_holdout = test_groups
            else:
                # One-snapshot registries have no observation-time axis.
                # Approval date is used only as a deterministic chronological
                # fallback; random row splitting is intentionally avoided.
                ordered = known.assign(
                    _approval=pd.to_datetime(
                        known.get("approval_date"), errors="coerce")
                ).sort_values(["_approval", "building_id"])
                first, second = (
                    int(len(ordered) * .7), int(len(ordered) * .85))
                b_train = ordered.iloc[:first].drop(columns="_approval")
                b_val = ordered.iloc[first:second].drop(columns="_approval")
                b_test = ordered.iloc[second:].drop(columns="_approval")
                b_holdout = []
        unit_model = UnitCountModel.fit(
            b_train, b_val, UNIT_NUMERIC, UNIT_CATEGORICAL)

        l_train, l_val, l_test, l_holdout = _temporal_spatial_split(
            leases, "contract_year_month", "legal_dong")
        deposit_model = QuantileModel(
            DEPOSIT_NUMERIC, DEPOSIT_CATEGORICAL,
            random_state=seed).fit(l_train, "deposit", l_val)

        direct_sales = sales[
            sales["match_confidence"].isin(["exact", "high"])].copy()
        if len(direct_sales) < 100:
            raise ValueError(
                "at least 100 exact/high-confidence sales are required; "
                "medium/low partial-lot matches are not used as labels")
        s_train, s_val, s_test, s_holdout = _temporal_spatial_split(
            direct_sales, "contract_year_month", "legal_dong")
        value_model = QuantileModel(
            VALUE_NUMERIC, VALUE_CATEGORICAL,
            random_state=seed).fit(s_train, "sale_price", s_val)
        survey_split_years: dict[str, object] = {}
        if "survey_year" in survey:
            survey_year = pd.to_numeric(
                survey["survey_year"], errors="coerce")
            available_years = sorted(
                int(value) for value in survey_year.dropna().unique())
            if len(available_years) < 3:
                raise ValueError(
                    "survey requires at least three distinct years for a "
                    "year-preserving train/validation/test split")
            validation_year = available_years[-2]
            test_year = available_years[-1]
            survey_train = survey[survey_year < validation_year].copy()
            survey_validation = survey[
                survey_year == validation_year].copy()
            survey_test = survey[survey_year == test_year].copy()
            survey_split_years = {
                "train": available_years[:-2],
                "validation": validation_year,
                "test": test_year,
            }
        else:
            ordered_survey = survey.sample(frac=1, random_state=seed)
            first, second = (
                int(len(ordered_survey) * .7),
                int(len(ordered_survey) * .85),
            )
            survey_train = ordered_survey.iloc[:first].copy()
            survey_validation = ordered_survey.iloc[first:second].copy()
            survey_test = ordered_survey.iloc[second:].copy()
        if min(len(survey_train), len(survey_validation), len(survey_test)) < 30:
            raise ValueError(
                "survey must support year-preserving train/validation/test "
                "splits with at least 30 landlord rows per split")
        owner_prior = OwnerAssetPrior(min_group_size=30).fit(survey_train)
        population_calibration: dict[str, object] | None = None
        population_test_loss: float | None = None
        if data_kind == "actual":
            calibration_rng = np.random.default_rng(seed + 701)
            validation_property_value = survey_validation[
                "rental_real_estate_assets"].to_numpy(float)
            validation_k, _, _ = owner_prior.sample(
                validation_property_value, calibration_rng)
            population_calibration = choose_k_scale(
                survey_validation["rental_deposit_liability"].to_numpy(float),
                validation_property_value,
                validation_k,
                survey_validation["R_survey"].to_numpy(float),
                survey_validation["survey_weight"].to_numpy(float),
            )
            owner_prior.k_scale = float(population_calibration["k_scale"])

            test_rng = np.random.default_rng(seed + 702)
            test_property_value = survey_test[
                "rental_real_estate_assets"].to_numpy(float)
            test_k, _, _ = owner_prior.sample(
                test_property_value, test_rng)
            predicted_test_ratio = (
                survey_test["rental_deposit_liability"].to_numpy(float)
                / np.maximum(
                    test_property_value
                    * (1.0 + owner_prior.k_scale * test_k),
                    1e-6,
                )
            )
            population_test_loss = population_quantile_loss(
                predicted_test_ratio,
                survey_test["R_survey"].to_numpy(float),
                survey_test["survey_weight"].to_numpy(float),
            )

        metadata = {
            "model_version": cls.model_version,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "data_kind": data_kind,
            "seed": seed,
            "occupancy": {
                "low": {"alpha": 7.0, "beta": 3.0},
                "baseline": {"alpha": 17.0, "beta": 3.0},
                "high": {"alpha": 19.0, "beta": 1.0},
            },
            "building_random_effect_sigma": 0.12,
            "splits": {
                "building": [len(b_train), len(b_val), len(b_test)],
                "lease": [len(l_train), len(l_val), len(l_test)],
                "sale": [len(s_train), len(s_val), len(s_test)],
                "survey": [
                    len(survey_train), len(survey_validation), len(survey_test)],
                "survey_years": survey_split_years,
                "spatial_holdout": {
                    "building_legal_dong_codes": b_holdout,
                    "lease_legal_dongs": l_holdout,
                    "sale_legal_dongs": s_holdout,
                },
            },
            "validation": {
                "unit_count": unit_model.validation_metrics,
                "deposit": deposit_model.validation_,
                "property_value": value_model.validation_,
                "owner_prior_weighted_quantiles": owner_prior.weighted_quantiles(),
                "survey_validation_rows": len(survey_validation),
                "owner_prior_population_calibration": population_calibration,
                "baselines": {
                    "deposit": _deposit_baselines(l_train, l_val),
                    "property_value": _value_baselines(s_train, s_val),
                },
            },
            "test": {
                "unit_count": unit_model.evaluate(b_test),
                "deposit": deposit_model.evaluate(l_test, "deposit"),
                "property_value": value_model.evaluate(
                    s_test, "sale_price"),
                "individual_owner_assets": (
                    "not_evaluated_no_individual_label"),
                "survey_test_rows": len(survey_test),
                "owner_prior_population_quantile_loss": population_test_loss,
            },
            "provenance": {
                "building_registry": data_kind,
                "lease_rtms": data_kind,
                "sale_rtms": data_kind,
                "household_survey": data_kind,
                "direct_owner_wealth_label": False,
                "direct_building_survey_join": False,
            },
            "limitations": [
                "individual owner wealth is unobserved",
                "K is a survey-weighted conditional population prior",
                "partial RTMS lots are not exact building matches",
                "net-asset ratio is auxiliary and not the primary risk score",
            ],
        }
        return cls(unit_model, deposit_model, value_model, owner_prior, metadata)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        metadata_path = path.with_suffix(".metadata.json")
        import json
        metadata_path.write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2),
            encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path, *,
             allow_synthetic: bool = False) -> "OwnerAssetRatioPipeline":
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError("artifact is not an OwnerAssetRatioPipeline")
        if (model.metadata.get("data_kind") != "actual"
                and not allow_synthetic):
            raise RuntimeError(
                "synthetic smoke artifact cannot be used for real inference; "
                "pass allow_synthetic only in tests/examples")
        return model

    def infer(self, building: BuildingEstimateInput | dict, **kwargs) -> dict:
        return infer_ratio_distribution(self, building, **kwargs)

    def evaluation_summary(self) -> dict:
        return {
            "data_kind": self.metadata["data_kind"],
            "individual_owner_mae_reported": False,
            "reason": "individual owner total-assets labels do not exist",
            "splits": self.metadata.get("splits"),
            "validation": self.metadata["validation"],
            "test": self.metadata.get("test"),
            "population_calibration": self.metadata["validation"].get(
                "owner_prior_population_calibration"),
            "required_population_calibration": (
                self.metadata["data_kind"] == "actual"
                and not self.metadata["validation"].get(
                    "owner_prior_population_calibration")),
        }
