from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_poisson_deviance,
    mean_squared_error,
)


DEFAULT_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def _predict(model: object, X: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="X does not have valid feature names")
        return np.asarray(model.predict(X), dtype=float)


def weighted_quantile(values: np.ndarray, quantiles: Iterable[float],
                      weights: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    quantiles = np.asarray(list(quantiles), dtype=float)
    if weights is None:
        return np.quantile(values, quantiles)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[mask], weights[mask]
    if not len(values):
        return np.full(len(quantiles), np.nan)
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= weights.sum()
    return np.interp(quantiles, cumulative, values,
                     left=values[0], right=values[-1])


def pinball_loss(y: np.ndarray, prediction: np.ndarray, q: float,
                 weights: np.ndarray | None = None) -> float:
    error = np.asarray(y) - np.asarray(prediction)
    loss = np.maximum(q * error, (q - 1.0) * error)
    return float(np.average(loss, weights=weights))


class FrameEncoder:
    """Deterministic numeric/one-hot encoder with stored training columns."""

    def __init__(self, numeric: Iterable[str], categorical: Iterable[str]):
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.columns_: list[str] = []
        self.medians_: dict[str, float] = {}

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        encoded = self._frame(frame, fitting=True)
        self.columns_ = list(encoded.columns)
        return encoded.to_numpy(dtype=float)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        encoded = self._frame(frame, fitting=False)
        encoded = encoded.reindex(columns=self.columns_, fill_value=0.0)
        return encoded.to_numpy(dtype=float)

    def _frame(self, frame: pd.DataFrame, *, fitting: bool) -> pd.DataFrame:
        numeric = pd.DataFrame(index=frame.index)
        for name in self.numeric:
            values = pd.to_numeric(
                frame[name] if name in frame else np.nan, errors="coerce")
            if not isinstance(values, pd.Series):
                values = pd.Series(values, index=frame.index)
            if fitting:
                median = float(values.median()) if values.notna().any() else 0.0
                self.medians_[name] = median
            numeric[name] = values.fillna(self.medians_.get(name, 0.0))
        categorical = pd.DataFrame(index=frame.index)
        for name in self.categorical:
            categorical[name] = (
                frame[name] if name in frame else "missing")
            categorical[name] = categorical[name].fillna("missing").astype(str)
        dummies = pd.get_dummies(
            categorical, columns=self.categorical, dtype=float)
        return pd.concat([numeric, dummies], axis=1)


class NegativeBinomialRegressor:
    """NB2 regression with softplus mean and learned dispersion."""

    def __init__(self, l2: float = 1e-4, max_iter: int = 300):
        self.l2 = l2
        self.max_iter = max_iter
        self.coef_: np.ndarray | None = None
        self.alpha_: float = 1.0
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NegativeBinomialRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.center_ = np.nanmean(X, axis=0)
        self.scale_ = np.nanstd(X, axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        Z = (np.nan_to_num(X, nan=self.center_) - self.center_) / self.scale_
        Z = np.column_stack([np.ones(len(Z)), Z])

        def objective(params: np.ndarray) -> float:
            beta, log_alpha = params[:-1], params[-1]
            alpha = np.exp(np.clip(log_alpha, -8, 6))
            mu = np.logaddexp(0.0, Z @ beta) + 1e-8
            size = 1.0 / alpha
            log_prob = (
                gammaln(y + size) - gammaln(size) - gammaln(y + 1)
                + size * np.log(size / (size + mu))
                + y * np.log(mu / (size + mu))
            )
            return float(-log_prob.sum() + self.l2 * np.square(beta[1:]).sum())

        init = np.zeros(Z.shape[1] + 1)
        init[0] = np.log(np.expm1(max(float(np.mean(y)), 0.1)))
        result = minimize(
            objective, init, method="L-BFGS-B",
            options={"maxiter": self.max_iter})
        self.coef_ = result.x[:-1]
        self.alpha_ = float(np.exp(np.clip(result.x[-1], -8, 6)))
        self.converged_ = bool(result.success)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("model is not fitted")
        X = np.asarray(X, dtype=float)
        Z = (np.nan_to_num(X, nan=self.center_) - self.center_) / self.scale_
        Z = np.column_stack([np.ones(len(Z)), Z])
        return np.logaddexp(0.0, Z @ self.coef_)

    def sample(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        mean = self.predict(X)
        size = 1.0 / self.alpha_
        probability = size / (size + mean)
        return rng.negative_binomial(size, probability).astype(int)


@dataclass
class UnitCountModel:
    encoder: FrameEncoder
    selected_name: str
    selected_model: object
    validation_metrics: dict
    residual_std: float

    @classmethod
    def fit(cls, train: pd.DataFrame, validation: pd.DataFrame,
            numeric: Iterable[str], categorical: Iterable[str],
            target: str = "registered_units_observed") -> "UnitCountModel":
        encoder = FrameEncoder(numeric, categorical)
        X_train = encoder.fit_transform(train)
        X_val = encoder.transform(validation)
        y_train = pd.to_numeric(train[target], errors="coerce").to_numpy(float)
        y_val = pd.to_numeric(validation[target], errors="coerce").to_numpy(float)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="invalid value encountered in matmul")
            poisson = PoissonRegressor(alpha=1e-4, max_iter=500).fit(
                X_train, y_train)
        models: dict[str, object] = {
            "negative_binomial": NegativeBinomialRegressor().fit(X_train, y_train),
            "poisson": poisson,
            "hist_poisson": HistGradientBoostingRegressor(
                loss="poisson", max_iter=100, random_state=28).fit(
                    X_train, y_train),
        }
        try:
            from lightgbm import LGBMRegressor
            models["lightgbm_poisson"] = LGBMRegressor(
                objective="poisson", n_estimators=160, learning_rate=.05,
                num_leaves=24, min_child_samples=20,
                random_state=28, verbosity=-1,
            ).fit(X_train, y_train)
        except ImportError:
            pass
        metrics = {}
        for name, model in models.items():
            pred = np.clip(_predict(model, X_val), 1e-6, None)
            rounded = np.rint(pred)
            metrics[name] = {
                "mae": float(mean_absolute_error(y_val, pred)),
                "rmse": float(mean_squared_error(y_val, pred) ** 0.5),
                "poisson_deviance": float(mean_poisson_deviance(y_val, pred)),
                "within_1_accuracy": float(np.mean(np.abs(rounded - y_val) <= 1)),
                "within_2_accuracy": float(np.mean(np.abs(rounded - y_val) <= 2)),
            }
        selected = min(metrics, key=lambda name: metrics[name]["mae"])
        residual = y_val - np.clip(_predict(models[selected], X_val), 0, None)
        return cls(
            encoder=encoder, selected_name=selected,
            selected_model=models[selected], validation_metrics=metrics,
            residual_std=float(np.std(residual)),
        )

    def mean(self, frame: pd.DataFrame) -> np.ndarray:
        return np.clip(_predict(
            self.selected_model, self.encoder.transform(frame)), 0.1, 100)

    def sample(self, frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        X = self.encoder.transform(frame)
        if isinstance(self.selected_model, NegativeBinomialRegressor):
            values = self.selected_model.sample(X, rng)
        else:
            values = rng.poisson(np.clip(
                _predict(self.selected_model, X), 0.1, 100))
        return np.clip(values, 1, 100).astype(int)

    def evaluate(self, frame: pd.DataFrame,
                 target: str = "registered_units_observed") -> dict:
        truth = pd.to_numeric(frame[target], errors="coerce").to_numpy(float)
        prediction = np.clip(self.mean(frame), 1e-6, None)
        rounded = np.rint(prediction)
        return {
            "mae": float(mean_absolute_error(truth, prediction)),
            "rmse": float(mean_squared_error(truth, prediction) ** 0.5),
            "poisson_deviance": float(mean_poisson_deviance(truth, prediction)),
            "within_1_accuracy": float(np.mean(np.abs(rounded - truth) <= 1)),
            "within_2_accuracy": float(np.mean(np.abs(rounded - truth) <= 2)),
        }


class QuantileModel:
    """Independent quantile estimators with monotonic post-processing."""

    def __init__(self, numeric: Iterable[str], categorical: Iterable[str],
                 quantiles: Iterable[float] = DEFAULT_QUANTILES,
                 random_state: int = 28):
        self.encoder = FrameEncoder(numeric, categorical)
        self.quantiles = np.asarray(list(quantiles), dtype=float)
        self.random_state = random_state
        self.models: dict[float, object] = {}
        self.backend_: str = "unfitted"
        self.validation_: dict = {}
        self.interval_corrections_: dict[str, float] = {
            "50": 0.0, "80": 0.0, "90": 0.0}

    def _interval_indices(
        self, nominal: float,
    ) -> tuple[int, int] | None:
        lower = (1.0 - nominal) / 2.0
        upper = 1.0 - lower
        if (float(np.min(self.quantiles)) > lower + 1e-9
                or float(np.max(self.quantiles)) < upper - 1e-9):
            return None
        lower_index = int(np.argmin(np.abs(self.quantiles - lower)))
        upper_index = int(np.argmin(np.abs(self.quantiles - upper)))
        return lower_index, upper_index

    def fit(self, train: pd.DataFrame, target: str,
            validation: pd.DataFrame | None = None) -> "QuantileModel":
        clean = train[pd.to_numeric(train[target], errors="coerce").gt(0)].copy()
        X = self.encoder.fit_transform(clean)
        y = np.log1p(pd.to_numeric(clean[target], errors="coerce").to_numpy(float))
        for q in self.quantiles:
            try:
                from lightgbm import LGBMRegressor
                model = LGBMRegressor(
                    objective="quantile", alpha=float(q),
                    n_estimators=220, learning_rate=.04, num_leaves=28,
                    min_child_samples=25, reg_lambda=.1,
                    random_state=self.random_state, verbosity=-1,
                )
                self.backend_ = "lightgbm_quantile"
            except ImportError:
                model = HistGradientBoostingRegressor(
                    loss="quantile", quantile=float(q), max_iter=120,
                    learning_rate=0.06, l2_regularization=0.1,
                    random_state=self.random_state,
                )
                self.backend_ = "hist_gradient_boosting_quantile_fallback"
            self.models[float(q)] = model.fit(X, y)
        if validation is not None and len(validation):
            target_values = pd.to_numeric(validation[target], errors="coerce")
            mask = target_values.gt(0)
            truth = target_values[mask].to_numpy(float)
            before = self._raw_quantiles(validation.loc[mask])
            raw_log = np.sort(
                self._raw_log_quantiles(validation.loc[mask]), axis=1)
            truth_log = np.log1p(truth)
            intervals = (("50", .5), ("80", .8), ("90", .9))
            for label, nominal in intervals:
                indices = self._interval_indices(nominal)
                if indices is None:
                    continue
                lower_index, upper_index = indices
                score = np.maximum(
                    raw_log[:, lower_index] - truth_log,
                    truth_log - raw_log[:, upper_index])
                # Split-conformal finite-sample quantile with "higher"
                # interpolation. Negative corrections are unnecessary.
                level = min(
                    1.0,
                    np.ceil((len(score) + 1) * nominal) / len(score))
                self.interval_corrections_[label] = max(
                    0.0, float(np.quantile(score, level, method="higher")))
            predicted = self.predict_quantiles(validation.loc[mask])
            self.validation_ = {
                "pinball": {
                    str(q): pinball_loss(
                        np.log1p(truth), np.log1p(predicted[:, i]), float(q))
                    for i, q in enumerate(self.quantiles)
                },
                "crossing_rate_before": float(
                    np.mean(np.any(np.diff(before, axis=1) < 0, axis=1))),
                "crossing_rate_after": float(
                    np.mean(np.any(np.diff(predicted, axis=1) < 0, axis=1))),
                "conformal_log_corrections": dict(
                    self.interval_corrections_),
                "backend": self.backend_,
                "median_absolute_error": float(
                    np.median(np.abs(
                        truth - predicted[:, int(np.argmin(
                            np.abs(self.quantiles - .5)))]))),
            }
            sorted_before = np.sort(before, axis=1)
            for label, nominal in intervals:
                indices = self._interval_indices(nominal)
                if indices is None:
                    continue
                lower_index, upper_index = indices
                self.validation_[f"coverage_{label}"] = float(np.mean(
                    (truth >= predicted[:, lower_index])
                    & (truth <= predicted[:, upper_index])))
                self.validation_[
                    f"coverage_{label}_before_conformal"
                ] = float(np.mean(
                    (truth >= sorted_before[:, lower_index])
                    & (truth <= sorted_before[:, upper_index])))
        return self

    def _raw_log_quantiles(self, frame: pd.DataFrame) -> np.ndarray:
        X = self.encoder.transform(frame)
        return np.column_stack([
            _predict(self.models[float(q)], X)
            for q in self.quantiles
        ])

    def _raw_quantiles(self, frame: pd.DataFrame) -> np.ndarray:
        return np.expm1(self._raw_log_quantiles(frame)).clip(min=0)

    def predict_quantiles(self, frame: pd.DataFrame) -> np.ndarray:
        values = np.sort(self._raw_log_quantiles(frame), axis=1)
        corrections = getattr(self, "interval_corrections_", {})
        for label, nominal in (("90", .9), ("80", .8), ("50", .5)):
            indices = self._interval_indices(nominal)
            if indices is None:
                continue
            lower_index, upper_index = indices
            correction = corrections.get(label, 0.0)
            values[:, lower_index] -= correction
            values[:, upper_index] += correction
        return np.sort(np.expm1(values).clip(min=0), axis=1)

    def sample(self, frame: pd.DataFrame, rng: np.random.Generator,
               size: int | None = None) -> np.ndarray:
        quantile_values = self.predict_quantiles(frame)
        if size is None:
            size = len(frame)
        if len(frame) == 1 and size != 1:
            quantile_values = np.repeat(quantile_values, size, axis=0)
        elif len(frame) != size:
            raise ValueError("frame length must be one or equal to size")
        u = rng.uniform(0.0, 1.0, size)
        # Piecewise-linear inverse CDF.  The tails extend flat rather than
        # imposing a normal/log-normal distribution.
        sampled = np.empty(size, dtype=float)
        for i in range(size):
            sampled[i] = np.interp(
                u[i], self.quantiles, quantile_values[i],
                left=quantile_values[i, 0], right=quantile_values[i, -1])
        return np.clip(sampled, 0, None)

    def evaluate(self, frame: pd.DataFrame, target: str) -> dict:
        values = pd.to_numeric(frame[target], errors="coerce")
        mask = values.gt(0)
        truth = values[mask].to_numpy(float)
        predicted = self.predict_quantiles(frame.loc[mask])
        if not len(truth):
            return {"rows": 0}
        median_index = int(np.argmin(np.abs(self.quantiles - .5)))
        result = {
            "rows": len(truth),
            "pinball": {
                str(q): pinball_loss(
                    np.log1p(truth), np.log1p(predicted[:, i]), float(q))
                for i, q in enumerate(self.quantiles)
            },
            "median_absolute_error": float(
                np.median(np.abs(truth - predicted[:, median_index]))),
            "median_ape": float(np.median(
                np.abs(truth - predicted[:, median_index])
                / np.maximum(truth, 1e-6))),
        }
        for label, nominal in (("50", .5), ("80", .8), ("90", .9)):
            indices = self._interval_indices(nominal)
            if indices is None:
                continue
            lower_index, upper_index = indices
            result[f"coverage_{label}"] = float(np.mean(
                (truth >= predicted[:, lower_index])
                & (truth <= predicted[:, upper_index])))
        return result


class OwnerAssetPrior:
    """Survey-weighted empirical K prior with hierarchical fallback."""

    def __init__(self, min_group_size: int = 30):
        self.min_group_size = min_group_size
        self.k_scale = 1.0
        self.rows_: pd.DataFrame | None = None
        self.group_counts_: dict[str, int] = {}
        self.rental_asset_boundaries_: np.ndarray | None = None

    def fit(self, survey: pd.DataFrame) -> "OwnerAssetPrior":
        required = {
            "K_other", "R_survey", "L_debt", "survey_weight",
            "rental_real_estate_assets", "total_assets", "capital_region",
        }
        missing = required - set(survey.columns)
        if missing:
            raise ValueError(f"owner prior missing columns: {sorted(missing)}")
        rows = survey.copy()
        if rows["K_other"].lt(0).any():
            raise ValueError("negative K_other is not clipped; check schema mapping")
        rows["rental_asset_band"] = pd.qcut(
            rows["rental_real_estate_assets"], 5, labels=False,
            duplicates="drop").fillna(0).astype(int)
        rows["total_asset_band"] = pd.qcut(
            rows["total_assets"], 5, labels=False,
            duplicates="drop").fillna(0).astype(int)
        rows["deposit_band"] = pd.qcut(
            rows.get("rental_deposit_liability", rows["R_survey"]),
            5, labels=False, duplicates="drop").fillna(0).astype(int)
        if "home_count" not in rows:
            rows["home_count"] = np.nan
        self.rows_ = rows.reset_index(drop=True)
        self.rental_asset_boundaries_ = np.quantile(
            rows["rental_real_estate_assets"], [0.2, 0.4, 0.6, 0.8])
        self.group_counts_ = {
            "capital_region": int(rows["capital_region"].sum()),
            "national": int(len(rows)),
        }
        return self

    def _candidates(self, building_value: float,
                    capital_region: bool = True) -> tuple[pd.DataFrame, str]:
        if self.rows_ is None:
            raise RuntimeError("owner prior is not fitted")
        rows = self.rows_
        boundaries = self.rental_asset_boundaries_
        if boundaries is None:
            boundaries = np.quantile(
                rows["rental_real_estate_assets"], [0.2, 0.4, 0.6, 0.8])
        band = int(np.searchsorted(boundaries, building_value, side="right"))
        return self._candidates_for_band(band, capital_region)

    def _candidates_for_band(
        self,
        band: int,
        capital_region: bool = True,
    ) -> tuple[pd.DataFrame, str]:
        if self.rows_ is None:
            raise RuntimeError("owner prior is not fitted")
        rows = self.rows_
        exact = rows[
            (rows["capital_region"] == capital_region)
            & (rows["rental_asset_band"] == band)]
        if len(exact) >= self.min_group_size:
            return exact, "exact_group"
        relaxed = rows[rows["rental_asset_band"].between(
            max(0, band - 1), min(4, band + 1))]
        relaxed = relaxed[relaxed["capital_region"] == capital_region]
        if len(relaxed) >= self.min_group_size:
            return relaxed, "relaxed_asset_band"
        metro = rows[rows["capital_region"]]
        if capital_region and len(metro) >= self.min_group_size:
            return metro, "capital_region_landlords"
        return rows, "national_landlords"

    def sample(self, building_value: np.ndarray,
               rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, list[str]]:
        values = np.asarray(building_value, dtype=float)
        k = np.empty(len(values))
        debt = np.empty(len(values))
        fallback_values = np.empty(len(values), dtype=object)
        boundaries = self.rental_asset_boundaries_
        if boundaries is None:
            if self.rows_ is None:
                raise RuntimeError("owner prior is not fitted")
            boundaries = np.quantile(
                self.rows_["rental_real_estate_assets"],
                [0.2, 0.4, 0.6, 0.8],
            )
        bands = np.searchsorted(boundaries, values, side="right")
        # Candidate rows and weights are identical within a value band. Batch
        # sampling avoids rebuilding the same weighted distribution once per
        # Monte Carlo draw.
        for band in np.unique(bands):
            positions = np.flatnonzero(bands == band)
            candidates, fallback = self._candidates_for_band(
                int(band), True)
            weights = candidates["survey_weight"].to_numpy(float)
            weights = weights / weights.sum()
            chosen = rng.choice(
                len(candidates), size=len(positions), p=weights)
            sampled = candidates.iloc[chosen]
            k[positions] = np.maximum(
                0.0,
                sampled["K_other"].to_numpy(float) * self.k_scale,
            )
            debt[positions] = np.maximum(
                0.0, sampled["L_debt"].to_numpy(float))
            fallback_values[positions] = fallback
        return k, debt, fallback_values.tolist()

    def weighted_quantiles(self) -> dict:
        if self.rows_ is None:
            raise RuntimeError("owner prior is not fitted")
        weights = self.rows_["survey_weight"].to_numpy(float)
        return {
            name: dict(zip(
                ("p10", "p50", "p90"),
                weighted_quantile(
                    self.rows_[name].to_numpy(float), (0.1, 0.5, 0.9), weights),
            ))
            for name in ("K_other", "R_survey", "L_debt")
        }
