"""시간 순서를 보존하는 집값 모델 walk-forward 비교."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from .conformal import calibrate_intervals


def _params() -> dict:
    return dict(max_iter=180, learning_rate=0.045, max_leaf_nodes=15,
                min_samples_leaf=25, l2_regularization=0.8, random_state=42)


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict:
    error = np.abs(y - prediction)
    denominator = np.maximum(np.abs(y), .001)
    return {
        "mae_monthly_log_return": float(error.mean()),
        "mape_with_0_1pct_floor": float(np.mean(error / denominator)),
        "direction_accuracy": float(np.mean(np.sign(y) == np.sign(prediction))),
        "rows": int(len(y)),
    }


def walk_forward_backtest(frame, features: list[str], *, folds: int = 6) -> dict:
    months = sorted(int(value) for value in frame["deal_ym"].unique())
    validation_months = months[-min(folds, max(1, len(months) - 6)):]
    truth, predictions = [], {"seasonal_naive": [], "ridge": [], "hist_gbdt": []}
    try:
        from lightgbm import LGBMRegressor
        lightgbm_available = True
        predictions["lightgbm"] = []
    except ImportError:
        LGBMRegressor = None
        lightgbm_available = False
    lows, highs = [], []
    fold_rows = []
    for month in validation_months:
        train = frame[frame.deal_ym < month]
        test = frame[frame.deal_ym == month]
        if len(train) < 100 or test.empty:
            continue
        X_train, y_train = train[features], train["target"].to_numpy()
        X_test, y_test = test[features], test["target"].to_numpy()
        ridge = Ridge(alpha=2.0).fit(X_train, y_train)
        gbdt = HistGradientBoostingRegressor(
            loss="squared_error", **_params()).fit(X_train, y_train)
        low_model = HistGradientBoostingRegressor(
            loss="quantile", quantile=.1, **_params()).fit(X_train, y_train)
        high_model = HistGradientBoostingRegressor(
            loss="quantile", quantile=.9, **_params()).fit(X_train, y_train)
        truth.extend(y_test.tolist())
        predictions["seasonal_naive"].extend(test["ret_1"].to_numpy().tolist())
        predictions["ridge"].extend(ridge.predict(X_test).tolist())
        predictions["hist_gbdt"].extend(gbdt.predict(X_test).tolist())
        if lightgbm_available:
            lightgbm = LGBMRegressor(
                n_estimators=180, learning_rate=.04, num_leaves=15,
                min_child_samples=25, reg_lambda=.8, random_state=42,
                verbosity=-1,
            ).fit(X_train, y_train)
            predictions["lightgbm"].extend(lightgbm.predict(X_test).tolist())
        lows.extend(low_model.predict(X_test).tolist())
        highs.extend(high_model.predict(X_test).tolist())
        fold_rows.append({"validation_month": month, "train_rows": len(train),
                          "test_rows": len(test)})
    y = np.asarray(truth, dtype=float)
    if not len(y):
        raise ValueError("walk-forward validation rows are empty")
    scores = {name: _metrics(y, np.asarray(values, dtype=float))
              for name, values in predictions.items()}
    selected = min(scores, key=lambda name: scores[name]["mae_monthly_log_return"])
    return {
        "method": "expanding_window_walk_forward",
        "folds": fold_rows, "validation_months": validation_months,
        "model_metrics": scores, "selected_base_model": selected,
        "lightgbm_available": lightgbm_available,
        "conformal": calibrate_intervals(
            y, np.asarray(lows, dtype=float), np.asarray(highs, dtype=float)),
        "tft": {
            "status": "eligible" if len(frame) >= 5000 and len(months) >= 36 else "skipped",
            "reason": (
                "global panel has enough rows and months"
                if len(frame) >= 5000 and len(months) >= 36 else
                "TFT requires at least 5,000 supervised rows and 36 observed months; "
                "a simpler validated model is safer for the current panel"
            ),
        },
    }
