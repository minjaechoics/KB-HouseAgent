"""국토교통부 실거래 24~25개월로 지역별 월간 가격수익률 모델을 학습한다."""
from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge

from src import config
from src.market_forecast.backtest import walk_forward_backtest

FEATURES = [
    "ret_1", "ret_3", "ret_6", "vol_3", "log_count", "national_ret_1",
    "month_sin", "month_cos", "group_code",
]
SOURCES = [
    ("rtms_apt_trade.csv", "apartment", "excluUseAr"),
    ("rtms_offi_trade.csv", "officetel", "excluUseAr"),
    ("rtms_sh_trade.csv", "single_multi", "totalFloorAr"),
]


class SeasonalNaiveRegressor:
    """Persistable no-training baseline using the latest one-month return."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return pd.to_numeric(X["ret_1"], errors="coerce").fillna(0).to_numpy()


def _load_source(path: Path, group: str, area_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    price = pd.to_numeric(
        frame["dealAmount"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    area = pd.to_numeric(frame[area_column], errors="coerce")
    result = pd.DataFrame({
        "lawd_cd": frame["lawd_cd"].astype(str).str.zfill(5).str[:5],
        "deal_ym": pd.to_numeric(frame["deal_ym"], errors="coerce"),
        "price_per_m2": price / area,
        "group": group,
    }).replace([np.inf, -np.inf], np.nan).dropna()
    return result[(result["price_per_m2"] > 10) & (result["price_per_m2"] < 100000)]


def build_feature_panel(raw_dir: Path | None = None) -> pd.DataFrame:
    raw_dir = raw_dir or config.DATA_RAW / "real_estate"
    transactions = pd.concat([
        _load_source(raw_dir / filename, group, area)
        for filename, group, area in SOURCES
    ], ignore_index=True)
    monthly = transactions.groupby(
        ["group", "lawd_cd", "deal_ym"], as_index=False
    ).agg(median_price_m2=("price_per_m2", "median"), count=("price_per_m2", "size"))
    monthly["date"] = pd.to_datetime(
        monthly["deal_ym"].astype(int).astype(str), format="%Y%m")
    # 거래가 없는 달을 건너뛴 diff가 1개월 수익률로 오인되지 않도록 각 지역을
    # 실제 달력 월로 재색인한다. 가격은 채워 넣지 않으며 관측월만 학습에 쓴다.
    completed = []
    for (group, lawd_cd), part in monthly.groupby(["group", "lawd_cd"]):
        part = part.set_index("date").sort_index()
        calendar = pd.date_range(part.index.min(), part.index.max(), freq="MS")
        part = part.reindex(calendar)
        part["group"], part["lawd_cd"] = group, lawd_cd
        completed.append(part.rename_axis("date").reset_index())
    monthly = pd.concat(completed, ignore_index=True)
    monthly["deal_ym"] = monthly["date"].dt.strftime("%Y%m").astype(int)
    monthly["observed"] = monthly["median_price_m2"].notna()
    monthly["raw_log_price"] = np.log(monthly["median_price_m2"])
    # 희소 지역의 월별 표본 잡음을 줄이되 미래값을 쓰지 않는 후행 3개월 중앙값.
    monthly["log_price"] = monthly.groupby(
        ["group", "lawd_cd"])["raw_log_price"].transform(
            lambda s: s.rolling(3, min_periods=1).median())
    monthly.loc[~monthly["observed"], "log_price"] = np.nan
    grouped = monthly.groupby(["group", "lawd_cd"], group_keys=False)
    monthly["ret_1"] = grouped["log_price"].diff(1)
    monthly["ret_3"] = grouped["log_price"].diff(3) / 3.0
    monthly["ret_6"] = grouped["log_price"].diff(6) / 6.0
    monthly["vol_3"] = grouped["ret_1"].transform(
        lambda s: s.rolling(3, min_periods=2).std())
    # 한 달 중위가격의 표본 잡음을 줄이기 위해 향후 3개월 누적 로그수익률을
    # 월평균으로 환산한 값을 예측한다.
    monthly["target"] = (grouped["log_price"].shift(-3) - monthly["log_price"]) / 3.0
    national = monthly.groupby(["group", "date"])["log_price"].median().groupby(
        level=0).diff().rename("national_ret_1")
    monthly = monthly.join(national, on=["group", "date"])
    month = monthly["deal_ym"].astype(int) % 100
    monthly["month_sin"] = np.sin(2 * math.pi * month / 12)
    monthly["month_cos"] = np.cos(2 * math.pi * month / 12)
    monthly["log_count"] = np.log1p(monthly["count"])
    codes = {name: index for index, name in enumerate(sorted(monthly["group"].unique()))}
    monthly["group_code"] = monthly["group"].map(codes)
    for column in ["ret_1", "ret_3", "ret_6", "national_ret_1", "target"]:
        monthly[column] = monthly[column].clip(-0.10, 0.10)
    return monthly.reset_index(drop=True)


def build_training_frame(raw_dir: Path | None = None) -> pd.DataFrame:
    return build_feature_panel(raw_dir).dropna(
        subset=FEATURES + ["target"]).reset_index(drop=True)


def train(output: Path | None = None) -> dict:
    output = output or config.MODELS_DIR / "house_price_forecast.joblib"
    panel = build_feature_panel()
    frame = panel.dropna(subset=FEATURES + ["target"]).reset_index(drop=True)
    cutoff = sorted(frame["deal_ym"].unique())[-3]
    train_frame, test_frame = frame[frame.deal_ym < cutoff], frame[frame.deal_ym >= cutoff]
    params = dict(max_iter=180, learning_rate=0.045, max_leaf_nodes=15,
                  min_samples_leaf=25, l2_regularization=0.8, random_state=42)
    validation_model = HistGradientBoostingRegressor(loss="squared_error", **params)
    validation_model.fit(train_frame[FEATURES], train_frame["target"])
    prediction = validation_model.predict(test_frame[FEATURES])
    mae = float(mean_absolute_error(test_frame["target"], prediction))
    backtest = walk_forward_backtest(frame, FEATURES)

    models = {}
    for name, loss, quantile in (
        ("low", "quantile", 0.1), ("base", "squared_error", None),
        ("high", "quantile", 0.9),
    ):
        if name == "base" and backtest["selected_base_model"] == "seasonal_naive":
            model = SeasonalNaiveRegressor()
        elif name == "base" and backtest["selected_base_model"] == "ridge":
            model = Ridge(alpha=2.0)
        elif name == "base" and backtest["selected_base_model"] == "lightgbm":
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(
                n_estimators=180, learning_rate=.04, num_leaves=15,
                min_child_samples=25, reg_lambda=.8, random_state=42,
                verbosity=-1,
            )
        else:
            model = HistGradientBoostingRegressor(
                loss=loss, quantile=quantile, **params)
        model.fit(frame[FEATURES], frame["target"])
        models[name] = model
    # 추론 특성은 정답이 존재하는 마지막 학습월이 아니라, 정답이 아직 없는
    # 최신 실거래 관측월까지 사용한다.
    inference = panel.dropna(subset=FEATURES).reset_index(drop=True)
    latest = inference.sort_values("deal_ym").groupby(
        ["group", "lawd_cd"], as_index=False).tail(1)
    fallback = latest.groupby("group", as_index=False)[FEATURES].median()
    fallback["lawd_cd"] = "national"
    fallback["deal_ym"] = fallback["group"].map(
        latest.groupby("group")["deal_ym"].max())
    artifact = {
        "version": "rtms_walkforward_conformal_v4", "models": models,
        "features": FEATURES, "latest": latest,
        "fallback": fallback, "groups": sorted(frame.group.unique()),
        "trained_rows": len(frame),
        "source_month_min": int(panel.loc[panel.observed, "deal_ym"].min()),
        "source_month_max": int(panel.loc[panel.observed, "deal_ym"].max()),
        "training_target_month_max": int(frame.deal_ym.max()),
        "inference_feature_month_max": int(inference.deal_ym.max()),
        "holdout_month_start": int(cutoff),
        "holdout_monthly_log_return_mae": mae,
        "walk_forward": backtest,
        "conformal": backtest["conformal"],
        "news_numeric_policy": (
            "live LLM news labels are qualitative evidence only; numeric price effects "
            "require a lagged historical news feature that improves walk-forward validation"
        ),
        "training_sources": [item[0] for item in SOURCES],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)
    metadata = {key: value for key, value in artifact.items()
                if key not in {"models", "latest", "fallback"}}
    output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    print(json.dumps(train(), ensure_ascii=False, indent=2))
