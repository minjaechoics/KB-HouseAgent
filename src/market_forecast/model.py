"""학습된 지역·주택유형 월간 가격수익률 모델 추론."""
from __future__ import annotations

import math
from pathlib import Path

import joblib

from src import config
from src.market_forecast.kb_complex import KBLandComplexPriceTool
from src.market_forecast.news import NewsSignalTool
from src.real_estate_feeds.rtms import RTMSPriceHistoryTool


def _group(house_type: str) -> str:
    if house_type == "아파트":
        return "apartment"
    if house_type == "오피스텔":
        return "officetel"
    return "single_multi"


class HousePriceForecaster:
    def __init__(self, model_path: Path | None = None,
                 news_tool: NewsSignalTool | None = None,
                 kb_tool: KBLandComplexPriceTool | None = None,
                 price_history_tool: RTMSPriceHistoryTool | None = None,
                 llm=None):
        self.model_path = model_path or config.MODELS_DIR / "house_price_forecast.joblib"
        self.news_tool = news_tool or NewsSignalTool(llm=llm)
        self.kb_tool = kb_tool or KBLandComplexPriceTool()
        self.price_history_tool = price_history_tool or RTMSPriceHistoryTool()
        self.artifact = joblib.load(self.model_path) if self.model_path.exists() else None

    def _row(self, prop: dict):
        if not self.artifact:
            return None, "model_missing"
        group = _group(str(prop.get("house_type") or prop.get("property_type") or ""))
        code = str(prop.get("legal_dong_code") or "").zfill(10)[:5]
        latest = self.artifact["latest"]
        exact = latest[(latest.group == group) & (latest.lawd_cd.astype(str) == code)]
        if len(exact):
            return exact.iloc[-1], "sgg_house_type_exact"
        fallback = self.artifact["fallback"]
        match = fallback[fallback.group == group]
        return (match.iloc[-1], "house_type_national_fallback") if len(match) else (None, "missing")

    def forecast(self, prop: dict, regional_market: dict | None = None) -> dict:
        # Official reported transactions are the primary, reproducible history.
        # KB is never fetched automatically without a separate data-use grant.
        price_history = self.price_history_tool.history(prop)
        row, coverage = self._row(prop)
        if row is None:
            time_series_base = 0.02
            rone_price = (regional_market or {}).get("price_index") or {}
            rone_supply = (regional_market or {}).get("supply_demand") or {}
            news = self.news_tool.assess(
                str(prop.get("sido") or ""), str(prop.get("gugun") or ""),
                str(prop.get("house_type") or "주택"),
                building_name=str(prop.get("building_name") or ""),
                market_context={
                    "time_series_annual_growth_rate": time_series_base,
                    "rtms_change_1m": price_history.get("change_1m"),
                    "rtms_change_period": price_history.get("change_period"),
                    "price_history_match_type": price_history.get("match_type"),
                    "rone_price_index": rone_price.get("latest_value"),
                    "rone_price_index_change_1m_points": rone_price.get("change_1m"),
                    "rone_supply_demand_index": rone_supply.get("latest_value"),
                },
            )
            adjustment = news["annual_adjustment_pct_point"] / 100.0
            base = time_series_base + adjustment
            return {
                "model_version": "fallback_no_artifact", "coverage": coverage,
                "time_series_annual_growth_rate": round(time_series_base, 4),
                "news_adjustment_rate": round(adjustment, 4),
                "annual_growth_rate": round(base, 4),
                "annual_low": round(base - 0.04, 4), "annual_high": round(base + 0.04, 4),
                "news": news, "price_history": price_history,
                "market_assessment": news.get("overall_assessment"), "training": None,
                "warning": "학습모형이 없어 보수적 기본 시나리오를 사용했습니다.",
            }
        X = row[self.artifact["features"]].astype(float).to_frame().T
        monthly = {name: float(model.predict(X)[0])
                   for name, model in self.artifact["models"].items()}
        # 짧은 원천기간의 월별 잡음이 장기 복리에서 폭발하지 않도록 월 ±1%로
        # 제한한다. 원시 오차와 제한 사실은 모델 카드에 공개한다.
        annual = {name: math.exp(max(-0.01, min(0.01, value)) * 12) - 1
                  for name, value in monthly.items()}
        time_series_annual = dict(annual)
        rone_price = (regional_market or {}).get("price_index") or {}
        rone_supply = (regional_market or {}).get("supply_demand") or {}
        news = self.news_tool.assess(
            str(prop.get("sido") or ""), str(prop.get("gugun") or ""),
            str(prop.get("house_type") or "주택"),
            building_name=str(prop.get("building_name") or ""),
            market_context={
                "time_series_annual_growth_rate": round(time_series_annual["base"], 4),
                "time_series_low": round(time_series_annual["low"], 4),
                "time_series_high": round(time_series_annual["high"], 4),
                "rtms_change_1m": price_history.get("change_1m"),
                "rtms_change_period": price_history.get("change_period"),
                "price_history_match_type": price_history.get("match_type"),
                "rone_price_index": rone_price.get("latest_value"),
                "rone_price_index_change_1m_points": rone_price.get("change_1m"),
                "rone_supply_demand_index": rone_supply.get("latest_value"),
            },
        )
        adjustment = float(news["annual_adjustment_pct_point"]) / 100.0
        annual = {name: max(-0.15, min(0.15, value + adjustment))
                  for name, value in time_series_annual.items()}
        low, high = sorted((annual["low"], annual["high"]))
        # Quantile models may cross on sparse sub-regions. Keep the displayed
        # base estimate inside the interval even when that happens.
        low = min(low, annual["base"])
        high = max(high, annual["base"])
        return {
            "model_version": self.artifact["version"], "coverage": coverage,
            "time_series_annual_growth_rate": round(time_series_annual["base"], 4),
            "news_adjustment_rate": round(adjustment, 4),
            "annual_growth_rate": round(annual["base"], 4),
            "annual_low": round(low, 4), "annual_high": round(high, 4),
            "monthly_base_log_return": round(monthly["base"], 6),
            "news": news, "price_history": price_history,
            "market_assessment": news.get("overall_assessment"),
            "training": {
                "rows": self.artifact["trained_rows"],
                "source_month_min": self.artifact["source_month_min"],
                "source_month_max": self.artifact["source_month_max"],
                "training_target_month_max": self.artifact.get("training_target_month_max"),
                "inference_feature_month_max": self.artifact.get("inference_feature_month_max"),
                "holdout_monthly_log_return_mae": round(
                    self.artifact["holdout_monthly_log_return_mae"], 6),
            },
            "warning": "예측구간을 포함한 연구용 전망이며 투자수익을 보장하지 않습니다.",
        }
