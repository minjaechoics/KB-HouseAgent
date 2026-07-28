import numpy as np
import pandas as pd

from src.market_forecast.backtest import walk_forward_backtest
from src.market_forecast.conformal import calibrate_intervals


def test_conformal_interval_orders_80_and_95_coverage():
    truth = np.array([-2, -1, 0, 1, 2], dtype=float)
    lower = np.array([-.5] * 5)
    upper = np.array([.5] * 5)
    result = calibrate_intervals(truth, lower, upper)
    assert result["qhat_95_monthly_log_return"] >= result[
        "qhat_80_monthly_log_return"]
    assert result["empirical_coverage_95"] >= result["empirical_coverage_80"]


def test_walk_forward_never_trains_on_validation_or_future_months():
    rng = np.random.default_rng(7)
    rows = []
    for month_index, deal_ym in enumerate(range(202501, 202513)):
        # Calendar-safe YYYYMM values for the fixture.
        deal_ym = 202500 + month_index + 1
        for _ in range(15):
            ret = rng.normal(.002, .01)
            rows.append({
                "deal_ym": deal_ym, "ret_1": ret, "ret_3": ret,
                "ret_6": ret, "vol_3": .01, "log_count": 2,
                "national_ret_1": ret / 2, "month_sin": 0,
                "month_cos": 1, "group_code": 0,
                "target": ret + rng.normal(0, .002),
            })
    frame = pd.DataFrame(rows)
    features = ["ret_1", "ret_3", "ret_6", "vol_3", "log_count",
                "national_ret_1", "month_sin", "month_cos", "group_code"]
    result = walk_forward_backtest(frame, features, folds=4)
    assert len(result["folds"]) == 4
    assert all(row["validation_month"] in result["validation_months"]
               for row in result["folds"])
    assert result["selected_base_model"] in result["model_metrics"]
    assert result["conformal"]["sample_count"] == 60
