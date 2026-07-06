import numpy as np
import pandas as pd

from src.features import add_lag_features, county_year_counts
from src.forecast import evaluate, lgbm_forecast, naive_forecast


def test_evaluate_perfect_prediction_has_zero_error():
    metrics = evaluate([10, 20, 30], [10, 20, 30])
    assert metrics["mae"] == 0
    assert metrics["rmse"] == 0
    assert metrics["mape"] == 0


def test_evaluate_mape_ignores_zero_actuals():
    metrics = evaluate([0, 10], [5, 12])
    assert abs(metrics["mape"] - 20.0) < 1e-6


def test_naive_forecast_uses_prior_year_as_prediction():
    county_year = county_year_counts(
        pd.DataFrame({"county": ["KINGS"] * 2, "model_year": [2020, 2021]})
    )
    lagged = add_lag_features(county_year, n_lags=1)
    out = naive_forecast(lagged, holdout_years=[2021])
    assert out.iloc[0]["prediction"] == lagged[lagged["model_year"] == 2020].iloc[0]["new_registrations"]


def test_lgbm_forecast_runs_end_to_end_on_synthetic_series():
    rng = np.random.default_rng(0)
    counties = ["KINGS", "QUEENS", "BRONX"]
    rows = []
    for county in counties:
        base = rng.integers(50, 200)
        for i, year in enumerate(range(2015, 2027)):
            rows.append({"county": county, "model_year": year, "new_registrations": base + i * 15})

    county_year = pd.DataFrame(rows)
    county_year["cumulative_registrations"] = county_year.groupby("county")["new_registrations"].cumsum()
    lagged = add_lag_features(county_year, n_lags=2)

    preds, model = lgbm_forecast(lagged, holdout_years=[2025, 2026])
    assert len(preds) == len(counties) * 2
    assert (preds["prediction"] >= 0).all()
    assert model is not None
