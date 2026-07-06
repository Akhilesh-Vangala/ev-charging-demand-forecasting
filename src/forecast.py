import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import Holt


def evaluate(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan
    bias = float(np.mean(y_pred - y_true))
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    variance_component = (max(rmse ** 2 - bias ** 2, 0)) ** 0.5
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse,
        "mape": mape,
        "bias": bias,
        "variance_component": variance_component,
    }


def bias_variance_by_group(preds: pd.DataFrame, group_col: str = "county") -> pd.DataFrame:
    rows = []
    for group, sub in preds.groupby(group_col):
        metrics = evaluate(sub["new_registrations"], sub["prediction"])
        metrics[group_col] = group
        metrics["n"] = len(sub)
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values("rmse", ascending=False)


def naive_forecast(lagged: pd.DataFrame, holdout_years: list[int]) -> pd.DataFrame:
    rows = lagged[lagged["model_year"].isin(holdout_years)].copy()
    rows["prediction"] = rows["lag_1"]
    return rows[["county", "model_year", "new_registrations", "prediction"]]


def lgbm_forecast(lagged: pd.DataFrame, holdout_years: list[int], n_lags: int = 2) -> pd.DataFrame:
    df = lagged.dropna(subset=[f"lag_{i}" for i in range(1, n_lags + 1)]).copy()
    df["county_code"] = df["county"].astype("category").cat.codes

    feature_cols = ["year_index", "county_code"] + [f"lag_{i}" for i in range(1, n_lags + 1)]
    train = df[~df["model_year"].isin(holdout_years)]
    test = df[df["model_year"].isin(holdout_years)]

    model = LGBMRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        min_child_samples=5,
        verbosity=-1,
    )
    model.fit(train[feature_cols], train["new_registrations"])

    test = test.copy()
    test["prediction"] = model.predict(test[feature_cols]).clip(min=0)
    return test[["county", "model_year", "new_registrations", "prediction"]], model


def rf_forecast(lagged: pd.DataFrame, holdout_years: list[int], n_lags: int = 2):
    df = lagged.dropna(subset=[f"lag_{i}" for i in range(1, n_lags + 1)]).copy()
    df["county_code"] = df["county"].astype("category").cat.codes

    feature_cols = ["year_index", "county_code"] + [f"lag_{i}" for i in range(1, n_lags + 1)]
    train = df[~df["model_year"].isin(holdout_years)]
    test = df[df["model_year"].isin(holdout_years)]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=3,
        random_state=0,
    )
    model.fit(train[feature_cols], train["new_registrations"])

    test = test.copy()
    test["prediction"] = model.predict(test[feature_cols]).clip(min=0)
    return test[["county", "model_year", "new_registrations", "prediction"]], model


def ols_trend_forecast(lagged: pd.DataFrame, holdout_years: list[int]) -> pd.DataFrame:
    train = lagged[~lagged["model_year"].isin(holdout_years)]
    test = lagged[lagged["model_year"].isin(holdout_years)].copy()

    predictions = []
    for county, test_rows in test.groupby("county"):
        county_train = train[train["county"] == county]
        if len(county_train) < 2:
            predictions.append(pd.Series(county_train["new_registrations"].mean() if len(county_train) else 0, index=test_rows.index))
            continue
        model = LinearRegression()
        model.fit(county_train[["year_index"]], county_train["new_registrations"])
        preds = model.predict(test_rows[["year_index"]])
        predictions.append(pd.Series(np.clip(preds, 0, None), index=test_rows.index))

    test["prediction"] = pd.concat(predictions).sort_index()
    return test[["county", "model_year", "new_registrations", "prediction"]]


def holt_forecast(lagged: pd.DataFrame, holdout_years: list[int]) -> pd.DataFrame:
    train = lagged[~lagged["model_year"].isin(holdout_years)]
    test = lagged[lagged["model_year"].isin(holdout_years)].copy()
    holdout_years_sorted = sorted(holdout_years)

    predictions = {}
    for county, county_train in train.groupby("county"):
        series = county_train.sort_values("model_year")["new_registrations"]
        if len(series) < 3 or series.sum() == 0:
            predictions[county] = {y: series.iloc[-1] if len(series) else 0.0 for y in holdout_years_sorted}
            continue
        try:
            model = Holt(series.values, initialization_method="estimated").fit(optimized=True)
            forecast = model.forecast(len(holdout_years_sorted))
        except Exception:
            forecast = np.repeat(series.iloc[-1], len(holdout_years_sorted))
        predictions[county] = dict(zip(holdout_years_sorted, np.clip(forecast, 0, None)))

    test["prediction"] = test.apply(lambda r: predictions[r["county"]][r["model_year"]], axis=1)
    return test[["county", "model_year", "new_registrations", "prediction"]]
