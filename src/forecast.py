import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "mape": mape,
    }


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
