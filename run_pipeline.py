import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.clean import clean_gazetteer, clean_registrations, clean_stations
from src.features import add_lag_features, county_year_counts, zip_supply_demand
from src.forecast import (
    bias_variance_by_group,
    evaluate,
    holt_forecast,
    lgbm_forecast,
    naive_forecast,
    ols_trend_forecast,
    rf_forecast,
)
from src.ingest import fetch_charging_stations, fetch_ev_registrations, fetch_zcta_gazetteer

PROCESSED_DIR = Path("data/processed")
FIGURES_DIR = Path("reports/figures")
NYC_COUNTIES = ["BRONX", "KINGS", "NEW YORK", "QUEENS", "RICHMOND"]
HOLDOUT_YEARS = [2024, 2025]
COMPLETE_YEAR_CUTOFF = 2025

BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"
VIOLET = "#4a3aa7"
RED = "#e34948"
GRID = "#e1e0d9"
MUTED = "#898781"
INK_SECONDARY = "#52514e"

MODEL_COLORS = {
    "Naive (last year)": MUTED,
    "OLS trend": YELLOW,
    "Holt linear": VIOLET,
    "Random Forest": AQUA,
    "LightGBM": BLUE,
}

plt.rcParams.update(
    {
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "grid.color": GRID,
        "text.color": "#0b0b0b",
        "font.size": 10,
    }
)


def build_datasets():
    registrations = clean_registrations(fetch_ev_registrations())
    stations = clean_stations(fetch_charging_stations())
    gazetteer = clean_gazetteer(fetch_zcta_gazetteer())
    return registrations, stations, gazetteer


def run_forecasts(registrations: pd.DataFrame):
    county_year = county_year_counts(registrations[registrations["model_year"] <= COMPLETE_YEAR_CUTOFF])
    lagged = add_lag_features(county_year, n_lags=2)

    naive_preds = naive_forecast(lagged, HOLDOUT_YEARS)
    ols_preds = ols_trend_forecast(lagged, HOLDOUT_YEARS)
    holt_preds = holt_forecast(lagged, HOLDOUT_YEARS)
    rf_preds, rf_model = rf_forecast(lagged, HOLDOUT_YEARS)
    lgbm_preds, lgbm_model = lgbm_forecast(lagged, HOLDOUT_YEARS)

    preds_by_model = {
        "Naive (last year)": naive_preds,
        "OLS trend": ols_preds,
        "Holt linear": holt_preds,
        "Random Forest": rf_preds,
        "LightGBM": lgbm_preds,
    }
    metrics_by_model = {
        name: evaluate(preds["new_registrations"], preds["prediction"]) for name, preds in preds_by_model.items()
    }

    return {
        "county_year": county_year,
        "preds_by_model": preds_by_model,
        "metrics_by_model": metrics_by_model,
        "lgbm_model": lgbm_model,
        "rf_model": rf_model,
        "feature_cols": ["year_index", "county_code", "lag_1", "lag_2"],
    }


def plot_growth_curves(county_year: pd.DataFrame, out_path: Path):
    totals = county_year.groupby("county")["new_registrations"].sum().sort_values(ascending=False)
    top_counties = totals.head(8).index
    colors = [BLUE, AQUA, YELLOW, VIOLET, RED, "#e87ba4", "#eb6834", "#008300"]

    max_year = county_year["model_year"].max()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for county, color in zip(top_counties, colors):
        sub = county_year[county_year["county"] == county]
        ax.plot(sub["model_year"], sub["cumulative_registrations"], marker="o", markersize=3, label=county.title(), color=color, linewidth=1.8)

    ax.axvspan(2023.5, max_year + 0.5, color=GRID, alpha=0.5, zorder=0, label="Slowdown period")
    ax.set_xlabel("Model year")
    ax.set_ylabel("Cumulative registered EVs (NY State)")
    ax.set_title("EV registration growth by county, top 8 by volume")
    ax.grid(axis="y", linewidth=0.6, alpha=0.6)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_model_comparison(metrics_by_model: dict, out_path: Path):
    models = list(metrics_by_model.keys())
    rmse_vals = [metrics_by_model[m]["rmse"] for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [MODEL_COLORS[m] for m in models]
    bars = ax.bar(models, rmse_vals, color=colors, width=0.6)
    ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=9)
    ax.set_ylabel("Validation RMSE (registrations)")
    ax.set_title(f"Holdout RMSE by model, model years {HOLDOUT_YEARS}")
    ax.grid(axis="y", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_bias_variance(metrics_by_model: dict, out_path: Path):
    models = list(metrics_by_model.keys())
    bias = [metrics_by_model[m]["bias"] for m in models]
    variance = [metrics_by_model[m]["variance_component"] for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width / 2, bias, width, label="Bias", color=RED)
    ax.bar(x + width / 2, variance, width, label="Variance component", color=BLUE)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Registrations")
    ax.set_title("Bias vs. variance component of holdout error")
    ax.legend(frameon=False)
    ax.grid(axis="y", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(model, feature_cols, out_path: Path):
    importances = model.feature_importances_
    order = np.argsort(importances)
    labels = [feature_cols[i] for i in order]
    values = importances[order]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh(labels, values, color=BLUE)
    ax.set_xlabel("LightGBM total gain")
    ax.set_title("Feature importance, county-year registration forecast")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_worst_counties(bv_table: pd.DataFrame, out_path: Path, top_n: int = 10):
    top = bv_table.head(top_n).iloc[::-1]
    labels = top["county"].str.title()
    y = np.arange(len(labels))
    height = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(y + height / 2, top["bias"], height, color=RED, label="Bias")
    ax.barh(y - height / 2, top["variance_component"], height, color=BLUE, label="Variance component")
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Registrations")
    ax.set_title("Worst-predicted counties, LightGBM holdout\n(bias and variance components of RMSE)")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_nyc_demand_gap(zip_table: pd.DataFrame, out_path: Path, top_n: int = 20):
    ranked = zip_table[zip_table["registered_evs"] >= 20].sort_values("evs_per_total_port", ascending=False)
    top = ranked.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top["zip"].astype(str), top["evs_per_total_port"], color=BLUE)
    ax.set_xlabel("Registered EVs per public charging port")
    ax.set_title("NYC zip codes with the largest EV-to-port gap\n(zips with 20+ registered EVs)")
    ax.grid(axis="x", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    registrations, stations, gazetteer = build_datasets()
    registrations.to_csv(PROCESSED_DIR / "registrations_clean.csv", index=False)
    stations.to_csv(PROCESSED_DIR / "stations_clean.csv", index=False)

    results = run_forecasts(registrations)
    results["county_year"].to_csv(PROCESSED_DIR / "county_year_counts.csv", index=False)

    lgbm_bv = bias_variance_by_group(results["preds_by_model"]["LightGBM"])
    lgbm_bv.to_csv(PROCESSED_DIR / "lgbm_county_bias_variance.csv", index=False)

    zip_table = zip_supply_demand(registrations, stations, gazetteer)
    zip_table.to_csv(PROCESSED_DIR / "zip_supply_demand.csv", index=False)

    nyc_zip_table = zip_table.merge(
        registrations[["zip", "county"]].drop_duplicates("zip"), on="zip", how="left"
    )
    nyc_zip_table = nyc_zip_table[nyc_zip_table["county"].isin(NYC_COUNTIES)]
    nyc_zip_table.to_csv(PROCESSED_DIR / "nyc_zip_supply_demand.csv", index=False)

    plot_growth_curves(results["county_year"], FIGURES_DIR / "county_growth_curves.png")
    plot_model_comparison(results["metrics_by_model"], FIGURES_DIR / "model_comparison.png")
    plot_bias_variance(results["metrics_by_model"], FIGURES_DIR / "bias_variance.png")
    plot_feature_importance(results["lgbm_model"], results["feature_cols"], FIGURES_DIR / "feature_importance.png")
    plot_worst_counties(lgbm_bv, FIGURES_DIR / "worst_counties.png")
    plot_nyc_demand_gap(nyc_zip_table, FIGURES_DIR / "nyc_demand_gap.png")

    summary = {
        "total_registered_evs": int(len(registrations)),
        "total_charging_stations": int(len(stations)),
        "counties_covered": int(registrations["county"].nunique()),
        "zips_covered": int(registrations["zip"].nunique()),
        "metrics_by_model": results["metrics_by_model"],
        "holdout_years": HOLDOUT_YEARS,
        "worst_counties_lgbm": lgbm_bv.head(5).to_dict(orient="records"),
        "nyc_zips_with_zero_dcfc_and_50plus_evs": int(
            ((nyc_zip_table["dcfc_ports"] == 0) & (nyc_zip_table["registered_evs"] >= 50)).sum()
        ),
    }
    with open(PROCESSED_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
