import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.clean import clean_gazetteer, clean_registrations, clean_stations
from src.features import add_lag_features, county_year_counts, zip_supply_demand
from src.forecast import evaluate, lgbm_forecast, naive_forecast
from src.ingest import fetch_charging_stations, fetch_ev_registrations, fetch_zcta_gazetteer

PROCESSED_DIR = Path("data/processed")
FIGURES_DIR = Path("reports/figures")
NYC_COUNTIES = ["BRONX", "KINGS", "NEW YORK", "QUEENS", "RICHMOND"]
HOLDOUT_YEARS = [2024, 2025]
COMPLETE_YEAR_CUTOFF = 2025


def build_datasets():
    registrations = clean_registrations(fetch_ev_registrations())
    stations = clean_stations(fetch_charging_stations())
    gazetteer = clean_gazetteer(fetch_zcta_gazetteer())
    return registrations, stations, gazetteer


def run_forecasts(registrations: pd.DataFrame):
    county_year = county_year_counts(registrations[registrations["model_year"] <= COMPLETE_YEAR_CUTOFF])
    lagged = add_lag_features(county_year, n_lags=2)

    naive_preds = naive_forecast(lagged, HOLDOUT_YEARS)
    lgbm_preds, model = lgbm_forecast(lagged, HOLDOUT_YEARS)

    naive_metrics = evaluate(naive_preds["new_registrations"], naive_preds["prediction"])
    lgbm_metrics = evaluate(lgbm_preds["new_registrations"], lgbm_preds["prediction"])

    return {
        "county_year": county_year,
        "naive_preds": naive_preds,
        "lgbm_preds": lgbm_preds,
        "naive_metrics": naive_metrics,
        "lgbm_metrics": lgbm_metrics,
        "model": model,
    }


def plot_growth_curves(county_year: pd.DataFrame, out_path: Path):
    totals = county_year.groupby("county")["new_registrations"].sum().sort_values(ascending=False)
    top_counties = totals.head(8).index

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for county in top_counties:
        sub = county_year[county_year["county"] == county]
        ax.plot(sub["model_year"], sub["cumulative_registrations"], marker="o", markersize=3, label=county.title())

    ax.set_xlabel("Model year")
    ax.set_ylabel("Cumulative registered EVs (NY State)")
    ax.set_title("EV registration growth by county, top 8 by volume")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_forecast_error(naive_metrics: dict, lgbm_metrics: dict, out_path: Path):
    labels = ["MAE", "RMSE", "MAPE (%)"]
    naive_vals = [naive_metrics["mae"], naive_metrics["rmse"], naive_metrics["mape"]]
    lgbm_vals = [lgbm_metrics["mae"], lgbm_metrics["rmse"], lgbm_metrics["mape"]]

    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    width = 0.35
    ax.bar([i - width / 2 for i in x], naive_vals, width, label="Naive (last year)")
    ax.bar([i + width / 2 for i in x], lgbm_vals, width, label="LightGBM")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_title(f"Holdout forecast error, model years {HOLDOUT_YEARS}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_nyc_demand_gap(zip_table: pd.DataFrame, out_path: Path, top_n: int = 20):
    ranked = zip_table[zip_table["registered_evs"] >= 20].sort_values("evs_per_total_port", ascending=False)
    top = ranked.head(top_n)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top["zip"].astype(str), top["evs_per_total_port"], color="#3a6ea5")
    ax.invert_yaxis()
    ax.set_xlabel("Registered EVs per public charging port")
    ax.set_title("NYC zip codes with the largest EV-to-port gap\n(zips with 20+ registered EVs)")
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

    zip_table = zip_supply_demand(registrations, stations, gazetteer)
    zip_table.to_csv(PROCESSED_DIR / "zip_supply_demand.csv", index=False)

    nyc_zip_table = zip_table.merge(
        registrations[["zip", "county"]].drop_duplicates("zip"), on="zip", how="left"
    )
    nyc_zip_table = nyc_zip_table[nyc_zip_table["county"].isin(NYC_COUNTIES)]
    nyc_zip_table.to_csv(PROCESSED_DIR / "nyc_zip_supply_demand.csv", index=False)

    plot_growth_curves(results["county_year"], FIGURES_DIR / "county_growth_curves.png")
    plot_forecast_error(results["naive_metrics"], results["lgbm_metrics"], FIGURES_DIR / "forecast_error_comparison.png")
    plot_nyc_demand_gap(nyc_zip_table, FIGURES_DIR / "nyc_demand_gap.png")

    summary = {
        "total_registered_evs": int(len(registrations)),
        "total_charging_stations": int(len(stations)),
        "counties_covered": int(registrations["county"].nunique()),
        "zips_covered": int(registrations["zip"].nunique()),
        "naive_metrics": results["naive_metrics"],
        "lgbm_metrics": results["lgbm_metrics"],
        "holdout_years": HOLDOUT_YEARS,
        "nyc_zips_with_zero_dcfc_and_50plus_evs": int(
            ((nyc_zip_table["dcfc_ports"] == 0) & (nyc_zip_table["registered_evs"] >= 50)).sum()
        ),
    }
    with open(PROCESSED_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
