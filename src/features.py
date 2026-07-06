import numpy as np
import pandas as pd


def county_year_counts(registrations: pd.DataFrame) -> pd.DataFrame:
    counts = (
        registrations.groupby(["county", "model_year"])
        .size()
        .rename("new_registrations")
        .reset_index()
    )

    counties = counts["county"].unique()
    years = np.arange(counts["model_year"].min(), counts["model_year"].max() + 1)
    full_index = pd.MultiIndex.from_product([counties, years], names=["county", "model_year"])
    counts = counts.set_index(["county", "model_year"]).reindex(full_index, fill_value=0).reset_index()

    counts = counts.sort_values(["county", "model_year"])
    counts["cumulative_registrations"] = counts.groupby("county")["new_registrations"].cumsum()
    return counts


def add_lag_features(county_year: pd.DataFrame, n_lags: int = 2) -> pd.DataFrame:
    out = county_year.sort_values(["county", "model_year"]).copy()
    for lag in range(1, n_lags + 1):
        out[f"lag_{lag}"] = out.groupby("county")["new_registrations"].shift(lag)
    out["year_index"] = out["model_year"] - out["model_year"].min()
    return out


def zip_supply_demand(registrations: pd.DataFrame, stations: pd.DataFrame, gazetteer: pd.DataFrame) -> pd.DataFrame:
    demand = registrations.groupby("zip").size().rename("registered_evs").reset_index()

    supply = (
        stations.groupby("zip")
        .agg(
            station_count=("zip", "size"),
            l1_ports=("ev_level1_evse_num", "sum"),
            l2_ports=("ev_level2_evse_num", "sum"),
            dcfc_ports=("ev_dc_fast_count", "sum"),
        )
        .reset_index()
    )

    merged = demand.merge(supply, on="zip", how="left").merge(gazetteer, on="zip", how="left")
    for col in ["station_count", "l1_ports", "l2_ports", "dcfc_ports"]:
        merged[col] = merged[col].fillna(0)

    merged["total_ports"] = merged["l1_ports"] + merged["l2_ports"] + merged["dcfc_ports"]
    merged["evs_per_dcfc_port"] = merged["registered_evs"] / merged["dcfc_ports"].replace(0, np.nan)
    merged["evs_per_total_port"] = merged["registered_evs"] / merged["total_ports"].replace(0, np.nan)
    merged["ev_density_per_sqmi"] = merged["registered_evs"] / merged["land_sqmi"].replace(0, np.nan)
    return merged
