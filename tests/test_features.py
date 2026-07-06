import pandas as pd

from src.features import add_lag_features, county_year_counts, zip_supply_demand


def test_county_year_counts_fills_missing_years_with_zero():
    regs = pd.DataFrame(
        {
            "county": ["KINGS", "KINGS", "QUEENS"],
            "model_year": [2020, 2022, 2021],
        }
    )
    out = county_year_counts(regs)
    kings_2021 = out[(out["county"] == "KINGS") & (out["model_year"] == 2021)]
    assert kings_2021.iloc[0]["new_registrations"] == 0


def test_county_year_counts_cumulative_sum_is_monotonic():
    regs = pd.DataFrame({"county": ["KINGS"] * 3, "model_year": [2020, 2020, 2021]})
    out = county_year_counts(regs)
    cumulative = out.sort_values("model_year")["cumulative_registrations"].tolist()
    assert cumulative == sorted(cumulative)
    assert cumulative[-1] == 3


def test_add_lag_features_shifts_within_county():
    county_year = pd.DataFrame(
        {
            "county": ["KINGS", "KINGS", "QUEENS"],
            "model_year": [2020, 2021, 2020],
            "new_registrations": [10, 20, 5],
            "cumulative_registrations": [10, 30, 5],
        }
    )
    out = add_lag_features(county_year, n_lags=1)
    kings_2021 = out[(out["county"] == "KINGS") & (out["model_year"] == 2021)]
    assert kings_2021.iloc[0]["lag_1"] == 10
    queens_2020 = out[(out["county"] == "QUEENS") & (out["model_year"] == 2020)]
    assert pd.isna(queens_2020.iloc[0]["lag_1"])


def test_zip_supply_demand_handles_zero_ports():
    registrations = pd.DataFrame({"zip": ["11201", "11201", "10001"]})
    stations = pd.DataFrame(
        {
            "zip": ["10001"],
            "ev_level1_evse_num": [0],
            "ev_level2_evse_num": [2],
            "ev_dc_fast_count": [0],
        }
    )
    gazetteer = pd.DataFrame({"zip": ["11201", "10001"], "land_sqmi": [1.0, 2.0]})

    out = zip_supply_demand(registrations, stations, gazetteer)
    row_11201 = out[out["zip"] == "11201"].iloc[0]
    assert row_11201["dcfc_ports"] == 0
    assert pd.isna(row_11201["evs_per_dcfc_port"])

    row_10001 = out[out["zip"] == "10001"].iloc[0]
    assert row_10001["registered_evs"] == 1
    assert row_10001["evs_per_total_port"] == 0.5
