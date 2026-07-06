import pandas as pd

from src.clean import clean_gazetteer, clean_registrations, clean_stations


def test_clean_registrations_filters_out_of_range_years():
    df = pd.DataFrame(
        {
            "zip": ["10001", "10002-1234", "07030"],
            "county": ["new york", "KINGS", "out-of-state"],
            "model_year": [2022, 1998, 2023],
        }
    )
    out = clean_registrations(df)
    assert list(out["county"]) == ["NEW YORK"]
    assert out.iloc[0]["zip"] == "10001"


def test_clean_registrations_extracts_five_digit_zip():
    df = pd.DataFrame({"zip": ["10001-4567"], "county": ["KINGS"], "model_year": [2020]})
    out = clean_registrations(df)
    assert out.iloc[0]["zip"] == "10001"


def test_clean_stations_fills_missing_ports_with_zero():
    df = pd.DataFrame(
        {
            "zip": ["11201"],
            "ev_level1_evse_num": [None],
            "ev_level2_evse_num": ["2"],
            "ev_dc_fast_count": [None],
            "latitude": ["40.7"],
            "longitude": ["-73.9"],
        }
    )
    out = clean_stations(df)
    assert out.iloc[0]["ev_level1_evse_num"] == 0
    assert out.iloc[0]["ev_dc_fast_count"] == 0
    assert out.iloc[0]["ev_level2_evse_num"] == 2


def test_clean_gazetteer_renames_and_pads_zip():
    df = pd.DataFrame({"GEOID": ["601"], "ALAND_SQMI": ["10.5"], "AWATER": ["0"]})
    out = clean_gazetteer(df)
    assert list(out.columns) == ["zip", "land_sqmi"]
    assert out.iloc[0]["zip"] == "00601"
