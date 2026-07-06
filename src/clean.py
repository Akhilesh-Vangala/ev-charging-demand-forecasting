import pandas as pd

MIN_MODEL_YEAR = 2011
MAX_MODEL_YEAR = 2026


def clean_registrations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["zip"] = out["zip"].astype(str).str.extract(r"(\d{5})")[0]
    out["county"] = out["county"].astype(str).str.strip().str.upper()
    out["model_year"] = pd.to_numeric(out["model_year"], errors="coerce")

    out = out.dropna(subset=["zip", "county", "model_year"])
    out = out[out["county"] != "OUT-OF-STATE"]
    out = out[(out["model_year"] >= MIN_MODEL_YEAR) & (out["model_year"] <= MAX_MODEL_YEAR)]
    out["model_year"] = out["model_year"].astype(int)
    return out.reset_index(drop=True)


def clean_stations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["zip"] = out["zip"].astype(str).str.extract(r"(\d{5})")[0]
    out = out.dropna(subset=["zip"])

    port_cols = ["ev_level1_evse_num", "ev_level2_evse_num", "ev_dc_fast_count"]
    for col in port_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    return out.reset_index(drop=True)


def clean_gazetteer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={"GEOID": "zip", "ALAND_SQMI": "land_sqmi"})
    out["zip"] = out["zip"].astype(str).str.zfill(5)
    out["land_sqmi"] = pd.to_numeric(out["land_sqmi"], errors="coerce")
    return out[["zip", "land_sqmi"]].reset_index(drop=True)
