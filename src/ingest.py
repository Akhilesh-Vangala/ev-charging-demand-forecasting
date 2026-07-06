import io
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DMV_ENDPOINT = "https://data.ny.gov/resource/w4pv-hbkt.json"
STATIONS_ENDPOINT = "https://data.ny.gov/resource/7rrd-248n.json"
GAZETTEER_URL = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_zcta_national.zip"


def _paginate(url, params, page_size=5000, max_pages=200):
    rows = []
    offset = 0
    for _ in range(max_pages):
        page_params = dict(params)
        page_params["$limit"] = page_size
        page_params["$offset"] = offset
        resp = requests.get(url, params=page_params, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        offset += page_size
        if len(batch) < page_size:
            break
        time.sleep(0.2)
    return rows


def fetch_ev_registrations(force=False):
    cache_path = RAW_DIR / "ev_registrations.csv"
    if cache_path.exists() and not force:
        return pd.read_csv(cache_path, dtype={"zip": str})

    params = {
        "$where": "fuel_type='ELECTRIC' AND record_type='VEH'",
        "$select": "zip,county,model_year,body_type,make",
    }
    rows = _paginate(DMV_ENDPOINT, params)
    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)
    return df


def fetch_charging_stations(force=False):
    cache_path = RAW_DIR / "charging_stations.csv"
    if cache_path.exists() and not force:
        return pd.read_csv(cache_path, dtype={"zip": str})

    params = {
        "$select": (
            "zip,city,latitude,longitude,ev_level1_evse_num,ev_level2_evse_num,"
            "ev_dc_fast_count,ev_network,open_date"
        )
    }
    rows = _paginate(STATIONS_ENDPOINT, params)
    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)
    return df


def fetch_zcta_gazetteer(force=False):
    cache_path = RAW_DIR / "zcta_gazetteer.csv"
    if cache_path.exists() and not force:
        return pd.read_csv(cache_path, dtype={"GEOID": str})

    resp = requests.get(GAZETTEER_URL, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = [n for n in zf.namelist() if n.lower().endswith(".txt")][0]
        with zf.open(name) as f:
            df = pd.read_csv(f, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df.to_csv(cache_path, index=False)
    return df


if __name__ == "__main__":
    regs = fetch_ev_registrations()
    stations = fetch_charging_stations()
    gaz = fetch_zcta_gazetteer()
    print("registrations:", len(regs))
    print("stations:", len(stations))
    print("zcta rows:", len(gaz))
