# EV Charging Demand Forecasting (New York State)

A reproducible pipeline that ingests public New York State data on registered electric
vehicles and public charging infrastructure, forecasts county-level EV adoption, and
flags zip codes where charging supply is falling behind demand.

## Why this exists

Utilization data for individual DC fast charging stations (session counts, dwell time,
uptime) is not public. Companies that need it license it from vendors like Paren or
Atlas Public Policy. This project does not have access to that data, so instead of
faking it, it uses the two things New York State does publish: who owns an EV and
where, and where the public chargers are. The result is a demand-side proxy for
utilization pressure rather than a direct utilization model, and the README and report
are explicit about that distinction throughout.

## Data sources

- **NY DMV Vehicle Registrations** ([data.ny.gov, w4pv-hbkt](https://data.ny.gov/resource/w4pv-hbkt.json)) -
  a live snapshot of active vehicle registrations statewide. Filtered to `fuel_type =
  ELECTRIC` and `record_type = VEH` (excludes boats), giving ~211k electric vehicles
  across every NY county and roughly 1,900 zip codes.
- **NY Electric Vehicle Charging Stations** ([data.ny.gov, 7rrd-248n](https://data.ny.gov/resource/7rrd-248n.json)) -
  ~5,500 public charging locations statewide with Level 1, Level 2, and DC fast port
  counts by zip code.
- **Census Gazetteer ZCTA file** ([census.gov](https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_zcta_national.zip)) -
  land area per zip code, used to compute density rather than relying on population
  data that isn't freely available without an API key.

All three are pulled live by `src/ingest.py` and cached under `data/raw/` so re-running
the pipeline doesn't hammer the source APIs.

## What the pipeline does

1. **Ingest** - pull all three sources, paginating through the Socrata APIs.
2. **Clean** - normalize zip codes to 5 digits, standardize county names, drop
   out-of-state and clearly invalid rows, coerce port counts to numeric.
3. **Forecast** - build a county-by-model-year registration time series, then compare
   a naive last-year baseline against a LightGBM regression (lag features + year index)
   on a held-out 2024-2025 window.
4. **Rank** - compute a zip-level "registered EVs per public charging port" ratio across
   NYC's five counties to surface zip codes with a real gap between demand and supply.

Run it end to end:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
```

Outputs land in `data/processed/` (cleaned tables, `summary.json`) and
`reports/figures/` (charts referenced in `reports/REPORT.md`).

## Testing

```bash
python -m pytest tests/ -v
```

Tests cover the cleaning rules, feature construction (lag features, zip-level ratio
edge cases like zero ports), and the forecasting/evaluation functions, all against
small synthetic fixtures so they don't depend on network access.

## Key finding, and why it matters more than the model

Statewide EV registrations grew every year from 2018 through 2023, then **declined for
two consecutive years** (2024 and 2025). A model trained on the pre-2024 acceleration -
including the LightGBM model in this repo - overshoots badly once that trend reverses,
while a naive "assume no growth" baseline ends up more accurate almost by accident. Full
numbers and interpretation are in [`reports/REPORT.md`](reports/REPORT.md).

That result is the actual point of this project: a forecasting model is only as good as
its ability to notice when the regime it was trained on has ended, and it's worth
reporting that limitation plainly rather than picking a holdout window that flatters the
model.

## Limitations

- Registration counts are a proxy for adoption, not for charging session volume or
  utilization. A vehicle being registered says nothing about how often its owner uses
  public DC fast charging versus charging at home.
- "Model year" reflects a vehicle's model year, not necessarily the calendar year it was
  purchased or first registered, so the year-over-year series is a reasonable but
  imperfect stand-in for a true adoption curve.
- Charging station counts come from a self-reported directory and may lag real-world
  installations or decommissions by weeks to months.
