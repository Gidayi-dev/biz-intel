# Runbook: how to run & test the project

Two ways to use this project:

- **Option A — just see the results** (recommended for a demo / evaluation,
no network needed): run the dashboard or the CLI queries against the
already-built database.
- **Option B — re-run the full pipeline** (needs internet): rebuild
everything from the raw OSM fetch up.

Everything below uses the project venv. From the project root
(`biz-intel/`), the Python interpreter is:

```
venv\Scripts\python
```

---

## Prerequisites (first time only)

```powershell
# From the project root
venv\Scripts\python -m pip install -r requirements.txt   # if a requirements file exists
venv\Scripts\python -m pip install streamlit             # needed for the dashboard
```

`streamlit` is already installed in this venv. `numpy`, `pandas` and
`requests` are the only other runtime deps the scripts need.

---



## Option A — test the finished model (no network)

The database `data/processed/biz_intel.db` is already built. Two interfaces:

### A1. Interactive dashboard (best for "test as a user")

The spec dashboard is **`creative_engine.py`**:

```powershell
venv\Scripts\python -m streamlit run creative_engine.py
```



Your browser opens at **[http://localhost:8501](http://localhost:8501)**. The dashboard
implements the two required query modes directly against `data/processed/biz_intel.db`:

| Control                | What it does                                                                    |
| ---------------------- | ------------------------------------------------------------------------------- |
| **Mode selector**      | "I have a location" (rank categories for a place) vs "I have a business type" (rank locations for a category) |
| **Mode 1 — location**  | Dropdown validated against the `locations` table — arbitrary text is rejected   |
| **Mode 2 — category**  | Category list from the real taxonomy (12 categories incl. Tier-2 `laundry`)     |
| **Free-text box (optional)** | Parses into a (location, category) pair and routes to the real lookups      |
| **Ranked tables**      | Sorted by computed model output (`gap_residual` / `businesses_per_1000_people`), glass-box — every score shows its inputs and formula |
| **Summary**            | Generative, grounded strictly in the row values (never invented)               |
| **Disclaimer**         | Permanent: gap/opportunity signal from currently-mapped OSM + 2019 census, **not** a profitability/success prediction; Tier-2 counts are floors |

**`dashboard.py`** is the separate pipeline-validation dashboard (five tabs, live
integrity checks) and is still runnable as a second app on another port:

```powershell
venv\Scripts\python -m streamlit run dashboard.py --server.port 8502
```



| Tab                 | What it's for                                                                       |
| ------------------- | ----------------------------------------------------------------------------------- |
| **📈 Overview**     | KPIs (30 locations, 12 categories, businesses mapped), health pills, location table |
| **📍 By location**  | Pick a place → see which categories are most underserved there                      |
| **🏷️ By category** | Pick a category → see which places are most underserved for it                      |
| **✅ Completeness**  | Live integrity checks + coverage heatmap ("M" = never fetched)                      |
| **🗂️ Raw & logs**  | The pipeline's own fetch-failure / join / dedup logs                                |




### A2. CLI queries (the two "product" modes)

Mode 1 — *"I have an area, what should I open here?"*

```powershell
venv\Scripts\python scripts\09_recommend.py --location Kilimani
venv\Scripts\python scripts\09_recommend.py --location Kasarani
```

Mode 2 — *"I want to open a [category], where is it underserved?"*

```powershell
venv\Scripts\python scripts\09_recommend.py --category salon
venv\Scripts\python scripts\09_recommend.py --category supermarket
```

Regenerate the written summary report:

```powershell
venv\Scripts\python scripts\09_recommend.py --summary
```

Every output row carries flags you should know how to read:

- `[tier2]` — informal category; the OSM count is a **floor**, not a true count
- `[fin]` — financing-confound (boda boda stages); entry may be driven by
asset financing, not market gap
- `[bbox]` — this location's OSM data came from a geocoded bounding box,
not an OSM admin boundary (lower fidelity)

---



## Option B — re-run the full pipeline (needs internet)

Run the numbered scripts in order. They are **resumable** — safe to interrupt
and re-run.

```powershell
# 1. Resolve each census location to a geography (OSM admin area or bbox)
venv\Scripts\python scripts\04_resolve_areas.py

# 2. Fetch OSM business nodes for every location × category
#    (skips files already on disk; retries any that previously failed)
venv\Scripts\python scripts\05_fetch_osm_locations.py

# 3. Flatten raw JSON into a clean business table
venv\Scripts\python scripts\06_clean_categorize.py

# 4. Build the SQLite database (locations, businesses, features)
venv\Scripts\python scripts\07_build_db.py

# 5. Assign underserved / moderate / saturated bands per category
venv\Scripts\python scripts\08_cluster.py

# 6. Regenerate the summary report
venv\Scripts\python scripts\09_recommend.py --summary
```

Notes:

- `05` is the slow, network-heavy step (30 locations × 12 categories, ~1s
apart against the public Overpass API). It writes one file per
location/category under `data/raw/locations/` and skips existing files.
- `07` never invents data: a location/category that failed to fetch simply
has **no** feature row (never a fabricated zero-count).
- Scripts `01`–`03` are the original Kilimani-only pilot; they are kept for
reference but are **not** part of the current pipeline.

---



## What "working as intended" looks like — quick sanity checklist

After a run, confirm:

1. **The three integrity checks pass** (also shown live in the dashboard's
  Completeness tab):
  - every feature row has a raw fetch file behind it
  - every raw fetch reached the feature table
  - `business_count` matches the actual business rows
2. **Recommendation output reads sensibly**: rankings are sorted ascending
  by businesses-per-1,000; flags are present; no tracebacks.
3. `fetch_failures.log` — entries here are *cumulative*; a listed failure
  that now has a file on disk was recovered on a later run. Only failures
   with **no** raw file still matter.

---



## Troubleshooting


| Problem                                               | Fix                                                                                                                                          |
| ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `streamlit run` errors "streamlit: command not found" | use `venv\Scripts\python -m streamlit run creative_engine.py`                                                                                 |
| Port 8501 already in use                              | `venv\Scripts\python -m streamlit run creative_engine.py --server.port 8502`                                                                  |
| Overpass fetch fails (HTTP 429 / 504)                 | re-run `05` — it retries; files already fetched are skipped                                                                                  |
| A location shows all zeros                            | it may be a genuine gap, OR the OSM area/bbox is under-mapped — check the Completeness tab heatmap for an "M" (never fetched) vs a real zero |
| Want to force re-resolution of geographies            | `venv\Scripts\python scripts\04_resolve_areas.py --force`                                                                                    |


---



## Project layout (where the data lives)

```
data/external/                census input (nairobi_locations_census.csv)
data/raw/locations/<name>/<category>.json    raw OSM fetch per location×category
data/processed/area_resolutions.json         how each location was geocoded
data/processed/businesses_clean.csv          cleaned business table
data/processed/biz_intel.db                  the SQLite database (the dashboard reads this)
data/processed/*.log                         fetch/join/dedup logs
RESULTS_SUMMARY.md                           generated summary report
creative_engine.py                           the spec dashboard (two query modes)
dashboard.py                                 pipeline-validation dashboard (5 tabs)
app.py                                       earlier dashboard variant (see CHANGELOG)
scripts/01..12                               the pipeline (models: 10-12, summaries.py)
```

