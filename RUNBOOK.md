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
has two top-level tabs:

| Tab                          | What it is                                                                 |
| ---------------------------- | -------------------------------------------------------------------------- |
| **Market-gap explorer**      | The original two query modes (below), against `data/processed/biz_intel.db` |
| **Business idea fit**        | The recommendation engine: ranks categories per location, or scores your idea (see below) |

The **Market-gap explorer** tab implements the two required query modes directly
against `data/processed/biz_intel.db`:

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



## Recommendation engine (scripts 20–24)

The market-gap system upgrades into a **transparent recommendation engine** that
ranks, for each Nairobi location, which business category to consider opening
next — from only free public data. Every number in a ranking traces back to a
real column; the per-term contributions are emitted as their own columns so the
arithmetic is hand-checkable.

### The score (glass-box)

```
normalized_gap          = (predicted_count − business_count) z-scored per category   # positive = underserved
normalized_log_pop      = ln(population) z-scored across locations
normalized_competitors  = competitors_500m z-scored per category
normalized_viirs        = viirs_mean z-scored across locations

score = w_gap·n_gap + w_pop·n_log_pop − w_comp·n_competitors + w_viirs·n_viirs
        − penalty_tier2 − penalty_bbox
```

Defaults (configurable in `config/enrichment.yml`): `w_gap=0.6`, `w_pop=0.25`,
`w_comp=0.4`, `w_viirs=0.2`, `penalty_tier2=0.4`, `penalty_bbox=0.3`. The
tier-2 penalty subtracts for informal/under-mapped categories (their OSM counts
are a floor); the bbox penalty subtracts for locations sourced from a geocoded
bounding box rather than an OSM admin boundary (lower fidelity).

### Run the scripts, in order

```powershell
# 20. OSM enrichment: competitors within 100m/500m per category, plus
#     per-location anchors / bus stops / total POIs  (pure requests, no raster)
venv\Scripts\python scripts\20_enrich_osm.py

# 21. WorldPop population enrichment (needs rasterio/rasterstats + internet)
venv\Scripts\python scripts\21_enrich_worldpop.py

# 22. VIIRS nightlights enrichment (needs rasterio/rasterstats + internet)
venv\Scripts\python scripts\22_enrich_viirs.py

# 23. Merge features + the three enrichments into one wide table
venv\Scripts\python scripts\23_merge_enrichments.py

# 24. Rank categories per location; write data/processed/recommendations.csv
venv\Scripts\python scripts\24_recommender_prototype.py
```

Outputs, all under `data/processed/`: `osm_enrichment.csv`,
`worldpop_enrichment.csv`, `viirs_enrichment.csv`, `enriched_features.csv`
(plus an `enriched_features` table in `biz_intel.db` when writable), and
`recommendations.csv`.

### Offline / no-raster behaviour (by design)

- Scripts **21** and **22** need `rasterio` + `rasterstats` *and* internet to
  download the public rasters. If either is missing (the normal offline case)
  they write **NULL** cells and log a `SKIPPED` note — they never crash and
  never fabricate a zero.
- The recommender (24) drops a missing term: a `NaN` normalized term contributes
  `0` and the *available* weights are renormalized so the total weight magnitude
  is preserved. A location missing nightlights is not silently penalised; the
  gap + population terms still carry the ranking.
- Raw OSM responses are cached under `data/raw/enrichment/osm/` (resumable);
  rasters under `data/raw/enrichment/worldpop/` and `.../viirs/`. Results are
  cached locally and never re-published.

### CLI knobs (script 24)

```powershell
# print the top 8 per location instead of 5
venv\Scripts\python scripts\24_recommender_prototype.py --top 8

# just one location
venv\Scripts\python scripts\24_recommender_prototype.py --location Kilimani --top 5

# reweight the four drivers (gap,pop,comp,viirs)
venv\Scripts\python scripts\24_recommender_prototype.py --weights 0.5,0.3,0.5,0.2

# switch the standardization
venv\Scripts\python scripts\24_recommender_prototype.py --normalize minmax
```

`--top` and `--location` only change what is *printed*; `recommendations.csv`
always holds the full ranked table (rank 1..N within each location), so the
dashboard can re-rank under different weights without re-running the scripts.

### In the dashboard — the "Business idea fit" tab

The **Business idea fit** tab reads `data/processed/recommendations.csv`
(cached). If it is absent it shows "run scripts 20–24 first" — the tab never
invents data. It is a **recommendation product** for a person deciding what to
open, built on top of the script-24 score but reworked for a non-technical user.

**Two entry modes** (a radio at the top):

- **"I have no idea yet"** — pick a place, see its categories ranked best-first
  (top-K), each with a verdict badge and the glass-box breakdown.
- **"I have an idea"** — pick a place *and* type a business type (e.g. "salon",
  "boda boda", "supermarket"). It returns a direct verdict for that idea and
  ranks it against the other categories there.

**"Anywhere" location** — the "Where?" box is free text, not a fixed dropdown.
It resolves what you type against the 30 covered Nairobi census areas via
`scripts/verdicts.match_location` (exact → substring → fuzzy). If a place is
**not** covered (e.g. Ruaka, which is in Kiambu County), it says so plainly and
lists the closest covered areas — it never fabricates a match.

**Verdict bands** — every cell is labelled from its score (thresholds fixed in
`scripts/verdicts.py`):

| Verdict | Score | Meaning |
| --- | --- | --- |
| **Strong fit** | ≥ +0.25 | mapped supply well below the model's expectation |
| **Good fit** | 0 … +0.25 | modest positive gap |
| **Weak fit** | −1 … 0 | "less recommended" — near/below expected supply |
| **Not recommended** | < −1 | oversaturated — supply already exceeds expectation |
| **Not fit** | *(override)* | `financing_confound_flag == 1` (e.g. boda boda) — gap signal unreliable |

**Charts** — a ranked horizontal bar chart (colored by verdict) and a
gap-vs-competition quadrant. The quadrant's "sweet spot" is top-left
(underserved *and* little existing supply). Because the 500 m competitor buffer
is `NULL` in the offline case, the quadrant's x-axis falls back to
`business_count` (mapped supply) as the saturation proxy and says so.

Weight sliders are moved into a collapsed "Tune the scoring (advanced)"
expander; moving them re-ranks every score client-side with the same
renormalization as script 24.

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
data/raw/enrichment/                          cached enrichment rasters + raw OSM responses
data/processed/area_resolutions.json         how each location was geocoded
data/processed/businesses_clean.csv          cleaned business table
data/processed/biz_intel.db                  the SQLite database (the dashboard reads this)
data/processed/*.log                         fetch/join/dedup/enrichment logs
data/processed/osm_enrichment.csv            script 20 output
data/processed/worldpop_enrichment.csv       script 21 output (NULL offline)
data/processed/viirs_enrichment.csv          script 22 output (NULL offline)
data/processed/enriched_features.csv         script 23 output (merged wide table)
data/processed/recommendations.csv           script 24 output (ranked, glass-box)
config/enrichment.yml                        weights/penalties/paths for scripts 20-24
RESULTS_SUMMARY.md                           generated summary report
creative_engine.py                           the spec dashboard (two tabs)
dashboard.py                                 pipeline-validation dashboard (5 tabs)
app.py                                       earlier dashboard variant (see CHANGELOG)
scripts/01..12                               the pipeline (models: 10-12, summaries.py)
scripts/enrichment_common.py                 shared helpers for scripts 20-24
scripts/20..24                               enrichment + recommender scripts
scripts/verdicts.py                          score → verdict-band classifier (Business idea fit)
tests/test_verdicts.py                       pins the verdict contract (offline)
```

