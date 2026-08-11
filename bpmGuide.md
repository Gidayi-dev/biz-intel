# AI-Powered Business Location & Market Gap Intelligence
## Project Guide (Capstone)

---

## 1. Project Goal

Given a location in Kenya (starting pilot: Kilimani/Kileleshwa, Nairobi;
target: national), and a business category, tell the user:

- **How saturated** that category is in that area relative to local
  population/demand
- **Which categories are relatively underserved** in that area

Two output modes, same underlying computation, sorted differently:

1. **Location -> recommendation**: "I have an area, what should I open here?"
2. **Category -> recommendation**: "I want to open a salon, where in Kenya
   is that underserved?"

This is a **gap/opportunity signal**, not a profitability or success
prediction. The model never claims a category will make money or
survive, only that supply looks low relative to demand. That framing
is deliberate, not a limitation to apologize for. See Section 4.

---

## 2. What's Explicitly OUT of Scope (and why)

| Cut | Reason |
|---|---|
| Business survival prediction | Needs longitudinal open/close data we don't have. Also, survival for many ground-level categories (boda boda especially) is driven by financing structure (e.g. lipa mdogo mdogo asset loans), a variable no available dataset captures. Modeling survival without it would be misleading, not just incomplete. |
| Social media scraping as a core data source | Requires solving five separate NLP problems (geolocation extraction, business-type classification, claim-type classification, dedup, Sheng/code-switching) each needing its own labeled data. Too large for this timeline; also structurally biased toward the formal, urban, online-literate segment, which is the opposite of who the problem statement targets. |
| Google Places API as a primary source | Paid past a small free tier, and the free-data mandate (Section 3) rules it out for the core pipeline. May be revisited as an optional cross-check only if budget allows. |
| Business registration bulk data (eCitizen/BRS) | No free bulk export exists at present, lookup-only. |

---

## 3. Data Sources (Free-Only Stack)

| Source | What it gives us | Access method | Status |
|---|---|---|---|
| **OpenStreetMap** | Business/POI locations, tagged by type | Overpass API (pilot, small area) -> Geofabrik `.osm.pbf` national extract (scale-up) | Confirmed free, ready to use |
| **KNBS Population Census (2019)** | Population/household counts per ward/sub-county | Published tables from KNBS | Confirmed free, need to lock exact admin-level granularity |
| **KNBS MSME Survey (2016)** | National/sector-level informality context | KNBS site PDF/tables | Free; likely context only, not row-level modeling data, confirm whether any usable microdata exists |
| **County reports (CIDPs) / KNCCI / MSEA reports** | Narrative context, validation of model output | Manual reading, a few reports only | Not modeling data, used for problem framing and sanity-checking model output against real reporting |
| ~~Kenya Open Data Portal~~ | -- | -- | Treated as unavailable (long offline history) |

**Known structural limitation to state up front, not hide:** OSM
coverage is uneven and skews toward formal, permanent, signed
businesses. Informal categories (vibandaski, mama mboga, boda boda
stages) are structurally under-mapped. A "zero results" reading in
these categories more often means "not mapped" than "doesn't exist."
The model treats informal-category density as a **floor**, not a
true count, and this caveat goes directly into the write-up.

---

## 4. Business Category Taxonomy (Two-Tier)

**Tier 1 - Formal, OSM-reliable** (safe to treat density as roughly
representative): supermarket, pharmacy, restaurant, hairdresser/salon,
clothes shop, hardware/duka.

**Tier 2 - Informal, structurally under-mapped** (treat OSM count as
a floor, flag explicitly in output): mama mboga (`shop=greengrocer`/
`shop=kiosk`), vibandaski (`amenity=fast_food`, often unmapped),
kinyozi (`shop=hairdresser`, same tag as salons, can't distinguish),
mtumba/secondhand clothes, laundromat (`shop=laundry`), cyber cafe,
water vendor, posho mill, boda boda stage (usually unmapped).

**Special flag - financing-confound categories** (boda boda and
similar asset-financed informal work): saturation may not predict
entry behavior, since financing access, not market gap, can be the
dominant driver of who enters. These get a separate note in the
output rather than being scored identically to market-driven
categories.

*(This taxonomy is a living list, expect to revise it once the pilot
OSM pull shows what's actually tagged in practice.)*

---

## 5. Pipeline / Architecture

```
[OSM pull]  ->  [clean + categorize into taxonomy]  -> -+
                                                          |
[Census pull] -> [clean, admin-level match]  -----------+
                                                          |
                                                          v
                                    [spatial join: attach census
                                     figures to OSM points/areas]
                                                          |
                                                          v
                                    [feature table: density-per-
                                     capita by area x category]
                                                          |
                                                          v
                                    [clustering: saturated / moderate
                                     / underserved bands per category]
                                                          |
                                                          v
                                    [recommendation layer: sort by
                                     area (mode 1) or category (mode 2)]
                                                          |
                                                          v
                                    [validation: spot-check against
                                     county reports / local knowledge]
```

---

## 6. Tools & Stack

- **Data pull**: `requests` (Overpass API pilot), `pyrosm` or `osmnx`
  (national `.osm.pbf` extract)
- **Geo processing**: `geopandas`, `shapely` (spatial joins between
  OSM points and census admin polygons)
- **Modeling**: `pandas`/`numpy` for feature tables, `scikit-learn`
  for clustering (start with k-means or DBSCAN, revisit after seeing
  real feature distributions)
- **Validation**: manual review, no special tooling, reading a
  handful of CIDP/KNCCI reports against model output
- **Delivery** (TBD, decide once modeling is working): could be a
  simple dashboard (Streamlit) or a written report with maps, decide
  this once there's something to show, not before

---

## 7. Milestones

| # | Milestone | Deliverable | Depends on |
|---|---|---|---|
| M0 | Project understanding overhaul | This document | -- (done) |
| M1 | Pilot OSM pull | Real data for Kilimani/Kileleshwa, sanity-checked by eye | -- |
| M2 | Census pull + spatial join | Population figures correctly attached to the pilot area | M1 |
| M3 | Feature table | Density-per-capita by category for the pilot area | M2 |
| M4 | Clustering model v1 | Saturated/moderate/underserved bands for pilot area | M3 |
| M5 | Recommendation layer | Both output modes working on pilot data | M4 |
| M6 | Validation pass | Model output checked against Nairobi CIDP / known local conditions | M5 |
| M7 | Scale to national | Swap Overpass pilot pull for Geofabrik national extract, rerun M2-M6 | M6 |
| M8 | Delivery format | Dashboard or report, decided based on what M1-M7 actually produced | M7 |

We are at **M0, about to start M1**. Each milestone should produce
something real and checkable before moving to the next, not a
theoretical plan for the next one.

---

## 8. Open Questions to Resolve at the Meeting

- Exact census admin-level granularity available (ward vs sub-county)
- Whether KNBS MSME 2016 has any row-level microdata or only aggregate PDFs
- Whether MSEA's MSE database has any accessible sample
- What delivery format the evaluators actually expect (dashboard vs report vs both)