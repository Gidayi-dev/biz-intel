# Known Limitations & Data-Quality Notes

*Companion to RESULTS_SUMMARY.md. This is a gap/opportunity signal, not a
profitability or survival prediction. Every claim below is a documented
caveat a reviewer should read before trusting any specific number. Sections
1-11 cover data and clustering; Sections 12-14 cover the trained models
(regression, classifier, neural network).*

## 0. Dataset scope vs the spec: 30 locations, 12 categories

The spec targeted 25 locations / ~11 categories. The census table that was
actually loaded contains **30 locations** and the pipeline fetches **12
categories** (the 11 listed plus motorcycle_taxi as its own financing-confound
category). The extra locations are nested units (Roysambu in Kasarani, Kayole
in Embakasi, Laini Saba in Kibera, etc.), which the nested-geography handling
(Section 4) accounts for. Nothing was dropped to force the numbers to match
the spec's nominal counts.

---

## 1. Tier-2 categories are structurally under-mapped (the big one)

OSM coverage skews toward formal, permanent, signed businesses. The
taxonomy treats these as Tier 1 and trusts their density as roughly
representative:

> supermarket, pharmacy, restaurant, hairdresser, clothes, hardware

The Tier-2 informal categories below are structurally under-mapped in OSM.
Their counts are a **floor**, not a true count. A low or zero value here
more often means "not mapped" than "doesn't exist":

> salon/beauty (`shop=beauty`), greengrocer/mama mboga (`shop=greengrocer`),
> kiosk (`shop=kiosk`), fast_food/vibandaski (`amenity=fast_food`),
> laundry (`shop=laundry`)

Every output row for these categories carries a `[tier2]` flag. Do not
compare a tier-2 number to a tier-1 number directly; the mapping
probabilities are not comparable. The whole point of the tier split is to
stop a reviewer from reading `kiosk: 0.01/1000` as "no kiosks exist here."

Observed effect on this dataset (30 Nairobi locations x 12 categories,
3,680 mapped business rows):

| category | locations with >=1 mapped business | note |
|---|---|---|
| restaurant | 26/30 | Tier 1, well mapped |
| supermarket | 27/30 | Tier 1, well mapped |
| pharmacy | 25/30 | Tier 1, well mapped |
| hardware | 25/30 | Tier 1 |
| hairdresser | 24/30 | Tier 1 |
| clothes | 20/30 | Tier 1 |
| fast_food | 22/30 | Tier 2 |
| kiosk | 17/30 | Tier 2 |
| greengrocer | 15/30 | Tier 2 |
| salon | 14/30 | Tier 2 |
| laundry | 9/30 | Tier 2, very sparse |
| motorcycle_taxi | 0/30 | see Section 3 |

## 2. Financing-confound category: boda boda is unmapped everywhere

The one category flagged as financing-confound — motorcycle_taxi (boda
boda stage, `amenity=motorcycle_taxi`) — returned **zero mapped businesses
in all 30 locations**. This is consistent with the project guide's note
that boda boda stages are effectively unmapped in OSM.

Consequences:

- The `financing_confound_flag` never fires on real rows (0 of 3,680),
  because there is no mapped data to flag.
- `features` rows for motorcycle_taxi are all zero-count, and `08_cluster`
  correctly left its cluster_label NULL ("too sparse for 3-way clustering").
- **Do not interpret the zero as "no boda boda supply here."** It is a
  total absence of OSM coverage for that tag. If boda boda is in scope for
  the business question, this data source cannot answer it; a survey would
  be needed.
- The financing-confound reasoning still applies to whichever informal
  transport-adjacent categories appear in future data: entry can be driven
  by asset financing (e.g. lipa mdogo mdogo loans), not market gap, so
  density is a weaker signal for those.

## 3. Geography resolution fidelity: 21 of 30 locations are bbox, not admin boundary

`data/processed/area_resolutions.json` records how each census location
name was resolved to a geography. Two methods:

- **area (9 locations):** matched an OSM administrative boundary. Higher
  fidelity. Embakasi, Eastleigh, Umoja, Kasarani, Woodley, Bahati,
  Makadara, Mathare, Njiru.
- **bbox (21 locations):** no admin boundary existed for the name, so
  Nominatim geocoded it to a bounding box. Lower fidelity — the box is a
  geocoder guess at the extent and may over- or under-cover the census
  unit. Every row sourced from a bbox carries a `[bbox]` flag.

80% of mapped business rows (2,961 of 3,680) come from bbox fetches, so
the lower-fidelity method dominates the dataset. This is the single
largest systematic source of location-extent error.

### Locations flagged suspect (fetched, but flagged for LIMITATIONS)

- **Umoja** — matched an admin boundary (admin_level=9) that contained
  only 1 mapped shop/amenity node. Either the boundary is nearly empty or
  it doesn't match the census unit. Its counts should be read with strong
  skepticism.
- **Laini Saba** — Nominatim geocoded it as a *village* type, a coarse
  match for a census location. Fetched via bbox and flagged `[bbox]`.
- **Central** — originally geocoded to a single amenity ("Central bar",
  a bar on Juja Road), which would have zeroed the whole location.
  Manually re-resolved on 2026-08-09 to the **Nairobi Central ward**
  boundary (Starehe) and refetched; 141 businesses recovered. The
  re-resolution is itself a judgment call and should be verified against
  the census definition of the "Central" location.

### Locations confirmed empty (not failures)

- **Bahati** — its admin boundary is valid and contains 35 mapped
  shop/amenity nodes, but they are schools (17), fuel stations (4),
  marketplaces (3), places of worship (2), etc. — **none** in the 12
  target categories. Its zero counts are a genuine "confirmed empty for
  these categories," not a fetch failure. `join_mismatches.log` records
  this as "no fetched business data."

## 4. Nested/overlapping geographies: the same business counts in multiple census units

The census file lists both sub-county-scale units and their constituent
locations (Embakasi vs Kayole/Umoja; Kasarani vs Roysambu; Kibera vs
Laini Saba; Mathare vs Huruma). OSM admin boundaries for the larger units
contain the smaller ones, and some bboxes overlap adjacent areas. Verified
examples: all 95 of Roysambu's mapped nodes also fall inside Kasarani's
boundary; all 30 of Laini Saba's fall inside Kibera's; 13 of Kayole's 18
fall inside Embakasi's.

`06_clean_categorize.py` therefore deduplicates **within** a location
(one node tagged with two category queries is counted once) but keeps the
same business in **both** locations when its two fetch geographies both
contain it. Consequences:

- A shop in Kayole is counted in both Kayole's and Embakasi's totals.
  That is intentional per-unit attribution: the shop serves both units,
  and the per-capita ratio is computed against each unit's own population.
- When comparing two locations, remember their geographies may overlap;
  the counts are not counts of *distinct* businesses across the city.

This was a deliberate fix. The earlier global osm_id dedup silently
zeroed nested locations (Roysambu, Laini Saba showed 0 despite having
data) because the containing boundary's fetch was processed first. See
the `dedup` docstring in `06_clean_categorize.py`.

## 5. Census name joins

All 30 census locations joined cleanly to the resolved-geography manifest
by name; no business row was dropped for an unknown location. One note in
`join_mismatches.log` (Bahati, see Section 3). No "Nairobi West" vs
"Nairobi West Estate" style mismatch occurred — every census name matched
either an OSM area name or a Nominatim geocode.

## 6. Missing land-area data

`data/external/nairobi_locations_census.csv` intentionally contains
**no land-area column** (the KNBS source table had extraction artifacts
for it, and we do not fabricate numbers). Consequently:

- `businesses_per_sqkm` is **not** computed anywhere in the pipeline.
- The `kilimani_features.csv` pilot file contains a
  `business_per_sqkm` figure for Kilimani only; that came from a
  hand-drawn pilot bbox and is not reproduced at scale.

If density-per-area is needed, land area per location is a gap to fill
from another source, not something to invent.

## 7. Fetch reliability history

All 360 (location x category) fetches were eventually completed and saved
(`data/raw/locations/<location>/<category>.json`). The public Overpass
instances had transient failure spells during collection. The events
below are recorded in `data/processed/fetch_failures.log`, which is
append-only history:

- Kasarani / laundry, Woodley / hairdresser, Woodley / motorcycle_taxi,
  Mathare / salon, Mathare / greengrocer, Mathare / laundry,
  Makadara / kiosk — each failed all three endpoints once (last was
  HTTP 504 from the .fr mirror), then succeeded on retry.

Additionally, three Njiru files (clothes, hairdresser, hardware) were
found zero-filled (invalid JSON) after the original collection was
interrupted by the machine shutting down; they were deleted and refetched
successfully on 2026-08-09. Their original counts were recovered (2, 2, 3
respectively).

Final state: **every location x category has a valid fetch file.** No
zero-count in the database is a fabricated "confirmed empty" for a
location/category that actually failed.

## 8. Pilot (Kilimani) vs pipeline counts differ — by design, but be aware

The pilot hand-validated Kilimani with a tight hand-drawn bbox and only 6
categories (`01_fetch_osm.py`, `data/processed/osm_pilot_kilimani_clean.csv`,
`kilimani_features.csv`). The scale-out pipeline resolves Kilimani via a
wider geocoded bbox and fetches 12 categories. So pipeline Kilimani counts
are higher than pilot counts (e.g. 111 pharmacies mapped vs 12 in the
pilot). This is expected — different fetch extents — and is not a bug.
The pilot remains the human-validated reference for the *approach*; the
pipeline numbers supersede it.

## 9. Clustering method & sparse categories

Per category, locations are binned into terciles of
`businesses_per_1000_people` (bottom = underserved, middle = moderate,
top = saturated). Terciles were chosen over k-means because the
distributions are zero-heavy and right-skewed, and terciles are robust to
that shape and trivially explainable. Diagnostics:
`data/processed/cluster_diagnostics.csv`.

Caveats:

- **laundry** has only 9/30 locations with any mapped business and its
  tercile bounds collapse to 0 / 0. The "moderate" band is empty; the
  labels reduce to "any mapped laundromat = saturated". Treat laundry
  bands as ordinal ("has vs hasn't"), not as three meaningful levels.
- **motorcycle_taxi** is unclustered (Section 2).
- **salon, kiosk, greengrocer** are thin (14, 17, 15 of 30 non-zero);
  their terciles are driven mostly by zeros vs a small number of mapped
  businesses. Read their bands with the tier-2 caveat (Section 1) on top.

## 10. Small counts are noisy

Per-1000 ratios on small counts are noisy. A single mapped restaurant in
a location of ~30k people moves the ratio by ~0.03/1000; the differences
between adjacent ranks in a ranking table are often within one business's
worth of data. Rankings are a *directional* signal, not a precise
ordering.

## 11. What this model does NOT claim

- It does not predict profitability, survival, or success of any business.
  The guide (Section 2 of bpmGuide.md) excludes this explicitly: survival
  needs longitudinal open/close data we don't have, and for several
  ground-level categories entry is driven by financing access, not market
  gap.
- It does not claim a low density guarantees a good location. It reports
  only that *supply looks low relative to the local population*.
- It does not claim OSM counts are complete for any category; tier-1
  counts are roughly representative, tier-2 counts are a floor.

## 12. Trained regression: overdispersion and model choice

- Counts are zero-heavy and right-skewed. The Poisson baseline failed its
  overdispersion test decisively (Pearson chi2/dof = 25.0), so the final
  model is a Negative Binomial (nb2) with estimated dispersion
  alpha = 2.04. Alpha is stored as `lnalpha` in statsmodels; the pipeline
  converts it with `exp(lnalpha)` (see CHANGELOG).
- GroupKFold(5) grouped by location_id: rows sharing a location share the
  same population/households, so a random split would leak. With 360 rows
  across 30 locations, CV MAE = 11.37, RMSE = 26.19. MAE is the headline;
  R2 is not reported for count data (misleading at skewed distributions).
- **The intercept and category shifts are log-scale rates.** `motorcycle_taxi`
  has an implied rate ~24 log-points below the supermarket reference
  because it is unmapped everywhere (Section 2), not because there is
  evidence of high latent demand. Read category-rate shifts with that
  caveat.
- `predicted_count` and `gap_residual` written into `features` come from a
  model refit on all 360 rows (the same protocol's final step), so they are
  in-sample values, not CV predictions. The CV numbers above are the honest
  generalization estimate; the in-sample residuals are for *ranking*, not
  for claiming CV-quality accuracy on individual cells.

## 13. Supervised classifier: tiny labeled set, weak result, reported honestly

- The classifier does NOT use `features.cluster_label` (that would be
  circular — the label was derived from the same density data). Supervision
  is a hand-labeled set of 34 `(location, category)` pairs across 11
  locations, labeled **assistant domain knowledge informed by mapped
  density and Nairobi market context — NOT cross-checked against a county
  CIDP report** (none was available in the repo).
- GroupKFold(3) CV: accuracy 0.44, macro precision 0.40 / recall 0.41 /
  F1 0.40. The moderate class is essentially unrecoverable (recall 0.10);
  underserved (16/34) and saturated (8/34) separate somewhat better. This
  is a weak exploratory result and is reported as such rather than tuned
  into a misleading number.
- Because tier-2 labels already absorb the OSM undercount, the classifier
  must not be read as predicting true informal-market saturation.

## 14. Glass-box vs black-box comparison

- A small MLP (16 hidden units, log1p target, early stopping) was trained
  under the SAME GroupKFold(5) folds and scored on the same MAE/RMSE as the
  Negative Binomial GLM. Results: MLP MAE 9.40 vs GLM MAE 11.37; MLP RMSE
  27.04 vs GLM RMSE 26.19. The winner depends on the metric; neither model
  is favored in the write-up.
- The MLP's log1p back-transform and early stopping make its error profile
  different from the GLM's (better on average error, worse on tail cells).
  With 360 rows the difference is within what hyper-parameter luck could
  produce; the honest statement is "both are within a few MAE points".
- MLP training is stochastic; `random_state=42` makes the run reproducible,
  but a different seed would move the MLP numbers slightly.
