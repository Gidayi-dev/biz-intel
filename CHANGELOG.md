# Changelog

*Documenting what the modeling + dashboard phase extended or changed, so a
reviewer can see the delta against the original pipeline. The Kilimani pilot
scripts `01_fetch_osm.py`–`03_*` are intentionally kept: they are the
human-validated pilot for the approach and the spec requires not deleting
them without explanation (see the "pilot vs pipeline" note in LIMITATIONS.md
Section 8).*

## 2026-08-11 — dashboard rebuilt to spec (glass-box, no mock data)

### `creative_engine.py` — now the canonical spec dashboard
- Rebuilt in place; the old "creative engine" (mock idea generator with random
  seeding, prototype trend chart, invented confidence scores) is **gone**.
- **No mock/placeholder data anywhere.** Every number traces to
  `data/processed/biz_intel.db` (`businesses_per_1000_people`,
  `predicted_count`, `gap_residual`, `cluster_label`).
- **Both required output modes implemented** as real ranked lists:
  Mode 1 location → categories, Mode 2 category → locations, sorted by the
  model's gap/density columns.
- **Glass-box design:** each card shows its inputs + formula
  (population, business_count, predicted_count → gap), so no score is
  unverifiable. Model context comes from `model_eval_regression.json`
  (CV MAE 11.37 — "gaps smaller than the MAE sit within model noise").
- **Tier-2 badges on every card** (`laundry` included — shop=laundry was an
  explicit Tier-2 category); `[fin]` financing-confound flag visibly flagged
  for `motorcycle_taxi`; `[bbox]` flagged per location.
- **Mode 1 input validated against the `locations` table** (dropdown, no
  arbitrary text). Mode 2 category list is the real 12-category taxonomy.
- **Optional free-text box** parses into a (location, category) pair via
  `CATEGORY_ALIASES` ("laundromat"→"laundry") and routes to the real lookups.
- **Generative summary** from `scripts/summaries.py` — grounded strictly in
  real row values, never invented.
- **Permanent visible disclaimer:** gap/opportunity signal based on currently
  mapped OSM data + 2019 census figures, not a profitability/success
  prediction; Tier-2 counts are undercounts (floors).
- "Brainstorm alternatives" section (unvalidated generic startup advice)
  removed from the main flow.
- `dashboard.py` (validation dashboard, 5 tabs) and `app.py` (earlier spec
  attempt) remain as separate apps; RUNBOOK updated so `creative_engine.py`
  is the primary "test as a user" entry.

## 2026-08-10 — trained models, dashboard, docs

### Pipeline scripts added
- **`scripts/10_model_regression.py`** — trained expected-count regression.
  Protocol per spec: Poisson baseline -> Pearson chi2/dof overdispersion check
  (25.0, so negative_binomial chosen) -> Negative Binomial (nb2, BFGS solver),
  GroupKFold(5) grouped by `location_id`, MAE/RMSE (R2 deliberately not used
  for count data). Writes `predicted_count` + `gap_residual` into `features`
  and `data/processed/model_eval_regression.json`. Results: CV MAE 11.37,
  RMSE 26.19, alpha 2.04.
- **`scripts/11_classify.py`** — supervised saturation classifier on a
  hand-labeled set (NOT cluster-derived labels, which would be circular).
  34 pairs / 11 locations, GroupKFold(3), LogisticRegression + StandardScaler.
  Weak honest result (acc 0.44) reported as-is. Writes
  `model_eval_classifier.json`.
- **`scripts/12_nn_compare.py`** — small MLP vs the Negative Binomial GLM on
  the same GroupKFold(5) folds and same MAE/RMSE. MLP MAE 9.40 vs GLM 11.37;
  MLP RMSE 27.04 vs GLM 26.19 — winner depends on the metric, reported
  side-by-side without favoring either. Writes `model_eval_nn.json`.
- **`scripts/summaries.py`** — generative summary layer. Template paragraphs
  grounded strictly in computed numbers (`business_count`,
  `businesses_per_1000_people`, `cluster_label`, `predicted_count`,
  `gap_residual`); sign-aware (surpluses are never called "gaps"),
  tier-2/financing/bbox notes attached, and always ends with the
  gap-not-success caveat. Used by `09_recommend.py`'s results, `app.py`, and
  the creative engine.

### Schema / database (`scripts/07_build_db.py`)
- `features` gained two additive columns: `predicted_count REAL`,
  `gap_residual REAL` (populated by `10_model_regression.py`).
- New table `model_labels (location_id, category, hand_label, labeled_by,
  notes)` created. It exists for spec conformance; the classifier keeps its
  label source in-code (`scripts/11_classify.py`, `HAND_LABELS`) because the
  labels are knowledge-based and not CIDP-cross-checked.
- The `businesses` primary key is `(osm_id, location_id)`, an intentional
  deviation from the spec's single-column `osm_id INTEGER PRIMARY KEY`:
  Nairobi census units nest (Roysambu in Kasarani, Kayole in Embakasi, Laini
  Saba in Kibera), so the same OSM node legitimately belongs to more than one
  unit and must be count-able in each. See LIMITATIONS.md Section 4.

### `scripts/09_recommend.py`
- `write_summary` now appends a **Trained-model evaluation** section (regression,
  classifier, NN) loaded from the eval JSONs, so RESULTS_SUMMARY.md cannot
  drift from the trained outputs.
- `scripts/summaries.py` is the new shared summary source; the CLI ranking
  modes are unchanged.

### Dashboard
- **`app.py`** at the repo root — the spec dashboard. Location selector +
  category selector, ranked gap tables (by model `gap_residual`), generative
  summary per view, tier2/fin/bbox badges, permanent gap-not-success note,
  reads `biz_intel.db` read-only. `dashboard.py` (validation dashboard, 5
  tabs) and `creative_engine.py` (idea engine) remain as separate apps.
- All Streamlit apps use `width="stretch"` (not the deprecated
  `use_container_width`, removed after 2025-12-31).

### Docs
- `RESULTS_SUMMARY.md` regenerated with the trained-model evaluation section.
- `LIMITATIONS.md` extended: spec-vs-reality scope (30 locations / 12
  categories), regression overdispersion/alpha notes, classifier small-sample
  honesty note, NN comparison caveats.

### Fixes
- `10_model_regression.py` stored alpha as `NaN` because statsmodels
  `NegativeBinomialResults` exposes `lnalpha`, not `.alpha`. Now computed as
  `exp(lnalpha)` (2.04) — also fixes the invalid `NaN` token in the JSON.
- Negative Binomial fits use `method="bfgs", maxiter=1000`: the default
  Newton solver stalls on this zero-heavy, overdispersed data (a category
  coefficient blew up to ~ -24 before the fix).
