"""Recommendation layer: two query modes against the features table.

Mode 1 -- location -> categories: given a location, rank categories by how
  underserved they are there (lowest businesses-per-1000 first).
Mode 2 -- category -> locations: given a category, rank all 30 locations by
  how underserved they are for it.

Every row is flagged:
  [tier2]  -- informal category; OSM count is a FLOOR, not a true count.
  [fin]    -- financing-confound category (boda boda stages); entry may be
              driven by asset financing, not market gap.
  [bbox]   -- location's OSM data came from a geocoded bbox, not an admin
              boundary; lower fidelity.

This is a gap/opportunity signal only. Never phrase output as a success or
profitability prediction.

CLI:
  python 09_recommend.py --location Kilimani
  python 09_recommend.py --category salon
  python 09_recommend.py --summary   # write RESULTS_SUMMARY.md
"""
import argparse
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "processed" / "biz_intel.db"
SUMMARY_PATH = ROOT / "RESULTS_SUMMARY.md"

FLAG_TIER2 = " [tier2]"
FLAG_FIN = " [fin]"
FLAG_BBOX = " [bbox]"

# Tier-2 categories: informal, under-mapped -- count is a floor.
TIER2 = {"salon", "greengrocer", "kiosk", "fast_food", "laundry", "motorcycle_taxi"}
# Financing-confound categories.
CONFOUND = {"motorcycle_taxi"}


def _fmt(label: str, flags: list[str]) -> str:
    return label + "".join(flags)


def recommend_categories_for_location(conn: sqlite3.Connection, location: str, limit: int = 12):
    """Rank categories for one location, most-underserved first."""
    rows = conn.execute(
        """
        SELECT f.category, f.business_count, f.businesses_per_1000_people, f.cluster_label,
               l.geo_method
        FROM features f JOIN locations l USING (location_id)
        WHERE l.location_name = ?
        ORDER BY f.businesses_per_1000_people ASC
        """, (location,)
    ).fetchall()
    if not rows:
        print(f"No data for location {location!r}")
        return []

    print(f"\n{location} -- categories ranked most-underserved first:")
    print(f"{'category':<15} {'count':>6} {'per_1000':>9} {'band':<12} flags")
    out = []
    for category, count, per_1000, cluster, geo in rows:
        flags = []
        if category in TIER2:
            flags.append(FLAG_TIER2)
        if category in CONFOUND:
            flags.append(FLAG_FIN)
        if geo == "bbox":
            flags.append(FLAG_BBOX)
        band = cluster or "n/a"
        print(f"{_fmt(category, flags):<24} {count:>6} {per_1000:>9.2f} {band:<12}")
        out.append((category, count, per_1000, cluster, flags))
    return out[:limit]


def recommend_locations_for_category(conn: sqlite3.Connection, category: str, limit: int = 10):
    """Rank locations for one category, most-underserved first."""
    rows = conn.execute(
        """
        SELECT l.location_name, f.business_count, f.businesses_per_1000_people,
               f.cluster_label, l.geo_method
        FROM features f JOIN locations l USING (location_id)
        WHERE f.category = ?
        ORDER BY f.businesses_per_1000_people ASC
        """, (category,)
    ).fetchall()
    if not rows:
        print(f"No data for category {category!r}")
        return []

    flags = []
    if category in TIER2:
        flags.append(FLAG_TIER2)
    if category in CONFOUND:
        flags.append(FLAG_FIN)

    print(f"\n{_fmt(category, flags)} -- locations ranked most-underserved first:")
    print(f"{'location':<18} {'count':>6} {'per_1000':>9} {'band':<12} flags")
    out = []
    for loc_name, count, per_1000, cluster, geo in rows:
        loc_flags = list(flags)
        if geo == "bbox":
            loc_flags.append(FLAG_BBOX)
        band = cluster or "n/a"
        print(f"{_fmt(loc_name, loc_flags):<26} {count:>6} {per_1000:>9.2f} {band:<12}")
        out.append((loc_name, count, per_1000, cluster, loc_flags))
    return out[:limit]


def _modeling_section() -> list[str]:
    """Modeling headline results, loaded from the eval JSONs so the report
    can never drift from the trained models' actual outputs."""
    import json

    lines = ["## Trained-model evaluation", ""]
    files = {
        "regression": ROOT / "data" / "processed" / "model_eval_regression.json",
        "classifier": ROOT / "data" / "processed" / "model_eval_classifier.json",
        "nn": ROOT / "data" / "processed" / "model_eval_nn.json",
    }
    loaded = {}
    for key, path in files.items():
        if path.exists():
            try:
                loaded[key] = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                loaded[key] = None

    reg = loaded.get("regression")
    if reg:
        lines.append(
            f"- **Count regression ({reg.get('model_type', 'n/a')}, "
            f"n={reg['n_obs']}):** overdispersion test Pearson chi2/dof = "
            f"{reg['pearson_chi2_per_dof']:.1f} (much > 1) -> Negative "
            f"Binomial used. Dispersion alpha = {reg['alpha']:.2f}. "
            f"GroupKFold(5) CV MAE = {reg['cv_mae']:.2f}, "
            f"RMSE = {reg['cv_rmse']:.2f} (MAE is the headline; R2 is not used "
            f"for count data). Categories with the lowest implied rate vs "
            f"supermarket: motorcycle_taxi, laundry, salon, greengrocer."
        )
        lines.append("")
    else:
        lines.append("- _(regression eval file missing)_")
        lines.append("")

    cls = loaded.get("classifier")
    if cls:
        lines.append(
            f"- **Supervised saturation classifier (hand-labeled):** "
            f"{cls['n_labels']} labeled (location, category) pairs across "
            f"{cls['n_locations']} locations. GroupKFold(3) CV: accuracy "
            f"{cls['accuracy']:.2f}, macro precision {cls['precision_macro']:.2f} "
            f"/ recall {cls['recall_macro']:.2f} / F1 {cls['f1_macro']:.2f}. "
            f"The labeled set is small (34 pairs) and class imbalance is "
            f"severe; treat this as a weak exploratory result, reported "
            f"honestly (labels are knowledge-based, not county-CIDP "
            f"cross-checked)."
        )
        lines.append("")
    else:
        lines.append("- _(classifier eval file missing)_")
        lines.append("")

    nn = loaded.get("nn")
    if nn and reg:
        lines.append(
            f"- **Glass-box vs black-box (same folds, same metrics):** "
            f"MLP (16-unit) CV MAE = {nn['mlp_cv_mae']:.2f} vs "
            f"Negative Binomial GLM CV MAE = {reg['cv_mae']:.2f}; "
            f"MLP RMSE = {nn['mlp_cv_rmse']:.2f} vs GLM RMSE = "
            f"{reg['cv_rmse']:.2f}. The winner depends on the metric -- "
            f"reported as-is, no model is favored."
        )
        lines.append("")
    else:
        lines.append("- _(NN eval file missing)_")
        lines.append("")

    return lines


def write_summary(conn: sqlite3.Connection) -> None:
    """Generate RESULTS_SUMMARY.md: top underserved per a few locations and
    per a few categories, plus trained-model evaluation highlights."""
    notable_locations = ["Kilimani", "Kasarani", "Embakasi", "Kibera", "Eastleigh", "Karen"]
    notable_categories = ["supermarket", "pharmacy", "restaurant", "salon", "kiosk"]

    lines = [
        "# Nairobi Business Market-Gap Results Summary",
        "",
        "*Generated from data/processed/biz_intel.db by scripts/09_recommend.py. "
        "This is a gap/opportunity signal, not a profitability or survival prediction.*",
        "",
        "**Legend for flags:** `[tier2]` informal category - the OSM count is a "
        "floor, not a true count (salon, greengrocer, kiosk, fast_food, laundry, "
        "motorcycle_taxi). `[fin]` financing-confound category (boda boda stages) - "
        "entry may be driven by asset financing, not market gap. `[bbox]` location "
        "fetched from a geocoded bounding box, not an OSM admin boundary - lower "
        "fidelity. Numbers are businesses per 1,000 people. Full caveats in "
        "LIMITATIONS.md.",
        "",
    ]

    lines.append("## Most-underserved categories in notable locations")
    lines.append("")
    for loc in notable_locations:
        found = [r for r in conn.execute(
            "SELECT f.category, f.businesses_per_1000_people, l.geo_method "
            "FROM features f JOIN locations l USING (location_id) "
            "WHERE l.location_name = ? ORDER BY f.businesses_per_1000_people ASC", (loc,)
        ).fetchall()]
        if not found:
            lines.append(f"### {loc}")
            lines.append("_(no data)_")
            lines.append("")
            continue
        lines.append(f"### {loc}")
        for category, per_1000, geo in found[:5]:
            flags = []
            if category in TIER2:
                flags.append(FLAG_TIER2)
            if category in CONFOUND:
                flags.append(FLAG_FIN)
            if geo == "bbox":
                flags.append(FLAG_BBOX)
            lines.append(f"- {category} ({per_1000:.2f}/1000){''.join(flags)}")
        lines.append("")

    lines.append("## Most-underserved locations in notable categories")
    lines.append("")
    for category in notable_categories:
        found = [r for r in conn.execute(
            "SELECT l.location_name, f.businesses_per_1000_people, l.geo_method "
            "FROM features f JOIN locations l USING (location_id) "
            "WHERE f.category = ? ORDER BY f.businesses_per_1000_people ASC", (category,)
        ).fetchall()]
        lines.append(f"### {category}")
        if not found:
            lines.append("_(no data)_")
            lines.append("")
            continue
        for loc, per_1000, geo in found[:5]:
            flags = []
            if category in TIER2:
                flags.append(FLAG_TIER2)
            if category in CONFOUND:
                flags.append(FLAG_FIN)
            if geo == "bbox":
                flags.append(FLAG_BBOX)
            lines.append(f"- {loc} ({per_1000:.2f}/1000){''.join(flags)}")
        lines.append("")

    lines.extend(_modeling_section())
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {SUMMARY_PATH}")


def main():
    ap = argparse.ArgumentParser(description="Recommendation queries against the features table")
    ap.add_argument("--location", help="rank categories for this location")
    ap.add_argument("--category", help="rank locations for this category")
    ap.add_argument("--summary", action="store_true", help="write RESULTS_SUMMARY.md")
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)
    if args.location:
        recommend_categories_for_location(conn, args.location)
    elif args.category:
        recommend_locations_for_category(conn, args.category)
    elif args.summary:
        write_summary(conn)
    else:
        ap.print_help()
    conn.close()


if __name__ == "__main__":
    main()
