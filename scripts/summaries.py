"""Generative summary layer for the market-gap dashboard.

Plain-language, template-generated paragraphs for a location or a category,
grounded STRICTLY in computed numbers from the features table:
  business_count, businesses_per_1000_people, cluster_label,
  predicted_count, gap_residual.

Rules (from the project spec):
  - This is a gap/opportunity SIGNAL from mapped OSM density. Never phrase
    output as a profitability, survival, or success prediction.
  - Tier-2 categories are under-mapped: their counts are a FLOOR. State that
    where it matters.
  - Financing-confound categories (motorcycle_taxi) can be driven by asset
    financing, not market gap.
  - bbox locations are lower-fidelity (geocoded box, not admin boundary).
  - No fabricated numbers: every value in the text comes from the DB.

Used by scripts/09_recommend.py (RESULTS_SUMMARY), app.py (dashboard), and
the creative engine. Nothing here makes requests; it only reads the DB.
"""
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "processed" / "biz_intel.db"

TIER2 = {"salon", "greengrocer", "kiosk", "fast_food", "laundry", "motorcycle_taxi"}
CONFOUND = {"motorcycle_taxi"}

CAVEAT = ("This is a mapped-density gap/opportunity signal, not a demand, "
          "profitability, or survival forecast.")

FLAG_TIER2 = " [tier2]"
FLAG_FIN = " [fin]"
FLAG_BBOX = " [bbox]"


def flags_for(category: str, geo_method: str | None) -> list[str]:
    """Structured flags for a feature row (tier2 / financing / bbox fidelity)."""
    flags = []
    if category in TIER2:
        flags.append(FLAG_TIER2)
    if category in CONFOUND:
        flags.append(FLAG_FIN)
    if geo_method == "bbox":
        flags.append(FLAG_BBOX)
    return flags


def _conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def _human(category: str) -> str:
    return category.replace("_", " ")


def _feature_rows(conn: sqlite3.Connection, location: str | None = None,
                  category: str | None = None) -> list[dict]:
    """Rows with every number the summaries cite, plus location metadata."""
    where, params = [], []
    if location is not None:
        where.append("l.location_name = ?")
        params.append(location)
    if category is not None:
        where.append("f.category = ?")
        params.append(category)
    sql = (
        "SELECT f.location_id, f.category, f.business_count, "
        "       f.businesses_per_1000_people, f.cluster_label, "
        "       f.predicted_count, f.gap_residual, "
        "       l.location_name, l.population, l.geo_method "
        "FROM features f JOIN locations l USING (location_id)"
    )
    if where:
        sql += " WHERE " + " AND ".join(where)
    rows = conn.execute(sql, params).fetchall()
    out = []
    for (lid, cat, count, per_1000, cluster, pred, gap, loc_name,
         pop, geo) in rows:
        out.append({
            "location_id": lid,
            "location": loc_name,
            "category": cat,
            "count": int(count or 0),
            "per_1000": float(per_1000 or 0.0),
            "predicted": float(pred) if pred is not None else None,
            "gap": float(gap) if gap is not None else None,
            "band": cluster or "n/a",
            "population": int(pop or 0),
            "geo_method": geo,
            "flags": flags_for(cat, geo),
        })
    return out


def _ranked_by_gap(rows: list[dict]) -> list[dict]:
    """Rank by model gap residual (most negative = most scarce) first."""
    return sorted(rows, key=lambda r: r["gap"] if r["gap"] is not None
                  else float("inf"))


def summarize_location(conn: sqlite3.Connection, location: str,
                       top_n: int = 5) -> dict:
    """Plain-language summary of the most-underserved categories in a location."""
    rows = _feature_rows(conn, location=location)
    if not rows:
        return {"subject": location, "mode": "location", "text": "",
                "rows": [], "total_mapped": 0, "total_predicted": None,
                "total_gap": None}

    ranked = _ranked_by_gap(rows)
    unders = [r for r in ranked if r["gap"] is not None and r["gap"] < -0.05]
    has_underserved = bool(unders)
    # Present the genuine gaps; if a location has none, show the smallest
    # surpluses (closest to parity) instead -- never call a surplus a "gap".
    shown = (unders or ranked)[:top_n]

    # Aggregate across all fetched categories for this location.
    total_mapped = sum(r["count"] for r in rows)
    preds = [r["predicted"] for r in rows if r["predicted"] is not None]
    total_predicted = sum(preds) if preds else None
    total_gap = total_mapped - total_predicted if total_predicted is not None else None

    geo = rows[0]["geo_method"]
    geo_note = (" (note: OSM data for this location came from a geocoded "
                "bounding box, so counts are lower-fidelity)" if geo == "bbox"
                else "")

    frags = [_gap_fragment(r, _human(r["category"])) for r in shown]

    if has_underserved:
        text = (f"In {location}, the widest mapped gaps vs the model's "
                f"expected count are: {'; '.join(frags)}.{geo_note} ")
    else:
        text = (f"{location} has no category below its expected count on "
                f"mapped data; the categories closest to their expected count "
                f"are: {'; '.join(frags)}.{geo_note} ")
    if total_predicted is not None:
        text += _aggregate_sentence(total_mapped, total_predicted, total_gap,
                                    len(rows))
    text += CAVEAT
    return {"subject": location, "mode": "location", "text": text, "rows": shown,
            "total_mapped": total_mapped,
            "total_predicted": round(total_predicted, 1) if total_predicted is not None else None,
            "total_gap": round(total_gap, 1) if total_gap is not None else None}


def summarize_category(conn: sqlite3.Connection, category: str,
                       top_n: int = 5) -> dict:
    """Plain-language summary of the most-underserved locations for a category."""
    rows = _feature_rows(conn, category=category)
    if not rows:
        return {"subject": category, "mode": "category", "text": "",
                "rows": [], "total_mapped": 0, "total_predicted": None,
                "total_gap": None}

    ranked = _ranked_by_gap(rows)
    unders = [r for r in ranked if r["gap"] is not None and r["gap"] < -0.05]
    has_underserved = bool(unders)
    shown = (unders or ranked)[:top_n]

    total_mapped = sum(r["count"] for r in rows)
    preds = [r["predicted"] for r in rows if r["predicted"] is not None]
    total_predicted = sum(preds) if preds else None
    total_gap = total_mapped - total_predicted if total_predicted is not None else None

    frags = [_gap_fragment(r, r["location"]) for r in shown]

    lead = ""
    if " [fin]" in flags_for(category, None):
        lead = ("This category can be driven by asset financing, not market "
                "gap; treat the signal with caution. ")
    elif " [tier2]" in flags_for(category, None):
        lead = ("This is a tier-2 (informal) category: mapped counts are a "
                "floor, so the true gap is at least as large as shown. ")

    if has_underserved:
        text = (f"For {_human(category)}, the most-underserved locations by "
                f"mapped gap are: {'; '.join(frags)}. " + lead)
    else:
        text = (f"No location is below its expected count for "
                f"{_human(category)} on mapped data; the locations closest to "
                f"parity are: {'; '.join(frags)}. " + lead)
    if total_predicted is not None:
        text += _aggregate_sentence(total_mapped, total_predicted, total_gap,
                                    len(rows))
    text += CAVEAT
    return {"subject": category, "mode": "category", "text": text, "rows": shown,
            "total_mapped": total_mapped,
            "total_predicted": round(total_predicted, 1) if total_predicted is not None else None,
            "total_gap": round(total_gap, 1) if total_gap is not None else None}


def _gap_fragment(r: dict, name: str) -> str:
    """One grounded fragment: '<name> (mapped N, P.PP/1,000, gap G.G)'. Notes
    financing confound before the tier-2 floor, then bbox fidelity."""
    gap_str = f"{r['gap']:+.1f}" if r["gap"] is not None else "n/a"
    note = ""
    if " [fin]" in r["flags"]:
        note = "; financing-confound category"
    elif " [tier2]" in r["flags"]:
        note = "; tier-2 category is under-mapped, count is a floor"
    if " [bbox]" in r["flags"]:
        note += " (lower-fidelity geocoded box)"
    return (f"{name} (mapped {r['count']}, "
            f"{r['per_1000']:.2f}/1,000 people, gap {gap_str}{note})")


def _aggregate_sentence(total_mapped: int, total_predicted: float,
                        total_gap: float, n_cells: int) -> str:
    """Net-position sentence with the sign spelled out in words."""
    if total_gap < 0:
        pos = f"a net shortfall of {abs(total_gap):.1f}"
    else:
        pos = f"mapped density is {total_gap:.1f} above the expected count"
    return (f"Across {n_cells} fetched cells, {total_mapped} businesses are "
            f"mapped against an expected {total_predicted:.1f} -- {pos}. ")


def aggregate_metrics(conn: sqlite3.Connection) -> dict:
    """Whole-dataset headline numbers for the dashboard header."""
    n_locs = conn.execute("SELECT COUNT(*) FROM locations").fetchone()[0]
    n_cats = conn.execute("SELECT COUNT(DISTINCT category) FROM features").fetchone()[0]
    n_biz = conn.execute("SELECT COUNT(*) FROM businesses").fetchone()[0]
    n_feat = conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    n_pred = conn.execute(
        "SELECT COUNT(*) FROM features WHERE predicted_count IS NOT NULL").fetchone()[0]
    return {
        "n_locations": int(n_locs),
        "n_categories": int(n_cats),
        "n_businesses": int(n_biz),
        "n_feature_rows": int(n_feat),
        "n_predicted_rows": int(n_pred),
    }


def demo() -> None:
    conn = _conn()
    try:
        print("== aggregate ==")
        print(aggregate_metrics(conn))
        print("\n== location summary: Kilimani ==")
        print(summarize_location(conn, "Kilimani")["text"])
        print("\n== location summary: Kibera ==")
        print(summarize_location(conn, "Kibera")["text"])
        print("\n== category summary: supermarket ==")
        print(summarize_category(conn, "supermarket")["text"])
        print("\n== category summary: salon ==")
        print(summarize_category(conn, "salon")["text"])
    finally:
        conn.close()


if __name__ == "__main__":
    demo()
