"""Flatten per-location raw OSM JSON into a single clean business table.

Reads data/raw/locations/<location>/<category>.json (written by 05) and
writes data/processed/businesses_clean.csv with columns:
  osm_id, location, category, tier, financing_confound_flag, lat, lon,
  name, geo_method

Tier and financing_confound_flag come from the taxonomy in overpass_client.
Rows are deduplicated by osm_id (an OSM node can match two category queries,
e.g. a node tagged both shop=beauty and amenity=fast_food); the first
category seen wins and the conflict is logged.

geo_method records whether the row came from an OSM admin area or a
geocoded bbox, so bbox-sourced rows can be flagged in the final output.
"""
import csv
import json
from pathlib import Path

from overpass_client import CATEGORIES

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "locations"
OUT_CSV = ROOT / "data" / "processed" / "businesses_clean.csv"
CONFLICT_LOG = ROOT / "data" / "processed" / "dedup_conflicts.log"

COLUMNS = ["osm_id", "location", "category", "tier", "financing_confound_flag",
           "lat", "lon", "name", "geo_method"]


def load_raw() -> list[dict]:
    rows = []
    for loc_dir in sorted(RAW_DIR.iterdir()):
        if not loc_dir.is_dir():
            continue
        for raw_file in sorted(loc_dir.glob("*.json")):
            category = raw_file.stem
            if category not in CATEGORIES:
                print(f"  WARN: unknown category file {raw_file.name}, skipping")
                continue
            try:
                payload = json.loads(raw_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                print(f"  WARN: unparseable {raw_file}, skipping")
                continue
            method = payload.get("method", "area")
            meta = CATEGORIES[category]
            for element in payload.get("elements", []):
                tags = element.get("tags", {})
                rows.append({
                    "osm_id": element["id"],
                    "location": loc_dir.name,
                    "category": category,
                    "tier": meta["tier"],
                    "financing_confound_flag": int(meta["financing_confound"]),
                    "lat": element.get("lat"),
                    "lon": element.get("lon"),
                    "name": tags.get("name"),
                    "geo_method": method,
                })
    return rows


def dedup(rows: list[dict]) -> list[dict]:
    """Deduplicate by (location, osm_id), NOT osm_id alone.

    Within a location, one node matching two category queries (e.g. tagged
    both shop=beauty and amenity=fast_food) is kept once. But the same
    business found in TWO different locations' fetches is kept in BOTH.

    This matters because Nairobi census units nest: Roysambu sits inside
    the Kasarani admin boundary, Kayole/Umoja inside Embakasi, Laini Saba
    inside Kibera. A global osm_id dedup attributes every nested business
    to whichever location sorts first alphabetically, silently zeroing the
    nested location's count (Roysambu -> 0, Laini Saba -> 0) even though
    its fetch succeeded. Per-location counts must reflect each location's
    own geography, so the same mapped business is legitimately counted for
    both a ward and the sub-county that contains it.
    """
    seen: dict[tuple[str, int], str] = {}
    conflicts: list[str] = []
    out = []
    for row in rows:
        key = (row["location"], row["osm_id"])
        if key in seen:
            if seen[key] != row["category"]:
                conflicts.append(
                    f"{row['location']}: osm_id {row['osm_id']} seen as {seen[key]} and "
                    f"{row['category']} (kept {seen[key]})"
                )
            continue
        seen[key] = row["category"]
        out.append(row)
    if conflicts:
        CONFLICT_LOG.write_text("\n".join(conflicts) + "\n", encoding="utf-8")
        print(f"  {len(conflicts)} dedup conflicts logged to {CONFLICT_LOG}")
    return out


def main():
    rows = load_raw()
    print(f"Raw rows: {len(rows)}")
    rows = dedup(rows)
    print(f"After dedup: {len(rows)}")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {OUT_CSV}")

    # Quick summary per location x category for eyeballing.
    from collections import Counter
    by_loc = Counter(r["location"] for r in rows)
    print("\nBusinesses per location:")
    for loc, n in sorted(by_loc.items()):
        print(f"  {loc}: {n}")


if __name__ == "__main__":
    main()
