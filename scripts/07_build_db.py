"""Build data/processed/biz_intel.db from census + cleaned business rows.

Tables (schema from the project spec, with one additive column):
  locations  -- one row per census location (geo_method added to record
                whether the location's OSM data came from an admin area or
                a geocoded bbox, so bbox rows can be flagged downstream).
  businesses -- one row per (mapped business node, location) pair. The PK
                is (osm_id, location_id), not osm_id alone: Nairobi census
                units nest (Roysambu inside Kasarani, Kayole inside
                Embakasi, Laini Saba inside Kibera), so the same business
                legitimately falls inside more than one unit and counts
                toward each unit's own per-capita density (see 06).
  features   -- one row per (location, category): count + density, cluster
                label filled in by 08.

Census name joins are logged: any business location or census row that does
not join cleanly is recorded (not silently dropped).
"""
import csv
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CENSUS_CSV = ROOT / "data" / "external" / "nairobi_locations_census.csv"
BIZ_CSV = ROOT / "data" / "processed" / "businesses_clean.csv"
DB_PATH = ROOT / "data" / "processed" / "biz_intel.db"
JOIN_LOG = ROOT / "data" / "processed" / "join_mismatches.log"

SCHEMA = """
CREATE TABLE IF NOT EXISTS locations (
  location_id INTEGER PRIMARY KEY,
  location_name TEXT,
  sub_county TEXT,
  population INTEGER,
  households INTEGER,
  geo_method TEXT
);
CREATE TABLE IF NOT EXISTS businesses (
  osm_id INTEGER,
  location_id INTEGER REFERENCES locations(location_id),
  category TEXT,
  tier TEXT,
  lat REAL,
  lon REAL,
  name TEXT,
  financing_confound_flag BOOLEAN DEFAULT 0,
  PRIMARY KEY (osm_id, location_id)
);
CREATE TABLE IF NOT EXISTS features (
  location_id INTEGER REFERENCES locations(location_id),
  category TEXT,
  business_count INTEGER,
  businesses_per_1000_people REAL,
  predicted_count REAL,
  gap_residual REAL,
  cluster_label TEXT,
  PRIMARY KEY (location_id, category)
);
CREATE TABLE IF NOT EXISTS model_labels (
  location_id INTEGER,
  category TEXT,
  hand_label TEXT,
  labeled_by TEXT,
  notes TEXT,
  PRIMARY KEY (location_id, category)
);
"""


def load_census() -> list[dict]:
    with open(CENSUS_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_businesses() -> list[dict]:
    with open(BIZ_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    census = load_census()
    businesses = load_businesses()

    # location_name -> geo_method (from area_resolutions.json)
    manifest_path = ROOT / "data" / "processed" / "area_resolutions.json"
    manifest = {}
    if manifest_path.exists():
        import json
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    geo_method = {name: entry.get("method", "area")
                  for name, entry in manifest.items()}

    conn = sqlite3.connect(DB_PATH)
    conn.executescript("DROP TABLE IF EXISTS features;")
    conn.executescript("DROP TABLE IF EXISTS businesses;")
    conn.executescript("DROP TABLE IF EXISTS locations;")
    conn.executescript(SCHEMA)

    loc_id = {}
    for i, row in enumerate(census, start=1):
        name = row["location"].strip()
        loc_id[name] = i
        conn.execute(
            "INSERT INTO locations (location_id, location_name, sub_county, population, households, geo_method)"
            " VALUES (?,?,?,?,?,?)",
            (i, name, row["sub_county"].strip(),
             int(row["population"]), int(row["households_total"]),
             geo_method.get(name)),
        )

    # Insert businesses; log any whose location is not in the census.
    mismatches = []
    inserted = 0
    for row in businesses:
        loc_name = row["location"]
        if loc_name not in loc_id:
            mismatches.append(f"business location {loc_name!r} not in census")
            continue
        conn.execute(
            "INSERT OR REPLACE INTO businesses (osm_id, location_id, category, tier, lat, lon, name, financing_confound_flag)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (int(row["osm_id"]), loc_id[loc_name], row["category"], row["tier"],
             row["lat"], row["lon"], row["name"], int(row["financing_confound_flag"])),
        )
        inserted += 1

    # Features: one row per (location x category) that was actually fetched.
    # A row with business_count=0 means "confirmed empty" only if the fetch
    # succeeded. A fetch that failed (or was never run) produces NO row --
    # never fabricate a zero-count for a location/category we couldn't query.
    # The raw json file's existence is the reliable "this was fetched" signal
    # (05 writes a file only on success and logs failures separately).
    from collections import Counter
    from overpass_client import CATEGORIES

    RAW_DIR = ROOT / "data" / "raw" / "locations"
    per_loc_cat = Counter((row["location"], row["category"]) for row in businesses)
    feature_rows = 0
    skipped = 0
    for name, lid in loc_id.items():
        pop = int(next(r["population"] for r in census if r["location"].strip() == name))
        for category in CATEGORIES:
            fetched = (RAW_DIR / name / f"{category}.json").exists()
            if not fetched:
                skipped += 1
                continue
            count = per_loc_cat.get((name, category), 0)
            per_1000 = count / pop * 1000 if pop else 0.0
            conn.execute(
                "INSERT INTO features (location_id, category, business_count, businesses_per_1000_people)"
                " VALUES (?,?,?,?)",
                (lid, category, count, round(per_1000, 6)),
            )
            feature_rows += 1
    print(f"feature rows: {feature_rows}; (location, category) combos with no fetch: {skipped}")

    conn.commit()

    # Log census rows with no business data at all (for the reviewer's benefit).
    locs_with_biz = {row["location"] for row in businesses}
    for name in loc_id:
        if name not in locs_with_biz:
            mismatches.append(f"census location {name!r} has no fetched business data (failed or empty)")

    if mismatches:
        JOIN_LOG.write_text("\n".join(mismatches) + "\n", encoding="utf-8")
        print(f"{len(mismatches)} join notes logged to {JOIN_LOG}")

    # Verify
    for table in ("locations", "businesses", "features"):
        n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"{table}: {n} rows")
    conn.close()
    print(f"\nWrote {DB_PATH}")


if __name__ == "__main__":
    main()
