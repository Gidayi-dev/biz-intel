"""Merge model features + the three enrichment tables into one wide table.

Base = the `features` table (one row per fetched (location, category) pair),
which carries the regression inputs business_count / predicted_count and the
location population. We left-join, in order:

  - OSM enrichment  (location_id, category) -> competitors / anchors / bus stops
  - WorldPop        (location_id)           -> population_sum/mean/density
  - VIIRS           (location_id)           -> nightlight mean/median

Missing enrichment cells stay NULL (read back as NaN downstream), never 0, so
the recommender can tell "no data" from "zero competitors".

Output: data/processed/enriched_features.csv (+ an `enriched_features` table in
biz_intel.db when the DB is writable).
"""
import sqlite3
import sys

import pandas as pd

from enrichment_common import load_config, write_csv, log_failure
from overpass_client import CATEGORIES

FIELDNAMES = [
    "location_id", "location_name", "category", "tier",
    "financing_confound_flag", "geo_method", "population",
    "business_count", "predicted_count",
    "competitors_100m", "competitors_500m",
    "anchors_market_500m", "bus_stops_500m", "total_pois",
    "population_sum_wp", "pop_mean_wp", "pop_density_wp",
    "viirs_mean", "viirs_median",
]


def load_db_features(cfg) -> pd.DataFrame:
    db_path = cfg["paths"]["db"]
    if not db_path.exists():
        log_failure(f"MERGE: DB not found at {db_path} -- cannot load features")
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    try:
        feat = pd.read_sql_query(
            "SELECT location_id, category, business_count, predicted_count "
            "FROM features", conn)
        locs = pd.read_sql_query(
            "SELECT location_id, location_name, population, geo_method "
            "FROM locations", conn)
    finally:
        conn.close()

    if feat.empty:
        log_failure("MERGE: `features` table is empty -- run 07 + 10 first")
        return pd.DataFrame()

    df = feat.merge(locs, on="location_id", how="left")
    # Attach tier / financing flag from the shared category taxonomy.
    df["tier"] = df["category"].map(
        lambda c: CATEGORIES.get(c, {}).get("tier"))
    df["financing_confound_flag"] = df["category"].map(
        lambda c: int(CATEGORIES.get(c, {}).get("financing_confound", False)))
    return df


def read_optional(cfg, key) -> pd.DataFrame:
    path = cfg["paths"][key]
    if not path.exists():
        log_failure(f"MERGE: {key} missing at {path} -- cells will be NULL")
        return pd.DataFrame()
    return pd.read_csv(path)


def main():
    cfg = load_config()
    out_path = cfg["paths"]["merge_out"]
    db_path = cfg["paths"]["db"]

    df = load_db_features(cfg)
    if df.empty:
        print("No features to merge -- writing an empty header-only CSV.")
        write_csv(out_path, [], FIELDNAMES)
        sys.exit(1)

    osm = read_optional(cfg, "osm_out")
    wp = read_optional(cfg, "worldpop_out")
    viirs = read_optional(cfg, "viirs_out")

    # OSM keyed on (location_id, category); WorldPop/VIIRS on location_id.
    if not osm.empty:
        osm_cols = [c for c in
                    ["location_id", "category", "competitors_100m",
                     "competitors_500m", "anchors_market_500m",
                     "bus_stops_500m", "total_pois"]
                    if c in osm.columns]
        df = df.merge(osm[osm_cols], on=["location_id", "category"], how="left")
    else:
        for c in ["competitors_100m", "competitors_500m",
                  "anchors_market_500m", "bus_stops_500m", "total_pois"]:
            df[c] = pd.NA

    for key, cols in (("worldpop_out",
                       ["population_sum_wp", "pop_mean_wp", "pop_density_wp"]),
                      ("viirs_out", ["viirs_mean", "viirs_median"])):
        src = {"worldpop_out": wp, "viirs_out": viirs}[key]
        if not src.empty:
            keep = [c for c in ["location_id"] + cols if c in src.columns]
            df = df.merge(src[keep], on="location_id", how="left")
        else:
            for c in cols:
                df[c] = pd.NA

    df = df[FIELDNAMES]

    # Write CSV (NULL cells stay empty).
    write_csv(out_path, df.to_dict("records"), FIELDNAMES)

    # Optional: also persist to the DB for dashboard queries.
    try:
        conn = sqlite3.connect(db_path)
        df.to_sql("enriched_features", conn, if_exists="replace", index=False)
        conn.close()
        print(f"Wrote {len(df)} rows -> {out_path} and table `enriched_features`")
    except Exception as exc:
        log_failure(f"MERGE: DB write skipped ({exc})")
        print(f"Wrote {len(df)} rows -> {out_path} (DB table skipped: {exc})")


if __name__ == "__main__":
    main()
