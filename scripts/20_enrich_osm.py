"""OSM enrichment: competitor density + context counts per (location, category).

For each of the 30 census locations this computes, using only the public
Overpass API (via scripts/overpass_client.py):

  - competitors_100m / competitors_500m: count of the SAME category's nodes
    within 100 m / 500 m of the location's commercial centre (the "around"
    queries).
  - anchors_market_500m: supermarket / mall / department-store nodes within
    500 m (a proxy for an established shopping anchor nearby).
  - bus_stops_500m: bus-stop nodes within 500 m (foot-traffic proxy).
  - total_pois: total shop/amenity nodes over the location's full extent.
    For area-sourced locations this reuses the cached `probe_count` from
    area_resolutions.json (no new query); for bbox-sourced locations it is a
    count over the stored bbox.

Output: data/processed/osm_enrichment.csv with one row per (location, category).

Resumable: each category's raw counts are cached to
data/raw/enrichment/osm/<location>/<category>.json and each location's context
to .../<location>/context.json. Files already on disk are skipped unless
`--force` is passed. A cell that cannot be fetched (offline / endpoint
exhaustion) is written as NULL -- never a fabricated zero -- and only cached
when at least one real value was obtained, so a later re-run retries it.
"""
import argparse
import json
import time
from pathlib import Path

from enrichment_common import (load_config, load_locations, load_manifest,
                               load_businesses_centroids, location_geometry,
                               write_csv, OSM_CACHE, log_failure, ROOT)
from overpass_client import CATEGORIES, post_query, SLEEP_BETWEEN_QUERIES

FIELDNAMES = [
    "location_id", "location_name", "category",
    "competitors_100m", "competitors_500m",
    "anchors_market_500m", "bus_stops_500m", "total_pois",
    "raw_json_cached_path",
]

ANCHOR_TAG = '[shop~"^(supermarket|mall|department_store)$"]'
BUS_STOP_TAG = '[highway=bus_stop]'


def _count_from_result(result: dict) -> int:
    """Overpass `out count` returns one element of type "count"."""
    for el in result.get("elements", []):
        if el.get("type") == "count":
            return int(el["tags"]["total"])
    return 0


def _tag_body(category: str) -> str:
    """CATEGORIES tags look like `node["shop"="x"]`; strip the leading node."""
    return CATEGORIES[category]["tag"].replace("node", "", 1)


def _around_count(lat: float, lon: float, radius: int, tag_body: str) -> int:
    query = (
        "[out:json][timeout:60];\n"
        f"node(around:{radius},{lat:.6f},{lon:.6f}){tag_body};\n"
        "out count;"
    )
    return _count_from_result(post_query(query))


def _bbox_poi_count(bbox) -> int:
    s, w, n, e = bbox
    query = (
        "[out:json][timeout:60];\n"
        f"(node({s},{w},{n},{e})[\"shop\"];\n"
        f" node({s},{w},{n},{e})[\"amenity\"];);\n"
        "out count;"
    )
    return _count_from_result(post_query(query))


def _cached_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_or_fetch_context(loc_name, geo, manifest, force) -> dict:
    """Per-location context (anchors / bus_stops / total_pois), cached once."""
    ctx_path = OSM_CACHE / loc_name / "context.json"
    if not force:
        cached = _cached_json(ctx_path)
        if cached is not None:
            return cached

    ctx = {"anchors_market_500m": None, "bus_stops_500m": None,
           "total_pois": None, "center": geo["center"], "bbox": geo["bbox"],
           "method": geo["method"]}
    got_real = False

    # total_pois: area locations already carry a cached probe_count.
    if geo["method"] == "area":
        pc = manifest.get(loc_name, {}).get("probe_count")
        if pc is not None:
            ctx["total_pois"] = int(pc)
            got_real = True
    elif geo["bbox"]:
        try:
            ctx["total_pois"] = _bbox_poi_count(geo["bbox"])
            got_real = True
            time.sleep(SLEEP_BETWEEN_QUERIES)
        except Exception as exc:
            log_failure(f"OSM total_pois {loc_name}: {exc}")

    center = geo["center"]
    if center:
        lat, lon = center
        try:
            ctx["anchors_market_500m"] = _around_count(lat, lon, 500, ANCHOR_TAG)
            got_real = True
            time.sleep(SLEEP_BETWEEN_QUERIES)
        except Exception as exc:
            log_failure(f"OSM anchors {loc_name}: {exc}")
        try:
            ctx["bus_stops_500m"] = _around_count(lat, lon, 500, BUS_STOP_TAG)
            got_real = True
            time.sleep(SLEEP_BETWEEN_QUERIES)
        except Exception as exc:
            log_failure(f"OSM bus_stops {loc_name}: {exc}")

    # Only persist when we actually obtained a real value; a fully-failed
    # context (offline) is left uncached so a later run retries it.
    if got_real:
        ctx_path.parent.mkdir(parents=True, exist_ok=True)
        ctx_path.write_text(json.dumps(ctx), encoding="utf-8")
    return ctx


def _load_or_fetch_category(loc_name, category, center, near_m, far_m, force):
    """Per-category competitor counts, cached under data/raw/enrichment/osm/."""
    cat_path = OSM_CACHE / loc_name / f"{category}.json"
    if not force:
        cached = _cached_json(cat_path)
        if cached is not None:
            return cached, _rel(cat_path)

    data = {"competitors_100m": None, "competitors_500m": None}
    if center:
        lat, lon = center
        tag_body = _tag_body(category)
        try:
            data["competitors_100m"] = _around_count(lat, lon, near_m, tag_body)
            time.sleep(SLEEP_BETWEEN_QUERIES)
        except Exception as exc:
            log_failure(f"OSM competitors_100m {loc_name}/{category}: {exc}")
        try:
            data["competitors_500m"] = _around_count(lat, lon, far_m, tag_body)
            time.sleep(SLEEP_BETWEEN_QUERIES)
        except Exception as exc:
            log_failure(f"OSM competitors_500m {loc_name}/{category}: {exc}")

    # Cache when we fetched at least one real count, or when there is no centre
    # (deterministic: nothing to query). A fully-failed fetch stays uncached.
    if (not center) or data["competitors_100m"] is not None \
            or data["competitors_500m"] is not None:
        cat_path.parent.mkdir(parents=True, exist_ok=True)
        cat_path.write_text(json.dumps(data), encoding="utf-8")
    return data, _rel(cat_path) if cat_path.exists() else ""


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def main():
    parser = argparse.ArgumentParser(description="OSM enrichment (competitor + context counts)")
    parser.add_argument("--force", action="store_true",
                        help="re-fetch even if cached files exist")
    args = parser.parse_args()

    cfg = load_config()
    near_m = int(cfg["buffers"]["near_m"])
    far_m = int(cfg["buffers"]["far_m"])
    out_path = cfg["paths"]["osm_out"]

    manifest = load_manifest()
    centroids = load_businesses_centroids()

    rows = []
    for loc_id, loc_name in load_locations():
        geo = location_geometry(manifest, loc_name, centroids)
        ctx = _load_or_fetch_context(loc_name, geo, manifest, args.force)
        for category in CATEGORIES:
            data, cached = _load_or_fetch_category(
                loc_name, category, geo["center"], near_m, far_m, args.force)
            rows.append({
                "location_id": loc_id,
                "location_name": loc_name,
                "category": category,
                "competitors_100m": data["competitors_100m"],
                "competitors_500m": data["competitors_500m"],
                "anchors_market_500m": ctx["anchors_market_500m"],
                "bus_stops_500m": ctx["bus_stops_500m"],
                "total_pois": ctx["total_pois"],
                "raw_json_cached_path": cached,
            })

    write_csv(out_path, rows, FIELDNAMES)
    print(f"Wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
