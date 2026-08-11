"""Resolve each census location name to a geography for OSM fetching.

The manifest (data/processed/area_resolutions.json) records one of two
methods per location:

  method "area" -- the name matched an OSM administrative boundary
    (area_id, admin_level, probe_count). Used directly for area queries.
  method "bbox" -- no OSM administrative boundary exists for the name;
    Nominatim geocoded it to a bounding box. Bbox entries are lower
    fidelity than admin boundaries and are flagged as such.

Each entry also has a status:
  "resolved" -> fetch normally.
  "suspect"  -> fetch, but flag in LIMITATIONS (boundary may not match the
                census unit, or the geocode matched a coarse type).
  "failed"   -> no usable geography; location is SKIPPED. We never
                fabricate zero-counts for a failed location.

The manifest is manually editable: to override a bad geography, set the
entry's method/area_id/bbox by hand and re-run. Existing entries are not
re-resolved unless --force.
"""
import argparse
import csv
import json
import time
from pathlib import Path

from overpass_client import (
    LOCATION_ADMIN_LEVELS,
    NOMINATIM_SLEEP,
    geocode_bbox,
    post_query,
    probe_area_business_count,
)

ROOT = Path(__file__).resolve().parent.parent
CENSUS_CSV = ROOT / "data" / "external" / "nairobi_locations_census.csv"
MANIFEST = ROOT / "data" / "processed" / "area_resolutions.json"

# Below this many mapped shop/amenity nodes inside an admin boundary the
# resolution is downgraded to "suspect": a location with tens of thousands of
# people that has almost no mapped businesses almost certainly has a bad or
# mismatched boundary (the Kilimani location boundary was found to be empty
# of mapped data even though the surrounding area is well mapped).
SUSPECT_PROBE_THRESHOLD = 5

# Geocode match types that are acceptable as a clean "resolved" bbox.
GOOD_GEOCODE_TYPES = {"suburb", "neighbourhood", "city_district", "commercial"}


def load_location_names() -> list[str]:
    """Read the location column from the census CSV, preserving order."""
    with open(CENSUS_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [row["location"].strip() for row in rows]


def find_area_candidates(name: str) -> list[dict]:
    """Return candidate areas matching a location name, with their tags."""
    query = f"""
    [out:json][timeout:60];
    area["name"="{name}"];
    out tags;
    """
    result = post_query(query)
    candidates = []
    for element in result.get("elements", []):
        if element.get("type") != "area":
            continue
        candidates.append({"area_id": element["id"], "tags": element.get("tags", {})})
    return candidates


def score_candidate(candidate: dict) -> tuple[int, int]:
    """Rank a candidate area. Returns (is_admin, level_penalty).

    is_admin:      1 if boundary=administrative else 0.
    level_penalty: how far admin_level is from the ideal Nairobi location
                   level (7/8). Lower is better. admin_level 6 is a
                   sub-county -- too broad but kept as a fallback.
    """
    tags = candidate["tags"]
    is_admin = 1 if tags.get("boundary") == "administrative" else 0
    level = tags.get("admin_level")
    if is_admin and level:
        try:
            lv = int(level)
        except ValueError:
            lv = 99
        penalty = 0 if lv in (7, 8) else (10 if lv == 6 else 5)
        return (1, penalty)
    return (0, 100)


def resolve_by_area(name: str) -> dict | None:
    """Try to resolve a name to an OSM admin boundary. Returns entry or None."""
    candidates = find_area_candidates(name)
    if not candidates:
        return None
    scored = sorted(candidates, key=score_candidate, reverse=True)
    best = scored[0]
    tags = best["tags"]
    if tags.get("boundary") != "administrative":
        return None  # named features exist, but none is an admin boundary

    area_id = best["area_id"]
    level = tags.get("admin_level")
    probe = probe_area_business_count(area_id)
    if probe < SUSPECT_PROBE_THRESHOLD:
        status, reason = "suspect", (
            f"area exists (admin_level={level}) but contains only {probe} mapped "
            "shop/amenity nodes -- boundary may be empty or not match the census location"
        )
    else:
        status, reason = "resolved", (
            f"admin boundary (admin_level={level}) with {probe} mapped shop/amenity nodes"
        )
    return {
        "method": "area",
        "area_id": area_id,
        "admin_level": level,
        "probe_count": probe,
        "status": status,
        "reason": reason,
        "area_name": tags.get("name"),
    }


def resolve_by_geocode(name: str) -> dict | None:
    """Try to geocode a name to a bbox via Nominatim. Returns entry or None."""
    geo = geocode_bbox(name)
    if geo is None:
        return None
    gtype = geo["geocode_type"]
    if gtype in GOOD_GEOCODE_TYPES:
        status, reason = "resolved", f"geocoded to a {gtype} bbox"
    else:
        status, reason = "suspect", (
            f"geocoded to a {gtype}-type bbox -- coarse match, verify against the census unit"
        )
    return {
        "method": "bbox",
        "bbox": geo["bbox"],
        "geocode_name": geo["geocode_name"],
        "geocode_type": gtype,
        "status": status,
        "reason": reason,
    }


def resolve_location(name: str) -> dict:
    """Resolve a location name: OSM admin boundary first, bbox fallback."""
    try:
        entry = resolve_by_area(name)
        if entry:
            return entry
    except RuntimeError as exc:
        # Overpass is down for this query; fall through to geocoding rather
        # than marking the location failed outright.
        print(f"    (area lookup failed: {exc}; trying geocode)")

    try:
        entry = resolve_by_geocode(name)
        if entry:
            return entry
    except RuntimeError as exc:
        return {"method": None, "status": "failed", "reason": f"geocode failed: {exc}"}

    return {"method": None, "status": "failed",
            "reason": "no OSM admin boundary and no Nairobi geocode match found"}


def main():
    ap = argparse.ArgumentParser(description="Resolve census locations to geographies")
    ap.add_argument("--force", action="store_true", help="re-resolve already-resolved locations")
    args = ap.parse_args()

    names = load_location_names()
    print(f"{len(names)} census locations")

    manifest = {}
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    for i, name in enumerate(names, 1):
        if name in manifest and not args.force:
            print(f"[{i}/{len(names)}] {name}: already resolved, skipping (use --force to redo)")
            continue
        print(f"[{i}/{len(names)}] resolving {name} ...", end=" ", flush=True)
        entry = resolve_location(name)
        entry["location"] = name
        manifest[name] = entry
        status = entry["status"]
        if entry.get("method") == "area":
            detail = f"area {entry['area_id']} (admin_level={entry['admin_level']}, probe={entry['probe_count']})"
        elif entry.get("method") == "bbox":
            detail = f"bbox {entry.get('bbox')} ({entry.get('geocode_type')})"
        else:
            detail = entry["reason"]
        print(f"{status} -> {detail}")
        # pace Nominatim calls (the Overpass client has its own pacing)
        if entry.get("method") == "bbox":
            time.sleep(NOMINATIM_SLEEP)
        # Write incrementally so an interrupted run never loses progress;
        # existing entries are skipped on re-run.
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nWrote {MANIFEST}")

    counts = {"resolved": 0, "suspect": 0, "failed": 0}
    methods = {"area": 0, "bbox": 0}
    for e in manifest.values():
        counts[e["status"]] = counts.get(e["status"], 0) + 1
        if e.get("method"):
            methods[e["method"]] = methods.get(e["method"], 0) + 1
    print("Status summary:", counts)
    print("Method summary:", methods)


if __name__ == "__main__":
    main()
