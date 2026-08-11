"""Fetch OSM business nodes for every resolved location x category.

Resumable: each (location, category) response is saved to its own file at
    data/raw/locations/<location>/<category>.json
and already-present files are skipped, so an interrupted run can simply be
re-run to pick up where it left off.

Only "resolved" and "suspect" locations are fetched (see area_resolutions.json).
"failed" locations are skipped and logged -- we never fabricate zero-counts
for a failed location. Each location is fetched by its recorded method:
admin area id (method=area) or geocoded bbox (method=bbox).

Failures are appended to data/processed/fetch_failures.log and are never
written as empty result files (an empty file would look identical to a
confirmed-empty area).
"""
import argparse
import json
import time
from pathlib import Path

from overpass_client import CATEGORIES, fetch_nodes, fetch_nodes_bbox

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "processed" / "area_resolutions.json"
RAW_DIR = ROOT / "data" / "raw" / "locations"
FAIL_LOG = ROOT / "data" / "processed" / "fetch_failures.log"

# Small sleep between every Overpass query, matching 01_fetch_osm.py's
# politeness cadence. We import the constant from the shared client.
from overpass_client import SLEEP_BETWEEN_QUERIES  # noqa: E402


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def fetchable_locations(manifest: dict) -> list[dict]:
    """Locations we are willing to fetch: resolved + suspect."""
    out = []
    for loc, entry in manifest.items():
        if entry.get("status") in ("resolved", "suspect"):
            out.append({"location": loc, "entry": entry})
    return out


def main():
    ap = argparse.ArgumentParser(description="Fetch OSM business nodes per location")
    ap.add_argument("--location", help="only fetch this one location (by census name)")
    ap.add_argument("--category", help="only fetch this one category")
    args = ap.parse_args()

    manifest = load_manifest()
    locations = fetchable_locations(manifest)
    if args.location:
        locations = [l for l in locations if l["location"].lower() == args.location.lower()]
    if args.category:
        assert args.category in CATEGORIES, f"unknown category {args.category}"

    print(f"{len(locations)} locations to fetch x {len(CATEGORIES)} categories")

    failures = []
    if FAIL_LOG.exists():
        failures = FAIL_LOG.read_text(encoding="utf-8").strip().splitlines()

    total = len(locations) * len(CATEGORIES)
    done = 0
    for loc in locations:
        name = loc["location"]
        entry = loc["entry"]
        method = entry.get("method")
        loc_dir = RAW_DIR / name
        loc_dir.mkdir(parents=True, exist_ok=True)
        for category in CATEGORIES:
            done += 1
            out_file = loc_dir / f"{category}.json"
            if out_file.exists():
                continue  # resumable: already fetched
            try:
                if method == "area":
                    elements = fetch_nodes(category, entry["area_id"])
                elif method == "bbox":
                    elements = fetch_nodes_bbox(category, entry["bbox"])
                else:
                    raise RuntimeError(f"unknown geography method {method!r}")
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump({"elements": elements, "method": method}, f)
                print(f"[{done}/{total}] {name} / {category}: {len(elements)}", flush=True)
            except RuntimeError as exc:
                msg = f"{name} / {category}: {exc}"
                print(f"[{done}/{total}] {name} / {category}: FAILED ({exc})", flush=True)
                failures.append(msg)
            time.sleep(SLEEP_BETWEEN_QUERIES)

    if failures:
        FAIL_LOG.write_text("\n".join(failures) + "\n", encoding="utf-8")
        print(f"\n{len(failures)} failures appended to {FAIL_LOG}")
    else:
        print("\nNo failures.")


if __name__ == "__main__":
    main()
