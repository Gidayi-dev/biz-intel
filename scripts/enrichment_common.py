"""Shared, dependency-free helpers for the enrichment + recommender scripts.

Reused by scripts 20 (OSM), 21 (WorldPop), 22 (VIIRS), 23 (merge) and
24 (recommender). Everything here runs with only the standard library, so the
core pipeline (OSM counts, merge, scoring) works even when the optional raster
dependencies (rasterio / rasterstats) are not installed.

Conventions (kept consistent with the rest of the pipeline):
  - location_id is the census row order (1..30), the same id 07_build_db.py
    assigns to the `locations` table -- so enrichment outputs join cleanly
    onto the `features` table keyed by (location_id, category).
  - Missing/undownloadable enrichment cells are written as empty (NULL), never
    a fabricated zero. The recommender drops missing terms and renormalizes.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "enrichment.yml"
CENSUS_CSV = ROOT / "data" / "external" / "nairobi_locations_census.csv"
BIZ_CSV = ROOT / "data" / "processed" / "businesses_clean.csv"
MANIFEST_JSON = ROOT / "data" / "processed" / "area_resolutions.json"

CACHE = ROOT / "data" / "raw" / "enrichment"
OSM_CACHE = CACHE / "osm"
WORLDPOP_CACHE = CACHE / "worldpop"
VIIRS_CACHE = CACHE / "viirs"

FAILURE_LOG = ROOT / "data" / "processed" / "enrichment_failures.log"


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def resolve_path(p):
    """Turn a config path into an absolute Path, relative to the project root."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def _strip_comment(line: str) -> str:
    """Remove a trailing '#' comment, respecting single/double quotes."""
    in_s, quote = False, None
    for i, ch in enumerate(line):
        if ch in "\"'":
            if in_s and quote == ch:
                in_s, quote = False, None
            elif not in_s:
                in_s, quote = True, ch
        elif ch == "#" and not in_s:
            return line[:i].rstrip()
    return line.rstrip()


def _parse_scalar(s: str):
    """Coerce a YAML scalar to bool / None / list / int / float / str."""
    s = s.strip()
    if s == "":
        return ""
    low = s.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "none", "~"):
        return None
    if len(s) >= 2 and s[0] in "\"'" and s[-1] == s[0]:
        return s[1:-1]
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        return [] if not inner else [_parse_scalar(x) for x in inner.split(",")]
    try:
        return float(s) if ("." in s or "e" in low) else int(s)
    except ValueError:
        return s


def _parse_yaml_minimal(text: str) -> dict:
    """Minimal parser for our controlled, two-space-indented YAML subset.

    Handles top-level scalars and one level of indented maps; no anchors,
    flow maps, or multiline strings. PyYAML is used instead when installed
    (see load_config), so this only ever sees our own config file.
    """
    root = {}
    stack = [(root, -1)]
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        content = _strip_comment(raw.strip())
        if ":" not in content:
            continue
        key, _, val = content.partition(":")
        key = key.strip()
        val = val.strip()
        while stack[-1][1] >= indent:
            stack.pop()
        parent, _ = stack[-1]
        if val == "":
            child = {}
            parent[key] = child
            stack.append((child, indent))
        else:
            parent[key] = _parse_scalar(val)
    return root


_CONFIG = None


def load_config(force: bool = False) -> dict:
    """Read config/enrichment.yml, resolving `paths` to absolute Paths.

    Tries PyYAML first (optional fast path), else falls back to the minimal
    parser above -- so no hard dependency on PyYAML.
    """
    global _CONFIG
    if _CONFIG is not None and not force:
        return _CONFIG
    text = CONFIG_PATH.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        cfg = yaml.safe_load(text)
    except Exception:
        cfg = _parse_yaml_minimal(text)
    if "paths" in cfg:
        cfg["paths"] = {k: resolve_path(v) for k, v in cfg["paths"].items()}
    if "raster" in cfg and "cache_dir" in cfg["raster"]:
        cfg["raster"]["cache_dir"] = resolve_path(cfg["raster"]["cache_dir"])
    _CONFIG = cfg
    return cfg


# ---------------------------------------------------------------------------
# small IO / misc helpers
# ---------------------------------------------------------------------------
def write_csv(path, rows, fieldnames):
    """Write a list of dicts to a CSV, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def optional_import(name):
    """Import a module if available, else return None (no exception)."""
    import importlib
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def log_failure(msg: str) -> None:
    """Append a human-readable note to the enrichment failure log."""
    FAILURE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(FAILURE_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in metres between two (lat, lon) points."""
    from math import asin, cos, radians, sin, sqrt
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371000.0 * 2 * asin(sqrt(a))


# ---------------------------------------------------------------------------
# geography helpers
# ---------------------------------------------------------------------------
def load_manifest() -> dict:
    """area_resolutions.json keyed by location name (empty dict if absent)."""
    if not MANIFEST_JSON.exists():
        return {}
    return json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))


def load_locations() -> list[tuple[int, str]]:
    """(location_id, location_name) in census/DB order (id starts at 1)."""
    with open(CENSUS_CSV, encoding="utf-8") as f:
        return [(i, r["location"].strip())
                for i, r in enumerate(csv.DictReader(f), start=1)]


def load_businesses_centroids() -> dict[str, list[tuple[float, float]]]:
    """location name -> list of (lat, lon) for every mapped business.

    This is the offline-safe source of an area-location's commercial centre
    of gravity: the around-N queries in script 20 are centred on where the
    mapped businesses actually are, which is a better anchor for "competitors
    within 500 m" than the centroid of a large admin boundary.
    """
    out = {}
    if not BIZ_CSV.exists():
        return out
    with open(BIZ_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                lat, lon = float(r["lat"]), float(r["lon"])
            except (KeyError, ValueError):
                continue
            out.setdefault(r["location"], []).append((lat, lon))
    return out


def location_geometry(manifest: dict, name: str,
                      businesses_lookup: dict | None = None) -> dict:
    """Resolve a location to {center, bbox, method} for the enrichment queries.

    - bbox locations: centre = midpoint of the stored geocoded box; bbox = the
      stored box (the same extent the original fetch used).
    - area locations: centre + bbox derived from the location's mapped
      businesses (offline-safe; anchors the around-queries on commercial
      activity). If the location has no mapped businesses (e.g. Bahati), both
      are None and the script degrades those cells to NULL.
    """
    entry = manifest.get(name)
    if not entry:
        return {"center": None, "bbox": None, "method": "area"}
    method = entry.get("method", "area")
    if method == "bbox":
        bbox = entry.get("bbox")
        if not bbox:
            return {"center": None, "bbox": None, "method": method}
        s, w, n, e = bbox
        return {"center": ((s + n) / 2.0, (w + e) / 2.0),
                "bbox": bbox, "method": method}
    pts = (businesses_lookup or {}).get(name, [])
    if pts:
        lats = [p[0] for p in pts]
        lons = [p[1] for p in pts]
        return {"center": (sum(lats) / len(lats), sum(lons) / len(lons)),
                "bbox": [min(lats), min(lons), max(lats), max(lons)],
                "method": method}
    return {"center": None, "bbox": None, "method": method}
