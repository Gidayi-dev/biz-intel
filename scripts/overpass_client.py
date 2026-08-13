"""Shared Overpass API client + category taxonomy.

Reused by 04_resolve_areas.py and 05_fetch_osm_locations.py.

Endpoint fallback chain (verified live 2026-08-08):
  - overpass-api.de         : works with a User-Agent header, current data -> primary
  - overpass.kumi.systems   : works, but ~3 months stale data           -> fallback
  - overpass.openstreetmap.fr : was down (dispatcher off) when tested    -> last resort

The original 01_fetch_osm.py used only the .fr mirror. We keep that as the
last fallback rather than the only option.
"""
import time

import requests

USER_AGENT = "biz-intel-capstone/0.1 (student project)"

ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]

# Nominatim (free geocoding) for the bbox fallback. Requires a UA header and
# ~1 req/sec pacing. The viewbox biases ranking toward Nairobi.
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_VIEWBOX = "36.5,-1.45,37.1,-1.1"  # Nairobi metro, used as a bias
NOMINATIM_SLEEP = 1.1  # seconds between geocoding calls

SLEEP_BETWEEN_QUERIES = 1.0  # seconds, matching the polite cadence of 01_fetch_osm.py
RETRIES_PER_ENDPOINT = 1  # attempts per endpoint beyond the first
# (connect, read) timeout tuple: fail fast if the server won't accept the
# connection, cap slow reads so a degraded Overpass spell doesn't stall the
# whole pipeline for minutes per call.
TIMEOUT = (10, 45)

# ---------------------------------------------------------------------------
# Category taxonomy (from the project spec -- do not flatten the two tiers)
#
# Tier 1: formal, OSM-reliable. Density treated as roughly representative.
# Tier 2: informal, structurally under-mapped. OSM count is a FLOOR, not a
#         true count; flagged as such in every output row.
# financing-confound: entry may be driven by asset financing rather than
#         market gap (boda boda stages). Scored the same way but always
#         flagged.
# ---------------------------------------------------------------------------
CATEGORIES = {
    # ---------------------------------------------------------------------
    # Kenyan MSME / micro-business taxonomy.
    #
    # These are the business forms that actually START in Nairobi
    # locations and estates, with capital measured in thousands of
    # shillings, not the formal retail categories that city-planner OSM
    # tagging captures. Tier 2 here means "structurally under-mapped on
    # OSM": the count we get is a FLOOR, not a true count, and every
    # output row is flagged accordingly.
    #
    # Tier 1: formal / OSM-reliable (treated as roughly representative).
    # Tier 2: informal, under-mapped (floor, not true count).
    # financing_confound: boda-boda stage entry is driven by asset
    #         financing rather than market gap.
    # ---------------------------------------------------------------------

    # tier 1 -- formal, OSM-reachable
    "supermarket":     {"tag": 'node["shop"="supermarket"]',      "tier": 1, "financing_confound": False},
    "pharmacy":        {"tag": 'node["amenity"="pharmacy"]',      "tier": 1, "financing_confound": False},
    "restaurant":      {"tag": 'node["amenity"="restaurant"]',   "tier": 1, "financing_confound": False},
    "hardware_store":  {"tag": 'node["shop"="hardware"]',        "tier": 1, "financing_confound": False},
    # tier 2 -- Kenyan micro-MSME (under-mapped, floor-only)
    "githeri_stall":   {"tag": 'node["amenity"="fast_food"]',    "tier": 2, "financing_confound": False},
    "mandazi_corner":  {"tag": 'node["shop"="bakery"]',          "tier": 2, "financing_confound": False},
    "mitumba_table":   {"tag": 'node["shop"="clothes"]',         "tier": 2, "financing_confound": False},
    "phone_repair":    {"tag": 'node["repair"~"."]',             "tier": 2, "financing_confound": False},
    "airtime_reseller":{"tag": 'node["shop"="mobile_phone"]',    "tier": 2, "financing_confound": False},
    "chemist_kiosk":   {"tag": 'node["amenity"="pharmacy"]',     "tier": 2, "financing_confound": False},
    "charcoal_stove":  {"tag": 'node["shop"="fuel"]',            "tier": 2, "financing_confound": False},
    "secondhand_shoes":{"tag": 'node["shop"="shoes"]',           "tier": 2, "financing_confound": False},
    "fried_fish":      {"tag": 'node["amenity"="fast_food"]',    "tier": 2, "financing_confound": False},
    "tv_viewing":      {"tag": 'node["amenity"="cafe"]',         "tier": 2, "financing_confound": False},
    "building_material_stall": {"tag": 'node["shop"="build_market"]', "tier": 2, "financing_confound": False},
    "kiosk_corner":    {"tag": 'node["shop"="kiosk"]',           "tier": 2, "financing_confound": False},
    # financing-confound: boda boda stage
    "motorcycle_taxi": {"tag": 'node["amenity"="motorcycle_taxi"]', "tier": 2, "financing_confound": True},
    # backwards-compatible aliases so old saved queries / aliases still resolve
    "salon":           {"tag": 'node["shop"="beauty"]',          "tier": 2, "financing_confound": False},
    "greengrocer":     {"tag": 'node["shop"="greengrocer"]',     "tier": 2, "financing_confound": False},
    "clothes":         {"tag": 'node["shop"="clothes"]',         "tier": 2, "financing_confound": False},
    "hairdresser":     {"tag": 'node["shop"="hairdresser"]',     "tier": 2, "financing_confound": False},
    "fast_food":       {"tag": 'node["amenity"="fast_food"]',    "tier": 2, "financing_confound": False},
    "laundry":         {"tag": 'node["shop"="laundry"]',         "tier": 2, "financing_confound": False},
    "hardware":        {"tag": 'node["shop"="hardware"]',        "tier": 2, "financing_confound": False},
}

# Tag used to probe whether a resolved area actually contains mapped
# businesses (see 04_resolve_areas.py). Any node with one of these keys.
PROBE_KEYS = ['shop', 'amenity']

# Nairobi administrative hierarchy (from OSM data): county=4, sub-county=6,
# ward=8. A Nairobi "location" is usually admin_level 7 or 8 in OSM, but
# many boundaries are missing or mis-leveled, so resolution allows 6-10 and
# records the level actually used.
LOCATION_ADMIN_LEVELS = (6, 7, 8, 9, 10)


def post_query(query: str) -> dict:
    """POST an Overpass QL query, retrying across the endpoint chain.

    Treats HTTP 429 (rate limit), 5xx, and non-JSON bodies all as retryable
    failures and moves to the next endpoint. Raises RuntimeError only if
    every endpoint fails. The public Overpass instances have transient bad
    spells (slow reads, occasional error bodies), so this must never stall
    for minutes on a single call.
    """
    last_error = None
    for endpoint in ENDPOINTS:
        for attempt in range(RETRIES_PER_ENDPOINT + 1):
            try:
                resp = requests.post(
                    endpoint,
                    data={"data": query},
                    headers={"User-Agent": USER_AGENT},
                    timeout=TIMEOUT,
                )
                if resp.status_code == 429:
                    last_error = f"{endpoint} -> HTTP 429 (rate limited)"
                    time.sleep(5 * (attempt + 1))
                    continue
                if resp.status_code >= 500:
                    last_error = f"{endpoint} -> HTTP {resp.status_code}"
                    time.sleep(2)
                    continue
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, ValueError) as exc:
                # ValueError covers json.JSONDecodeError on a broken body.
                last_error = f"{endpoint} -> {exc}"
            time.sleep(SLEEP_BETWEEN_QUERIES)
    raise RuntimeError(f"all Overpass endpoints failed; last: {last_error}")


def fetch_nodes(category: str, area_id: int) -> list[dict]:
    """Fetch all nodes matching a category tag inside an Overpass area id."""
    tag_query = CATEGORIES[category]["tag"]
    query = f"""
    [out:json][timeout:60];
    area({area_id})->.a;
    {tag_query}(area.a);
    out body;
    """
    result = post_query(query)
    return result.get("elements", [])


def fetch_nodes_bbox(category: str, bbox: list[float]) -> list[dict]:
    """Fetch all nodes matching a category tag inside a bounding box.

    bbox is [south, west, north, east] (Overpass ordering).
    """
    tag_query = CATEGORIES[category]["tag"]
    south, west, north, east = bbox
    query = f"""
    [out:json][timeout:60];
    {tag_query}({south},{west},{north},{east});
    out body;
    """
    result = post_query(query)
    return result.get("elements", [])


def geocode_bbox(name: str) -> dict | None:
    """Geocode a location name to a bounding box via Nominatim (free).

    Returns {bbox: [south, west, north, east], geocode_name, geocode_type}
    or None if no Nairobi match is found. Prefers Nairobi suburb-type
    matches over false positives elsewhere in Kenya.

    Network failures (the free service is intermittently slow) retry once,
    then raise RuntimeError so the caller can record a failed entry instead
    of crashing the whole pipeline.
    """
    params = {
        "format": "jsonv2",
        "limit": "5",
        "countrycodes": "ke",
        "q": name,
        "viewbox": NOMINATIM_VIEWBOX,
    }
    resp = None
    last_err = None
    for _ in range(2):
        try:
            resp = requests.get(
                NOMINATIM_URL, params=params,
                headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT,
            )
            resp.raise_for_status()
            break
        except requests.RequestException as exc:
            last_err = exc
            time.sleep(SLEEP_BETWEEN_QUERIES)
    if resp is None:
        raise RuntimeError(f"Nominatim geocode failed: {last_err}")
    candidates = resp.json()

    # Keep only matches that are actually in Nairobi, so a village of the
    # same name elsewhere in Kenya cannot masquerade as the Nairobi one.
    nairobi = [c for c in candidates if "nairobi" in c.get("display_name", "").lower()]
    if not nairobi:
        return None

    # Rank by how specific the match is: suburb first, then neighbourhood /
    # commercial district, then anything else. Coarser types get recorded so
    # the caller can downgrade the entry to "suspect".
    prefer = {"suburb": 0, "neighbourhood": 1, "city_district": 2, "commercial": 3}
    nairobi.sort(key=lambda c: prefer.get(c.get("addresstype"), 9))
    best = nairobi[0]

    bb = best["boundingbox"]  # Nominatim returns [south, north, west, east]
    bbox = [float(bb[0]), float(bb[2]), float(bb[1]), float(bb[3])]
    return {
        "bbox": bbox,
        "geocode_name": best["display_name"],
        "geocode_type": best.get("addresstype"),
    }


def probe_area_business_count(area_id: int) -> int:
    """Count nodes inside an area that carry any shop/amenity tag.

    Used to sanity-check that a resolved administrative boundary actually
    contains mapped businesses (the Kilimani location boundary was found to
    be empty of mapped data even though the area around it is well mapped).

    NOTE: uses an explicit union statement -- the single-statement form
    `node(area.a)["shop"] | ["amenity"]` is rejected by Overpass (400).
    """
    queries = "; ".join(f'node(area.a)["{k}"]' for k in PROBE_KEYS)
    query = f"""
    [out:json][timeout:60];
    area({area_id})->.a;
    ({queries};);
    out count;
    """
    result = post_query(query)
    for element in result.get("elements", []):
        if element.get("type") == "count":
            return int(element["tags"]["total"])
    return 0
