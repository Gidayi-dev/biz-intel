import requests
import json
import time

# "https://overpass.kumi.systems/api/interpreter"
# "https://overpass.openstreetmap.fr/api/interpreter"
OVERPASS_URL = "https://overpass.openstreetmap.fr/api/interpreter"

BBOX = (-1.30, 36.78, -1.27, 36.82)

CATEGORIES = {
    "hairdresser": 'node["shop"="hairdresser"]',
    "salon": 'node["shop"="beauty"]',
    "restaurant": 'node["amenity"="restaurant"]',
    "supermarket": 'node["shop"="supermarket"]',
    "pharmacy": 'node["amenity"="pharmacy"]',
    "clothes": 'node["shop"="clothes"]',
}


def fetch_category(tag_query: str, bbox: tuple) -> list[dict]:
    """Fetch all OSM nodes matching a tag query within a bounding box."""
    south, west, north, east = bbox
    query = f"""
    [out:json][timeout:25];
    {tag_query}({south},{west},{north},{east});
    out body;
    """
    headers = {"User-Agent": "biz-intel-capstone/0.1 (student project)"}
    resp = requests.post(
        OVERPASS_URL, data={"data": query}, headers=headers, timeout=30
    )
    resp.raise_for_status()
    return resp.json().get("elements", [])


def main():
    results = {}
    for category, tag_query in CATEGORIES.items():
        print(f"Fetching {category}...")
        elements = fetch_category(tag_query, BBOX)
        results[category] = elements
        print(f"  -> {len(elements)} results")
        time.sleep(1)  # be polite to the free public Overpass instance

    with open("../data/raw/osm_pilot_kilimani.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved to data/raw/osm_pilot_kilimani.json")
    print("Total businesses pulled:", sum(len(v) for v in results.values()))


if __name__ == "__main__":
    main()