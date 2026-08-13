"""WorldPop population enrichment per location (optional raster dependency).

Zonal population statistics (sum / mean / density) for each of the 30 census
locations from the public WorldPop 2020 Kenya unadjusted population GeoTIFF.

This is the one step with a hard dependency on optional packages: it needs
`rasterio` + `rasterstats` (NOT in the base venv) AND internet to download the
GeoTIFF. If either is missing -- the normal offline case -- it writes all-NULL
rows and a SKIPPED note to the failure log, and never crashes. The recommender
then simply drops the population term for any missing location.

Output: data/processed/worldpop_enrichment.csv
Columns: location_id, population_sum_wp, pop_mean_wp, pop_density_wp
"""
import json
import time
from math import cos, radians

from enrichment_common import (load_config, load_locations, load_manifest,
                               load_businesses_centroids, location_geometry,
                               write_csv, optional_import, log_failure,
                               WORLDPOP_CACHE)

FIELDNAMES = ["location_id", "population_sum_wp", "pop_mean_wp", "pop_density_wp"]


def _bbox_polygon(bbox):
    """GeoJSON Polygon for a [south, west, north, east] box (lon/lat order)."""
    s, w, n, e = bbox
    return {"type": "Polygon",
            "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}


def _bbox_area_sqkm(bbox):
    """Approximate bbox area in km^2 (equirectangular at the box's latitude)."""
    s, w, n, e = bbox
    lat_mid = (s + n) / 2.0
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * cos(radians(lat_mid))
    return abs((n - s) * km_per_deg_lat * (e - w) * km_per_deg_lon)


def _ensure_tif(url: str, filename: str, cache_dir) -> str | None:
    """Download the GeoTIFF if not already cached; None on any failure."""
    import requests
    tif_path = cache_dir / filename
    if tif_path.exists():
        return str(tif_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {filename} from {url} ...")
    resp = requests.get(url, stream=True, timeout=60)
    if resp.status_code != 200:
        log_failure(f"WorldPop download HTTP {resp.status_code}: {url}")
        return None
    with open(tif_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    return str(tif_path)


def main():
    cfg = load_config()
    out_path = cfg["paths"]["worldpop_out"]
    raster_cfg = cfg.get("raster", {})
    cache_dir = raster_cfg.get("cache_dir", WORLDPOP_CACHE)
    url = raster_cfg.get("worldpop_url")
    filename = raster_cfg.get("worldpop_file", "ken_ppp_2020_UNadj.tif")

    rasterio = optional_import("rasterio")
    rasterstats = optional_import("rasterstats")

    locations = load_locations()
    if rasterio is None or rasterstats is None or not url:
        log_failure("SKIPPED WorldPop enrichment: rasterio/rasterstats missing "
                    "(offline core -- population term will be NULL)")
        rows = [{"location_id": loc_id, "population_sum_wp": None,
                 "pop_mean_wp": None, "pop_density_wp": None}
                for loc_id, _ in locations]
        write_csv(out_path, rows, FIELDNAMES)
        print(f"WorldPop enrichment skipped (raster deps unavailable). "
              f"Wrote {len(rows)} NULL rows -> {out_path}")
        return

    tif_path = _ensure_tif(url, filename, cache_dir)
    if tif_path is None:
        rows = [{"location_id": loc_id, "population_sum_wp": None,
                 "pop_mean_wp": None, "pop_density_wp": None}
                for loc_id, _ in locations]
        write_csv(out_path, rows, FIELDNAMES)
        print(f"WorldPop download failed. Wrote {len(rows)} NULL rows -> {out_path}")
        return

    from rasterstats import zonal_stats

    manifest = load_manifest()
    centroids = load_businesses_centroids()

    rows = []
    for loc_id, loc_name in locations:
        geo = location_geometry(manifest, loc_name, centroids)
        bbox = geo["bbox"]
        if not bbox:
            rows.append({"location_id": loc_id, "population_sum_wp": None,
                         "pop_mean_wp": None, "pop_density_wp": None})
            continue
        try:
            stats = zonal_stats(_bbox_polygon(bbox), tif_path,
                                stats=["sum", "mean"])[0]
            pop_sum = stats.get("sum")
            pop_mean = stats.get("mean")
            density = (pop_sum / _bbox_area_sqkm(bbox)) if pop_sum is not None else None
            rows.append({"location_id": loc_id,
                         "population_sum_wp": pop_sum,
                         "pop_mean_wp": pop_mean,
                         "pop_density_wp": density})
        except Exception as exc:
            log_failure(f"WorldPop zonal_stats {loc_name}: {exc}")
            rows.append({"location_id": loc_id, "population_sum_wp": None,
                         "pop_mean_wp": None, "pop_density_wp": None})
        time.sleep(0.1)

    write_csv(out_path, rows, FIELDNAMES)
    print(f"Wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
