"""VIIRS nightlights enrichment per location (optional raster dependency).

Zonal mean / median nightlight radiance for each of the 30 census locations
from the public NASA EOG VIIRS annual composite (2020). Nightlights are a
proxy for economic activity / built-up density -- higher radiance suggests a
more established commercial area.

Same degrade contract as script 21: if `rasterio`/`rasterstats` are missing or
the composite can't be downloaded (the normal offline case), all cells are
written as NULL and a SKIPPED note is logged. The recommender drops the term
and renormalizes rather than crashing.

Output: data/processed/viirs_enrichment.csv
Columns: location_id, viirs_mean, viirs_median
"""
import time

from enrichment_common import (load_config, load_locations, load_manifest,
                               load_businesses_centroids, location_geometry,
                               write_csv, optional_import, log_failure,
                               VIIRS_CACHE)

FIELDNAMES = ["location_id", "viirs_mean", "viirs_median"]


def _bbox_polygon(bbox):
    s, w, n, e = bbox
    return {"type": "Polygon",
            "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}


def _ensure_tif(url: str, filename: str, cache_dir) -> str | None:
    import requests
    tif_path = cache_dir / filename
    if tif_path.exists():
        return str(tif_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {filename} from {url} ...")
    resp = requests.get(url, stream=True, timeout=60)
    if resp.status_code != 200:
        log_failure(f"VIIRS download HTTP {resp.status_code}: {url}")
        return None
    with open(tif_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    return str(tif_path)


def main():
    cfg = load_config()
    out_path = cfg["paths"]["viirs_out"]
    raster_cfg = cfg.get("raster", {})
    cache_dir = raster_cfg.get("cache_dir", VIIRS_CACHE)
    url = raster_cfg.get("viirs_url")
    filename = raster_cfg.get("viirs_file", "viirs_2020_avg_masked.tif")

    rasterio = optional_import("rasterio")
    rasterstats = optional_import("rasterstats")

    locations = load_locations()
    if rasterio is None or rasterstats is None or not url:
        log_failure("SKIPPED VIIRS enrichment: rasterio/rasterstats missing "
                    "(offline core -- nightlights term will be NULL)")
        rows = [{"location_id": loc_id, "viirs_mean": None, "viirs_median": None}
                for loc_id, _ in locations]
        write_csv(out_path, rows, FIELDNAMES)
        print(f"VIIRS enrichment skipped (raster deps unavailable). "
              f"Wrote {len(rows)} NULL rows -> {out_path}")
        return

    tif_path = _ensure_tif(url, filename, cache_dir)
    if tif_path is None:
        rows = [{"location_id": loc_id, "viirs_mean": None, "viirs_median": None}
                for loc_id, _ in locations]
        write_csv(out_path, rows, FIELDNAMES)
        print(f"VIIRS download failed. Wrote {len(rows)} NULL rows -> {out_path}")
        return

    from rasterstats import zonal_stats

    manifest = load_manifest()
    centroids = load_businesses_centroids()

    rows = []
    for loc_id, loc_name in locations:
        geo = location_geometry(manifest, loc_name, centroids)
        bbox = geo["bbox"]
        if not bbox:
            rows.append({"location_id": loc_id, "viirs_mean": None,
                         "viirs_median": None})
            continue
        try:
            stats = zonal_stats(_bbox_polygon(bbox), tif_path,
                                stats=["mean", "median"])[0]
            rows.append({"location_id": loc_id,
                         "viirs_mean": stats.get("mean"),
                         "viirs_median": stats.get("median")})
        except Exception as exc:
            log_failure(f"VIIRS zonal_stats {loc_name}: {exc}")
            rows.append({"location_id": loc_id, "viirs_mean": None,
                         "viirs_median": None})
        time.sleep(0.1)

    write_csv(out_path, rows, FIELDNAMES)
    print(f"Wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
