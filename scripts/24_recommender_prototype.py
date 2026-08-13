"""Transparent, glass-box recommendation engine (prototype).

Reads data/processed/enriched_features.csv (from script 23) and ranks, for each
Nairobi location, which business category to consider opening next. The score is
a hand-computable linear combination of four standardized, public-data signals
plus two transparent penalties -- every number in the output traces back to a
real column, and the per-term contributions are emitted as separate columns so
the dashboard can show the exact arithmetic.

    normalized_gap          = (predicted_count - business_count) z per category
                              (positive = the model expects more than is mapped)
    normalized_log_pop      = ln(population) z across locations
    normalized_competitors  = competitors_500m z per category
    normalized_viirs        = viirs_mean z across locations

    score = w_gap*n_gap + w_pop*n_log_pop - w_comp*n_comp + w_viirs*n_viirs
            - penalty_tier2 - penalty_bbox

Missing data handling (offline / no raster deps): a NaN normalized term
contributes 0 and the *available* weights are renormalized to preserve the total
weight magnitude, so a location missing nightlights is not silently penalized.
The renormalization is transparent -- each term column already carries its scaled
weight, so the printed formula above stays exactly what the code computes.

Output: data/processed/recommendations.csv -- the FULL ranked table (rank 1..N
within each location), so the dashboard can re-rank under different weights.
CLI `--top` / `--location` only affect what is printed, not what is written.
"""
import argparse
import math
import sys

import numpy as np
import pandas as pd

from enrichment_common import load_config, write_csv, log_failure
from summaries import TIER2, CONFOUND

OUTPUT_COLUMNS = [
    "location_id", "location_name", "category", "tier",
    "financing_confound_flag", "geo_method",
    "rank", "score",
    "n_gap", "n_log_pop", "n_competitors", "n_viirs",
    "gap_term", "pop_term", "comp_term", "viirs_term",
    "tier2_penalty", "bbox_penalty",
    "predicted_count", "business_count", "gap",
    "competitors_500m", "competitors_100m",
    "population", "log_pop", "viirs_mean",
]


def standardize(s: pd.Series, method: str) -> pd.Series:
    """z-score or minmax over `s`, NaN-skipping.

    All-missing -> NaN, so the recommender treats the term as *absent* and
    renormalizes the remaining weights; a present-but-degenerate spread -> 0
    (genuine no-signal, not missing).
    """
    s = pd.to_numeric(s, errors="coerce")
    out = pd.Series(0.0, index=s.index)
    valid = s.notna()
    if not valid.any():
        return pd.Series(np.nan, index=s.index)
    if method == "minmax":
        lo, hi = s.min(), s.max()
        if pd.isna(lo) or pd.isna(hi) or hi == lo:
            return out
        return (s - lo) / (hi - lo)
    mu, sd = s.mean(), s.std()
    if pd.isna(mu) or pd.isna(sd) or sd == 0:
        return out
    return (s - mu) / sd


def parse_weights(spec: str) -> dict:
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 4:
        raise SystemExit(f"--weights needs 4 comma-separated values (got {len(parts)})")
    return {"gap": float(parts[0]), "pop": float(parts[1]),
            "comp": float(parts[2]), "viirs": float(parts[3])}


def build_recommendations(df: pd.DataFrame, weights: dict, penalties: dict,
                          method: str) -> pd.DataFrame:
    out = df.copy()

    # --- raw inputs --------------------------------------------------------
    out["gap"] = out["predicted_count"] - out["business_count"]
    out["log_pop"] = out["population"].apply(
        lambda p: math.log(p) if pd.notna(p) and p > 0 else np.nan)

    # --- per-category standardization (gap, competitors) --------------------
    out["n_gap"] = out.groupby("category")["gap"].transform(
        lambda s: standardize(s, method))
    out["n_competitors"] = out.groupby("category")["competitors_500m"].transform(
        lambda s: standardize(s, method))

    # --- per-location (across locations) standardization (log_pop, viirs) ---
    # These are constant within a location; standardize the unique per-location
    # values and broadcast back so each location gets a meaningful z-score.
    loc_level = out.groupby("location_id").agg(
        log_pop=("log_pop", "first"), viirs=("viirs_mean", "first"))
    loc_level["n_log_pop"] = standardize(loc_level["log_pop"], method)
    loc_level["n_viirs"] = standardize(loc_level["viirs"], method)
    out = out.merge(loc_level[["n_log_pop", "n_viirs"]],
                    left_on="location_id", right_index=True, how="left")

    # --- renormalized weighted terms ---------------------------------------
    # A row missing a term (NaN) contributes 0 for that term; the remaining
    # terms' weights are scaled so the total weight magnitude is preserved.
    total_w = sum(weights.values())
    n_map = {"gap": "n_gap", "pop": "n_log_pop",
             "comp": "n_competitors", "viirs": "n_viirs"}
    term_map = {"gap": "gap_term", "pop": "pop_term",
                "comp": "comp_term", "viirs": "viirs_term"}

    row_avail = pd.Series(0.0, index=out.index)
    for k, col in n_map.items():
        row_avail += out[col].notna().astype(float) * weights[k]
    scale = (total_w / row_avail).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    for key, col in n_map.items():
        sign = -1.0 if key == "comp" else 1.0
        out[term_map[key]] = sign * weights[key] * scale * out[col].fillna(0.0)

    # --- penalties ---------------------------------------------------------
    tier2_pen = float(penalties["tier2"])
    bbox_pen = float(penalties["bbox"])
    tier_num = pd.to_numeric(out["tier"], errors="coerce")
    out["tier2_penalty"] = np.where(tier_num.eq(2), tier2_pen, 0.0)
    out["bbox_penalty"] = np.where(out["geo_method"].astype(str) == "bbox",
                                   bbox_pen, 0.0)

    out["score"] = (out["gap_term"] + out["pop_term"]
                    + out["comp_term"] + out["viirs_term"]
                    - out["tier2_penalty"] - out["bbox_penalty"])

    # --- rank within location ----------------------------------------------
    out["rank"] = out.groupby("location_id")["score"].rank(
        ascending=False, method="first").astype(int)
    out = out.sort_values(["location_id", "rank"])
    return out


def print_table(df: pd.DataFrame, top: int, location: str | None) -> None:
    view = df if location is None else df[df["location_name"] == location]
    if view.empty:
        print(f"No rows for location {location!r}")
        return
    locs = sorted(view["location_id"].unique())
    for lid in locs:
        rows = view[view["location_id"] == lid].head(top)
        name = rows.iloc[0]["location_name"]
        print(f"\n== {name} ==")
        cols = ["rank", "category", "score", "n_gap", "n_log_pop",
                "n_competitors", "n_viirs", "gap_term", "pop_term",
                "comp_term", "viirs_term", "tier", "geo_method"]
        print(rows[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def main():
    cfg = load_config()
    in_path = cfg["paths"]["merge_out"]
    out_path = cfg["paths"]["recommendations_out"]
    normalize_default = cfg.get("normalize", "z")
    top_default = int(cfg.get("top_k", 5))

    ap = argparse.ArgumentParser(description="Rank categories per location (glass-box)")
    ap.add_argument("--top", type=int, default=top_default,
                    help=f"rows per location to PRINT (default {top_default})")
    ap.add_argument("--weights", type=str, default=None,
                    help="gap,pop,comp,viirs (default from config)")
    ap.add_argument("--normalize", choices=["z", "minmax"], default=normalize_default)
    ap.add_argument("--location", type=str, default=None,
                    help="print only this location")
    args = ap.parse_args()

    if not in_path.exists():
        log_failure(f"RECOMMENDER: {in_path} missing -- run scripts 20-23 first")
        print(f"{in_path} not found. Run scripts 20-23 first.")
        sys.exit(1)

    df = pd.read_csv(in_path)
    if df.empty:
        print(f"{in_path} is empty -- nothing to rank.")
        sys.exit(1)

    weights = cfg["weights"] if args.weights is None else parse_weights(args.weights)
    penalties = cfg["penalties"]

    rec = build_recommendations(df, weights, penalties, args.normalize)
    rec = rec[OUTPUT_COLUMNS]
    write_csv(out_path, rec.to_dict("records"), OUTPUT_COLUMNS)
    print(f"Wrote {len(rec)} ranked rows -> {out_path}")

    print_table(rec, args.top, args.location)


if __name__ == "__main__":
    main()
