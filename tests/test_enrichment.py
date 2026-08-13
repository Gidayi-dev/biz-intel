"""Offline tests for the merge + recommender scripts (no network, no raster).

Builds a synthetic 2-location x 2-category fixture, runs scripts 23 and 24 as
importable functions against it, and asserts the glass-box contract:

  - output files exist and carry the required columns;
  - a larger model gap (predicted - mapped) raises the score;
  - more competitors lower the score;
  - tier-2 and bbox rows carry the configured penalties;
  - score == sum(terms) - penalties (exact identity).

Uses only the stdlib test runner plus pandas/numpy (already in the base venv).
"""
import importlib.util
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import enrichment_common as ec  # noqa: E402

WEIGHTS = {"gap": 0.6, "pop": 0.25, "comp": 0.4, "viirs": 0.2}
PENALTIES = {"tier2": 0.4, "bbox": 0.3}


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M23 = _load_script("m23", SCRIPTS / "23_merge_enrichments.py")
M24 = _load_script("m24", SCRIPTS / "24_recommender_prototype.py")


class EnrichmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="biz_intel_test_"))
        self.cfg = {
            "paths": {
                "db": self.tmp / "biz_intel.db",
                "osm_out": self.tmp / "osm_enrichment.csv",
                "worldpop_out": self.tmp / "worldpop_enrichment.csv",
                "viirs_out": self.tmp / "viirs_enrichment.csv",
                "merge_out": self.tmp / "enriched_features.csv",
                "recommendations_out": self.tmp / "recommendations.csv",
            },
            "weights": dict(WEIGHTS),
            "penalties": dict(PENALTIES),
            "normalize": "z",
            "top_k": 5,
        }
        ec._CONFIG = self.cfg
        self._build_db()
        self._build_enrichments()

    def tearDown(self):
        ec._CONFIG = None
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    # -- fixtures -----------------------------------------------------------
    def _build_db(self):
        conn = sqlite3.connect(self.cfg["paths"]["db"])
        conn.executescript(
            "CREATE TABLE locations ("
            "  location_id INTEGER, location_name TEXT,"
            "  population INTEGER, geo_method TEXT);"
            "CREATE TABLE features ("
            "  location_id INTEGER, category TEXT,"
            "  business_count REAL, predicted_count REAL);"
        )
        # location 1 = area, location 2 = bbox (to exercise the bbox penalty)
        conn.execute("INSERT INTO locations VALUES (1,'Alpha',100000,'area')")
        conn.execute("INSERT INTO locations VALUES (2,'Beta',100000,'bbox')")
        rows = [
            # (lid, category, business_count, predicted_count)
            (1, "supermarket", 10, 20),   # gap +10 (underserved)
            (2, "supermarket", 20, 10),   # gap -10 (over-served)
            (1, "salon", 10, 20),         # gap +10, tier2, fewer competitors
            (2, "salon", 10, 20),         # gap +10, tier2, more competitors
        ]
        conn.executemany(
            "INSERT INTO features VALUES (?,?,?,?)", rows)
        conn.commit()
        conn.close()

    def _build_enrichments(self):
        # OSM: same gaps across the salon pair, but different competitor counts
        # so the competition effect is the only thing that can move the score.
        osm = pd.DataFrame([
            {"location_id": 1, "category": "supermarket",
             "competitors_100m": 0, "competitors_500m": 0,
             "anchors_market_500m": 1, "bus_stops_500m": 2, "total_pois": 50},
            {"location_id": 2, "category": "supermarket",
             "competitors_100m": 0, "competitors_500m": 0,
             "anchors_market_500m": 1, "bus_stops_500m": 2, "total_pois": 50},
            {"location_id": 1, "category": "salon",
             "competitors_100m": 0, "competitors_500m": 0,
             "anchors_market_500m": 1, "bus_stops_500m": 2, "total_pois": 50},
            {"location_id": 2, "category": "salon",
             "competitors_100m": 0, "competitors_500m": 10,
             "anchors_market_500m": 1, "bus_stops_500m": 2, "total_pois": 50},
        ])
        osm.to_csv(self.cfg["paths"]["osm_out"], index=False)

        wp = pd.DataFrame([
            {"location_id": 1, "population_sum_wp": 90000.0,
             "pop_mean_wp": 1.0, "pop_density_wp": 1000.0},
            {"location_id": 2, "population_sum_wp": 90000.0,
             "pop_mean_wp": 1.0, "pop_density_wp": 1000.0},
        ])
        wp.to_csv(self.cfg["paths"]["worldpop_out"], index=False)

        # Equal VIIRS across locations -> std 0 -> normalized term = 0, so the
        # nightlight term does not confound the gap/competition assertions.
        viirs = pd.DataFrame([
            {"location_id": 1, "viirs_mean": 5.0, "viirs_median": 5.0},
            {"location_id": 2, "viirs_mean": 5.0, "viirs_median": 5.0},
        ])
        viirs.to_csv(self.cfg["paths"]["viirs_out"], index=False)

    # -- tests --------------------------------------------------------------
    def test_merge_writes_required_columns_and_joins(self):
        M23.main()
        df = pd.read_csv(self.cfg["paths"]["merge_out"])
        self.assertTrue(self.cfg["paths"]["merge_out"].exists())
        for col in ["location_id", "location_name", "category", "tier",
                    "geo_method", "population", "business_count",
                    "predicted_count", "competitors_500m",
                    "population_sum_wp", "viirs_mean"]:
            self.assertIn(col, df.columns)
        self.assertEqual(len(df), 4)
        # tier mapped from the category taxonomy
        self.assertEqual(df.loc[df.category == "salon", "tier"].iloc[0], 2)
        self.assertEqual(df.loc[df.category == "supermarket", "tier"].iloc[0], 1)

    def test_recommender_scoring_direction(self):
        M23.main()
        merged = pd.read_csv(self.cfg["paths"]["merge_out"])
        rec = M24.build_recommendations(merged, WEIGHTS, PENALTIES, "z")

        # score identity: score == sum(terms) - penalties
        lhs = rec["score"].to_numpy()
        rhs = (rec["gap_term"] + rec["pop_term"] + rec["comp_term"]
               + rec["viirs_term"] - rec["tier2_penalty"] - rec["bbox_penalty"]).to_numpy()
        np.testing.assert_allclose(lhs, rhs, rtol=1e-9)

        # larger model gap -> higher score (same category, both non-bbox terms
        # compared within supermarket where the only difference is the gap)
        alpha_super = rec[(rec.location_name == "Alpha")
                          & (rec.category == "supermarket")].iloc[0]
        beta_super = rec[(rec.location_name == "Beta")
                         & (rec.category == "supermarket")].iloc[0]
        self.assertGreater(alpha_super["n_gap"], beta_super["n_gap"])
        self.assertGreater(alpha_super["score"], beta_super["score"])

        # more competitors -> lower score (salon pair: same gap, same tier,
        # differing only in competitors_500m)
        alpha_salon = rec[(rec.location_name == "Alpha")
                          & (rec.category == "salon")].iloc[0]
        beta_salon = rec[(rec.location_name == "Beta")
                         & (rec.category == "salon")].iloc[0]
        self.assertGreater(beta_salon["n_competitors"], alpha_salon["n_competitors"])
        self.assertLess(beta_salon["score"], alpha_salon["score"])

    def test_penalties_applied(self):
        M23.main()
        merged = pd.read_csv(self.cfg["paths"]["merge_out"])
        rec = M24.build_recommendations(merged, WEIGHTS, PENALTIES, "z")

        tier2_rows = rec[rec.category == "salon"]
        tier1_rows = rec[rec.category == "supermarket"]
        self.assertTrue((tier2_rows["tier2_penalty"] == PENALTIES["tier2"]).all())
        self.assertTrue((tier1_rows["tier2_penalty"] == 0.0).all())

        bbox_rows = rec[rec.geo_method == "bbox"]
        area_rows = rec[rec.geo_method == "area"]
        self.assertTrue((bbox_rows["bbox_penalty"] == PENALTIES["bbox"]).all())
        self.assertTrue((area_rows["bbox_penalty"] == 0.0).all())

    def test_recommender_cli_writes_file(self):
        M23.main()
        saved_argv = list(sys.argv)
        sys.argv = ["24_recommender_prototype.py", "--top", "5"]
        try:
            M24.main()
        finally:
            sys.argv = saved_argv

        out = pd.read_csv(self.cfg["paths"]["recommendations_out"])
        for col in ["location_id", "location_name", "category", "rank", "score",
                    "n_gap", "n_log_pop", "n_competitors", "n_viirs",
                    "gap_term", "pop_term", "comp_term", "viirs_term",
                    "tier2_penalty", "bbox_penalty"]:
            self.assertIn(col, out.columns)
        # rank within each location is 1..N sorted by descending score
        for lid in out.location_id.unique():
            sub = out[out.location_id == lid].sort_values("rank")
            self.assertEqual(list(sub["rank"]), list(range(1, len(sub) + 1)))
            self.assertTrue((sub["score"].diff().dropna() <= 0).all())

    def test_missing_enrichment_is_nan_not_zero(self):
        # Remove VIIRS -> the merged cell must read back as NaN, not 0.
        self.cfg["paths"]["viirs_out"].unlink()
        M23.main()
        merged = pd.read_csv(self.cfg["paths"]["merge_out"])
        self.assertTrue(merged["viirs_mean"].isna().all())
        self.assertFalse((merged["viirs_mean"] == 0).any())

    def test_standardize_all_missing_is_nan(self):
        # An entirely-missing term must stay NaN (not 0) so the recommender can
        # tell "no data" from "zero signal" and renormalize the other weights.
        out = M24.standardize(pd.Series([np.nan, np.nan, np.nan]), "z")
        self.assertTrue(out.isna().all())
        self.assertFalse((out == 0.0).any())

    def test_uniformly_missing_terms_renormalize(self):
        # Offline case: competitors + nightlights entirely absent. The gap and
        # population weights must be scaled up so the total weight magnitude is
        # preserved, and the missing terms must contribute exactly 0.
        M23.main()
        merged = pd.read_csv(self.cfg["paths"]["merge_out"])
        merged["competitors_500m"] = np.nan
        merged["viirs_mean"] = np.nan
        rec = M24.build_recommendations(merged, WEIGHTS, PENALTIES, "z")

        self.assertTrue(rec["n_competitors"].isna().all())
        self.assertTrue(rec["n_viirs"].isna().all())

        total = sum(WEIGHTS.values())
        # log_pop is constant across the fixture (degenerate -> 0.0, still
        # "present"), so the available terms are gap + pop only.
        avail = WEIGHTS["gap"] + WEIGHTS["pop"]
        scale = total / avail
        alpha_super = rec[(rec.location_name == "Alpha")
                          & (rec.category == "supermarket")].iloc[0]
        self.assertAlmostEqual(alpha_super["gap_term"],
                               WEIGHTS["gap"] * scale * alpha_super["n_gap"],
                               places=9)
        self.assertEqual(alpha_super["comp_term"], 0.0)
        self.assertEqual(alpha_super["viirs_term"], 0.0)


if __name__ == "__main__":
    unittest.main()
