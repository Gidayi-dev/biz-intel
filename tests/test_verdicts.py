"""Offline tests for the Business-Idea-Fit verdict classifier (scripts/verdicts.py).

Pins the contract of the four verdict bands + the "Not fit" override, and the
honest "anywhere" location matcher (which must never fabricate a location).
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import verdicts as v  # noqa: E402

COVERED = ["Kilimani", "Kasarani", "Ruai", "Kawangware", "Nairobi West",
           "Roysambu"]


class VerdictClassifyTests(unittest.TestCase):
    def test_band_boundaries(self):
        # exact thresholds fall into the higher band (>= comparisons)
        self.assertEqual(v.classify(0.25, False), v.VERDICT_STRONG)
        self.assertEqual(v.classify(1.50, False), v.VERDICT_STRONG)
        self.assertEqual(v.classify(0.00, False), v.VERDICT_GOOD)
        self.assertEqual(v.classify(0.24, False), v.VERDICT_GOOD)
        self.assertEqual(v.classify(-1.00, False), v.VERDICT_WEAK)
        self.assertEqual(v.classify(-0.01, False), v.VERDICT_WEAK)
        self.assertEqual(v.classify(-1.01, False), v.VERDICT_NOT_RECOMMENDED)
        self.assertEqual(v.classify(-5.00, False), v.VERDICT_NOT_RECOMMENDED)

    def test_financing_confound_overrides_everything(self):
        # even a very high score is "Not fit" when financing-confounded
        self.assertEqual(v.classify(1.50, True), v.VERDICT_NOT_FIT)
        self.assertEqual(v.classify(-5.00, True), v.VERDICT_NOT_FIT)
        # truthy int works too
        self.assertEqual(v.classify(0.9, 1), v.VERDICT_NOT_FIT)

    def test_nan_score_is_not_fit(self):
        self.assertEqual(v.classify(np.nan, False), v.VERDICT_NOT_FIT)
        self.assertEqual(v.classify(None, False), v.VERDICT_NOT_FIT)

    def test_rank_order_is_best_to_worst(self):
        self.assertEqual(v.VERDICT_RANK[v.VERDICT_STRONG], 0)
        self.assertEqual(v.VERDICT_RANK[v.VERDICT_GOOD], 1)
        self.assertEqual(v.VERDICT_RANK[v.VERDICT_WEAK], 2)
        self.assertEqual(v.VERDICT_RANK[v.VERDICT_NOT_RECOMMENDED], 3)
        self.assertEqual(v.VERDICT_RANK[v.VERDICT_NOT_FIT], 4)


class VerdictAddVerdictsTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "location_name": ["A", "A", "B"],
            "category": ["supermarket", "salon", "motorcycle_taxi"],
            "score": [0.40, -0.10, 0.90],
            "financing_confound_flag": [0, 0, 1],
        })

    def test_adds_verdict_and_rank_columns(self):
        out = v.add_verdicts(self.df)
        self.assertIn("verdict", out.columns)
        self.assertIn("verdict_rank", out.columns)
        self.assertEqual(len(out), 3)
        self.assertEqual(out.iloc[0]["verdict"], v.VERDICT_STRONG)
        self.assertEqual(out.iloc[1]["verdict"], v.VERDICT_WEAK)
        # financing confound overrides the high score
        self.assertEqual(out.iloc[2]["verdict"], v.VERDICT_NOT_FIT)

    def test_does_not_mutate_input(self):
        original = self.df.copy()
        v.add_verdicts(self.df)
        pd.testing.assert_frame_equal(self.df, original)


class MatchLocationTests(unittest.TestCase):
    def test_exact(self):
        r = v.match_location("Kilimani", COVERED)
        self.assertEqual(r["matched_name"], "Kilimani")
        self.assertEqual(r["method"], "exact")

    def test_exact_case_and_spacing(self):
        r = v.match_location("  nairobi  west ", COVERED)
        self.assertEqual(r["matched_name"], "Nairobi West")
        self.assertEqual(r["method"], "exact")

    def test_substring_covered_name_in_query(self):
        r = v.match_location("a salon in Kasarani", COVERED)
        self.assertEqual(r["matched_name"], "Kasarani")
        self.assertEqual(r["method"], "substring")

    def test_substring_query_fragment_in_name(self):
        r = v.match_location("kili", COVERED)
        self.assertEqual(r["matched_name"], "Kilimani")

    def test_fuzzy_typo(self):
        # "Kasarini" is a transposition typo that does NOT contain the full
        # name as a substring, so it must resolve via the fuzzy path.
        r = v.match_location("Kasarini", COVERED)
        self.assertEqual(r["matched_name"], "Kasarani")
        self.assertEqual(r["method"], "fuzzy")

    def test_out_of_coverage_is_honest_fallback(self):
        # Ruaka is NOT covered (Kiambu County). It must resolve to None and
        # list Ruai (the nearest covered name) -- never a fabricated match.
        r = v.match_location("Ruaka", COVERED)
        self.assertIsNone(r["matched_name"])
        self.assertIsNone(r["method"])
        self.assertTrue(r["alternatives"])
        self.assertEqual(r["alternatives"][0][0], "Ruai")

    def test_empty_input(self):
        r = v.match_location("", COVERED)
        self.assertIsNone(r["matched_name"])
        self.assertEqual(r["alternatives"], [])


if __name__ == "__main__":
    unittest.main()
