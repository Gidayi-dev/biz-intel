"""Business-Idea-Fit verdicts -- turn a glass-box score into a plain label.

The recommender (script 24) produces a continuous ``score`` per
(location, category) cell. For a person deciding whether to open a business
that number alone is too abstract. This module maps it to an interpretable,
hand-checkable verdict band and -- crucially -- a hard "Not fit" override for
the one case the market-gap signal cannot be trusted.

The bands are fixed, public, and documented below so a label never changes
silently with a re-run. Only two columns are read (``score`` and
``financing_confound_flag``), so the same classifier runs unchanged on the
dashboard's client-side re-scored table (different weights -> different score
-> possibly a different band, same transparent rules).

Verdicts (best -> worst):

    Strong fit      score >= +0.25   clearly underserved: gap is positive ~96% of the time
    Good fit        0.00 <= score < +0.25
    Weak fit        -1.00 <= score < 0.00   "less recommended"
    Not recommended score < -1.00    over-saturated: gap is negative ~88% of the time
    Not fit         financing_confound_flag == 1   hard override

The thresholds were chosen against the real recommendations.csv score
distribution (z-scored terms, weights sum ~1.45, penalties up to 0.7): the
+0.25 cut sits at roughly the 70th percentile and separates positive-gap cells
from the pack; the -1.00 cut sits at roughly the 23rd percentile and separates
strongly negative-gap cells. "Not fit" is applied *before* any band because a
financing-confounded category (boda boda / motorcycle taxi) is driven by asset
financing rather than market gap -- the model literally has no reliable signal
for it, so ranking it by score would be misleading.

This module is dependency-light (pandas + numpy + the stdlib ``difflib``) and
purely functional so ``tests/test_verdicts.py`` can pin the contract offline.
"""
from __future__ import annotations

import difflib

import numpy as np
import pandas as pd

# -- fixed, documented thresholds -------------------------------------------
STRONG_THRESHOLD = 0.25
GOOD_THRESHOLD = 0.00
WEAK_THRESHOLD = -1.00

VERDICT_STRONG = "Strong fit"
VERDICT_GOOD = "Good fit"
VERDICT_WEAK = "Weak fit"
VERDICT_NOT_RECOMMENDED = "Not recommended"
VERDICT_NOT_FIT = "Not fit"

# Best -> worst. Used for ordered legends and stable sorting.
VERDICT_ORDER = [
    VERDICT_STRONG,
    VERDICT_GOOD,
    VERDICT_WEAK,
    VERDICT_NOT_RECOMMENDED,
    VERDICT_NOT_FIT,
]

# Small int for sorting; lower = better.
VERDICT_RANK = {v: i for i, v in enumerate(VERDICT_ORDER)}

# (background, foreground) hex -- matches the dashboard's status tokens so the
# product reads as one visual language. Kept here so CLI/tests share the palette.
VERDICT_COLORS = {
    VERDICT_STRONG: ("#dcf3e8", "#0d6b4a"),
    VERDICT_GOOD: ("#e6f4f1", "#0d6b6b"),
    VERDICT_WEAK: ("#fff4e5", "#9a5b00"),
    VERDICT_NOT_RECOMMENDED: ("#ffe9e9", "#a32727"),
    VERDICT_NOT_FIT: ("#efefee", "#3f3f3e"),
}

# One plain-English sentence per band, so the label is never a bare word.
VERDICT_HINT = {
    VERDICT_STRONG: "Clear opportunity — mapped supply is well below what the "
                    "model expects for this area.",
    VERDICT_GOOD: "Reasonable opportunity — a modest positive gap.",
    VERDICT_WEAK: "Less recommended — near or below the expected supply; "
                  "enter with caution.",
    VERDICT_NOT_RECOMMENDED: "Oversaturated — mapped supply already exceeds "
                             "the model's expectation.",
    VERDICT_NOT_FIT: "Not assessable as a market-gap play — asset-financing "
                     "confound, so the gap signal is unreliable.",
}


def classify(score: float, is_financing_confound: bool | int) -> str:
    """Return the verdict band for a single cell.

    ``is_financing_confound`` is applied first (hard override); a NaN score is
    also "Not fit" because the cell cannot be assessed.
    """
    if is_financing_confound:
        return VERDICT_NOT_FIT
    if score is None or pd.isna(score):
        return VERDICT_NOT_FIT
    if score >= STRONG_THRESHOLD:
        return VERDICT_STRONG
    if score >= GOOD_THRESHOLD:
        return VERDICT_GOOD
    if score >= WEAK_THRESHOLD:
        return VERDICT_WEAK
    return VERDICT_NOT_RECOMMENDED


def add_verdicts(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with ``verdict`` (str) and ``verdict_rank`` (int).

    Requires columns ``score`` and ``financing_confound_flag``; all other
    columns pass through untouched.
    """
    out = df.copy()
    is_confound = (
        pd.to_numeric(out["financing_confound_flag"], errors="coerce")
        .fillna(0).astype(int) == 1
    )
    scores = pd.to_numeric(out["score"], errors="coerce")
    out["verdict"] = [
        classify(float(s), bool(c)) for s, c in zip(scores, is_confound)
    ]
    out["verdict_rank"] = out["verdict"].map(VERDICT_RANK)
    return out


def verdict_badge(verdict: str) -> str:
    """Inline HTML span for a verdict label (uses VERDICT_COLORS)."""
    bg, fg = VERDICT_COLORS.get(verdict, VERDICT_COLORS[VERDICT_NOT_FIT])
    return (
        f"<span class='badge' style='background:{bg};color:{fg};"
        f"border:1px solid {fg}40'>{verdict}</span>"
    )


# ---------------------------------------------------------------------------
# "Anywhere" location resolution -- honest, offline, never fabricated.
# ---------------------------------------------------------------------------
def match_location(text: str, loc_names: list[str]) -> dict:
    """Resolve free text to a covered location name.

    Returns a dict with:
      - ``matched_name``: a real covered location name, or ``None``.
      - ``method``: "exact" | "substring" | "fuzzy" | ``None``.
      - ``alternatives``: list of ``(name, similarity)`` nearest covered names,
        populated only when nothing matched -- used for the honest
        "not in coverage, closest areas are X" fallback (e.g. "Ruaka").

    This never invents a location: an unknown place resolves to ``None`` with
    the nearest *real* covered names listed alongside their similarity, so the
    user can see exactly why (and how) it fell through.
    """
    if not text or not text.strip():
        return {"matched_name": None, "method": None, "alternatives": []}

    q = " ".join(text.strip().split())
    ql = q.lower()
    lower_to_name = {name.lower(): name for name in loc_names}

    # 1) exact, case-insensitive full-name match ("Kilimani" / "nairobi west").
    if ql in lower_to_name:
        return {"matched_name": lower_to_name[ql], "method": "exact",
                "alternatives": []}

    # 2) substring: the longest covered name inside the query ("a salon in
    #    Kilimani"), or a query fragment (>=3 chars) inside a covered name
    #    ("kili" -> Kilimani). Longest fragment wins.
    best_name, best_len = None, 0
    for name in loc_names:
        nl = name.lower()
        if nl in ql and len(nl) > best_len:
            best_name, best_len = name, len(nl)
        if len(ql) >= 3 and ql in nl and len(nl) > best_len:
            best_name, best_len = name, len(nl)
    if best_name:
        return {"matched_name": best_name, "method": "substring",
                "alternatives": []}

    # 3) fuzzy -- catches typos / near-misses on the full name.
    close = difflib.get_close_matches(ql, list(lower_to_name), n=1, cutoff=0.78)
    if close:
        return {"matched_name": lower_to_name[close[0]], "method": "fuzzy",
                "alternatives": []}

    # 4) no match -- honest fallback: nearest real names by similarity.
    scored = sorted(
        ((name, difflib.SequenceMatcher(None, ql, name.lower()).ratio())
         for name in loc_names),
        key=lambda t: -t[1],
    )[:3]
    alternatives = [(name, round(ratio, 3)) for name, ratio in scored]
    return {"matched_name": None, "method": None, "alternatives": alternatives}


def format_alternatives(alternatives: list[tuple[str, float]]) -> str:
    """Human-readable 'closest covered areas' line for the fallback message."""
    if not alternatives:
        return ""
    parts = [f"{name} ({pct:.0%} name match)" for name, pct in alternatives]
    return ", ".join(parts)
