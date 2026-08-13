"""Nairobi Market-Gap Intelligence -- two query modes, glass-box.

Run:  venv\\Scripts\\python -m streamlit run creative_engine.py

This is the spec dashboard for the biz-intel pipeline. It reads
data/processed/biz_intel.db READ-ONLY and presents the two required output
modes as ranked tables computed from the `features` table:

  Mode 1 -- "I have a location"       location -> ranked categories
  Mode 2 -- "I have a business type"  category -> ranked locations

Glass-box rules (this is a location-intelligence product, not a black box):
  * Every number on screen is a real column from `features` joined to
    `locations`: business_count, businesses_per_1000_people, predicted_count,
    gap_residual, cluster_label, population, households, geo_method. Nothing
    is generated or invented; there is no mock data and no seeded randomness.
  * The derivation of every headline number is shown next to it:
        per 1,000 = business_count / population * 1000
        gap       = business_count - predicted_count   (regression residual)
        band      = per-category tercile of per-1,000 across all 30 locations
  * Tier-2 (informal, under-mapped) categories and the financing-confound
    category (motorcycle_taxi) are visibly flagged on every row and card.
  * The optional natural-language box only PARSES free text into a
    (location, category) pair and routes to a real query. It never generates
    analysis itself.
  * The only model-derived quantities shown are the regression residual
    (gap_residual) and the fitted model's CV MAE
    (data/processed/model_eval_regression.json). There is no invented
    "confidence" score.

DISCLAIMER (always on screen): this is a gap/opportunity signal based on
currently mapped OSM data and 2019 census figures -- NOT a profitability,
survival, or success prediction. Tier-2 counts are a floor, not a true count.
"""
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "processed" / "biz_intel.db"
EVAL_PATH = ROOT / "data" / "processed" / "model_eval_regression.json"

# Single source of truth for the taxonomy lives in the pipeline scripts --
# import it so this app can never drift from what the scripts actually fetch.
sys.path.insert(0, str(ROOT / "scripts"))
from overpass_client import CATEGORIES  # noqa: E402
from summaries import summarize_category, summarize_location  # noqa: E402

TIER2 = {c for c, m in CATEGORIES.items() if m["tier"] == 2}
CONFOUND = {c for c, m in CATEGORIES.items() if m["financing_confound"]}
CATEGORY_LABEL = {c: c.replace("_", " ") for c in CATEGORIES}

# ---------------------------------------------------------------------------
# Dataviz palette (validated default; see the dataviz skill palette.md)
# ---------------------------------------------------------------------------
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
SURFACE = "#fcfcfb"
BORDER = "rgba(11,11,11,0.10)"
# Band colors -- categorical identity, fixed order, never cycled.
BAND_COLORS = {"underserved": "#2a78d6", "moderate": "#eb6834", "saturated": "#1baf7a"}
BAND_HINT = {
    "underserved": "below the middle third of Nairobi locations for this category",
    "moderate": "in the middle third of Nairobi locations for this category",
    "saturated": "in the top third of Nairobi locations for this category",
}
# Status tokens for the disclaimer / badges (icon + label pairing, not color alone).
STATUS_WARN_BG, STATUS_WARN_FG = "#fff4e5", "#9a5b00"
STATUS_BAD_BG, STATUS_BAD_FG = "#ffe9e9", "#a32727"
STATUS_GOOD_BG, STATUS_GOOD_FG = "#dcf3e8", "#0d6b4a"

st.set_page_config(
    page_title="Nairobi Market-Gap Intelligence",
    page_icon="📍",
    layout="wide",
)

st.markdown("""
<style>
  .stApp { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }

  .app-title { font-size: 26px; font-weight: 700; letter-spacing: -0.02em; }
  .app-sub   { font-size: 14px; color: #52514e; margin-top: 4px; }

  .disclaimer {
    background: #fff4e5;
    border: 1px solid #f0d9b8;
    border-radius: 12px;
    padding: 12px 16px;
    font-size: 13.5px;
    color: #5b4a2a;
    margin-bottom: 18px;
  }
  .disclaimer b { color: #4a3a17; }

  .card {
    background: #fcfcfb;
    border: 1px solid rgba(11,11,11,0.10);
    border-radius: 14px;
    padding: 18px 20px;
  }
  .card-title { font-size: 13px; font-weight: 650; color: #52514e;
                text-transform: uppercase; letter-spacing: 0.04em; }

  .badge { display: inline-block; border-radius: 9px; padding: 1px 7px;
           font-size: 0.72rem; font-weight: 600; margin-right: 4px; }
  .badge-tier2 { background: #fff4e5; color: #9a5b00; }
  .badge-fin   { background: #ffe9e9; color: #a32727; }
  .badge-bbox  { background: #e8effb; color: #274a7d; }
  .badge-tier1 { background: #efefee; color: #3f3f3e; }

  .why-row  { display: flex; justify-content: space-between; gap: 12px;
              font-size: 13px; padding: 6px 0; border-top: 1px solid #efeee9; }
  .why-row .k { color: #52514e; }
  .why-row .v { font-variant-numeric: tabular-nums; font-weight: 600; color: #0b0b0b; }
  .why-row .v.mono { font-family: ui-monospace, "Cascadia Mono", monospace; font-weight: 600; }
  .why-formula { font-size: 12.5px; color: #898781; padding: 8px 0 0;
                 border-top: 1px solid #efeee9; font-variant-numeric: tabular-nums; }

  .section-title { font-size: 19px; font-weight: 700; margin: 0; }
  .section-sub   { font-size: 13px; color: #898781; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading (read-only, cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_db() -> dict | None:
    """Read every table the dashboard needs, once, into DataFrames.

    Returns None only if the DB file is missing -- the app then shows a clear
    "run the pipeline first" error instead of inventing data.
    """
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        locs = pd.read_sql_query(
            "SELECT location_id, location_name, sub_county, population, "
            "households, geo_method FROM locations ORDER BY location_name", conn)
        feat = pd.read_sql_query("SELECT * FROM features", conn)
    finally:
        conn.close()

    feat = feat.merge(locs, on="location_id", how="left")
    feat["tier"] = feat["category"].map(
        {c: m["tier"] for c, m in CATEGORIES.items()})
    feat["financing_confound"] = feat["category"].isin(CONFOUND)
    feat["tier2"] = feat["category"].isin(TIER2)
    return {"locations": locs, "features": feat}


@st.cache_data
def load_model_eval() -> dict | None:
    """Real fitted-model evaluation (for the glass-box 'model context' note)."""
    if not EVAL_PATH.exists():
        return None
    try:
        return json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Free-text parser: map words to a real location / category, or reject.
# It never generates analysis -- it only resolves an exact match from the
# tables the app reads, so arbitrary text can never masquerade as an area.
# ---------------------------------------------------------------------------
# Alias list per category: canonical key first, then common phrasings.
# "laundromat" is explicitly mapped to "laundry" (the shop=laundry tier-2
# category) -- the reason the old dashboard said it was "not a tracked sector".
CATEGORY_ALIASES: dict[str, list[str]] = {
    "laundry": ["laundry", "laundromat", "launderette", "dry cleaning", "drycleaner",
                "dry cleaner"],
    "motorcycle_taxi": ["motorcycle taxi", "motorbike taxi", "boda boda", "boda-boda",
                        "boda", "piki piki", "motorbike"],
    "supermarket": ["supermarket", "grocery store", "groceries", "grocery"],
    "pharmacy": ["pharmacy", "chemist", "drugstore", "drug store", "chemist's"],
    "restaurant": ["restaurant", "eatery", "eating place", "food court"],
    "hairdresser": ["hairdresser", "barber", "barbershop", "haircut", "barbers"],
    "clothes": ["clothes", "clothing", "clothes shop", "apparel", "boutique", "fashion"],
    "hardware": ["hardware", "hardware store", "building supplies", "ironmongery"],
    "salon": ["hair salon", "beauty salon", "beauty shop", "salon", "beautician"],
    "greengrocer": ["greengrocer", "greengrocery", "greengrocer's", "vegetable shop",
                    "produce"],
    "kiosk": ["kiosk", "duka", "convenience store", "corner shop"],
    "fast_food": ["fast food", "fastfood", "takeaway", "take away", "quick service"],
}
_ALIAS_LOOKUP = [(alias, cat) for cat, aliases in CATEGORY_ALIASES.items()
                 for alias in aliases]


def resolve_category(text: str) -> tuple[str | None, str]:
    """Return (canonical category, matched alias) or (None, '') if no match."""
    low = text.lower()
    best, best_alias = None, ""
    for alias, cat in _ALIAS_LOOKUP:
        if alias in low and len(alias) > len(best_alias):
            best, best_alias = cat, alias
    return best, best_alias


def resolve_location(text: str, loc_names: list[str]) -> tuple[str | None, str]:
    """Return (exact location name, matched fragment) or (None, '') if no match.

    Matches only against the real locations table -- the longest real name
    that appears in the text wins. Unknown text resolves to None (never a
    fabricated 'custom area').
    """
    low = text.lower()
    best, best_frag = None, ""
    for name in loc_names:
        frag = name.lower()
        if frag in low and len(frag) > len(best_frag):
            best, best_frag = name, frag
    return best, best_frag


# ---------------------------------------------------------------------------
# Ranked queries (real rows, ranked by model gap, most-underserved first)
# ---------------------------------------------------------------------------
def rank_location(data: dict, location: str, sort_key: str) -> pd.DataFrame:
    df = data["features"]
    df = df[df["location_name"] == location].copy()
    order = {
        "model gap": ["gap_residual", "businesses_per_1000_people"],
        "density per 1,000": ["businesses_per_1000_people", "gap_residual"],
        "mapped count": ["business_count", "gap_residual"],
    }
    cols = order.get(sort_key, order["model gap"])
    return df.sort_values(cols, ascending=True, na_position="last")


def rank_category(data: dict, category: str, sort_key: str) -> pd.DataFrame:
    df = data["features"]
    df = df[df["category"] == category].copy()
    order = {
        "model gap": ["gap_residual", "businesses_per_1000_people"],
        "density per 1,000": ["businesses_per_1000_people", "gap_residual"],
        "mapped count": ["business_count", "gap_residual"],
    }
    cols = order.get(sort_key, order["model gap"])
    return df.sort_values(cols, ascending=True, na_position="last")


# ---------------------------------------------------------------------------
# Flag / badge helpers
# ---------------------------------------------------------------------------
def flag_text(tier2: bool, fin: bool, geo: str) -> str:
    parts = []
    if tier2:
        parts.append("[tier2]")
    if fin:
        parts.append("[fin]")
    if geo == "bbox":
        parts.append("[bbox]")
    return " ".join(parts)


def badge_html(tier2: bool, fin: bool, geo: str, tier: int | None = None) -> str:
    out = []
    if fin:
        out.append("<span class='badge badge-fin'>financing-confound</span>")
    if tier2:
        out.append("<span class='badge badge-tier2'>tier-2 · under-mapped</span>")
    elif tier is not None:
        out.append(f"<span class='badge badge-tier1'>tier-{tier}</span>")
    if geo == "bbox":
        out.append("<span class='badge badge-bbox'>bbox · lower fidelity</span>")
    return "".join(out) if out else ""


def band_badge(band: str | None) -> str:
    if not band:
        return ("<span class='badge badge-tier1'>no band · too sparse</span>")
    color = BAND_COLORS[band]
    return (f"<span class='badge' style='background:{color}1a;color:{color};"
            f"border:1px solid {color}40'>{band}</span>")


# ---------------------------------------------------------------------------
# Glass-box renderers
# ---------------------------------------------------------------------------
def render_why_row(label: str, value: str, mono: bool = False) -> str:
    cls = "v mono" if mono else "v"
    return (f"<div class='why-row'><span class='k'>{label}</span>"
            f"<span class='{cls}'>{value}</span></div>")


def render_cell_card(r: pd.Series, model_eval: dict | None) -> None:
    """The full glass-box card for one (location, category) cell.

    Shows every number the headline derives from, with the arithmetic, and the
    badges the row carries -- never a bare score.
    """
    pop = int(r["population"])
    count = int(r["business_count"])
    per_1000 = float(r["businesses_per_1000_people"])
    pred = r["predicted_count"]
    gap = r["gap_residual"]
    band = r["cluster_label"]
    tier2 = bool(r["tier2"])
    fin = bool(r["financing_confound"])
    geo = r["geo_method"]

    rows = [
        render_why_row("Location", r["location_name"]),
        render_why_row("Population (2019 census)", f"{pop:,}"),
        render_why_row("Mapped businesses (OSM)", f"{count}"),
        render_why_row("Per 1,000 people", f"{per_1000:.4f}", mono=True),
        render_why_row("Expected (model)", "n/a" if pd.isna(pred) else f"{pred:.2f}",
                       mono=True),
        render_why_row("Gap residual", "n/a" if pd.isna(gap) else f"{gap:+.2f}",
                       mono=True),
    ]

    formula_lines = [f"per 1,000 = {count} ÷ {pop:,} × 1,000 = {per_1000:.4f}"]
    if not pd.isna(pred) and not pd.isna(gap):
        formula_lines.append(
            f"gap = {count} mapped − {pred:.2f} expected = {gap:+.2f} "
            f"(Negative Binomial GLM, GroupKFold-5)")

    band_line = ""
    if band:
        band_line = (f"Band <b>{band}</b>: {BAND_HINT[band]} "
                     f"(terciles of per-1,000 across all 30 locations).")
    else:
        band_line = ("Band <b>n/a</b>: this category has fewer than 5 locations "
                     "with a nonzero count, so a 3-way split is not meaningful.")

    flags_html = badge_html(tier2, fin, geo, int(r["tier"]))
    flag_note = []
    if fin:
        flag_note.append(
            "financing-confound: this category (boda boda / motorcycle taxi) can be "
            "driven by asset financing rather than market gap -- treat the signal "
            "with caution.")
    if tier2:
        flag_note.append(
            "tier-2 informal category: OSM is under-mapped here, so the mapped "
            "count is a floor, not a true count.")
    if geo == "bbox":
        flag_note.append(
            "bbox: this location's OSM data came from a geocoded bounding box, "
            "not an admin boundary -- lower fidelity.")

    model_note = ""
    if model_eval:
        model_note = (
            f"Model context: {model_eval.get('model_type', 'regression')} on "
            f"{model_eval.get('n_obs', '?')} cells, GroupKFold(5) CV MAE = "
            f"{model_eval.get('cv_mae', '?')}. Gaps smaller than the MAE sit "
            f"within model noise -- read their sign, not their magnitude.")

    note_html = ""
    if flag_note:
        note_html = "<div style='margin-top:10px;font-size:13px;color:#52514e'>" + \
            " · ".join(flag_note) + "</div>"

    st.markdown(
        f"<div class='card'>"
        f"<div class='card-title'>Focused cell · {r['category']} in {r['location_name']} "
        f"&nbsp; {flags_html}</div>"
        f"{''.join(rows)}"
        f"<div class='why-formula'>{'<br>'.join(formula_lines)}</div>"
        f"{'<div class=why-formula>' + band_line + '</div>' if band_line else ''}"
        f"{note_html}"
        f"{f'<div class=why-formula>{model_note}</div>' if model_note else ''}"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_why_panel(r: pd.Series, model_eval: dict | None,
                     headline: str) -> None:
    """Compact 'why' panel under a ranked table's top row."""
    pop = int(r["population"])
    count = int(r["business_count"])
    per_1000 = float(r["businesses_per_1000_people"])
    pred = r["predicted_count"]
    gap = r["gap_residual"]
    band = r["cluster_label"]

    formula_lines = [f"per 1,000 = {count} ÷ {pop:,} × 1,000 = {per_1000:.4f}"]
    if not pd.isna(pred) and not pd.isna(gap):
        formula_lines.append(
            f"gap = {count} mapped − {pred:.2f} expected = {gap:+.2f} "
            f"(Negative Binomial GLM, GroupKFold-5)")

    band_line = ""
    if band:
        band_line = (f"Band <b>{band}</b>: {BAND_HINT[band]} (terciles of "
                     f"per-1,000 across all 30 locations).")
    else:
        band_line = ("Band <b>n/a</b>: fewer than 5 locations with a nonzero "
                     "count for this category.")

    flags = badge_html(bool(r["tier2"]), bool(r["financing_confound"]),
                       r["geo_method"], int(r["tier"]))
    st.markdown(
        f"<div class='card'>"
        f"<div class='card-title'>Why the top row ranks first · {headline}</div>"
        f"<div class='why-formula' style='border-top:0;padding-top:0'>"
        f"{'<br>'.join(formula_lines)}</div>"
        f"{'<div class=why-formula>' + band_line + '</div>' if band_line else ''}"
        f"<div style='margin-top:8px'>{flags}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_table(df: pd.DataFrame, by_location: bool) -> None:
    """Styled ranked table for either mode, with the flag columns visible."""
    show = df.copy()
    show["tier_col"] = show["category"].map(CATEGORY_LABEL)
    if by_location:
        # Mode 1: one location, categories as rows.
        show["_label"] = show["tier_col"]
        col_order = ["_label", "tier", "business_count", "population",
                     "businesses_per_1000_people", "predicted_count",
                     "gap_residual", "cluster_label", "flag_text"]
    else:
        # Mode 2: one category, locations as rows.
        show["_label"] = show["location_name"]
        col_order = ["_label", "tier", "business_count", "population",
                     "businesses_per_1000_people", "predicted_count",
                     "gap_residual", "cluster_label", "flag_text"]
    show["flag_text"] = [
        flag_text(bool(t2), bool(fin), geo)
        for t2, fin, geo in zip(show["tier2"], show["financing_confound"],
                                show["geo_method"])]
    st.dataframe(
        show[col_order],
        width="stretch",
        hide_index=True,
        column_config={
            "_label": st.column_config.TextColumn(
                "Category" if by_location else "Location"),
            "tier": st.column_config.NumberColumn("Tier", format="%d"),
            "business_count": st.column_config.NumberColumn("Mapped", format="%d"),
            "population": st.column_config.NumberColumn("Population", format="%d"),
            "businesses_per_1000_people": st.column_config.NumberColumn(
                "Per 1,000", format="%.3f"),
            "predicted_count": st.column_config.NumberColumn("Expected", format="%.2f"),
            "gap_residual": st.column_config.NumberColumn("Gap", format="%+.2f"),
            "cluster_label": st.column_config.TextColumn("Band"),
            "flag_text": st.column_config.TextColumn("Flags"),
        },
    )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main() -> None:
    st.markdown(
        "<div class='app-title'>📍 Nairobi Market-Gap Intelligence</div>"
        "<div class='app-sub'>Ranked market-gap signals for the 30 mapped "
        "Nairobi census locations and the 12 tracked categories — computed "
        "from the pipeline database, every number traceable to a real column.</div>",
        unsafe_allow_html=True)
    st.write("")

    st.markdown(
        "<div class='disclaimer'><b>DISCLAIMER — read this before interpreting "
        "anything.</b> This is a <b>gap / opportunity signal</b> based on "
        "<b>currently mapped OpenStreetMap data</b> and <b>2019 Kenya Census "
        "figures</b>. It is <b>NOT a profitability, survival, or success "
        "prediction</b>. Tier-2 (informal) categories — salon, greengrocer, "
        "kiosk, fast food, laundry, boda boda — are <b>under-mapped</b> in OSM, "
        "so their counts are a <b>floor, not a true count</b>.</div>",
        unsafe_allow_html=True)

    data = load_db()
    if data is None:
        st.error(
            "Data file not found (data/processed/biz_intel.db). Run the "
            "pipeline first — see RUNBOOK.md Option B — then reload this app."
        )
        return
    model_eval = load_model_eval()

    loc_names = data["locations"]["location_name"].tolist()
    cat_names = sorted(CATEGORIES)

    # ------------------------------------------------------------------ query
    st.markdown("#### Build your query")
    c_nl, c_mode = st.columns([3, 2])
    with c_nl:
        nl_text = st.text_input(
            "Natural language (optional)",
            placeholder='e.g. "laundromat in Roysambu", or just "Kilimani" or "pharmacy"',
            label_visibility="collapsed",
            key="nl_query")
    with c_mode:
        mode = st.radio(
            "Or choose a mode",
            ["I have a location", "I have a business type"],
            index=0, horizontal=True, key="mode_radio",
            label_visibility="collapsed")

    # Route: the NL box, when it resolves, wins. It never invents; it only
    # picks real values from the tables, then runs the same ranked query.
    nl_loc, nl_loc_frag = (None, "")
    nl_cat, nl_cat_alias = (None, "")
    if nl_text and nl_text.strip():
        nl_loc, nl_loc_frag = resolve_location(nl_text.strip(), loc_names)
        nl_cat, nl_cat_alias = resolve_category(nl_text.strip())

    both_resolved = nl_loc and nl_cat
    loc_only = nl_loc and not nl_cat
    cat_only = nl_cat and not nl_loc

    if nl_text and nl_text.strip() and not (both_resolved or loc_only or cat_only):
        st.warning(
            "I couldn't match that to a real location or category. Locations "
            "are validated against the mapped set; categories against the "
            "tracked taxonomy. Try e.g. 'laundromat in Roysambu', 'Kilimani', "
            "or 'pharmacy'."
        )

    st.divider()

    # ----------------------------------------------------------------- mode 1
    mode1_on = (mode == "I have a location") or loc_only
    mode2_on = (mode == "I have a business type") or cat_only

    if both_resolved:
        _render_focused(data, model_eval, nl_loc, nl_cat, nl_cat_alias,
                        nl_loc_frag)
        st.divider()
        _render_mode2(data, model_eval, nl_cat, highlight=nl_loc,
                      note=f"from your sentence ('{nl_cat_alias}')")
        return

    if mode1_on:
        _render_mode1(data, model_eval, loc_names, nl_loc, nl_loc_frag)
    if mode2_on:
        _render_mode2(data, model_eval, nl_cat,
                      note=f"from your sentence ('{nl_cat_alias}')" if nl_cat else None)

    st.divider()
    _render_footer(model_eval)


def _render_focused(data: dict, model_eval: dict | None, loc: str, cat: str,
                    cat_alias: str, loc_frag: str) -> None:
    st.markdown("#### Focused cell")
    st.caption(
        f"Both a location ({loc!r}) and a category ({cat_alias!r} → "
        f"<b>{cat}</b>) were matched from your sentence. Here is that exact "
        f"cell from the features table, with every number it derives from.",
        unsafe_allow_html=True)
    row = data["features"][(data["features"]["location_name"] == loc)
                           & (data["features"]["category"] == cat)]
    if row.empty:
        st.info(
            f"No feature row for {loc} + {cat} — the OSM fetch for that "
            "location×category never produced a row (failed or not run), so "
            "the model cannot score it. Nothing is invented for missing data."
        )
        return
    render_cell_card(row.iloc[0], model_eval)


def _render_mode1(data: dict, model_eval: dict | None, loc_names: list[str],
                  nl_loc: str | None, nl_loc_frag: str) -> None:
    st.markdown("#### Mode 1 · I have a location")
    st.caption(
        "Pick one of the 30 mapped locations → ranked categories for it, "
        "most-underserved first. Input is validated against the locations "
        "table — arbitrary text is never treated as a real area.")

    idx = loc_names.index(nl_loc) if nl_loc in loc_names else 0
    location = st.selectbox("Location", loc_names, index=idx, key="mode1_loc")
    if nl_loc and nl_loc_frag:
        st.caption(f"ℹ️ Routed from your sentence ('{nl_loc_frag}').")

    row = data["features"][data["features"]["location_name"] == location]
    if row.empty:
        st.info("No feature rows for this location — it had no fetched data.")
        return

    sort_key = st.radio("Rank by", ["model gap", "density per 1,000", "mapped count"],
                        index=0, horizontal=True, key="mode1_sort")
    df = rank_location(data, location, sort_key)
    pop = int(row["population"].iloc[0])
    method = row["geo_method"].iloc[0]
    st.caption(f"Population {pop:,} · geo source: {method}")

    render_why_panel(df.iloc[0], model_eval,
                     f"#{df.iloc[0]['category']} in {location}")
    st.write("")
    render_table(df, by_location=True)
    st.write("")

    # Generative summary -- grounded strictly in the rows above.
    conn = _ro_conn()
    try:
        summary = summarize_location(conn, location)
    finally:
        conn.close()
    if summary["text"]:
        st.markdown(f"**What the numbers say** — {summary['text']}")


def _render_mode2(data: dict, model_eval: dict | None, nl_cat: str | None,
                  highlight: str | None = None, note: str | None = None) -> None:
    st.markdown("#### Mode 2 · I have a business type")
    st.caption(
        "Pick one of the 12 tracked categories (the real Tier 1 + Tier 2 "
        "taxonomy) → ranked locations for it, most-underserved first.")

    cat_idx = cat_names.index(nl_cat) if nl_cat in cat_names else 0
    category = st.selectbox("Category", cat_names, index=cat_idx, key="mode2_cat",
                            format_func=lambda c: CATEGORY_LABEL[c])
    if note:
        st.caption(f"ℹ️ {note}")

    meta = CATEGORIES[category]
    st.caption(
        f"Tier {meta['tier']} · "
        f"{'financing-confound' if meta['financing_confound'] else 'market-driven'} · "
        f"OSM tag: `{meta['tag']}`")

    row = data["features"][data["features"]["category"] == category]
    if row.empty:
        st.info("No feature rows for this category.")
        return

    sort_key = st.radio("Rank by", ["model gap", "density per 1,000", "mapped count"],
                        index=0, horizontal=True, key="mode2_sort")
    df = rank_category(data, category, sort_key)

    render_why_panel(df.iloc[0], model_eval,
                     f"{category} · {df.iloc[0]['location_name']}")
    st.write("")
    render_table(df, by_location=False)
    if highlight:
        st.caption(f"★ {highlight!r} is the location you named.")

    st.write("")
    conn = _ro_conn()
    try:
        summary = summarize_category(conn, category)
    finally:
        conn.close()
    if summary["text"]:
        st.markdown(f"**What the numbers say** — {summary['text']}")


def _ro_conn() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def _render_footer(model_eval: dict | None) -> None:
    st.markdown("#### How every number is derived")
    st.markdown(
        "| Column | Derivation | Source |\n"
        "|---|---|---|\n"
        "| **Mapped** | `business_count` — count of mapped OSM nodes in the location×category | `features` |\n"
        "| **Per 1,000** | `business_count ÷ population × 1,000` | `features` |\n"
        "| **Expected** | `predicted_count` — Negative Binomial GLM on log-population, households, geo method, category | `features` (fitted by `10_model_regression.py`) |\n"
        "| **Gap** | `gap_residual = business_count − predicted_count` — the regression residual. Negative = below expected (underserved) | `features` |\n"
        "| **Band** | per-category tercile of per-1,000 across the 30 locations (33rd / 67th percentile); `n/a` if <5 locations have a nonzero count | `features` (assigned by `08_cluster.py`) |\n"
        "| **Tier** | 1 = formal/OSM-reliable; 2 = informal/under-mapped (count is a floor) | taxonomy (`overpass_client.py`) |\n"
        "| **Flags** | `[tier2]` informal · `[fin]` financing-confound · `[bbox]` lower-fidelity geocode | taxonomy + `locations.geo_method` |\n"
    )
    if model_eval:
        st.caption(
            f"Fitted model: {model_eval.get('model_type')} on "
            f"{model_eval.get('n_obs')} cells across {model_eval.get('n_locations')} "
            f"locations × {model_eval.get('n_categories')} categories; "
            f"GroupKFold(5) CV MAE = {model_eval.get('cv_mae')} (RMSE "
            f"{model_eval.get('cv_rmse')}). R² is not reported — it is misleading "
            f"for count data."
        )
    st.caption(
        "Sources: Kenya 2019 Census (population, households) and OpenStreetMap "
        "node counts fetched per location/category. This is a gap/opportunity "
        "signal based on currently mapped OSM data and 2019 census figures — "
        "not a profitability or success prediction. Tier-2 counts are "
        "undercounts."
    )


if __name__ == "__main__":
    main()
