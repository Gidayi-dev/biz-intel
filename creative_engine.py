"""
Creative Project Recommendation Engine
--------------------------------------

A Streamlit app that turns a free-form *location* + *industry* into:

  1. a numerical model predictive confidence score (0-100),
  2. a structural market-trend chart (mock array data, seeded per query),
  3. a generated creative business alternative path for brainstorming.

Design intent
-------------
This is a *front-end experiment layer* over the biz-intel pipeline. It reads
data/processed/biz_intel.db READ-ONLY when available to ground the score in
real market-gap data, and falls back to deterministic mock data otherwise, so
the app never hard-crashes on a missing database.

Production-grade error handling
-------------------------------
- Empty / whitespace-only text inputs degrade to a friendly default + note
  (never a traceback).
- Every user-triggered code path is wrapped in try/except; a failure renders
  st.error and the app keeps running.
- Trend data is MOCK by design and seeded per (location, industry), so the
  same query always returns the same chart.

Run:
    venv\\Scripts\\python -m streamlit run creative_engine.py
"""
import hashlib
import random
import sqlite3
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths & taxonomy (single source of truth lives in the pipeline scripts)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "processed" / "biz_intel.db"

# Pull the category taxonomy from the pipeline so this app can never drift
# from what the scripts actually fetch.
sys.path.insert(0, str(ROOT / "scripts"))
try:
    from overpass_client import CATEGORIES  # noqa: E402
except Exception:  # pragma: no cover - the scripts are part of the repo
    # Minimal fallback taxonomy if the scripts module cannot be imported
    # (e.g. the app is deployed as a standalone folder).
    CATEGORIES = {c: {"tier": 1, "financing_confound": False} for c in [
        "supermarket", "pharmacy", "restaurant", "hairdresser", "clothes",
        "hardware", "salon", "greengrocer", "kiosk", "fast_food", "laundry",
        "motorcycle_taxi",
    ]}

KNOWN_INDUSTRIES = sorted(CATEGORIES)

# ---------------------------------------------------------------------------
# Dataviz palette (validated default, see the dataviz skill)
# ---------------------------------------------------------------------------
COLOR_DEMAND = "#2a78d6"   # blue
COLOR_SUPPLY = "#eb6834"   # orange
COLOR_AQUA = "#1baf7a"     # success / opportunity
COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#898781"
COLOR_CARD_BG = "#fcfcfb"

# ---------------------------------------------------------------------------
# App config + modern scannable styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Creative Project Recommendation Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  /* base: keep the system sans, reset streamlit chrome for a kiosk feel */
  .stApp { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }

  .app-title { font-size: 26px; font-weight: 700; letter-spacing: -0.02em; }
  .app-sub   { font-size: 14px; color: #52514e; margin-top: 4px; }

  /* input-block cards in the sidebar */
  .input-card {
    background: #fcfcfb;
    border: 1px solid rgba(11,11,11,0.10);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 14px;
  }
  .input-card-head { font-size: 13px; font-weight: 650; color: #0b0b0b;
                     display: flex; align-items: center; gap: 8px; }
  .input-card-sub  { font-size: 12px; color: #898781; margin-top: 2px;
                     margin-bottom: 10px; }

  /* main-area cards */
  .card {
    background: #fcfcfb;
    border: 1px solid rgba(11,11,11,0.10);
    border-radius: 14px;
    padding: 18px 20px;
    height: 100%;
  }
  .card-title { font-size: 13px; font-weight: 650; color: #52514e;
                text-transform: uppercase; letter-spacing: 0.04em; }

  /* confidence score */
  .score-number { font-size: 52px; font-weight: 700; line-height: 1;
                  font-variant-numeric: tabular-nums; letter-spacing: -0.03em; }
  .score-label  { font-size: 13px; color: #52514e; margin-top: 6px; }
  .score-factor { display: flex; justify-content: space-between;
                  font-size: 13px; padding: 6px 0;
                  border-top: 1px solid #efeee9; }

  /* pills / chips */
  .pill { display: inline-block; padding: 2px 11px; border-radius: 999px;
          font-size: 12px; font-weight: 500; }
  .pill-blue   { background: #e3eefc; color: #1c5cab; }
  .pill-orange { background: #fdeadd; color: #b34a1a; }
  .pill-green  { background: #dcf3e8; color: #0d6b4a; }

  .section-title { font-size: 19px; font-weight: 700; margin: 0; }
  .section-sub   { font-size: 13px; color: #898781; margin-top: 2px; }

  /* brainstorm success path */
  .success-path {
    background: linear-gradient(135deg, #0d6b4a, #1baf7a);
    border-radius: 14px;
    padding: 22px 24px;
    color: #ffffff;
  }
  .success-path .path-kicker { font-size: 12px; font-weight: 650;
                               text-transform: uppercase; letter-spacing: 0.05em;
                               opacity: 0.85; }
  .success-path .path-name  { font-size: 24px; font-weight: 700;
                              margin: 6px 0 4px; letter-spacing: -0.01em; }
  .success-path .path-tagline { font-size: 14px; opacity: 0.92; }
  .phase { display: flex; gap: 12px; align-items: flex-start;
           margin-top: 14px; }
  .phase-num { background: rgba(255,255,255,0.18); border-radius: 999px;
               width: 26px; height: 26px; flex: none;
               display: flex; align-items: center; justify-content: center;
               font-size: 13px; font-weight: 650; }
  .phase-title { font-weight: 650; font-size: 14px; }
  .phase-body  { font-size: 13px; opacity: 0.9; margin-top: 1px; }

  /* secondary idea card */
  .idea-card { background: #fcfcfb; border: 1px solid rgba(11,11,11,0.10);
               border-radius: 12px; padding: 16px 18px; }
  .idea-icon { font-size: 22px; }
  .idea-name { font-weight: 650; font-size: 15px; margin-top: 6px; }
  .idea-pitch { font-size: 13px; color: #52514e; margin-top: 3px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data loading (read-only, graceful fallback)
# ---------------------------------------------------------------------------
@st.cache_data
def load_market_context() -> dict | None:
    """Load a read-only snapshot of the pipeline DB. Returns None if missing.

    We wrap the whole thing so a missing/corrupt DB (or a deployment without
    the data folder) degrades the app to mock mode instead of crashing it.
    """
    try:
        if not DB_PATH.exists():
            return None
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        try:
            locs = pd.read_sql_query(
                "SELECT location_id, location_name, sub_county, population, "
                "geo_method FROM locations", conn)
            feat = pd.read_sql_query(
                "SELECT location_id, category, business_count, "
                "businesses_per_1000_people, cluster_label FROM features", conn)
        finally:
            conn.close()

        name_to_id = dict(zip(locs["location_name"], locs["location_id"]))
        loc_map = locs.set_index("location_name").to_dict("index")
        feat_map = {}
        for _, r in feat.iterrows():
            feat_map[(r["location_id"], r["category"])] = {
                "count": int(r["business_count"]),
                "per_1000": float(r["businesses_per_1000_people"]),
                "band": r["cluster_label"],
            }
        return {
            "names": sorted(name_to_id),
            "name_to_id": name_to_id,
            "locations": loc_map,
            "features": feat_map,
        }
    except Exception:
        return None


MARKET = load_market_context()


def stable_seed(*parts: str) -> int:
    """Deterministic seed from arbitrary strings (same input -> same output)."""
    key = "|".join(parts).strip().lower()
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# Input resolution — never crash on empty/missing text
# ---------------------------------------------------------------------------
def resolve_location(raw: str) -> tuple[str, bool, str]:
    """Normalise a free-form location string.

    Returns (display_name, matched_to_pilot, note). Empty / whitespace input
    and unknown names degrade gracefully instead of raising.
    """
    text = (raw or "").strip()
    if not text:
        return "Nairobi (general)", False, "No location entered — using a generic Nairobi baseline."
    if MARKET is not None:
        low = text.lower()
        hits = [n for n in MARKET["names"]
                if low == n.lower() or low in n.lower() or n.lower() in low]
        if hits:
            best = max(hits, key=len)
            return best, True, ""
        return text.title(), False, (
            f"'{text}' is not in the pilot area list — treated as a custom location.")
    return text.title(), False, ""


def resolve_industry(raw: str) -> tuple[str, bool, str]:
    """Normalise a free-form industry string the same way."""
    text = (raw or "").strip()
    if not text:
        return "local business", False, "No industry entered — using a general 'local business' lens."
    low = text.lower()
    exact = [k for k in KNOWN_INDUSTRIES if k == low]
    if exact:
        return exact[0], True, ""
    fuzzy = [k for k in KNOWN_INDUSTRIES if low in k or k in low]
    if fuzzy:
        best = max(fuzzy, key=len)
        return best, True, f"Matched '{text}' to the closest known sector: '{best}'."
    return text.title(), False, (
        f"'{text}' is not a tracked sector — using a general retail/services lens.")


# ---------------------------------------------------------------------------
# 1) Model predictive confidence score (numerical card)
# ---------------------------------------------------------------------------
def compute_confidence(loc_display: str, ind_display: str,
                       loc_matched: bool, ind_matched: bool) -> tuple[int, list]:
    """Return (score 0-100, [('label', '+/-delta'), ...]).

    Base is deterministic from the query; grounded in real pipeline data
    (density + cluster band) when the location/category is known, so the
    score is stable across reruns and moves with the data.
    """
    rng = random.Random(stable_seed(loc_display, ind_display))
    score = 62 + rng.randint(0, 16)          # 62-78 base
    factors: list[tuple[str, str]] = []

    if MARKET is not None and loc_matched and ind_matched:
        lid = MARKET["name_to_id"].get(loc_display)
        feat = MARKET["features"].get((lid, ind_display)) if lid is not None else None
        geo_method = MARKET["locations"].get(loc_display, {}).get("geo_method")
        if feat is not None:
            band = feat["band"]
            if band == "underserved":
                score += 14
                factors.append(("Underserved supply (cluster band)", "+14"))
            elif band == "moderate":
                score += 5
                factors.append(("Moderate supply (cluster band)", "+5"))
            elif band == "saturated":
                score -= 8
                factors.append(("Saturated supply (cluster band)", "-8"))
            if feat["per_1000"] < 0.05:
                score += 6
                factors.append(("Very thin mapped supply", "+6"))
            if geo_method == "bbox":
                score -= 3
                factors.append(("Geocoded bbox (lower fidelity)", "-3"))
            if feat["count"] == 0 and feat["per_1000"] == 0:
                score += 8
                factors.append(("Zero mapped businesses (gap or unmapped)", "+8"))
        else:
            factors.append(("No feature data for this pair", "+0"))
    elif not loc_matched or not ind_matched:
        factors.append(("Generalised input (mock baseline)", "+0"))

    # Clamp into a sane band and round.
    score = int(round(min(100, max(3, score))))
    return score, factors


# ---------------------------------------------------------------------------
# 2) Structural market trend chart (MOCK array data, seeded)
# ---------------------------------------------------------------------------
def build_trend_data(loc_display: str, ind_display: str) -> pd.DataFrame:
    """Deterministic mock demand/supply trend over 8 quarters.

    Supply structurally lags demand, which is the visual story the chart
    tells: a widening opportunity gap.
    """
    rng = random.Random(stable_seed(loc_display, ind_display, "trend"))
    demand = 58.0 + rng.uniform(0, 14)
    supply = demand * rng.uniform(0.55, 0.72)
    growth = rng.uniform(1.5, 3.0)
    rows = []
    for q in range(1, 9):
        demand += growth + rng.uniform(-0.8, 0.8)
        supply += (growth - 0.8) + rng.uniform(-1.2, 0.4)   # lags demand
        rows.append({
            "period": f"Q{q}",
            "Market demand": max(0.0, round(demand, 1)),
            "Local supply": max(0.0, round(supply, 1)),
        })
    return pd.DataFrame(rows)


def render_trend_chart(trend_df: pd.DataFrame) -> None:
    """Line chart: demand vs supply over quarters."""
    long = trend_df.melt(id_vars="period", var_name="series", value_name="index")
    chart = (
        alt.Chart(long)
        .mark_line(point=alt.OverlayMarkDef(size=55), strokeWidth=2.5)
        .encode(
            x=alt.X("period:N", title="Quarter",
                    axis=alt.Axis(labelAngle=0)),
            y=alt.Y("index:Q", title="Index (mock)", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=["Market demand", "Local supply"],
                    range=[COLOR_DEMAND, COLOR_SUPPLY]),
                legend=alt.Legend(title=None, orient="top")),
            tooltip=["period", "series", "index"],
        )
        .properties(height=300)
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False, domainColor="#c3c2b7",
                        labelColor="#52514e", titleColor="#52514e")
    )
    st.altair_chart(chart, width="stretch")


# ---------------------------------------------------------------------------
# 3) Creative business alternative path (brainstorm)
# ---------------------------------------------------------------------------
LOCATION_CHARACTER = {
    "kilimani":    {"icon": "🏙️", "tagline": "young professionals & upscale expat living"},
    "kasarani":    {"icon": "🏠", "tagline": "family suburbs around the sports complex"},
    "kibera":      {"icon": "🚶", "tagline": "dense high-footfall informal economy"},
    "eastleigh":   {"icon": "🛒", "tagline": "bustling trade hub with strong diaspora commerce"},
    "karen":       {"icon": "🌳", "tagline": "affluent low-density estates"},
    "cbd":         {"icon": "🏢", "tagline": "office-worker & commuter daytime crowds"},
    "embakasi":    {"icon": "🏗️", "tagline": "fast-growing residential sprawl"},
    "umoja":       {"icon": "🏘️", "tagline": "mid-density established estates"},
    "dandora":     {"icon": "🏗️", "tagline": "dense working-class residential area"},
    "mathare":     {"icon": "🚶", "tagline": "compact high-density informal neighbourhood"},
    "kawangware":  {"icon": "🚶", "tagline": "dense market-driven trading corridor"},
    "ruai":        {"icon": "🌾", "tagline": "expanding peri-urban settlement"},
}

INDUSTRY_LABEL = {
    "supermarket": "grocery & daily essentials",
    "pharmacy": "health & wellness",
    "restaurant": "dining & food service",
    "hairdresser": "grooming & styling",
    "salon": "beauty & grooming",
    "clothes": "apparel & fashion",
    "hardware": "building & DIY supplies",
    "greengrocer": "fresh produce",
    "kiosk": "convenience retail",
    "fast_food": "quick-service food",
    "laundry": "cleaning & laundry services",
    "motorcycle_taxi": "last-mile transport",
}

# Creative innovation frames ("pivots") used to brainstorm alternative paths.
PIVOTS = [
    {"icon": "🚚", "title": "On-Demand / Mobile", "hook":
     "bring the service to customers with doorstep or pop-up delivery"},
    {"icon": "🤝", "title": "Community Co-op", "hook":
     "pool demand and ownership through a residents' cooperative model"},
    {"icon": "📦", "title": "Subscription Bundle", "hook":
     "convert one-off purchases into a recurring bundled plan"},
    {"icon": "🎪", "title": "Rotating Pop-up", "hook":
     "share a rotating space with complementary businesses to cut rent"},
    {"icon": "📱", "title": "Tech-Enabled", "hook":
     "use a lightweight app / WhatsApp channel for booking, orders and payments"},
    {"icon": "🌙", "title": "Night & Early Hours", "hook":
     "serve the after-hours or dawn commuter window competitors ignore"},
    {"icon": "♻️", "title": "Circular & Waste-Free", "hook":
     "build reuse, refill and recycling into the core offer"},
    {"icon": "🧩", "title": "Retail + Experience", "hook":
     "pair the core product with an experience layer (classes, trials, lounge)"},
]

PHASES = [
    ("Validate", "run a low-cost pop-up or mobile pilot for 2-4 weeks and "
                 "measure real uptake before committing to a lease"),
    ("Ramp", "lock the model — subscription, memberships or pre-orders — to "
             "turn early demand into recurring revenue"),
    ("Scale", "expand to neighbouring wards or partner with complementary "
              "businesses to multiply reach without multiplying rent"),
]


def short_name(location: str) -> str:
    """First word of the location, minus noise words."""
    return location.split()[0]


def industry_label(ind_display: str) -> str:
    return INDUSTRY_LABEL.get(ind_display.lower(), ind_display.title())


def generate_ideas(loc_display: str, ind_display: str) -> list[dict]:
    """Return 3 brainstormed concepts, primary first."""
    rng = random.Random(stable_seed(loc_display, ind_display, "ideas"))
    char = LOCATION_CHARACTER.get(loc_display.lower(),
                                  {"icon": "🌆", "tagline": "a growing Nairobi neighbourhood"})
    sector = industry_label(ind_display)
    area = short_name(loc_display)

    pivots = PIVOTS.copy()
    rng.shuffle(pivots)

    ideas = []
    for pivot in pivots[:3]:
        ideas.append({
            "icon": pivot["icon"],
            "name": f"{area} {pivot['title']} {sector.title()}",
            "pivot": pivot["title"],
            "pitch": (
                f"A {sector} concept for {loc_display} ({char['tagline']}) that "
                f"{pivot['hook']}."
            ),
        })
    return ideas


# ---------------------------------------------------------------------------
# The "background" job triggered by the submit button
# ---------------------------------------------------------------------------
def run_recommendation(loc_display: str, ind_display: str,
                       loc_note: str, ind_note: str,
                       loc_matched: bool = False,
                       ind_matched: bool = False) -> dict:
    """Execute the full recommendation pipeline as a simulated background job.

    In a real deployment this is the seam where you'd push the work to a
    thread/queue (concurrent.futures, Celery, etc.). The progress widget
    makes the wait legible; the results are stored in session_state so the
    main view is stable across reruns.
    """
    steps = [
        "Resolving location geometry",
        "Loading sector supply profile",
        "Computing predictive confidence",
        "Projecting market trend",
        "Generating creative alternatives",
    ]
    progress = st.progress(0, text="Starting…")
    try:
        for i, label in enumerate(steps, start=1):
            progress.progress(i / len(steps), text=label)
            import time
            time.sleep(0.35)  # simulate model latency; remove in production
        progress.progress(1.0, text="Done")
    finally:
        progress.empty()

    trend_df = build_trend_data(loc_display, ind_display)
    score, factors = compute_confidence(loc_display, ind_display,
                                        loc_matched, ind_matched)
    ideas = generate_ideas(loc_display, ind_display)

    return {
        "location": loc_display,
        "industry": ind_display,
        "loc_note": loc_note,
        "ind_note": ind_note,
        "score": score,
        "factors": factors,
        "trend": trend_df,
        "ideas": ideas,
    }


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------
def render_header() -> None:
    st.markdown(
        "<div class='app-title'>🧠 Creative Project Recommendation Engine</div>"
        "<div class='app-sub'>Type a local area + industry, and get a scored "
        "opportunity signal plus creative business alternatives to brainstorm from.</div>",
        unsafe_allow_html=True)
    st.write("")


def render_confidence_card(score: int, factors: list) -> None:
    color = COLOR_AQUA if score >= 70 else (COLOR_SUPPLY if score >= 55 else "#d03b3b")
    factor_html = "".join(
        f"<div class='score-factor'><span>{label}</span><b style='color:{'#0ca30c' if d.startswith('+') else '#d03b3b'}'>{d}</b></div>"
        for label, d in factors)
    st.markdown(
        f"<div class='card'>"
        f"<div class='card-title'>🎯 Predictive confidence</div>"
        f"<div style='margin-top:12px'>"
        f"<span class='score-number' style='color:{color}'>{score}</span>"
        f"<span style='font-size:16px;color:#898781'> / 100</span></div>"
        f"<div class='score-label'>Model confidence that this pair is worth exploring</div>"
        f"<div style='margin:14px 0'>"
        f"<progress value='{score}' max='100' "
        f"style='width:100%;height:8px;border-radius:999px;"
        f"accent-color:{color}'></progress></div>"
        f"{factor_html}"
        f"</div></div>",
        unsafe_allow_html=True)


def render_primary_path(primary: dict, loc_display: str) -> None:
    phases_html = ""
    for n, (title, body) in enumerate(PHASES, start=1):
        phases_html += (
            f"<div class='phase'><div class='phase-num'>{n}</div>"
            f"<div><div class='phase-title'>{title}</div>"
            f"<div class='phase-body'>{body}</div></div></div>")
    st.markdown(
        f"<div class='success-path'>"
        f"<div class='path-kicker'>✅ Recommended creative path for {loc_display}</div>"
        f"<div class='path-name'>{primary['icon']} {primary['name']}</div>"
        f"<div class='path-tagline'>{primary['pitch']}</div>"
        f"{phases_html}"
        f"</div>",
        unsafe_allow_html=True)


def render_idea_card(idea: dict) -> None:
    st.markdown(
        f"<div class='idea-card'>"
        f"<div class='idea-icon'>{idea['icon']}</div>"
        f"<div class='idea-name'>{idea['name']}</div>"
        f"<div class='idea-pitch'>{idea['pitch']}</div>"
        f"<div style='margin-top:10px'><span class='pill pill-blue'>{idea['pivot']}</span></div>"
        f"</div>",
        unsafe_allow_html=True)


def render_input_chips(loc: str, ind: str) -> None:
    st.markdown(
        f"<span class='pill pill-blue'>📍 {loc}</span> &nbsp; "
        f"<span class='pill pill-orange'>💼 {ind}</span>",
        unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar — two text + dropdown input blocks
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧭 Build your query")
    st.markdown("Fill either field — or both — and press **Generate**.")

    # ---- Input block 1: location --------------------------------------
    st.markdown(
        "<div class='input-card'>"
        "<div class='input-card-head'>📍 Location</div>"
        "<div class='input-card-sub'>Any local area name, e.g. Kilimani</div>",
        unsafe_allow_html=True)
    loc_text = st.text_input(
        "Location (type freely)", value="", key="loc_text",
        placeholder="e.g. Kilimani, Kasarani, Kibera…",
        label_visibility="collapsed")
    loc_options = MARKET["names"] if MARKET is not None else []
    loc_sel = st.selectbox(
        "…or pick a pilot area", options=loc_options,
        placeholder="Choose a pilot area…", index=None,
        key="loc_sel", label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Input block 2: industry ---------------------------------------
    st.markdown(
        "<div class='input-card'>"
        "<div class='input-card-head'>💼 Industry / focus sector</div>"
        "<div class='input-card-sub'>Any business type, e.g. salon</div>",
        unsafe_allow_html=True)
    ind_text = st.text_input(
        "Industry (type freely)", value="", key="ind_text",
        placeholder="e.g. salon, pharmacy, restaurant…",
        label_visibility="collapsed")
    ind_sel = st.selectbox(
        "…or pick a sector", options=KNOWN_INDUSTRIES,
        placeholder="Choose a sector…", index=None,
        key="ind_sel", label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Submit ---------------------------------------------------------
    submitted = st.button(
        "🎯 Generate recommendation", type="primary",
        width="stretch")

    st.caption("Scores are grounded in real market-gap data where available; "
               "the trend chart is prototype mock data.")

# ---------------------------------------------------------------------------
# Main view — run the job on submit, then render results
# ---------------------------------------------------------------------------
render_header()

if submitted:
    # Resolve whichever inputs the user touched: typed text wins, else the
    # dropdown. Both empty is handled gracefully by resolve_* (never crashes).
    raw_loc = (loc_text or "").strip() or loc_sel
    raw_ind = (ind_text or "").strip() or ind_sel

    try:
        loc_display, loc_matched, loc_note = resolve_location(raw_loc)
        ind_display, ind_matched, ind_note = resolve_industry(raw_ind)
        result = run_recommendation(loc_display, ind_display,
                                    loc_note, ind_note,
                                    loc_matched, ind_matched)
        st.session_state["result"] = result
    except Exception as exc:  # production guard: never let the app die
        st.error(f"Something went wrong while generating. {exc}")
        st.session_state.pop("result", None)

result = st.session_state.get("result")

if result is None:
    # Empty state — inviting, never an error.
    st.info(
        "👋 **Welcome.** Type a **location** (e.g. *Kilimani*) and an "
        "**industry** (e.g. *salon*) in the sidebar, then press "
        "*Generate recommendation* to see a scored opportunity signal and "
        "creative business alternatives.")

else:
    loc_display = result["location"]
    ind_display = result["industry"]

    # Query chips + any graceful-degradation notes from empty/unknown input.
    st.markdown("#### ✨ Your query")
    render_input_chips(loc_display, ind_display)
    for note in (result.get("loc_note", ""), result.get("ind_note", "")):
        if note:
            st.caption(f"ℹ️ {note}")

    st.divider()

    # ---- Row: confidence score card + trend chart -----------------------
    st.markdown("#### 📊 Market signal")
    left, right = st.columns([1, 2])
    with left:
        render_confidence_card(result["score"], result.get("factors", []))
    with right:
        st.markdown(
            f"<div class='card'><div class='card-title'>📈 Structural market trend "
            f"— {ind_display} in {loc_display}</div>",
            unsafe_allow_html=True)
        render_trend_chart(result["trend"])
        st.caption("Mock projection: local supply structurally lags demand, "
                   "leaving an open opportunity window.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # ---- Creative business alternatives ---------------------------------
    st.markdown("#### 🧠 Brainstorm alternatives")
    primary, *others = result["ideas"]
    render_primary_path(primary, loc_display)
    st.write("")
    cols = st.columns(len(others))
    for col, idea in zip(cols, others):
        with col:
            render_idea_card(idea)

    st.caption("These are **brainstorm prompts**, not validated business plans. "
               "Pair them with the Completeness tab in the main dashboard to "
               "check real supply data before committing.")
