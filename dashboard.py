"""Streamlit dashboard for the Nairobi business market-gap pipeline.

Run:  venv\\Scripts\\python -m streamlit run dashboard.py

This is a *validation* dashboard: its job is to let you test whether the
pipeline is working as intended, not to prettify the result. It reads
data/processed/biz_intel.db READ-ONLY (never writes), so it is safe to
open while the pipeline scripts are being re-run.

Tabs:
  1. Overview        -- KPIs + pipeline-health summary
  2. By location     -- Mode 1: pick a place, see which categories are underserved
  3. By category     -- Mode 2: pick a category, see which places are underserved
  4. Completeness    -- live invariant checks + coverage heatmap (the "is it working?" tab)
  5. Raw & logs      -- the pipeline's own log files, unfiltered
"""
from pathlib import Path

import pandas as pd
import sqlite3
import sys

import streamlit as st

# ---- path setup ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "processed" / "biz_intel.db"
RAW_DIR = ROOT / "data" / "raw" / "locations"
LOGS = {
    "fetch_failures": ROOT / "data" / "processed" / "fetch_failures.log",
    "join_mismatches": ROOT / "data" / "processed" / "join_mismatches.log",
    "dedup_conflicts": ROOT / "data" / "processed" / "dedup_conflicts.log",
    "cluster_diagnostics": ROOT / "data" / "processed" / "cluster_diagnostics.csv",
}
LIMITATIONS = ROOT / "LIMITATIONS.md"

# The category taxonomy lives in the pipeline, not here -- import it so the
# dashboard can never drift from what the scripts actually fetch.
sys.path.insert(0, str(ROOT / "scripts"))
from overpass_client import CATEGORIES  # noqa: E402

TIER2 = {c for c, m in CATEGORIES.items() if m["tier"] == 2}
CONFOUND = {c for c, m in CATEGORIES.items() if m["financing_confound"]}

# ---- dataviz palette (validated default, see dataviz skill) ---------------
# Categorical slots for the three cluster bands (identity, fixed order).
BAND_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua
BAND_ORDER = ["underserved", "moderate", "saturated"]
# Sequential blue ramp for magnitude (density heatmap).
SEQ_RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#104281"]
# Status colors for the completeness checks.
GOOD, BAD = "#0ca30c", "#d03b3b"

st.set_page_config(page_title="Nairobi Business Market-Gap", page_icon="📊", layout="wide")


# ---- data loading (cached: re-runs on widget change hit the cache) --------
@st.cache_data
def load_db() -> dict[str, pd.DataFrame]:
    """Read every table the dashboard needs, once, into DataFrames."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        locs = pd.read_sql_query(
            "SELECT * FROM locations ORDER BY location_name", conn)
        feat = pd.read_sql_query(
            "SELECT * FROM features", conn)
        biz = pd.read_sql_query(
            "SELECT location_id, category, lat, lon, name FROM businesses", conn)
    finally:
        conn.close()

    # Join features to locations so every row carries name/pop/method.
    feat = feat.merge(
        locs[["location_id", "location_name", "sub_county", "population",
              "households", "geo_method"]],
        on="location_id", how="left")
    feat["tier"] = feat["category"].map(
        {c: m["tier"] for c, m in CATEGORIES.items()})
    feat["financing_confound_flag"] = feat["category"].isin(CONFOUND)
    return {"locations": locs, "features": feat, "businesses": biz}


@st.cache_data
def load_logs() -> dict[str, str]:
    """Read the pipeline's log files as plain text."""
    out = {}
    for key, path in LOGS.items():
        try:
            out[key] = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            out[key] = ""
    return out


def flag_str(tier: bool, confound: bool, geo: str) -> str:
    """Human-readable flag string for a row, mirroring 09_recommend.py."""
    parts = []
    if tier:
        parts.append("[tier2]")
    if confound:
        parts.append("[fin]")
    if geo == "bbox":
        parts.append("[bbox]")
    return " ".join(parts)


data = load_db()
feat = data["features"]
locs = data["locations"]
biz = data["businesses"]
logs = load_logs()

# ---- shared helpers --------------------------------------------------------
def kpi(label: str, value, sub: str = "", color: str = "#52514e") -> None:
    st.markdown(
        f"<div style='border:1px solid rgba(11,11,11,0.10);border-radius:10px;"
        f"padding:14px 18px;background:#fcfcfb'>"
        f"<div style='font-size:13px;color:#52514e'>{label}</div>"
        f"<div style='font-size:26px;font-weight:600;color:{color};"
        f"font-variant-numeric:tabular-nums'>{value}</div>"
        f"<div style='font-size:12px;color:#898781'>{sub}</div>"
        f"</div>", unsafe_allow_html=True)


def status_pill(text: str, ok: bool) -> None:
    color = GOOD if ok else BAD
    st.markdown(
        f"<div style='display:inline-block;padding:2px 10px;border-radius:999px;"
        f"font-size:13px;color:{color};border:1px solid {color};"
        f"font-weight:500'>{'✓ ' if ok else '✗ '}{text}</div>",
        unsafe_allow_html=True)


def render_explore_table(df: pd.DataFrame, by_location: bool) -> None:
    """Shared styled table for both explore tabs."""
    show = df.copy()
    show["flag"] = show.apply(
        lambda r: flag_str(r["category"] in TIER2,
                           r["category"] in CONFOUND,
                           r["geo_method"]),
        axis=1) if by_location else show.apply(
        lambda r: flag_str(r["category"] in TIER2,
                           r["category"] in CONFOUND,
                           r["geo_method"]),
        axis=1)
    cols = (["location_name", "sub_county", "population"] if not by_location
            else ["category"])
    st.dataframe(
        show[cols + ["business_count", "businesses_per_1000_people",
                     "cluster_label", "flag", "geo_method"]],
        width="stretch",
        column_config={
            "location_name": "Location",
            "category": "Category",
            "business_count": st.column_config.NumberColumn("Businesses",
                                                            format="%d"),
            "businesses_per_1000_people": st.column_config.NumberColumn(
                "Per 1000 people", format="%.3f"),
            "cluster_label": "Band",
            "flag": "Flags",
            "geo_method": "Geo source",
            "sub_county": "Sub-county",
            "population": st.column_config.NumberColumn("Population",
                                                        format="%d"),
        },
        hide_index=True,
    )


# ---- header ----------------------------------------------------------------
st.title("Nairobi Business Market-Gap Intelligence")
st.caption(
    "Validation dashboard for the OSM + census pipeline. "
    "This is a **gap/opportunity signal** — never a profitability or survival "
    "prediction. Numbers are businesses per 1,000 people.")

st.divider()

# ============================================================================
# TAB 1 -- OVERVIEW
# ============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Overview", "📍 By location", "🏷️ By category", "✅ Completeness", "🗂️ Raw & logs"])

with tab1:
    st.subheader("Pipeline health at a glance")
    col = st.columns(4)
    with col[0]:
        kpi("Locations covered", len(locs),
            f"{int(locs['geo_method'].eq('bbox').sum())} via bbox · "
            f"{int(locs['geo_method'].eq('area').sum())} via admin area")
    with col[1]:
        kpi("Categories", len(CATEGORIES),
            f"{len(TIER2)} informal (tier 2) · "
            f"{len(CATEGORIES) - len(TIER2)} formal (tier 1)")
    with col[2]:
        kpi("Businesses mapped", int(biz.shape[0]))
    with col[3]:
        labeled = int(feat["cluster_label"].notna().sum())
        kpi("Feature rows", int(feat.shape[0]),
            f"{labeled} with a cluster band · "
            f"{int(feat['cluster_label'].isna().sum())} too sparse to band")

    st.write("")
    st.markdown("#### What the checks say")
    n_fail_log = len([l for l in logs["fetch_failures"].splitlines() if l])
    n_join = len([l for l in logs["join_mismatches"].splitlines() if l])
    n_dedup = len([l for l in logs["dedup_conflicts"].splitlines() if l])

    c1, c2, c3 = st.columns(3)
    with c1:
        if n_fail_log == 0:
            status_pill("No fetch failures on record", True)
        else:
            status_pill(f"{n_fail_log} fetch failures in log "
                        "(may be stale — see Completeness tab)", False)
    with c2:
        if n_join == 0:
            status_pill("All census locations joined", True)
        else:
            status_pill(f"{n_join} join notes in log", False)
    with c3:
        if n_dedup == 0:
            status_pill("No dedup conflicts", True)
        else:
            status_pill(f"{n_dedup} dedup conflicts", False)
    st.write("")
    st.caption(
        "The fetch-failure log is **cumulative** — a transient failure on an "
        "earlier run stays listed even after a later run fetched the file. The "
        "Completeness tab checks the ground truth (does the raw file exist?) "
        "rather than the log.")

    st.markdown("#### Locations (census units)")
    loc_view = locs.copy()
    loc_view["total_biz"] = loc_view["location_id"].map(
        biz.groupby("location_id").size())
    loc_view["total_biz"] = loc_view["total_biz"].fillna(0).astype(int)
    loc_view["density_per_1000"] = (
        loc_view["total_biz"] / loc_view["population"] * 1000).round(3)
    st.dataframe(
        loc_view[["location_name", "sub_county", "population", "households",
                  "geo_method", "total_biz", "density_per_1000"]],
        width="stretch",
        column_config={
            "location_name": "Location",
            "population": st.column_config.NumberColumn("Population",
                                                        format="%d"),
            "households": st.column_config.NumberColumn("Households",
                                                        format="%d"),
            "geo_method": "Geo source",
            "total_biz": st.column_config.NumberColumn("Businesses mapped",
                                                       format="%d"),
            "density_per_1000": st.column_config.NumberColumn(
                "All-category density /1000", format="%.3f"),
        },
        hide_index=True,
    )

# ============================================================================
# TAB 2 -- BY LOCATION (MODE 1)
# ============================================================================
with tab2:
    st.subheader("Mode 1 · pick a place → which categories are underserved?")
    place = st.selectbox("Location", locs["location_name"].sort_values())
    row = feat[feat["location_name"] == place].sort_values(
        "businesses_per_1000_people")
    pop = int(locs.loc[locs["location_name"] == place, "population"].iloc[0])
    method = locs.loc[locs["location_name"] == place, "geo_method"].iloc[0]
    st.caption(f"Population {pop:,} · geo source: {method}")

    if row.empty:
        st.info("No feature rows for this location — it had no fetched data.")
    else:
        # Chart: horizontal bars, underserved first, colored by band.
        import altair as alt
        chart_df = row.copy()
        chart_df["band"] = chart_df["cluster_label"].fillna("no band")
        band_domain = BAND_ORDER + ["no band"]
        band_colors = dict(zip(BAND_ORDER, BAND_COLORS))
        band_colors["no band"] = "#c3c2b7"  # muted for unbanded
        chart = (
            alt.Chart(chart_df)
            .mark_bar(size=14)
            .encode(
                y=alt.Y("category:N", sort=alt.SortField(
                    "businesses_per_1000_people", order="ascending"),
                    axis=alt.Axis(labelLimit=160)),
                x=alt.X("businesses_per_1000_people:Q",
                        title="businesses per 1,000 people"),
                color=alt.Color("band:N", scale=alt.Scale(
                    domain=band_domain, range=[band_colors[b] for b in band_domain]),
                    legend=alt.Legend(title="Band")),
                tooltip=["category", "business_count", "businesses_per_1000_people",
                         "cluster_label", "geo_method"],
            )
            .properties(height=420)
        )
        st.altair_chart(chart, width="stretch")
        st.caption("Bars colored by cluster band (per-category terciles). "
                   "Gray = category too sparse across Nairobi to band.")

        render_explore_table(row, by_location=True)

# ============================================================================
# TAB 3 -- BY CATEGORY (MODE 2)
# ============================================================================
with tab3:
    st.subheader("Mode 2 · pick a category → which places are underserved?")
    cat = st.selectbox("Category", sorted(CATEGORIES))
    meta = CATEGORIES[cat]
    st.caption(
        f"Tier {meta['tier']} · "
        f"{'financing-confound' if meta['financing_confound'] else 'market-driven'} · "
        f"OSM tag: `{meta['tag']}`")

    row = feat[feat["category"] == cat].sort_values(
        "businesses_per_1000_people")
    if row.empty:
        st.info("No feature rows for this category.")
    else:
        import altair as alt
        chart_df = row.copy()
        chart_df["band"] = chart_df["cluster_label"].fillna("no band")
        band_domain = BAND_ORDER + ["no band"]
        band_colors = dict(zip(BAND_ORDER, BAND_COLORS))
        band_colors["no band"] = "#c3c2b7"
        chart = (
            alt.Chart(chart_df)
            .mark_bar(size=10)
            .encode(
                y=alt.Y("location_name:N", sort=alt.SortField(
                    "businesses_per_1000_people", order="ascending"),
                    axis=alt.Axis(labelLimit=160)),
                x=alt.X("businesses_per_1000_people:Q",
                        title="businesses per 1,000 people"),
                color=alt.Color("band:N", scale=alt.Scale(
                    domain=band_domain, range=[band_colors[b] for b in band_domain]),
                    legend=alt.Legend(title="Band")),
                tooltip=["location_name", "business_count",
                         "businesses_per_1000_people", "cluster_label",
                         "geo_method"],
            )
            .properties(height=560)
        )
        st.altair_chart(chart, width="stretch")
        st.caption("Bars colored by cluster band (per-category terciles). "
                   "Gray = category too sparse across Nairobi to band.")
        render_explore_table(row, by_location=False)

# ============================================================================
# TAB 4 -- COMPLETENESS (the "is it working?" tab)
# ============================================================================
with tab4:
    st.subheader("Live invariant checks")
    st.caption(
        "These mirror the pipeline's own contract: a failed fetch must never "
        "be recorded as a zero-count, counts must be consistent, and every "
        "fetched file must make it into the feature table.")

    # --- check 1: no fabricated feature rows ---------------------------------
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        loc_id = {r[0]: r[1] for r in conn.execute(
            "SELECT location_id, location_name FROM locations")}
        frows = set(conn.execute(
            "SELECT location_id, category FROM features"))
        raw_files = set()
        for d in RAW_DIR.iterdir():
            if d.is_dir():
                raw_files |= {(d.name, f.stem) for f in d.glob("*.json")}
        fabricated = [(loc_id.get(lid), c) for lid, c in frows
                      if (loc_id.get(lid), c) not in raw_files]
        orphaned = sorted(raw_files - {(loc_id[lid], c) for lid, c in frows})
    finally:
        conn.close()

    c1, c2, c3 = st.columns(3)
    with c1:
        if not fabricated:
            status_pill("No fabricated feature rows", True)
        else:
            status_pill(f"{len(fabricated)} fabricated rows", False)
    with c2:
        if not orphaned:
            status_pill("Every raw fetch reached the feature table", True)
        else:
            status_pill(f"{len(orphaned)} orphaned fetches", False)
    with c3:
        if biz.groupby(["location_id", "category"]).size().fillna(0).eq(
                feat.set_index(["location_id", "category"])["business_count"]).all():
            status_pill("Business counts match the features table", True)
        else:
            status_pill("Business counts mismatch", False)

    if fabricated:
        st.error("These feature rows have no raw fetch file behind them:")
        st.write(fabricated)
    if orphaned:
        st.warning("These raw files were never turned into feature rows:")
        st.write(orphaned)

    st.write("")
    st.markdown("#### Stale vs real fetch failures")
    st.caption(
        "The failure log is cumulative. These are the combos it lists; a raw "
        "file on disk means a later run recovered. The ones with a missing file "
        "are the real, current gaps.")
    if logs["fetch_failures"]:
        failed_rows = []
        for line in logs["fetch_failures"].splitlines():
            if " / " not in line:
                continue
            loc_part, rest = line.split(" / ", 1)
            cat_part = rest.split(":", 1)[0]
            missing = not (RAW_DIR / loc_part / f"{cat_part}.json").exists()
            failed_rows.append({"location": loc_part, "category": cat_part,
                                "recovered": not missing})
        fd = pd.DataFrame(failed_rows)
        st.dataframe(fd, width="stretch",
                     column_config={
                         "recovered": st.column_config.CheckboxColumn(
                             "Raw file on disk (recovered)", disabled=True)},
                     hide_index=True)
    else:
        status_pill("No fetch failures in log", True)

    st.write("")
    st.markdown("#### Coverage heatmap")
    st.caption(
        "Cell = businesses mapped for that (location × category). Lighter = "
        "fewer / none; darker = denser. 'M' = the file was never fetched.")

    # Build a complete grid with a "missing" sentinel for never-fetched combos.
    grid = (feat.pivot_table(
        index="location_name", columns="category",
        values="business_count", aggfunc="sum").reindex(
            index=sorted(feat["location_name"].unique()),
            columns=sorted(CATEGORIES)))
    raw_lookup = {}
    for d in RAW_DIR.iterdir():
        if d.is_dir():
            raw_lookup[d.name] = {f.stem for f in d.glob("*.json")}
    missing = grid.isna()
    grid = grid.fillna(-1)  # sentinel for never-fetched

    # Mark the heatmap; bbox-sourced locations get a bold row label.
    import altair as alt
    hm = grid.reset_index().melt(id_vars="location_name",
                                 var_name="category", value_name="count")
    hm["count"] = hm["count"].astype(int)
    hm["label"] = hm["count"].apply(
        lambda v: "M" if v == -1 else ("" if v == 0 else str(v)))
    hm["is_missing"] = hm["count"] == -1
    hm["count_plot"] = hm["count"].clip(lower=0)

    color_scale = alt.Scale(
        domain=list(range(0, 5)) + [5],
        range=[SEQ_RAMP[0], SEQ_RAMP[1], SEQ_RAMP[2],
               SEQ_RAMP[3], SEQ_RAMP[4], "#104281"],
        type="linear",
        clamp=True,
        domainMax=max(1, int(hm["count"].max())))
    base = alt.Chart(hm).mark_rect().encode(
        y=alt.Y("location_name:N", sort=list(grid.index),
                title=None, axis=alt.Axis(labelLimit=160)),
        x=alt.X("category:N", sort=sorted(CATEGORIES), title=None,
                axis=alt.Axis(labelLimit=120)),
        color=alt.condition(
            "datum.is_missing",
            alt.value("#e9e8e3"),  # surface-ish gray for never-fetched
            alt.Color("count_plot:Q", scale=color_scale, legend=None)),
        tooltip=["location_name", "category", "count"],
    )
    text = alt.Chart(hm).mark_text(size=9, color="#52514e").encode(
        y=alt.Y("location_name:N", sort=list(grid.index), title=None),
        x=alt.X("category:N", sort=sorted(CATEGORIES), title=None),
        text=alt.Text("label:N"),
    )
    st.altair_chart(base + text, width="stretch")
    st.caption(
        "Shade = count (darker is denser). 'M' = never fetched (failed on every "
        "run) — treated as unknown, not zero. The lightest shade means a "
        "confirmed-empty fetch.")

# ============================================================================
# TAB 5 -- RAW & LOGS
# ============================================================================
with tab5:
    st.subheader("Raw counts per location × category")
    st.caption("Element count directly from each raw OSM JSON file.")
    raw_counts = {}
    for d in sorted(RAW_DIR.iterdir()):
        if d.is_dir():
            for f in sorted(d.glob("*.json")):
                import json
                try:
                    payload = json.loads(f.read_text(encoding="utf-8"))
                    raw_counts[(d.name, f.stem)] = len(payload.get("elements", []))
                except (json.JSONDecodeError, OSError):
                    raw_counts[(d.name, f.stem)] = -1
    if raw_counts:
        rc = pd.DataFrame(
            [{"location": loc, "category": cat, "raw_elements": n}
             for (loc, cat), n in raw_counts.items()])
        rc = rc.pivot_table(index="location", columns="category",
                            values="raw_elements", aggfunc="max",
                            fill_value=0).reindex(
                                columns=sorted(CATEGORIES))
        st.dataframe(rc, width="stretch",
                     column_config={
                         c: st.column_config.NumberColumn(c, format="%d")
                         for c in sorted(CATEGORIES)},
                     hide_index=False)
        st.caption("Rows here should match the feature-table business counts "
                   "minus dedup conflicts (a node matching two category queries "
                   "is counted once).")
    else:
        st.info("No raw location files found under data/raw/locations/.")

    st.write("")
    for key, title in [
        ("fetch_failures", "Fetch failures log (cumulative)"),
        ("join_mismatches", "Join mismatches log"),
        ("dedup_conflicts", "Dedup conflicts log"),
        ("cluster_diagnostics", "Cluster diagnostics (CSV)"),
    ]:
        st.markdown(f"#### {title}")
        body = logs.get(key, "")
        if not body:
            st.caption("_(empty — nothing was ever logged)_")
        else:
            if key == "cluster_diagnostics":
                st.dataframe(pd.read_csv(ROOT / LOGS[key], dtype=str),
                             width="stretch", hide_index=True)
            else:
                st.code(body)
