"""
Nairobi Business Market-Gap Intelligence -- dashboard
-----------------------------------------------------

Two query modes over the trained model (data/processed/biz_intel.db):

  * By location  ->  rank that location's categories by mapped gap
                     (most-underserved first).
  * By category  ->  rank all locations for one category by mapped gap.

Every row carries its badges: [tier2] informal/undercounted, [fin]
financing-confound, [bbox] lower-fidelity geocoded box. The generative
summary layer (scripts/summaries.py) writes the plain-language paragraph,
grounded strictly in computed numbers.

GAP SIGNAL, NOT SUCCESS PREDICTION -- this is stated permanently at the top of
the app. Nothing here predicts profitability, survival, or business success.

Reads the DB read-only. Run:
    venv\\Scripts\\python -m streamlit run app.py
"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "processed" / "biz_intel.db"

# The summary layer lives in scripts/; keep the single source of truth there.
sys.path.insert(0, str(ROOT / "scripts"))
from summaries import (  # noqa: E402
    aggregate_metrics,
    flags_for,
    summarize_category,
    summarize_location,
)

# ---------------------------------------------------------------------------
# App config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Nairobi Market-Gap Intelligence",
    page_icon="📊",
    layout="wide",
)

BADGE_COLORS = {
    "tier2": ("#fff4e5", "#9a5b00"),  # informal / undercounted
    "fin": ("#ffe9e9", "#a32727"),     # financing confound
    "bbox": ("#e8effb", "#274a7d"),    # lower-fidelity geocode
}


def _badge(flag: str) -> str:
    key = flag.strip(" []")
    bg, fg = BADGE_COLORS.get(key, ("#efefee", "#3f3f3e"))
    return (f'<span style="background:{bg};color:{fg};border-radius:9px;'
            f'padding:1px 7px;font-size:0.72rem;font-weight:600;'
            f'margin-right:4px">{key}</span>')


def _badges_for(category: str, geo_method: str | None) -> str:
    return "".join(_badge(f) for f in flags_for(category, geo_method))


def load_db():
    """Connect read-only; return None (graceful) if the DB is missing."""
    if not DB_PATH.exists():
        return None
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def rank_rows(conn, location: str | None = None,
              category: str | None = None) -> pd.DataFrame:
    """Ranked rows, most-underserved (most negative gap) first."""
    where, params = [], []
    if location is not None:
        where.append("l.location_name = ?")
        params.append(location)
    if category is not None:
        where.append("f.category = ?")
        params.append(category)
    sql = (
        "SELECT l.location_name, f.category, f.business_count, "
        "       f.businesses_per_1000_people, f.cluster_label, "
        "       f.predicted_count, f.gap_residual, l.geo_method "
        "FROM features f JOIN locations l USING (location_id)"
    )
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY f.gap_residual ASC, f.businesses_per_1000_people ASC"
    df = pd.read_sql_query(sql, conn, params=params)
    if df.empty:
        return df
    df = df.rename(columns={
        "location_name": "Location",
        "category": "Category",
        "business_count": "Mapped",
        "businesses_per_1000_people": "Per 1,000",
        "cluster_label": "Band",
        "predicted_count": "Expected",
        "gap_residual": "Gap",
    })
    df["Gap"] = df["Gap"].round(1)
    df["Per 1,000"] = df["Per 1,000"].round(3)
    df["Expected"] = df["Expected"].round(1)
    df["Badges"] = [
        _badges_for(cat, geo) for cat, geo in zip(df["Category"], df["geo_method"])
    ]
    return df.drop(columns=["geo_method"])


def render_metrics(conn):
    m = aggregate_metrics(conn)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Locations", f"{m['n_locations']:,}")
    c2.metric("Categories", f"{m['n_categories']}")
    c3.metric("Mapped businesses", f"{m['n_businesses']:,}")
    c4.metric("Modelled cells", f"{m['n_feature_rows']:,} / {m['n_predicted_rows']:,} with pred.")


def main():
    # ------------------------------------------------------------------ header
    st.title("Nairobi Business Market-Gap Intelligence")
    st.markdown(
        "**This is a mapped-density gap/opportunity signal — it is NOT a "
        "profitability, survival, or success prediction.** Tier-2 (informal) "
        "categories (salon, greengrocer, kiosk, fast food, laundry, boda boda) "
        "are under-mapped in OpenStreetMap, so their counts are a **floor**, "
        "not a true count."
    )
    st.divider()

    conn = load_db()
    if conn is None:
        st.error(
            "Data file not found. Run the pipeline first so "
            "data/processed/biz_intel.db exists (see RUNBOOK.md)."
        )
        return
    try:
        render_metrics(conn)
        st.divider()

        locations = sorted(r[0] for r in conn.execute(
            "SELECT DISTINCT location_name FROM locations"))
        categories = sorted(r[0] for r in conn.execute(
            "SELECT DISTINCT category FROM features"))

        # ------------------------------------------------------------- tab: loc
        tab_loc, tab_cat = st.tabs(["By location", "By category"])

        with tab_loc:
            loc = st.selectbox("Select a location", locations, index=None,
                               placeholder="Choose a Nairobi location…",
                               key="loc_sel")
            if loc is None:
                st.info("Pick a location above to see its ranked market gaps.")
            else:
                df = rank_rows(conn, location=loc)
                if df.empty:
                    st.warning(f"No fetched data for {loc!r}.")
                else:
                    st.subheader(f"Ranked gaps in {loc}")
                    st.dataframe(df, width="stretch", hide_index=True)
                    summary = summarize_location(conn, loc)
                    if summary["text"]:
                        st.markdown(f"**Summary** — {summary['text']}")

        # ------------------------------------------------------------- tab: cat
        with tab_cat:
            cat = st.selectbox("Select a category", categories, index=None,
                               placeholder="Choose a category…",
                               key="cat_sel")
            if cat is None:
                st.info("Pick a category above to see where it is most scarce.")
            else:
                df = rank_rows(conn, category=cat)
                if df.empty:
                    st.warning(f"No data for category {cat!r}.")
                else:
                    st.subheader(f"Ranked gaps for {cat}")
                    st.dataframe(df, width="stretch", hide_index=True)
                    summary = summarize_category(conn, cat)
                    if summary["text"]:
                        st.markdown(f"**Summary** — {summary['text']}")

        st.divider()
        st.caption(
            "Sources: Kenya 2019 Census (population, households) and "
            "OpenStreetMap node counts fetched per location/category. "
            "Gap = mapped count minus the trained model's expected count "
            "(Negative Binomial GLM, GroupKFold-CV). Badges: tier2 = informal/"
            "undercounted (count is a floor); fin = financing-confound; "
            "bbox = lower-fidelity geocoded box."
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
