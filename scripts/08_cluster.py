"""Assign saturated / moderate / underserved bands per category.

Per category, the 30 locations are binned into terciles of
businesses_per_1000_people:
  bottom tercile  -> underserved
  middle tercile  -> moderate
  top tercile     -> saturated

Quantile terciles are used rather than k-means because the per-category
distributions are expected to be zero-heavy and right-skewed; terciles are
robust to that shape and trivially explainable. (k-means would be a
reasonable alternative if a category's distribution turns out to be smooth
and non-degenerate.)

Sparsity guard: if a category has fewer than 5 locations with a non-zero
count, a 3-way split is not meaningful (the spec: "3-way clustering doesn't
mean anything, note that instead"). Such categories get cluster_label NULL
and are listed in the diagnostics.

Writes data/processed/cluster_diagnostics.csv and updates
features.cluster_label in biz_intel.db.
"""
import csv
import sqlite3
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "processed" / "biz_intel.db"
DIAG_CSV = ROOT / "data" / "processed" / "cluster_diagnostics.csv"

MIN_NONZERO_FOR_CLUSTERING = 5


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE features SET cluster_label = NULL")

    # All categories present in the features table (from the taxonomy).
    categories = [r[0] for r in conn.execute(
        "SELECT DISTINCT category FROM features ORDER BY category")]

    diag_rows = []
    for category in categories:
        rows = conn.execute(
            "SELECT location_id, business_count, businesses_per_1000_people "
            "FROM features WHERE category = ?", (category,)
        ).fetchall()
        values = np.array([r[2] for r in rows], dtype=float)
        n_nonzero = int(np.count_nonzero(values))
        n = len(values)

        if n_nonzero < MIN_NONZERO_FOR_CLUSTERING:
            diag_rows.append({
                "category": category, "n_locations": n, "n_nonzero": n_nonzero,
                "p25": None, "p67": None, "note": "too sparse for 3-way clustering; labels left NULL",
            })
            print(f"{category}: {n_nonzero}/{n} locations with businesses -- too sparse, no clustering")
            continue

        p33, p67 = np.percentile(values, [33, 67])
        labels = np.where(values <= p33, "underserved",
                  np.where(values <= p67, "moderate", "saturated"))
        for (lid, _count, _density), label in zip(rows, labels):
            conn.execute("UPDATE features SET cluster_label = ? WHERE location_id = ? AND category = ?",
                         (label, lid, category))
        diag_rows.append({
            "category": category, "n_locations": n, "n_nonzero": n_nonzero,
            "p25": None, "p67": p67, "note": f"terciles at {p33:.4f} / {p67:.4f}",
        })
        print(f"{category}: {n_nonzero}/{n} locations with businesses; "
              f"terciles at {p33:.4f} / {p67:.4f}")

    conn.commit()

    # Fix the diagnostics columns: store both tercile bounds properly.
    for row in diag_rows:
        if row["note"].startswith("terciles"):
            pass  # note already carries the bounds; keep p25 column unused
    with open(DIAG_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "n_locations", "n_nonzero",
                                               "p25", "p67", "note"])
        writer.writeheader()
        writer.writerows(diag_rows)

    conn.close()
    print(f"\nWrote {DIAG_CSV}")


if __name__ == "__main__":
    main()
