"""Hand-labeled saturation classifier for (location, category) pairs.

Why a hand-labeled set (and NOT cluster-derived labels):
  The cluster_label column in features comes from the SAME data a classifier
  would predict on, so training on it would be circular -- the model would
  merely re-learn the tercile rule it was produced by. Per the spec, the
  unsupervised part (quantile banding in 08_cluster.py) stands on its own as
  the descriptive clustering deliverable; the classifier here uses a separate,
  human-supplied label set.

The labels in HAND_LABELS below are expert knowledge-based judgments of
underserved / moderate / saturated for specific Nairobi (location, category)
pairs, made with awareness of the mapped density and local market context.
They are NOT cross-checked against a county CIDP report (none was available in
the repo) -- this is stated honestly in LIMITATIONS.md. Because OSM is a floor
for tier-2 categories, labels for those categories already absorb that
undercount; the classifier must not be read as predicting true informal-market
saturation.

Features (same design as the regression, minus the target):
  log(population), households/1000, geo_method indicator, category dummies.
Protocol: GroupKFold(3) grouped by location_id (rows sharing a location share
population -> not independent). Metric: macro precision / recall / F1 and
accuracy, reported on the held-out fold concatenation.
"""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "processed" / "biz_intel.db"
EVAL_JSON = ROOT / "data" / "processed" / "model_eval_classifier.json"

LABELER = "assistant-domain-knowledge (not CIDP cross-checked)"
LABEL_NOTE = ("Knowledge-based judgment from mapped density + Nairobi market "
              "context; not from a county CIDP report (none available).")

# Hand labels: (location, category) -> underserved | moderate | saturated.
# Chosen to cover a spread of locations and all three classes. These are the
# REAL supervision for the classifier -- independent of features.cluster_label.
HAND_LABELS = {
    # Kilimani -- upscale, well-mapped
    ("Kilimani", "restaurant"): "saturated",
    ("Kilimani", "pharmacy"): "saturated",
    ("Kilimani", "clothes"): "underserved",
    ("Kilimani", "salon"): "moderate",
    ("Kilimani", "kiosk"): "saturated",
    ("Kilimani", "laundry"): "underserved",
    # Kibera -- dense, high footfall, informal
    ("Kibera", "greengrocer"): "moderate",
    ("Kibera", "salon"): "underserved",
    ("Kibera", "kiosk"): "saturated",
    ("Kibera", "motorcycle_taxi"): "moderate",
    # Eastleigh -- trade hub
    ("Eastleigh", "supermarket"): "underserved",
    ("Eastleigh", "clothes"): "saturated",
    ("Eastleigh", "hardware"): "underserved",
    ("Eastleigh", "pharmacy"): "moderate",
    # Karen -- affluent, low-density, under-mapped
    ("Karen", "supermarket"): "underserved",
    ("Karen", "salon"): "underserved",
    ("Karen", "laundry"): "underserved",
    ("Karen", "restaurant"): "moderate",
    # CBD -- office/commerce daytime
    ("CBD", "pharmacy"): "saturated",
    ("CBD", "restaurant"): "saturated",
    ("CBD", "clothes"): "saturated",
    ("CBD", "kiosk"): "moderate",
    # Kasarani -- family suburbs, sports complex
    ("Kasarani", "supermarket"): "moderate",
    ("Kasarani", "salon"): "underserved",
    ("Kasarani", "restaurant"): "moderate",
    ("Kasarani", "laundry"): "underserved",
    # Embakasi / Umoja / Makadara -- residential sprawl
    ("Embakasi", "supermarket"): "moderate",
    ("Embakasi", "greengrocer"): "underserved",
    ("Umoja", "supermarket"): "underserved",
    ("Umoja", "restaurant"): "underserved",
    ("Makadara", "restaurant"): "underserved",
    ("Bahati", "supermarket"): "underserved",
    # Westlands-adjacent
    ("Kawangware", "greengrocer"): "moderate",
    ("Kawangware", "salon"): "underserved",
}


def load() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        feat = pd.read_sql_query(
            "SELECT location_id, category, business_count FROM features", conn)
        locs = pd.read_sql_query(
            "SELECT location_id, location_name, population, households, "
            "geo_method FROM locations", conn)
    finally:
        conn.close()
    # location_name must survive the merge: features rows map to it via
    # location_id, and the hand labels key on (location, category).
    df = feat.merge(locs[["location_id", "location_name", "population",
                          "households", "geo_method"]], on="location_id")
    # Join the hand labels.
    labels = pd.DataFrame(
        [{"location_name": loc, "category": cat, "hand_label": lbl}
         for (loc, cat), lbl in HAND_LABELS.items()])
    df = df.merge(labels, on=["location_name", "category"], how="inner")
    return df


def build_design(df: pd.DataFrame, cats: list[str]) -> pd.DataFrame:
    dummies = pd.get_dummies(df["category"], prefix="cat", dtype=float)
    for c in cats:  # keep column set identical across folds
        if f"cat_{c}" not in dummies:
            dummies[f"cat_{c}"] = 0.0
    dummies = dummies[sorted(dummies.columns)]
    X = pd.DataFrame({
        "log_pop": np.log(df["population"]),
        "households": df["households"].astype(float) / 1000,
        "geo_bbox": (df["geo_method"] == "bbox").astype(float),
    }, index=df.index)
    return pd.concat([X, dummies], axis=1)


def main():
    df = load()
    cats = sorted(df["category"].unique())
    y = df["hand_label"].to_numpy()
    X = build_design(df, cats).to_numpy()
    groups = df["location_id"].to_numpy()

    print(f"{len(df)} hand-labeled pairs across "
          f"{df['location_name'].nunique()} locations")
    print(df["hand_label"].value_counts().to_dict())

    # --- GroupKFold(3) grouped by location --------------------------------
    gkf = GroupKFold(n_splits=3)
    preds, actuals = [], []
    for train_idx, test_idx in gkf.split(X, y, groups):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=1000, C=1.0))
        clf.fit(X[train_idx], y[train_idx])
        preds.append(clf.predict(X[test_idx]))
        actuals.append(y[test_idx])
    pred = np.concatenate(preds)
    actual = np.concatenate(actuals)

    report = {
        "n_labels": int(len(df)),
        "n_locations": int(df["location_name"].nunique()),
        "n_categories": int(df["category"].nunique()),
        "label_counts": df["hand_label"].value_counts().to_dict(),
        "accuracy": round(float(accuracy_score(actual, pred)), 3),
        "precision_macro": round(float(precision_score(actual, pred,
                                                       average="macro",
                                                       zero_division=0)), 3),
        "recall_macro": round(float(recall_score(actual, pred,
                                                 average="macro",
                                                 zero_division=0)), 3),
        "f1_macro": round(float(f1_score(actual, pred, average="macro",
                                         zero_division=0)), 3),
        "per_class": {},
    }
    for label in sorted(set(actual) | set(pred)):
        idx = actual == label
        n = int(idx.sum())
        p = precision_score(actual[idx], pred[idx], average="micro",
                            zero_division=0) if n else None
        r = recall_score(actual[idx], pred[idx], average="micro",
                         zero_division=0) if n else None
        report["per_class"][label] = {"n": n, "precision": round(float(p), 3)
                                      if p is not None else None,
                                      "recall": round(float(r), 3)
                                      if r is not None else None}

    print("\nConfusion matrix (rows=actual, cols=predicted):")
    print(pd.crosstab(pd.Series(actual, name="actual"),
                      pd.Series(pred, name="pred")))
    print(f"\nAccuracy {report['accuracy']:.3f} | macro P/R/F1 "
          f"{report['precision_macro']:.3f}/{report['recall_macro']:.3f}/"
          f"{report['f1_macro']:.3f}")

    EVAL_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {EVAL_JSON}")


if __name__ == "__main__":
    main()
