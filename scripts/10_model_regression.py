"""Trained regression for expected business count per (location, category).

Protocol (per the project spec):
  - Target: business_count (a count).
  - Features: log(population), households, geo-method indicator, and category
    dummies (supermarket is the reference category).
  - Cross-validation: GroupKFold(5) grouped by location_id -- rows from the
    same location share population and are NOT independent, so a plain random
    split would leak. With ~360 rows across 30 locations a single held-out
    split is unreliable; GroupKFold reuses every row in the CV average.
  - Baseline: Poisson GLM (statsmodels). Counts here are zero-heavy and
    right-skewed (e.g. 215 restaurants vs 0-5 clothes shops in one area), so
    overdispersion is expected: if Pearson chi2 / dof >> 1 we refit with a
    Negative Binomial (sm.NegativeBinomial, which estimates alpha).
  - Evaluation: MAE (+ RMSE) over the CV folds. R2 is deliberately NOT used
    (misleading for count data).

Writes predicted_count and gap_residual (= business_count - predicted_count)
back into the features table, and saves eval metrics to
data/processed/model_eval_regression.json for RESULTS_SUMMARY.
"""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "processed" / "biz_intel.db"
EVAL_JSON = ROOT / "data" / "processed" / "model_eval_regression.json"

REFERENCE_CATEGORY = "supermarket"
MIN_OVERDISPERSION = 1.5  # Pearson chi2/dof above this -> switch to NegBin


def load() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        feat = pd.read_sql_query(
            "SELECT location_id, category, business_count FROM features", conn)
        locs = pd.read_sql_query(
            "SELECT location_id, population, households, geo_method "
            "FROM locations", conn)
    finally:
        conn.close()
    return feat.merge(locs, on="location_id")


def build_design(df: pd.DataFrame) -> pd.DataFrame:
    """Fixed feature matrix for the whole dataset (folds re-index, no drift)."""
    cats = sorted(df["category"].unique())
    dummies = pd.get_dummies(df["category"], prefix="cat", dtype=float)
    dummies = dummies.drop(columns=[f"cat_{REFERENCE_CATEGORY}"])
    X = pd.DataFrame({
        "log_pop": np.log(df["population"]),
        "households": df["households"].astype(float) / 1000,  # scale for stability
        "geo_bbox": (df["geo_method"] == "bbox").astype(float),
    }, index=df.index)
    X = pd.concat([X, dummies], axis=1)
    X = sm.add_constant(X, has_constant="add")
    return X


def fit_model(family: str, y: np.ndarray, X: pd.DataFrame):
    if family == "poisson":
        return sm.GLM(y, X, family=sm.families.Poisson()).fit()
    # nb2 with alpha estimated. BFGS converges reliably on this zero-heavy
    # overdispersed data; the default Newton solver stalls (see CHANGELOG).
    return sm.NegativeBinomial(y, X).fit(disp=False, method="bfgs", maxiter=1000)


def pearson_chi2_dof(model, y: np.ndarray, X: pd.DataFrame, n_params: int) -> float:
    mu = model.predict(X)
    pearson = float(np.sum((y - mu) ** 2 / np.maximum(mu, 1e-9)))
    dof = max(1, len(y) - n_params)
    return pearson / dof


def run_cv(df: pd.DataFrame, family: str) -> tuple[float, float]:
    from sklearn.model_selection import GroupKFold

    X_full = build_design(df)
    y = df["business_count"].to_numpy(float)
    groups = df["location_id"].to_numpy()
    gkf = GroupKFold(n_splits=5)

    preds, actuals = [], []
    for train_idx, test_idx in gkf.split(X_full, y, groups):
        model = fit_model(family, y[train_idx], X_full.iloc[train_idx])
        preds.append(model.predict(X_full.iloc[test_idx]))
        actuals.append(y[test_idx])
    pred = np.concatenate(preds)
    actual = np.concatenate(actuals)
    mae = float(np.mean(np.abs(actual - pred)))
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    return mae, rmse


def main():
    df = load()
    y = df["business_count"].to_numpy(float)
    X = build_design(df)

    # --- step 1: Poisson baseline + overdispersion check -------------------
    poisson = fit_model("poisson", y, X)
    chi2_dof = pearson_chi2_dof(poisson, y, X, X.shape[1])
    family = "poisson" if chi2_dof < MIN_OVERDISPERSION else "negative_binomial"
    print(f"Pearson chi2/dof = {chi2_dof:.2f}  ->  model: {family}")

    mae, rmse = run_cv(df, family)
    print(f"GroupKFold CV  MAE = {mae:.2f}   RMSE = {rmse:.2f}")

    # --- step 2: final model on all data, write into features --------------
    final = fit_model(family, y, X)
    pred = final.predict(X)
    resid = y - pred

    conn = sqlite3.connect(DB_PATH)
    try:
        for i, (_, row) in enumerate(df.iterrows()):
            conn.execute(
                "UPDATE features SET predicted_count=?, gap_residual=? "
                "WHERE location_id=? AND category=?",
                (round(float(pred[i]), 4), round(float(resid[i]), 4),
                 int(row["location_id"]), row["category"]))
        conn.commit()
    finally:
        conn.close()

    # --- step 3: interpretable summary of the fitted model -----------------
    coef = pd.Series(final.params, index=X.columns)
    rate_cols = [c for c in coef.index if c.startswith("cat_")]
    rates = coef[rate_cols].sort_values()
    # sm.NegativeBinomial stores the log of the dispersion parameter; exp() is
    # the actual alpha. (getattr(..., "alpha", nan) silently returned NaN.)
    alpha = float(np.exp(final.lnalpha)) if family == "negative_binomial" else np.nan
    intercept = float(coef["const"])

    eval_out = {
        "model_type": family,
        "alpha": alpha,
        "pearson_chi2_per_dof": round(chi2_dof, 3),
        "n_obs": int(len(df)),
        "n_locations": int(df["location_id"].nunique()),
        "n_categories": int(df["category"].nunique()),
        "cv_mae": round(mae, 3),
        "cv_rmse": round(rmse, 3),
        "intercept": round(intercept, 3),
        "reference_category": REFERENCE_CATEGORY,
        "category_rate_shifts": {
            cat: round(float(rates[f"cat_{cat}"]), 3)
            for cat in sorted(df["category"].unique())
            if cat != REFERENCE_CATEGORY
        },
    }
    EVAL_JSON.write_text(json.dumps(eval_out, indent=2), encoding="utf-8")

    print(f"\nWrote predicted_count / gap_residual for {len(df)} rows")
    print(f"Wrote {EVAL_JSON}")
    print(f"\nTop 4 categories with lowest implied rate (most 'scarce'):")
    print(rates.head(4).to_string())


if __name__ == "__main__":
    main()
