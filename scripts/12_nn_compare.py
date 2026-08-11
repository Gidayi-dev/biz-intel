"""Glass-box (GLM) vs black-box (MLP) comparison on the count target.

Same target (expected business_count), same GroupKFold(5) folds grouped by
location_id, same metrics (MAE, RMSE) as scripts/10_model_regression.py, so
the two models are directly comparable.

Because counts are zero-heavy and right-skewed (mean ~10, variance ~760), the
MLP is trained on log1p(business_count) and its predictions are back-transformed
with expm1 before scoring -- a standard stabilisation for count targets. The
Negative Binomial GLM is the reference (its CV MAE is loaded from
model_eval_regression.json so both models are measured on identical folds).

At ~360 rows the two models are expected to land within a few MAE points of
each other. Whichever wins is a legitimate finding, reported as-is -- the spec
explicitly says don't force the NN to appear superior (and don't bury a
glass-box win either).
"""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "processed" / "biz_intel.db"
REGRESSION_JSON = ROOT / "data" / "processed" / "model_eval_regression.json"
EVAL_JSON = ROOT / "data" / "processed" / "model_eval_nn.json"

SEED = 42
N_SPLITS = 5


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


def build_design(df: pd.DataFrame, cats: list[str]) -> pd.DataFrame:
    dummies = pd.get_dummies(df["category"], prefix="cat", dtype=float)
    for c in cats:
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
    X = build_design(df, cats).to_numpy()
    y = df["business_count"].to_numpy(float)
    groups = df["location_id"].to_numpy()
    # log1p stabilisation for the MLP target; scored on the original scale.
    y_log = np.log1p(y)

    gkf = GroupKFold(n_splits=N_SPLITS)
    preds, actuals = [], []
    for train_idx, test_idx in gkf.split(X, y, groups):
        mlp = make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(16,), max_iter=3000,
                         random_state=SEED, early_stopping=True))
        mlp.fit(X[train_idx], y_log[train_idx])
        preds.append(np.expm1(mlp.predict(X[test_idx])))
        actuals.append(y[test_idx])
    pred = np.concatenate(preds)
    actual = np.concatenate(actuals)

    mae = float(np.mean(np.abs(actual - pred)))
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))

    # Load the GLM's identical-fold CV metrics for the side-by-side.
    glm = json.loads(REGRESSION_JSON.read_text(encoding="utf-8"))
    out = {
        "model": "mlp",
        "target": "business_count (log1p stabilised, expm1 back-transform)",
        "cv_protocol": f"GroupKFold({N_SPLITS}) grouped by location_id "
                       "(same folds as the regression)",
        "mlp_cv_mae": round(mae, 3),
        "mlp_cv_rmse": round(rmse, 3),
        "glm_cv_mae": glm["cv_mae"],
        "glm_cv_rmse": glm["cv_rmse"],
        "glm_model_type": glm["model_type"],
        "glm_wins_mae": glm["cv_mae"] <= mae,
        "n_obs": int(len(df)),
        "hidden_layer_sizes": [16],
    }
    EVAL_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("Glass-box (Negative Binomial GLM) vs black-box (MLP), "
          "identical GroupKFold folds, same MAE/RMSE on original count scale:")
    print(f"  GLM  ({glm['model_type']}):  MAE {glm['cv_mae']:.3f}  "
          f"RMSE {glm['cv_rmse']:.3f}")
    print(f"  MLP  (16-unit, log1p):      MAE {mae:.3f}  RMSE {rmse:.3f}")
    verdict = ("GLM" if glm["cv_mae"] <= mae else "MLP")
    print(f"  Verdict on MAE: {verdict} wins -- "
          "either outcome is reported honestly; the winner varies by metric "
          "and neither is forced.")
    print(f"\nWrote {EVAL_JSON}")


if __name__ == "__main__":
    main()
