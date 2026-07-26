import json
import pandas as pd

RAW_PATH = "../data/raw/osm_pilot_kilimani.json"
OUT_PATH = "../data/processed/osm_pilot_kilimani_clean.csv"


def flatten(raw: dict) -> pd.DataFrame:
    rows = []
    for category, entries in raw.items():
        for entry in entries:
            tags = entry.get("tags", {})
            rows.append({
                "osm_id": entry["id"],
                "category": category,
                "lat": entry["lat"],
                "lon": entry["lon"],
                "name": tags.get("name"),  # None if missing, don't guess
            })
    return pd.DataFrame(rows)


def main():
    with open(RAW_PATH) as f:
        raw = json.load(f)

    df = flatten(raw)

    print(f"Total rows: {len(df)}")
    print(f"\nRows missing a name (unnamed businesses):")
    missing_name = df["name"].isna().sum()
    print(f"  {missing_name} of {len(df)} ({missing_name/len(df)*100:.0f}%)")

    print("\nCount per category:")
    print(df["category"].value_counts())

    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved clean table to {OUT_PATH}")


if __name__ == "__main__":
    main()