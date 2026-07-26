import pandas as pd

CLEAN_OSM_PATH = "../data/processed/osm_pilot_kilimani_clean.csv"
OUT_PATH = "../data/processed/kilimani_features.csv"

CENSUS_KILIMANI = {
  "location": "Kilimani",
  "sub_county": "Westlands",
  "county": "Nairobi",
  "population": 82970,
  "households": 25723,
  "land_area_sqkm": 25.1,
}

def main():
  osm = pd.read_csv(CLEAN_OSM_PATH)

  counts = osm["category"].value_counts().rename("business_count")
  features = counts.to_frame().reset_index().rename(columns={"index": "category"})

  pop = CENSUS_KILIMANI["population"]
  area = CENSUS_KILIMANI["land_area_sqkm"]

  features["business_per_1000_people"] = (
    features["business_count"] / pop * 1000
  )
  features["business_per_sqkm"] = features["business_count"] / area

  features["location"] = CENSUS_KILIMANI["location"]
  features["population"] = pop

  features = features.sort_values("business_per_1000_people", ascending=False)

  print(f"Kilimani location: {pop:,} people, {area} sq km\n")
  print(features.to_string(index=False))

  features.to_csv(OUT_PATH, index=False)
  print(f"\nSaved feature table to {OUT_PATH}")

if __name__ == "__main__":
  main()