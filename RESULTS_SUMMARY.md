# Nairobi Business Market-Gap Results Summary

*Generated from data/processed/biz_intel.db by scripts/09_recommend.py. This is a gap/opportunity signal, not a profitability or survival prediction.*

**Legend for flags:** `[tier2]` informal category - the OSM count is a floor, not a true count (salon, greengrocer, kiosk, fast_food, laundry, motorcycle_taxi). `[fin]` financing-confound category (boda boda stages) - entry may be driven by asset financing, not market gap. `[bbox]` location fetched from a geocoded bounding box, not an OSM admin boundary - lower fidelity. Numbers are businesses per 1,000 people. Full caveats in LIMITATIONS.md.

## Most-underserved categories in notable locations

### Kilimani
- motorcycle_taxi (0.00/1000) [tier2] [fin] [bbox]
- clothes (0.06/1000) [bbox]
- salon (0.08/1000) [tier2] [bbox]
- laundry (0.08/1000) [tier2] [bbox]
- hardware (0.13/1000) [bbox]

### Kasarani
- laundry (0.00/1000) [tier2]
- motorcycle_taxi (0.00/1000) [tier2] [fin]
- salon (0.01/1000) [tier2]
- hardware (0.02/1000)
- greengrocer (0.02/1000) [tier2]

### Embakasi
- motorcycle_taxi (0.00/1000) [tier2] [fin]
- greengrocer (0.00/1000) [tier2]
- clothes (0.01/1000)
- salon (0.01/1000) [tier2]
- laundry (0.01/1000) [tier2]

### Kibera
- motorcycle_taxi (0.00/1000) [tier2] [fin] [bbox]
- clothes (0.02/1000) [bbox]
- salon (0.05/1000) [tier2] [bbox]
- laundry (0.07/1000) [tier2] [bbox]
- hardware (0.08/1000) [bbox]

### Eastleigh
- supermarket (0.00/1000)
- kiosk (0.00/1000) [tier2]
- motorcycle_taxi (0.00/1000) [tier2] [fin]
- hardware (0.00/1000)
- greengrocer (0.00/1000) [tier2]

### Karen
- hairdresser (0.00/1000) [bbox]
- kiosk (0.00/1000) [tier2] [bbox]
- laundry (0.00/1000) [tier2] [bbox]
- motorcycle_taxi (0.00/1000) [tier2] [fin] [bbox]
- clothes (0.03/1000) [bbox]

## Most-underserved locations in notable categories

### supermarket
- Eastleigh (0.00/1000)
- Bahati (0.00/1000)
- Makadara (0.00/1000)
- Umoja (0.00/1000)
- Githurai (0.01/1000) [bbox]

### pharmacy
- Kayole (0.00/1000) [bbox]
- Umoja (0.00/1000)
- Githurai (0.00/1000) [bbox]
- Bahati (0.00/1000)
- Ruai (0.00/1000) [bbox]

### restaurant
- Umoja (0.00/1000)
- Githurai (0.00/1000) [bbox]
- Bahati (0.00/1000)
- Makadara (0.00/1000)
- Laini Saba (0.03/1000) [bbox]

### salon
- Kawangware (0.00/1000) [tier2] [bbox]
- Umoja (0.00/1000) [tier2]
- Githurai (0.00/1000) [tier2] [bbox]
- Kariobangi North (0.00/1000) [tier2] [bbox]
- Roysambu (0.00/1000) [tier2] [bbox]

### kiosk
- Kawangware (0.00/1000) [tier2] [bbox]
- Waithaka (0.00/1000) [tier2] [bbox]
- Kayole (0.00/1000) [tier2] [bbox]
- Umoja (0.00/1000) [tier2]
- Eastleigh (0.00/1000) [tier2]

## Trained-model evaluation

- **Count regression (negative_binomial, n=360):** overdispersion test Pearson chi2/dof = 25.0 (much > 1) -> Negative Binomial used. Dispersion alpha = 2.04. GroupKFold(5) CV MAE = 11.37, RMSE = 26.19 (MAE is the headline; R2 is not used for count data). Categories with the lowest implied rate vs supermarket: motorcycle_taxi, laundry, salon, greengrocer.

- **Supervised saturation classifier (hand-labeled):** 34 labeled (location, category) pairs across 11 locations. GroupKFold(3) CV: accuracy 0.44, macro precision 0.40 / recall 0.41 / F1 0.40. The labeled set is small (34 pairs) and class imbalance is severe; treat this as a weak exploratory result, reported honestly (labels are knowledge-based, not county-CIDP cross-checked).

- **Glass-box vs black-box (same folds, same metrics):** MLP (16-unit) CV MAE = 9.40 vs Negative Binomial GLM CV MAE = 11.37; MLP RMSE = 27.04 vs GLM RMSE = 26.19. The winner depends on the metric -- reported as-is, no model is favored.
