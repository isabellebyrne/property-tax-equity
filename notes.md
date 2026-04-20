# Implementation Notes

Reference document for non-obvious design decisions in this codebase.

---

## Data pipeline (Cook County)

```
load_raw_data()
  → clean_assessed / clean_sales / clean_chars / clean_parcels / clean_census
  → merge_parcels()           # left-joins assessed + chars + parcels + census on pin
  → merge_spatial()      # left-joins spatial_features.parquet on pin (Cook only)
  → add_derived()        # assessment levels, market values, land_ratio, flags
  → create_subsets()          # splits into residential and training_set
```

For NYC and Philly: `build_nyc_datasets()` / `build_philly_datasets()` do the equivalent merge + derive in one step.

---

## Cook County tax structure

- **State equalizer**: 3.0163× (Illinois Department of Revenue, 2023). Applied as `AV × equalizer × rate`.
- **Assessment levels**: class < 500 = 10% of market value (residential); class ≥ 500 = 25% (commercial). `market_value = final_tot / level`.
- **Composite rate**: actual 2023 rates range from ~7% (Chicago) to >20% (south suburbs). Four-level fallback in `load_cook_tax_rates()`:
  1. `pin_tax_codes.parquet` + `tax_rates_by_code_2023.parquet` — 100% PIN match, actual rates
  2. `parcel_universe.tax_code` + same rates table
  3. Cached `pin_composite_rates_2023.parquet`
  4. Township-level estimates from `_COOK_TOWNSHIP_RATES` dict (~36 townships)
  - Rates capped at 40% to exclude two known erroneous tax codes (76.9% in source data).
- **PIN format**: 14-digit zero-padded string (`str.zfill(14)`). Raw data may be 10-digit (PIN10) or have other formats — always normalize on load.

---

## NYC RPTL §581 — AVM scope restriction

AVM training and prediction is restricted to **Tax Class 1** (PLUTO `landuse == '01'`, 1-3 family homes).

Class 2 properties (landuse 02/03: rental apartments, co-ops, condominiums) are assessed under a **statutory income capitalization methodology** per RPTL §581, not comparable sales. Applying a hedonic AVM-based assessment ratio to Class 2 is methodologically inappropriate.

Consequence in code:
- `training` in `build_nyc_datasets()` is filtered to `landuse == '01'`
- After `predict_all()`, Class 2/3 parcels get `predicted_market_value = market_value_total` → `assessment_ratio = 1.0` (neutral placeholder — no AVM opinion)
- OLS equity analysis is filtered to `landuse == '01'` only; including Class 2/3 (all ratio=1.0 by construction) would mechanically suppress disparity signal

Source: NYC Advisory Commission on Property Tax Reform (2020); IAAO Standard on Mass Appraisal.

---

## NYC assessment rates by tax class

| PLUTO landuse | Tax Class | Statutory rate | Description |
|---|---|---|---|
| 01 | Class 1 | 6% of market value | 1-3 family homes |
| 02 | Class 2 | 45% of market value | Walk-up apartments |
| 03 | Class 2 | 45% of market value | Elevator apartments / co-ops / condos |

`implied_market_value = assesstot / rate` is used as `market_value_total` for equity analysis and LVT simulation.

---

## Revenue-neutral LVT formula

```
lvt_mult = Σ(AV_i × rate_i) / Σ(land_AV_i × rate_i)
```

`lvt_mult` scales the land tax rate so total revenue is unchanged. Cook County uses per-parcel rates (`composite_rate` column); NYC and Philly use a flat rate. The formula is rate-weighted, so it handles the heterogeneous Cook County rate structure correctly.

---

## Feature lists

**Valuation model (Cook County) — `NUMERIC_FEATURES` + spatial:**
```
bldg_sf, hd_sf, age, beds, fbath, hbath, rooms, frpl, gar1_size, attic_fnsh
nearest_cta_stop_dist_ft, nearest_metra_stop_dist_ft, nearest_water_dist_ft,
lake_michigan_dist_ft, num_bus_stop_in_half_mile, num_pin_in_half_mile,
num_school_in_half_mile, num_foreclosure_in_half_mile_past_5_years,
env_flood_fs_factor, env_airport_noise_dnl
```
Spatial features come from `data/raw/spatial_features.parquet` and are only present in Cook County. They are absent from processed parquets if `data_preparation.ipynb` cell `aa000004` hasn't been re-run since `merge_spatial_features()` was added. `model_utils.py` filters feature lists to available columns before training.

**NYC valuation:** `bldgarea, lotarea, age, unitsres, numfloors` + `borough, bldgclass`

**Philly valuation:** `total_livable_area, total_area, age, number_of_bedrooms, number_of_bathrooms, number_of_rooms` + `zip_code, zoning, building_code_description_new`

**Classification (LVT beneficiary):** `market_value_total/land/bldg, assessment_ratio, bldg_sf, hd_sf, age, beds, fbath, rooms, median_household_income, pct_black, pct_hispanic, pct_owner_occupied`

**Clustering (tract level):** `median_ratio, median_land_ratio, median_tax_change_pct, pct_lvt_benefit, median_household_income, pct_black, pct_hispanic, pct_owner_occupied, median_predicted_value`

---

## Feature name derivation (XGBoost)

After fitting, XGBoost feature names are read from the fitted `ColumnTransformer` in `get_xgb_feature_names()`:
- Numeric names: `preprocessor.named_transformers_["num"].feature_names_in_`
- OHE names: `preprocessor.named_transformers_["cat"].get_feature_names_out()`

This is necessary because feature lists may be filtered before training (missing columns dropped), so passing the original lists in would cause a length mismatch.

---

## NYC Census GEOID construction

PLUTO stores `tract2010` as an integer. Small tracts store the value ÷ 100 (e.g., tract 11.23 → integer 1123); large tracts store the 6-digit FIPS directly (e.g., tract 1072.01 → integer 107201).

Rule: if `tract2010 × 100 < 1,000,000`, apply ×100; otherwise treat as already in FIPS format. Then zero-pad to 6 digits and prepend `'36' + county_fips`.

Borough → county FIPS: MN=061, BX=005, BK=047, QN=081, SI=085.

This achieves ~90% GEOID match with 2023 ACS (residual = tracts split between 2010 PLUTO geometry and 2023 ACS vintage).

---

## Philadelphia exemption and tax base

- `taxable_land + exempt_land = full land market value` (exempt_land = homestead exemption portion)
- `taxable_total = taxable_land + taxable_building` = the actual on-roll tax base (after exemptions)
- Philly targets 100% of market value → `fair_ratio = 1.0`, `state_equalizer = 1.0`
- LVT simulation uses `taxable_total` / `taxable_land` as the tax base (consistent with Cook `final_tot` / `final_land` and NYC `assesstot` / `assessland`)
- **Act 76 abatement**: parcels with `market_value_bldg == 0` and `market_value_total > 0` have building value absent from OPA data. Imputed using citywide median land share from non-abated parcels.

---

## Philadelphia Census GEOID

OPA stores `census_tract` as a plain integer (e.g., 73 → tract 73.00). Apply ×100 and zero-pad to 6 digits: `73 → 7300 → '007300'` → GEOID `'42101007300'` (state 42, county 101). Achieves ~65% match with 2023 ACS.

---

## NYC rolling sales filter

`clean_nyc_sales()`: price range $10K–$50M (NYC has multi-million transactions; Cook County uses $10K–$10M). One sale per BBL (most recent arm's-length transaction).

BBL construction: `boro (1 digit) + block (5 digits, zero-padded) + lot (4 digits, zero-padded)` = 10 digits.

---

## Philadelphia sales filter

`clean_philly_sales()`: price range $10K–$10M; restricted to 2019+ so training targets reflect current market values (not decade-old prices vs. current OPA assessments). Uses `adjusted_total_consideration` (partial-interest adjusted). Timestamps include UTC offset — stripped to naive datetime.
