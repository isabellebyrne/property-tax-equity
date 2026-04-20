"""Multi-city property tax equity analysis configuration."""
from pathlib import Path

_SRC_DIR = Path(__file__).parent
PROJECT_DIR = _SRC_DIR.parent

RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
FIGURES_DIR = PROJECT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

TARGET_YEAR = 2023
RAW_FILES = {
    "assessed": RAW_DIR / f"assessed_values_{TARGET_YEAR}.parquet",
    "sales": RAW_DIR / f"sales_2021_{TARGET_YEAR}.parquet",
    "chars": RAW_DIR / "property_chars_2019.parquet",
    "parcels": RAW_DIR / "parcel_universe.parquet",
    "census": RAW_DIR / f"census_acs_{TARGET_YEAR}.parquet",
}

PROCESSED_FILES = {
    "merged_all": PROCESSED_DIR / "merged_all_parcels.parquet",
    "residential": PROCESSED_DIR / "residential_parcels.parquet",
    "training_set": PROCESSED_DIR / "training_set.parquet",
    "with_predictions": PROCESSED_DIR / "residential_with_predictions.parquet",
    "with_lvt": PROCESSED_DIR / "residential_with_lvt.parquet",
    "tract_analysis": PROCESSED_DIR / "tract_level_analysis.parquet",
    "tracts_clusters": PROCESSED_DIR / "tracts_with_clusters.parquet",
}

STATE_EQUALIZER = 3.0163  # Cook County equalization factor (2023, official)
COMPOSITE_RATE = 0.095    # County-wide average; actual rates vary 7%–13%+

RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "bldg_sf", "hd_sf", "age", "beds", "fbath", "hbath",
    "rooms", "frpl", "gar1_size", "attic_fnsh",
]

CATEGORICAL_FEATURES = [
    "township_code", "nbhd", "type_resd", "ext_wall",
    "bsmt", "air", "cnst_qlty",
]

CLASSIFICATION_FEATURES = [
    "market_value_total", "market_value_land",
    "market_value_bldg", "assessment_ratio",
    "bldg_sf", "hd_sf", "age", "beds", "fbath", "rooms",
    "median_household_income", "pct_black", "pct_hispanic", "pct_owner_occupied",
]

CLUSTER_FEATURES = [
    "median_ratio", "median_land_ratio", "median_tax_change_pct",
    "pct_lvt_benefit", "median_household_income", "pct_black",
    "pct_hispanic", "pct_owner_occupied", "median_predicted_value",
]

CITIES: dict = {

    "cook": {
        "label": "Cook County, IL",
        "short": "cook",
        "parcel_id": "pin",
        "tax_base_total": "final_tot",
        "tax_base_land": "final_land",
        "fair_ratio": 1.0,
        "state_equalizer": 3.0163,
        "composite_rate": 0.095,
        "lat": "lat",
        "lon": "lon",
        "numeric_features": [
            "bldg_sf", "hd_sf", "age", "beds", "fbath", "hbath",
            "rooms", "frpl", "gar1_size", "attic_fnsh",
            # spatial features from spatial_features.parquet (Cook only)
            "nearest_cta_stop_dist_ft", "nearest_metra_stop_dist_ft",
            "nearest_water_dist_ft", "lake_michigan_dist_ft",
            "num_bus_stop_in_half_mile", "num_pin_in_half_mile",
            "num_school_in_half_mile", "num_foreclosure_in_half_mile_past_5_years",
            "env_flood_fs_factor", "env_airport_noise_dnl",
        ],
        "categorical_features": [
            "township_code", "nbhd", "type_resd", "ext_wall",
            "bsmt", "air", "cnst_qlty",
        ],
        "year_built_col": None,
        "sqft_col": "bldg_sf",
        "files": {
            "training": "training_set.parquet",
            "residential": "residential_parcels.parquet",
            "all_parcels": "merged_all_parcels.parquet",
            "residential_pred": "cook_residential_with_predictions.parquet",
            "tract_analysis": "cook_tract_level_analysis.parquet",
            "residential_lvt": "cook_residential_with_lvt.parquet",
            "tracts_clusters": "cook_tracts_with_clusters.parquet",
        },
    },

    "nyc": {
        "label": "New York City, NY",
        "short": "nyc",
        "parcel_id": "bbl",
        "tax_base_total": "assesstot",
        "tax_base_land": "assessland",
        "fair_ratio": 1.0,
        "state_equalizer": 1.0,
        # Actual FY2023 Class 1 rate ≈ 20.31% of AV; Class 2 ≈ 12.63% of AV.
        # 0.10 cancels in the revenue-neutral multiplier so distributional direction
        # (% benefit, benefit by group, Gini sign) is unaffected. Dollar amounts
        # (current_tax_est, lvt_tax_est, median tax_change) are understated ~50%
        # for Class 1. Do not cite NYC dollar figures without this caveat.
        "composite_rate": 0.10,
        "lat": "latitude",
        "lon": "longitude",
        "numeric_features": [
            "bldgarea", "lotarea", "age", "unitsres", "numfloors",
        ],
        "categorical_features": [
            "borough", "bldgclass",
        ],
        "year_built_col": "yearbuilt",
        "sqft_col": "bldgarea",
        "files": {
            "training": "nyc_training_set.parquet",
            "residential": "nyc_residential_parcels.parquet",
            "all_parcels": "nyc_all_parcels.parquet",
            "residential_pred": "nyc_residential_with_predictions.parquet",
            "tract_analysis": "nyc_tract_level_analysis.parquet",
            "residential_lvt": "nyc_residential_with_lvt.parquet",
            "tracts_clusters": "nyc_tracts_with_clusters.parquet",
        },
    },

    "philly": {
        "label": "Philadelphia, PA",
        "short": "philly",
        "parcel_id": "pin",
        # taxable_total (= taxable_land + taxable_building) is the on-roll tax base
        "tax_base_total": "taxable_total",
        "tax_base_land": "taxable_land",
        "fair_ratio": 1.0,
        "state_equalizer": 1.0,
        "composite_rate": 0.013,
        "lat": None,
        "lon": None,
        "numeric_features": [
            "total_livable_area", "total_area", "age",
            "number_of_bedrooms", "number_of_bathrooms", "number_of_rooms",
        ],
        "categorical_features": [
            "zip_code", "zoning", "building_code_description_new",
        ],
        "year_built_col": "year_built",
        "sqft_col": "total_livable_area",
        "files": {
            "training": "philly_training_set.parquet",
            "residential": "philly_residential_parcels.parquet",
            "all_parcels": "philly_all_parcels.parquet",
            "residential_pred": "philly_residential_with_predictions.parquet",
            "tract_analysis": "philly_tract_level_analysis.parquet",
            "residential_lvt": "philly_residential_with_lvt.parquet",
            "tracts_clusters": "philly_tracts_with_clusters.parquet",
        },
        # median_land_ratio excluded: Act 76 abatement imputation collapses it to constant 0.200
        "cluster_features_override": [
            "median_ratio", "median_tax_change_pct",
            "pct_lvt_benefit", "median_household_income", "pct_black",
            "pct_hispanic", "pct_owner_occupied", "median_predicted_value",
        ],
        # market_value_* excluded: derived from taxable_land/total via assessment_level,
        # encoding the LVT outcome (lvt_benefits = taxable_land/taxable_total < 1/lvt_mult) directly
        "classification_features_override": [
            "assessment_ratio",
            "total_livable_area",
            "total_area",
            "age",
            "number_of_bedrooms",
            "number_of_bathrooms",
            "number_of_rooms",
            "median_household_income",
            "pct_black",
            "pct_hispanic",
            "pct_owner_occupied",
        ],
    },
}


def get_city_config(city: str) -> dict:
    city = city.lower().strip()
    if city not in CITIES:
        raise ValueError(
            f"Unknown city '{city}'. Choose from: {list(CITIES.keys())}")
    return CITIES[city]
