import pandas as pd
import numpy as np
from .config import RAW_FILES, RAW_DIR, PROCESSED_DIR, NUMERIC_FEATURES, CATEGORICAL_FEATURES


def load_raw_data() -> dict:
    print("Loading raw data...")
    data = {}
    for name, path in RAW_FILES.items():
        data[name] = pd.read_parquet(path)
        print(f"  {name}: {data[name].shape}")
    return data


def clean_assessed(df: pd.DataFrame) -> pd.DataFrame:
    # Board of Review → Certified → Mailed
    df = df.copy()
    df["pin"] = df["pin"].astype(str).str.zfill(14)
    df["final_land"] = df["board_land"].fillna(
        df["certified_land"]).fillna(df["mailed_land"])
    df["final_bldg"] = df["board_bldg"].fillna(
        df["certified_bldg"]).fillna(df["mailed_bldg"])
    df["final_tot"] = df["board_tot"].fillna(
        df["certified_tot"]).fillna(df["mailed_tot"])
    return df[["pin", "class", "township_code", "township_name",
               "final_land", "final_bldg", "final_tot"]].copy()


def clean_sales(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pin"] = df["pin"].astype(str).str.zfill(14)
    df["sale_price"] = pd.to_numeric(df["sale_price"], errors="coerce")
    df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")
    df = df[
        (df["is_multisale"].astype(str).str.lower() == "false") &
        (df["sale_price"].between(10_000, 10_000_000))
    ]
    return (
        df.sort_values("sale_date")
          .drop_duplicates(subset="pin", keep="last")[["pin", "sale_price", "sale_date"]]
          .reset_index(drop=True)
    )


def clean_chars(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pin"] = df["pin"].astype(str).str.zfill(14)
    keep_cols = ["pin"] + [c for c in NUMERIC_FEATURES +
                           CATEGORICAL_FEATURES if c in df.columns]
    df = df[keep_cols].copy()
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_parcels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pin"] = df["pin"].astype(str).str.zfill(14)
    return df[["pin", "lat", "lon", "census_tract_geoid"]].copy()


def clean_census(df: pd.DataFrame) -> pd.DataFrame:
    # replace Census sentinel values (-666666666) with NaN
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        df.loc[df[col] < -1, col] = np.nan
    df["pct_black"] = (df["black_population"] /
                       df["total_population"] * 100).round(2)
    df["pct_hispanic"] = (df["hispanic_population"] /
                          df["total_population"] * 100).round(2)
    df["pct_white"] = (df["white_population"] /
                       df["total_population"] * 100).round(2)
    total_units = df["owner_occupied_units"] + df["renter_occupied_units"]
    df["pct_owner_occupied"] = (
        df["owner_occupied_units"] / total_units * 100).round(2)
    return df


def merge_parcels(assessed: pd.DataFrame, chars: pd.DataFrame,
                  parcels: pd.DataFrame, census: pd.DataFrame) -> pd.DataFrame:
    df = assessed.copy()
    df = df.merge(chars,   on="pin", how="left", suffixes=("", "_chars"))
    df = df.merge(parcels, on="pin", how="left")
    df = df.merge(census,  left_on="census_tract_geoid",
                  right_on="geoid", how="left")
    return df


def merge_spatial(df: pd.DataFrame, raw_dir=None) -> pd.DataFrame:
    # Cook County only — silently skipped if spatial_features.parquet not found
    from pathlib import Path
    path = (Path(raw_dir) if raw_dir else RAW_DIR) / "spatial_features.parquet"
    if not path.exists():
        print("  spatial_features.parquet not found — spatial features skipped")
        return df
    spatial = pd.read_parquet(path)
    spatial["pin"] = spatial["pin"].astype(str).str.zfill(14)
    merged = df.merge(spatial, on="pin", how="left")
    n = merged["nearest_cta_stop_dist_ft"].notna().sum()
    print(f"  Spatial features merged: {n:,}/{len(merged):,} PINs matched")
    return merged


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def _level(code):
        try:
            c = str(code).strip()
            if c.upper().startswith("EX"):
                return np.nan
            return 0.10 if int(c) < 500 else 0.25  # residential = 10%, commercial = 25%
        except (ValueError, TypeError):
            return np.nan

    df["assessment_level"] = df["class"].apply(_level)
    df["market_value_total"] = df["final_tot"] / df["assessment_level"]
    df["market_value_land"] = df["final_land"] / df["assessment_level"]
    df["market_value_bldg"] = df["final_bldg"] / df["assessment_level"]
    df["land_ratio"] = np.where(
        df["market_value_total"] > 0,
        df["market_value_land"] / df["market_value_total"],
        np.nan,
    )
    first_digit = df["class"].apply(lambda x: str(x).strip()[
                                    :1] if pd.notna(x) else "")
    df["is_residential"] = first_digit.isin(["2", "3"])
    df["is_vacant"] = first_digit == "1"
    df["is_commercial"] = first_digit.isin(["5", "6"])
    df["is_exempt"] = df["class"].apply(
        lambda x: str(x).strip().upper().startswith(
            "EX") if pd.notna(x) else False
    )
    return df


def create_subsets(df: pd.DataFrame, sales: pd.DataFrame):
    df = df.merge(sales, on="pin", how="left")
    df["has_sale"] = df["sale_price"].notna()

    residential = df[
        df["is_residential"] & ~df["is_exempt"] & (df["final_tot"] > 0)
    ].copy()

    training = residential[
        residential["has_sale"] &
        residential["bldg_sf"].notna() &
        (residential["bldg_sf"] > 0)
    ].copy()

    return residential, training


# Township-level composite rate estimates for Cook County (2023).
# Actual rates vary from ~7% (Chicago) to >20% (south suburbs).
# These approximations are much more accurate than a flat rate.
_COOK_TOWNSHIP_RATES = {
    "Rogers Park": 0.073, "West Chicago": 0.078, "South Chicago": 0.082,
    "Lake": 0.076, "Jefferson": 0.080, "Lake View": 0.071,
    "North Chicago": 0.074, "Hyde Park": 0.079,
    "Niles": 0.093, "Maine": 0.097, "Wheeling": 0.090,
    "Palatine": 0.088, "Schaumburg": 0.085, "Elk Grove": 0.092,
    "Barrington": 0.082, "Hanover": 0.087, "Northfield": 0.078,
    "New Trier": 0.075, "Evanston": 0.095,
    "Proviso": 0.105, "Riverside": 0.098, "Berwyn": 0.102,
    "Cicero": 0.108, "Oak Park": 0.092, "Leyden": 0.096,
    "Stickney": 0.105, "Lyons": 0.100,
    "Thornton": 0.118, "Bloom": 0.125, "Rich": 0.115,
    "Bremen": 0.112, "Calumet": 0.120, "Worth": 0.108,
    "Orland": 0.095, "Palos": 0.092, "Lemont": 0.088,
}
_COOK_RATE_DEFAULT = 0.095   # county-wide median fallback


def load_cook_tax_rates(raw_dir=None, processed_dir=None):
    from pathlib import Path
    raw_dir = Path(raw_dir) if raw_dir else RAW_DIR
    processed_dir = Path(processed_dir) if processed_dir else PROCESSED_DIR

    cache = processed_dir / "pin_composite_rates_2023.parquet"
    rates_path = raw_dir / "tax_rates_by_code_2023.parquet"

    def _load_rates_table():
        rates = pd.read_parquet(rates_path)
        rates["tax_code"] = rates["tax_code"].astype(str).str.strip()
        return rates[rates["composite_rate"].between(0.01, 0.40)][["tax_code", "composite_rate"]]

    def _merge_and_save(pin_tc, label):
        rates = _load_rates_table()
        merged = pin_tc.merge(rates, on="tax_code", how="left")
        n = merged["composite_rate"].notna().sum()
        med = merged["composite_rate"].median()
        print(f"  Tax-code rates ({label}): {n:,}/{len(merged):,} PINs "
              f"({n/len(merged)*100:.1f}%), median {med*100:.2f}%")
        merged["composite_rate"] = merged["composite_rate"].fillna(med)
        result = merged[["pin", "composite_rate"]]
        result.to_parquet(cache, index=False)
        return result

    # Option A: pin_tax_codes.parquet (best — 100% PIN coverage, actual rates)
    ptc_path = raw_dir / "pin_tax_codes.parquet"
    if ptc_path.exists() and rates_path.exists():
        ptc = pd.read_parquet(ptc_path)
        ptc["pin"] = ptc["pin"].astype(str).str.zfill(14)
        ptc["tax_code"] = ptc["tax_code"].astype(str).str.strip()
        return _merge_and_save(ptc[["pin", "tax_code"]].drop_duplicates("pin"), "pin_tax_codes")

    # Option B: tax_code column in parcel_universe
    parcel_path = raw_dir / "parcel_universe.parquet"
    if parcel_path.exists() and rates_path.exists():
        parcels = pd.read_parquet(parcel_path)
        parcels["pin"] = parcels["pin"].astype(str).str.zfill(14)
        if "tax_code" in parcels.columns:
            pin_tc = parcels[["pin", "tax_code"]].drop_duplicates("pin").copy()
            pin_tc["tax_code"] = pin_tc["tax_code"].astype(str).str.strip()
            return _merge_and_save(pin_tc, "parcel_universe")

    # Option C: previously cached result
    if cache.exists():
        df = pd.read_parquet(cache)
        print(f"Cook County tax rates loaded from cache: {len(df):,} PINs")
        return df

    # Option D: township-level estimates (last resort)
    if parcel_path.exists():
        parcels = pd.read_parquet(parcel_path)
        parcels["pin"] = parcels["pin"].astype(str).str.zfill(14)
        if "township_name" in parcels.columns:
            pin_twp = parcels[["pin", "township_name"]].drop_duplicates("pin").copy()
            pin_twp["composite_rate"] = (
                pin_twp["township_name"].map(_COOK_TOWNSHIP_RATES).fillna(_COOK_RATE_DEFAULT)
            )
            known = pin_twp["township_name"].isin(_COOK_TOWNSHIP_RATES).sum()
            print(f"  Township rates (fallback): {known:,}/{len(pin_twp):,} PINs matched")
            result = pin_twp[["pin", "composite_rate"]]
            result.to_parquet(cache, index=False)
            return result

    print("  No tax rate source found — per-parcel rates unavailable")
    return None


def add_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["majority_group"] = np.where(
        df["pct_black"] > 50, "Majority Black",
        np.where(df["pct_hispanic"] > 50, "Majority Hispanic",
                 np.where(df["pct_white"] > 50, "Majority White", "Mixed")),
    )
    df["income_quartile"] = pd.qcut(
        df["median_household_income"],
        q=4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"],
        duplicates="drop",
    )
    return df
