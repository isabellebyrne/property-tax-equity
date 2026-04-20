import time
import numpy as np
import pandas as pd
import requests


_ACS_VARS = {
    "B19013_001E": "median_household_income",
    "B03002_001E": "total_population",
    "B03002_003E": "white_population",
    "B03002_004E": "black_population",
    "B03002_012E": "hispanic_population",
    "B25003_002E": "owner_occupied_units",
    "B25003_003E": "renter_occupied_units",
    "B25077_001E": "median_home_value",
    "B25064_001E": "median_gross_rent",
}


def fetch_acs_tracts(
    state_fips: str,
    county_fips_list: list,
    year: int = 2023,
) -> pd.DataFrame:
    """Fetch ACS 5-year tract demographics from Census API; county_fips_list items are zero-padded 3-digit strings ('031'); includes 250ms throttle per request."""
    base = f"https://api.census.gov/data/{year}/acs/acs5"
    var_string = ",".join(_ACS_VARS.keys())
    frames = []

    for county in county_fips_list:
        url = (
            f"{base}?get=NAME,{var_string}"
            f"&for=tract:*&in=state:{state_fips}%20county:{county}"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        rows = resp.json()
        df = pd.DataFrame(rows[1:], columns=rows[0])
        df["census_tract_geoid"] = df["state"] + df["county"] + df["tract"]
        frames.append(df)
        time.sleep(0.25)   # stay polite to the API

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns=_ACS_VARS)
    combined = combined.drop(columns=["NAME", "state", "county", "tract"])
    for col in _ACS_VARS.values():
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
        combined[col] = combined[col].where(combined[col] >= 0, other=np.nan)
    return combined


def derive_census_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pop = df["total_population"].replace(0, np.nan)
    df["pct_black"] = df["black_population"] / pop * 100
    df["pct_hispanic"] = df["hispanic_population"] / pop * 100
    df["pct_white"] = df["white_population"] / pop * 100
    total_units = df["owner_occupied_units"] + df["renter_occupied_units"]
    df["pct_owner_occupied"] = df["owner_occupied_units"] / \
        total_units.replace(0, np.nan) * 100
    return df


_BORO_TO_COUNTY = {
    "MN": "061",   # Manhattan  (New York County)
    "BX": "005",   # Bronx      (Bronx County)
    "BK": "047",   # Brooklyn   (Kings County)
    "QN": "081",   # Queens     (Queens County)
    "SI": "085",   # Staten Island (Richmond County)
}


def _nyc_tract_geoid(borough_series: pd.Series, tract2010_series: pd.Series) -> pd.Series:
    # tract2010 stores FIPS ÷ 100 for small tracts; apply ×100 unless already 6-digit
    county = borough_series.str.strip().str.upper().map(_BORO_TO_COUNTY)
    tract = pd.to_numeric(tract2010_series, errors="coerce")
    scaled = tract * 100
    fips_int = np.where(scaled < 1_000_000, scaled, tract)
    fips_str = (
        pd.Series(fips_int, index=borough_series.index)
        .astype("Int64").astype(str).str.zfill(6)
        .replace("<NA>", np.nan)
    )
    return pd.Series(
        np.where(county.notna() & (fips_str != "nan"),
                 "36" + county + fips_str, np.nan),
        index=borough_series.index,
    )


def clean_nyc_pluto(pluto_csv_path: str) -> pd.DataFrame:
    """Loads MapPLUTO CSV and derives census_tract_geoid from borough + tract2010."""
    df = pd.read_csv(pluto_csv_path, low_memory=False)

    df.columns = df.columns.str.lower().str.replace(" ", "_")

    df["bbl"] = df["bbl"].astype(str).str.split(".").str[0].str.zfill(10)

    keep = [
        "bbl", "borough", "address", "assessland", "assesstot", "exempttot",
        "landuse", "bldgclass", "lotarea", "bldgarea", "resarea", "comarea",
        "unitsres", "unitstotal", "yearbuilt", "numfloors",
        "latitude", "longitude", "tract2010", "borocode",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()

    numeric_cols = [
        "assessland", "assesstot", "exempttot", "lotarea", "bldgarea",
        "resarea", "comarea", "unitsres", "unitstotal", "yearbuilt",
        "numfloors", "latitude", "longitude", "borocode", "tract2010",
    ]
    for col in numeric_cols:
        if col in df.columns:
            # PLUTO area columns use comma thousands-separators ("1,224") — strip before coerce
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["landuse"] = (
        pd.to_numeric(df.get("landuse"), errors="coerce")
        .astype("Int64").astype(str).str.zfill(2)
        .replace("<NA>", np.nan)
    )

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    df["census_tract_geoid"] = _nyc_tract_geoid(df["borough"], df["tract2010"])

    print(f"NYC PLUTO loaded: {len(df):,} parcels")
    print(
        f"  Census GEOID coverage: {df['census_tract_geoid'].notna().sum():,} ({df['census_tract_geoid'].notna().mean()*100:.1f}%)")
    return df


def clean_nyc_sales(sales_csv_path: str) -> pd.DataFrame:
    """Constructs BBL from BORO/BLOCK/LOT columns; arm's-length filter $10K–$50M, most recent per BBL."""
    df = pd.read_csv(sales_csv_path, low_memory=False)
    df.columns = df.columns.str.strip()

    boro = pd.to_numeric(df["BOROUGH"], errors="coerce").astype("Int64")
    block = pd.to_numeric(df["BLOCK"],   errors="coerce").astype("Int64")
    lot = pd.to_numeric(df["LOT"],     errors="coerce").astype("Int64")

    df["bbl"] = np.where(
        boro.notna() & block.notna() & lot.notna(),
        boro.astype(str) + block.astype(str).str.zfill(5) +
        lot.astype(str).str.zfill(4),
        np.nan,
    )
    df["sale_price"] = pd.to_numeric(df["SALE PRICE"], errors="coerce")
    df["sale_date"] = pd.to_datetime(df["SALE DATE"],  errors="coerce")

    df = df[(df["sale_price"] > 10_000) & (
        df["sale_price"] < 50_000_000)].copy()

    result = (
        df.dropna(subset=["bbl"])
          .sort_values("sale_date")
          .drop_duplicates(subset="bbl", keep="last")[["bbl", "sale_price", "sale_date"]]
          .reset_index(drop=True)
    )
    print(f"NYC Sales loaded: {len(result):,} arm's-length transactions")
    return result


def build_nyc_datasets(
    pluto: pd.DataFrame,
    sales: pd.DataFrame,
    census: pd.DataFrame,
) -> tuple:
    merged = pluto.merge(
        sales[["bbl", "sale_price", "sale_date"]], on="bbl", how="left")
    merged["has_sale"] = merged["sale_price"].notna()

    census_w_features = derive_census_features(census)
    merged = merged.merge(
        census_w_features, on="census_tract_geoid", how="left")

    # condos/co-ops store usable area in resarea, not bldgarea
    merged["bldgarea"] = np.where(
        merged["bldgarea"].isna() | (merged["bldgarea"] == 0),
        merged["resarea"],
        merged["bldgarea"],
    )

    # Per-landuse statutory assessment rates (% of market value):
    #   01 (1-2 family, Tax Class 1)      = 6%
    #   02 (walk-up apartments, Class 2)  = 45%
    #   03 (elevator apartments, Class 2) = 45%
    _NYC_RATES = {"01": 0.06, "02": 0.45, "03": 0.45}
    rate_series = merged["landuse"].map(_NYC_RATES).fillna(0.45)

    merged["implied_market_value"] = np.where(
        rate_series > 0,
        merged["assesstot"] / rate_series,
        np.nan,
    )
    merged["market_value_total"] = merged["implied_market_value"]
    merged["market_value_land"] = np.where(
        rate_series > 0,
        merged["assessland"] / rate_series,
        np.nan,
    )
    merged["market_value_bldg"] = (
        merged["market_value_total"] - merged["market_value_land"]
    ).clip(lower=0)
    merged["land_ratio"] = np.where(
        merged["market_value_total"] > 0,
        merged["market_value_land"] / merged["market_value_total"],
        np.nan,
    )

    merged["is_residential"] = merged["landuse"].isin(["01", "02", "03"])
    merged["is_vacant"] = merged["landuse"].isin(["10", "11"])
    merged["is_commercial"] = merged["landuse"].isin(["04", "05", "06"])
    # Only truly exempt parcels (pay $0 in taxes — churches, govt buildings).
    # Partial exemptions (STAR, 421-a, J-51, SCRIE) still pay property taxes.
    _exempttot = merged.get("exempttot",  pd.Series(
        0, index=merged.index)).fillna(0)
    _assesstot = merged.get("assesstot",  pd.Series(
        0, index=merged.index)).fillna(0)
    merged["is_exempt"] = (_exempttot >= _assesstot) & (_assesstot > 0)
    merged["is_exempt"] = merged["is_exempt"] | (_assesstot == 0)

    all_parcels = merged.copy()
    residential = merged[
        merged["is_residential"] & ~merged["is_exempt"] & (
            merged["market_value_total"] > 0)
    ].copy()
    # NYC AVM is restricted to Tax Class 1 (landuse 01, 1-3 family residential).
    # Class 2 properties (rental apartments, co-ops, condominiums) are assessed
    # under a statutory income capitalization methodology per RPTL §581, not
    # comparable sales. Hedonic AVM-based assessment ratios are methodologically
    # inappropriate for Class 2. This restriction is consistent with prior
    # literature on NYC assessment equity (NYC Advisory Commission, 2020).
    training = residential[
        residential["has_sale"] &
        residential["bldgarea"].notna() &
        (residential["bldgarea"] > 0) &
        (residential["landuse"] == "01")
    ].copy()

    n_class1 = (residential["landuse"] == "01").sum()
    n_class2 = (residential["landuse"].isin(["02", "03"])).sum()
    print(f"\nNYC datasets built:")
    print(f"  all_parcels:  {len(all_parcels):,}")
    print(f"  residential:  {len(residential):,}")
    print(f"    Class 1 (AVM scope):  {n_class1:,}")
    print(f"    Class 2 (income cap): {n_class2:,}  — excluded from AVM per RPTL §581")
    print(f"  training_set: {len(training):,}")
    print(
        f"  Census match: {merged['median_household_income'].notna().mean()*100:.1f}%")
    return all_parcels, residential, training


def _philly_tract_geoid(census_tract_series: pd.Series) -> pd.Series:
    # OPA stores census_tract as integer (e.g., 73 → tract 73.00); multiply ×100, pad to 6 digits
    ct = pd.to_numeric(census_tract_series, errors="coerce")
    fips = (
        (ct * 100).round()
        .astype("Int64").astype(str).str.zfill(6)
        .replace("<NA>", np.nan)
    )
    return pd.Series(
        np.where(fips != "nan", "42101" + fips, np.nan),
        index=census_tract_series.index,
    )


def clean_philly_assessments(assess_csv_path: str) -> pd.DataFrame:
    """Loads OPA assessment CSV; derives census_tract_geoid from OPA integer tract × 100."""
    df = pd.read_csv(assess_csv_path, low_memory=False)

    keep = [
        "parcel_number", "market_value", "taxable_land", "taxable_building",
        "exempt_land", "exempt_building",
        "census_tract", "year_built", "total_livable_area",
        "number_of_bedrooms", "number_of_bathrooms", "number_of_rooms",
        "total_area", "house_number", "street_name", "zip_code",
        "category_code", "zoning", "building_code_description_new",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()

    numeric_cols = [
        "market_value", "taxable_land", "taxable_building",
        "exempt_land", "exempt_building",
        "year_built", "total_livable_area", "number_of_bedrooms",
        "number_of_bathrooms", "number_of_rooms", "total_area", "category_code",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # parcel_number may be float due to NaN rows; coerce to Int64 then string
    df["pin"] = (
        pd.to_numeric(df["parcel_number"], errors="coerce")
        .astype("Int64").astype(str)
        .replace("<NA>", np.nan)
        .str.strip()
    )
    df = df.drop(columns=["parcel_number"])

    df["census_tract_geoid"] = _philly_tract_geoid(df["census_tract"])

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    print(f"Philadelphia assessments loaded: {len(df):,} parcels")
    print(
        f"  Census GEOID coverage: {df['census_tract_geoid'].notna().sum():,} ({df['census_tract_geoid'].notna().mean()*100:.1f}%)")
    return df


def clean_philly_sales(sales_csv_path: str) -> pd.DataFrame:
    """Arm's-length filter $10K–$10M; UTC timestamps stripped to naive; restricted to 2019+ for current market relevance."""
    df = pd.read_csv(sales_csv_path, low_memory=False)

    df["pin"] = (
        pd.to_numeric(df["opa_account_num"], errors="coerce")
        .astype("Int64").astype(str)
        .replace("<NA>", np.nan)
        .str.strip()
    )
    df["sale_price"] = pd.to_numeric(
        df["adjusted_total_consideration"], errors="coerce")
    # timestamps include UTC offset; strip to naive for consistency
    df["sale_date"] = (
        pd.to_datetime(df["document_date"], errors="coerce", utc=True)
        .dt.tz_localize(None)
    )

    df = df[(df["sale_price"] > 10_000) & (
        df["sale_price"] < 10_000_000)].copy()
    # restrict to 2019+ so training targets reflect current market, not decade-old prices
    df = df[df["sale_date"] >= pd.Timestamp("2019-01-01")].copy()

    result = (
        df.dropna(subset=["pin"])
          .sort_values("sale_date")
          .drop_duplicates(subset="pin", keep="last")[["pin", "sale_price", "sale_date"]]
          .reset_index(drop=True)
    )
    print(
        f"Philadelphia sales loaded: {len(result):,} arm's-length transactions (2019+)")
    return result


def build_philly_datasets(
    assess: pd.DataFrame,
    sales: pd.DataFrame,
    census: pd.DataFrame,
) -> tuple:
    merged = assess.merge(
        sales[["pin", "sale_price", "sale_date"]], on="pin", how="left")
    merged["has_sale"] = merged["sale_price"].notna()

    census_w_features = derive_census_features(census)
    merged = merged.merge(
        census_w_features, on="census_tract_geoid", how="left")

    # taxable_land + exempt_land = full land market value (exempt_land = homestead portion)
    merged["market_value_total"] = merged["market_value"]
    merged["market_value_land"] = merged["taxable_land"] + \
        merged["exempt_land"].fillna(0)
    merged["market_value_bldg"] = merged["taxable_building"] + \
        merged["exempt_building"].fillna(0)

    # Act 76 abatement: building value absent from OPA data for abated parcels —
    # impute using citywide median land share from non-abated parcels
    _abated = (merged["market_value_bldg"] == 0) & (
        merged["market_value_total"] > 0)
    _valid = (merged["market_value_bldg"] > 0) & (
        merged["market_value_total"] > 0)
    if _abated.sum() > 0 and _valid.sum() > 0:
        _med_land_share = float(
            (merged.loc[_valid, "market_value_land"] /
             merged.loc[_valid, "market_value_total"]).median()
        )
        merged.loc[_abated, "market_value_land"] = (
            merged.loc[_abated, "market_value_total"] * _med_land_share
        )
        merged.loc[_abated, "market_value_bldg"] = (
            merged.loc[_abated, "market_value_total"] * (1 - _med_land_share)
        )
        print(
            f"  Abated parcels imputed: {_abated.sum():,}  median land share: {_med_land_share:.3f}")

    # taxable_total = on-roll tax base (taxable_land + taxable_building, after exemptions)
    merged["taxable_total"] = merged["taxable_land"] + merged["taxable_building"]

    merged["land_ratio"] = np.where(
        merged["market_value_total"] > 0,
        merged["market_value_land"] / merged["market_value_total"],
        np.nan,
    )

    merged["is_residential"] = merged["category_code"].isin([1, 2])
    merged["is_vacant"] = merged["category_code"].isin([6, 12, 13])
    merged["is_commercial"] = merged["category_code"].isin(
        [4, 5, 7, 9, 10, 11, 14, 15])
    merged["is_exempt"] = (
        (merged["taxable_land"] == 0) &
        (merged["taxable_building"] == 0) &
        (merged["exempt_land"].fillna(0) == 0) &
        (merged["exempt_building"].fillna(0) == 0)
    )

    all_parcels = merged.copy()
    residential = merged[
        merged["is_residential"] & (merged["market_value_total"] > 0)
    ].copy()
    training = residential[
        residential["has_sale"] &
        pd.to_numeric(residential.get("total_livable_area"), errors="coerce").notna() &
        (pd.to_numeric(residential.get("total_livable_area"), errors="coerce") > 0)
    ].copy()

    print(f"\nPhiladelphia datasets built:")
    print(f"  all_parcels:  {len(all_parcels):,}")
    print(f"  residential:  {len(residential):,}")
    print(f"  training_set: {len(training):,}")
    print(
        f"  Census match: {merged['median_household_income'].notna().mean()*100:.1f}%")
    return all_parcels, residential, training
