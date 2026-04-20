import numpy as np
import pandas as pd
from .config import STATE_EQUALIZER, COMPOSITE_RATE


def compute_lvt(
    all_parcels: pd.DataFrame,
    state_equalizer: float = STATE_EQUALIZER,
    composite_rate=COMPOSITE_RATE,
    tax_base_total_col: str = "final_tot",
    tax_base_land_col: str = "final_land",
) -> tuple:
    # composite_rate may be a float (flat) or a column name (per-parcel rates).
    # Revenue-neutral multiplier: lvt_mult = Σ(AV_i × rate_i) / Σ(land_AV_i × rate_i)
    taxable = all_parcels[
        ~all_parcels["is_exempt"] & (all_parcels[tax_base_total_col] > 0)
    ].copy()

    if isinstance(composite_rate, str):
        rates = taxable[composite_rate]
    else:
        rates = composite_rate

    taxable["current_tax_est"] = taxable[tax_base_total_col] * state_equalizer * rates
    total_current_rev = taxable["current_tax_est"].sum()
    total_land_base = (taxable[tax_base_land_col] * state_equalizer * rates).sum()
    lvt_mult = total_current_rev / total_land_base

    taxable["lvt_tax_est"] = taxable[tax_base_land_col] * state_equalizer * rates * lvt_mult
    taxable["tax_change"] = taxable["lvt_tax_est"] - taxable["current_tax_est"]
    taxable["tax_change_pct"] = np.where(
        taxable["current_tax_est"] > 0,
        taxable["tax_change"] / taxable["current_tax_est"] * 100,
        0.0,
    )
    taxable["lvt_benefits"] = taxable["tax_change"] < 0

    revenue_diff = taxable["lvt_tax_est"].sum() - taxable["current_tax_est"].sum()
    print(f"LVT multiplier: {lvt_mult:.4f}x  (revenue check: ${revenue_diff:+,.0f})")
    return taxable, lvt_mult


def aggregate_lvt_to_tracts(res_lvt: pd.DataFrame) -> pd.DataFrame:
    return (
        res_lvt.groupby("census_tract_geoid")
        .agg(
            median_tax_change=("tax_change", "median"),
            median_tax_change_pct=("tax_change_pct", "median"),
            pct_lvt_benefit=("lvt_benefits", "mean"),
            median_current_tax=("current_tax_est", "median"),
            median_lvt_tax=("lvt_tax_est", "median"),
        )
        .assign(pct_lvt_benefit=lambda d: d["pct_lvt_benefit"] * 100)
        .reset_index()
    )


def gini_coefficient(values: np.ndarray) -> float:
    v = np.sort(values[values > 0])
    n = len(v)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * v) - (n + 1) * np.sum(v)) / (n * np.sum(v)))
