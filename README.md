# Assessment Inequity and Land Value Taxation

**A Multi-City Analysis of Property Tax Disparities in Cook County, New York City, and Philadelphia**

Krishna Singh · Isabelle Byrne  
MATH 7339 — Machine Learning & Statistical Learning Theory II, Northeastern University  
Professor He Wang\
Sponsor: James Frederiksen, Duke University

---

## Overview

Property taxes are the primary revenue source for U.S. local governments, but assessment practices produce systematic disparities along racial and income lines. This project asks whether a revenue-neutral shift from taxing land + improvements to taxing land only (a land-value tax, or LVT) would reduce those disparities.

We build a five-task ML pipeline across three cities and find that LVT benefits 61.3% of Cook County residential parcels — but majority-Black neighborhoods are the only racial group where fewer than half benefit. We trace this to high land-to-improvement ratios inherited from redlining and show that a $10,000 homestead exemption lifts all racial groups above 80% benefit while staying revenue-neutral.

## Key Findings

| Finding | Detail |
|---------|--------|
| Frederiksen hypothesis | **Supported.** 61.3% of residential parcels pay less under LVT |
| Assessment gap | Black tracts assessed at 0.630 vs White at 0.816 (18.6pp gap) |
| Racial mediation | Gap operates through spatial features (p=0.382 on pct_black with controls) |
| LVT racial disparity | Black 46.6% benefit vs White 64.2% — only group below 50% |
| Speculator impact | 0% of speculative parcels benefit (+$2,404 median increase) |
| GMM clustering | At k=2, recovers racial segregation from property data alone |
| Homestead exemption | $10K exemption: Black benefit jumps from 39.3% → 98.2% |
| Classification (w/o land ratio) | NN AUC=0.959, SVM=0.851, LogReg=0.857 |

## Pipeline

```
Task 1: Property Valuation ─── LightGBM R²=0.817 (Cook), 0.572 (NYC), 0.735 (Philly)
    │
Task 2: Assessment Bias ────── OLS on tract-level demographics, IAAO ratio statistics
    │
Task 3: LVT Simulation ────── Revenue-neutral, actual 2023 Clerk tax rates (4,464 codes)
    │
Task 4: Neighborhood Clustering ── PCA + GMM, BIC selection, k=2 and k=8
    │
Task 5: Classification ────── LogReg / SVM / Neural Network (with and without land ratio)
    │
Extensions: Homestead exemption, phased transition, owner identity analysis
```

## Data

All data are publicly available. No API keys required for Cook County or Philadelphia.

| Dataset | Source | Records |
|---------|--------|---------|
| Cook County assessments | [CCAO Socrata](https://datacatalog.cookcountyil.gov) | 1,864,161 |
| Cook County sales | CCAO Socrata | 287,490 |
| Cook County property chars | CCAO Socrata | 1,101,227 |
| Cook County spatial features | CCAO Socrata | 1,864,035 |
| Cook County tax rates | [Cook County Clerk](https://www.cookcountyclerk.com) | 4,464 codes |
| Cook County owner addresses | CCAO Socrata (3723-97qp) | 50,671,402 |
| NYC PLUTO + rolling sales | [NYC Open Data](https://data.cityofnewyork.us) | 705,173 |
| Philadelphia OPA | [OpenDataPhilly](https://www.opendataphilly.org) | 504,034 |
| Census ACS (5-year) | Census Bureau API | 1,332 tracts (Cook) |

## Repository Structure

```
├── src/
│   ├── config.py              # City-specific configs, feature lists, file paths
│   ├── data_utils.py          # Data loading, demographic grouping, tax rate merging
│   ├── model_utils.py         # AVM training, classifier training, prediction
│   ├── tax_utils.py           # LVT simulation, tract aggregation, Gini computation
│   └── viz_utils.py           # All plotting functions
│
├── notebooks/
│   ├── cook_county.ipynb      # Full 5-task pipeline for Cook County
│   ├── nyc.ipynb              # Full 5-task pipeline for NYC
│   └── philly.ipynb           # Full 5-task pipeline for Philadelphia
│
├── extensions/                # Additional Cook County analyses (Krishna)
│   ├── README.md
│   ├── 09_lightgbm_shap_upgrade.ipynb    # SHAP + reduced-feature classification
│   ├── 10_k2_transition_homestead.ipynb   # k=2 clustering, phased transition, exemption
│   └── 13_owner_identity_analysis.ipynb   # Mailing address analysis (50M records)
│
├── data/
│   ├── raw/                   # Downloaded parquets (not committed, see setup)
│   └── processed/             # Pipeline outputs (not committed)
│
├── figures/                   # All generated plots
├── report/
│   ├── main.tex               # IEEE-format paper
│   └── presentation.pptx      # Slide deck
└── README.md
```

## Setup

```bash
git clone https://github.com/isabellebyrne/property-tax-equity.git
cd property-tax-equity
pip install -r requirements.txt
```

### Data Download

Cook County data is downloaded via Socrata API (no key needed). The data acquisition cells in each notebook handle this automatically. Raw data is saved to `data/raw/` as parquet files and processed data to `data/processed/`.

Tax rates require a manual download of the Cook County Clerk's "2023 Tax Code Agency Rate" Excel file from [cookcountyclerk.com](https://www.cookcountyclerk.com). Place it in `data/raw/`.

### Running

Each city notebook runs independently. Open in Google Colab (recommended) or locally:

```bash
jupyter notebook notebooks/cook_county.ipynb
```

The `src/` package is imported at the top of each notebook. All paths are configured in `src/config.py`.

## Methods

**Valuation.** LightGBM gradient boosting on log-transformed sale prices. Cook County uses 28 features (16 building + 2 categorical + 10 spatial). NYC and Philadelphia use city-specific feature sets. 80/20 train-test split with StandardScaler and OneHotEncoder.

**Bias detection.** Tract-level OLS regression of median assessment ratio on demographics and spatial controls. IAAO ratio study statistics (COD, PRD, PRB) computed on training-set sales.

**LVT simulation.** Per-parcel revenue-neutral tax shift from total assessed value to land assessed value. Cook County uses actual composite rates from 4,464 tax codes (range 1.7%–32.1%, median 7.52%) with state equalizer 3.0163.

**Clustering.** PCA (5 components, ≥90% variance) followed by GMM with BIC selection on 9 tract-level features. Evaluated at k=2 through k=8.

**Classification.** Logistic regression, linear SVM, and neural network (128-64, ReLU, early stopping). Land value features excluded to create a genuine prediction task.

## Extensions (Krishna)

These notebooks in `extensions/` extend the base pipeline with Cook County deep-dives.

**SHAP + reduced-feature classification.** Feature importance analysis reveals foreclosure density, transit access, and Lake Michigan distance as top spatial predictors. Classification without land ratio shows NN AUC=0.959, demonstrating genuine nonlinearity in the relationship between observable property characteristics and LVT outcomes.

**k=2 clustering.** GMM at k=2 recovers a partition highly correlated with racial segregation (940 White/Hispanic tracts vs 330 Black tracts) without receiving any racial data as input. At k=8 (BIC-optimal), Black neighborhoods split further into middle-class tracts that benefit from LVT (76%) and deeply segregated low-income tracts that are hurt (11–31%).

**Homestead exemption.** A $10,000 exemption on land assessed value lifts majority-Black LVT benefit from 39.3% to 98.2% while remaining revenue-neutral. Phased implementation alone does not resolve the transition burden — majority-Black neighborhoods never cross 50% benefit at any phase.

**Owner identity.** Analysis of 50.7M mailing address records shows 75.2% county-wide absentee ownership. Among LVT losers in Black neighborhoods, 68.2% are absentee-owned, but the gap with winners is only 5.7pp. The transition burden falls on a mix of small landlords, estates, and owner-occupants — not concentrated among institutional speculators. This reinforces the necessity of the homestead exemption.

## Results at a Glance

### AVM Performance (LightGBM, best model)
| City | R² | MdAPE |
|------|-----|-------|
| Cook County | 0.817 | 15.2% |
| New York City | 0.572 | 15.6% |
| Philadelphia | 0.735 | 20.4% |

### LVT Simulation
| City | % Benefit | Black | White | Multiplier |
|------|-----------|-------|-------|------------|
| Cook County | 61.3% | 46.6% | 64.2% | 4.40× |
| New York City | 30.3% | 18.8% | 35.2% | 5.67× |
| Philadelphia | 58.2% | 52.3% | 64.2% | 3.55× |

### Classification (land value features excluded)
| City | LogReg AUC | SVM AUC | NN AUC |
|------|-----------|---------|--------|
| Cook County | 0.857 | 0.851 | 0.959 |
| New York City | 0.765 | 0.750 | 0.875 |
| Philadelphia | 0.816 | 0.813 | 0.870 |

## References

- Avenancio-León, C. and Howard, T. (2022). "The Assessment Gap: Racial Inequalities in Property Taxation." *QJE*, 137(3), 1383–1434.
- Ganong, P. and Shoag, D. (2017). "Why Has Regional Income Convergence in the U.S. Declined?" *J. Urban Economics*, 102, 76–90.
- Oates, W. and Schwab, R. (1997). "The Impact of Urban Land Taxation: The Pittsburgh Experience." *National Tax Journal*, 50(1), 1–21.
- Pappas, M. (2022). "Maps of Inequality." Cook County Treasurer's Office.
- Perry, A. et al. (2018). "The Devaluation of Assets in Black Neighborhoods." Brookings Institution.
- Berry, C. (2021). "Reassessing the Property Tax." University of Chicago Harris School.
- Cook County Assessor's Office. [ccao-data](https://github.com/ccao-data) (model-res-avm, ptaxsim).
- Cook County Clerk's Office. "2023 Cook County Tax Rates Released." July 2024.

## License

This project is for academic and research purposes. All data sources are publicly available. The CCAO's open-source tools are licensed under AGPL-3.
