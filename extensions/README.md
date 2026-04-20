# Extensions to Base Pipeline

These notebooks extend the core five-task pipeline with additional
analyses for Cook County:

## LightGBM + SHAP + Reduced-Feature Classification
- Original test for lightgbm model with shap and feature reduction
- SHAP feature importance (foreclosure density, transit, Lake Michigan)
- Classification WITHOUT land ratio: NN AUC=0.93, SVM=0.88, LogReg=0.82
- Demonstrates nonlinearity in the prediction task

## k=2 Clustering, Transition Analysis, Homestead Exemption
- GMM at k=2 recovers racial segregation from property data alone
- Phased transition: Black neighborhoods never cross 50% at any phase
- Homestead exemption: $10K lifts Black benefit from 39.3% to 98.2%

## Owner Identity Analysis
- Downloads 50.7M mailing address records (CCAO dataset 3723-97qp)
- 75.2% of parcels are absentee-owned county-wide
- Among LVT losers in Black neighborhoods: 68.2% absentee,
  but only 5.7pp gap with winners which means burden is real, and not speculator-driven
- Better data on ownership of land is stuck behind a paywall, which would provide better insights. Initial tests showed that speculators are definitely among the people who buy land in the LVT loser affected areas