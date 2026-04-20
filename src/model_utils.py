import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
import xgboost as xgb
try:
    import lightgbm as lgb
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

from .config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, RANDOM_STATE


def prep_data(
    df: pd.DataFrame,
    numeric_features=None,
    categorical_features=None,
    target: str = "sale_price",
):
    # target is log-transformed and clipped at 1st/99th percentile
    num = [c for c in (numeric_features or NUMERIC_FEATURES) if c in df.columns]
    cat = [c for c in (categorical_features or CATEGORICAL_FEATURES) if c in df.columns]
    missing = set(numeric_features or NUMERIC_FEATURES) - set(num)
    if missing:
        print(f"  [prep_data] dropping {len(missing)} numeric features not in data: {sorted(missing)}")
    all_feats = num + cat

    model_df = df[all_feats + [target]].copy()
    for col in num:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
        model_df[col] = model_df[col].fillna(model_df[col].median())
    for col in cat:
        model_df[col] = model_df[col].astype(str).fillna("Unknown")

    model_df["log_price"] = np.log1p(model_df[target])
    p01, p99 = model_df["log_price"].quantile([0.01, 0.99])
    model_df = model_df[(model_df["log_price"] >= p01) &
                        (model_df["log_price"] <= p99)]

    X = model_df[all_feats]
    y = model_df["log_price"]
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


def fit_regressors(
    X_train, X_test, y_train_log, y_test_log,
    numeric_features=None,
    categorical_features=None,
) -> tuple:
    # returns (trained_pipelines, results_df)
    num = [c for c in (numeric_features or NUMERIC_FEATURES) if c in X_train.columns]
    cat = [c for c in (categorical_features or CATEGORICAL_FEATURES) if c in X_train.columns]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(
                handle_unknown="ignore", max_categories=50, sparse_output=False,
            ), cat),
        ],
        remainder="drop",
    )
    models = {
        "Linear Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("regressor",    LinearRegression()),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("regressor",    RandomForestRegressor(
                n_estimators=200, max_depth=20, min_samples_leaf=10,
                n_jobs=-1, random_state=RANDOM_STATE,
            )),
        ]),
        "XGBoost": Pipeline([
            ("preprocessor", preprocessor),
            ("regressor",    xgb.XGBRegressor(
                n_estimators=500, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                reg_alpha=0.1, reg_lambda=1.0, n_jobs=-1,
                random_state=RANDOM_STATE, verbosity=0,
            )),
        ]),
    }
    if _HAS_LGBM:
        models["LightGBM"] = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor",    lgb.LGBMRegressor(
                n_estimators=500, max_depth=-1, learning_rate=0.05,
                num_leaves=255, subsample=0.8, colsample_bytree=0.8,
                min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0,
                n_jobs=-1, random_state=RANDOM_STATE, verbose=-1,
            )),
        ])

    y_test_dollars = np.expm1(y_test_log)
    trained, rows = {}, []
    for name, pipeline in models.items():
        print(f"  Training {name}...")
        pipeline.fit(X_train, y_train_log)
        y_pred = np.expm1(pipeline.predict(X_test))

        # remove NaN predictions before scoring
        valid_idx = np.isfinite(y_pred) & np.isfinite(y_test_dollars)
        y_pred_valid = y_pred[valid_idx]
        y_test_valid = y_test_dollars[valid_idx]

        if len(y_pred_valid) == 0:
            print(f"    WARNING: All predictions are invalid (NaN/Inf) for {name}")
            continue

        rows.append({
            "Model": name,
            "R²": r2_score(y_test_valid, y_pred_valid),
            "RMSE ($)": np.sqrt(mean_squared_error(y_test_valid, y_pred_valid)),
            "MAE ($)": mean_absolute_error(y_test_valid, y_pred_valid),
            "MdAPE (%)": np.median(np.abs(y_test_valid - y_pred_valid) / y_test_valid) * 100,
        })
        trained[name] = pipeline

    return trained, pd.DataFrame(rows)


def get_xgb_feature_names(pipeline) -> list:
    preprocessor = pipeline.named_steps["preprocessor"]
    num_names = list(preprocessor.named_transformers_["num"].feature_names_in_)
    cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out().tolist()
    return num_names + cat_names


def predict_all(pipeline, df: pd.DataFrame,
                numeric_features=None, categorical_features=None) -> np.ndarray:
    # returns predictions in dollar space, not log space
    num = [c for c in (numeric_features or NUMERIC_FEATURES) if c in df.columns]
    cat = [c for c in (categorical_features or CATEGORICAL_FEATURES) if c in df.columns]
    X = df[num + cat].copy()
    for col in num:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(X[col].median())
    for col in cat:
        X[col] = X[col].astype(str).fillna("Unknown")
    return np.expm1(pipeline.predict(X))


def fit_classifiers(
    X_train_scaled, X_test_scaled, y_train, y_test,
) -> tuple:
    # returns (trained_models, results_df)
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, C=1.0,
        ),
        # LinearSVC scales to 100k+ rows; wrapped for predict_proba support
        "SVM (Linear)": CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_STATE),
        ),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
            max_iter=300, early_stopping=True, validation_fraction=0.1,
            random_state=RANDOM_STATE, batch_size=256, learning_rate_init=0.001,
        ),
    }
    trained, rows = {}, []
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_prob),
        })
        trained[name] = model
    return trained, pd.DataFrame(rows)


def compute_iaao_metrics(
    sale_prices: np.ndarray,
    assessed_values: np.ndarray,
    assessment_level: float = 1.0,
) -> dict:
    """
    Compute IAAO ratio study statistics: COD, PRD, PRB.

    Parameters
    ----------
    sale_prices      : array of arm's-length sale prices (dollars)
    assessed_values  : array of official assessed values (dollars, same units as sale_prices
                       AFTER applying assessment_level — i.e. pass implied market value,
                       not the raw AV)
    assessment_level : scalar used to convert raw AV to implied market value.
                       Pass 1.0 if assessed_values are already implied market values.

    Returns
    -------
    dict with keys: median_ratio, cod, prd, prb, prb_pvalue, n
    """
    from scipy import stats as scipy_stats

    implied_mv = np.asarray(assessed_values, dtype=float) / assessment_level
    sp = np.asarray(sale_prices, dtype=float)

    mask = np.isfinite(implied_mv) & np.isfinite(sp) & (implied_mv > 0) & (sp > 0)
    implied_mv = implied_mv[mask]
    sp = sp[mask]

    ratios = implied_mv / sp
    median_r = float(np.median(ratios))

    cod = float(np.mean(np.abs(ratios - median_r)) / median_r * 100)

    mean_r = float(np.mean(ratios))
    weighted_mean_r = float(np.sum(ratios * sp) / np.sum(sp))
    prd = float(mean_r / weighted_mean_r)

    # PRB per IAAO 2013 Standard on Ratio Studies, Appendix B
    dep = (ratios - median_r) / median_r
    indep = 0.5 * np.log(sp) + 0.5 * np.log(implied_mv)
    indep_c = indep - indep.mean()
    result = scipy_stats.linregress(indep_c, dep)
    prb = float(result.slope)
    prb_pvalue = float(result.pvalue)

    return {
        "n": int(mask.sum()),
        "median_ratio": round(median_r, 4),
        "cod": round(cod, 2),
        "prd": round(prd, 4),
        "prb": round(prb, 4),
        "prb_pvalue": round(prb_pvalue, 4),
        "cod_pass": 5.0 <= cod <= 15.0,
        "prd_pass": 0.98 <= prd <= 1.03,
        "prb_pass": -0.05 <= prb <= 0.05,
    }


def iaao_to_dataframe(results_by_city: dict) -> pd.DataFrame:
    """
    results_by_city: {"Cook County": iaao_dict, "NYC": iaao_dict, ...}
    Returns a formatted summary DataFrame.
    """
    rows = []
    for city, r in results_by_city.items():
        rows.append({
            "City": city,
            "N Sales": r["n"],
            "Median Ratio": r["median_ratio"],
            "COD": r["cod"],
            "COD Pass": "✓" if r["cod_pass"] else "✗",
            "PRD": r["prd"],
            "PRD Pass": "✓" if r["prd_pass"] else "✗",
            "PRB": r["prb"],
            "PRB p-value": r["prb_pvalue"],
            "PRB Pass": "✓" if r["prb_pass"] else "✗",
        })
    return pd.DataFrame(rows)
