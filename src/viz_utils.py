import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay
from sklearn.inspection import permutation_importance


def plot_feature_importance(pipeline, title="", n=20, save_path=None):
    from .model_utils import get_xgb_feature_names
    names = get_xgb_feature_names(pipeline)
    importances = pipeline.named_steps["regressor"].feature_importances_

    imp_df = (
        pd.DataFrame(
            {"feature": names[:len(importances)], "importance": importances})
        .sort_values("importance", ascending=False)
        .head(n)
        .sort_values("importance")
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(imp_df["feature"], imp_df["importance"],
            color="steelblue", alpha=0.85)
    ax.set_xlabel("Feature importance (gain)")
    ax.set_title(title or f"XGBoost valuation model — top {n} features")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_actual_vs_predicted(y_true, y_pred, title="", sample_n=5000, save_path=None):
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_true), size=min(
        sample_n, len(y_true)), replace=False)
    y_t, y_p = np.asarray(y_true)[idx], np.asarray(y_pred)[idx]
    lim = max(y_t.max(), y_p.max()) * 1.05

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_t, y_p, alpha=0.15, s=5, color="steelblue")
    ax.plot([0, lim], [0, lim], "r--", linewidth=1, label="Perfect")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Actual sale price ($)")
    ax.set_ylabel("Predicted market value ($)")
    ax.set_title(title or "Actual vs Predicted")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_assessment_ratio_by_group(tract_stats, city_label="Cook County",
                                   fair_ratio=1.0, save_path=None):
    """fair_ratio controls both the dashed reference line and the shared y-axis ceiling."""
    race_order = ["Majority Black",
                  "Majority Hispanic", "Mixed", "Majority White"]
    race_colors = ["#EBE952", "#EF9F27", "#888780", "#50A0F0"]
    race_data = tract_stats.groupby("majority_group")[
        "median_ratio"].median().reindex(race_order)

    income_order = ["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"]
    inc_data = (
        tract_stats.groupby("income_quintile", observed=False)["median_ratio"]
        .median()
        .reindex([i for i in income_order if i in tract_stats["income_quintile"].cat.categories])
    )
    inc_colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(inc_data)))

    all_vals = np.concatenate(
        [race_data.dropna().values, inc_data.dropna().values, [fair_ratio]])
    y_max = float(np.nanmax(all_vals)) * 1.15
    text_offset = y_max * 0.02

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.bar(range(len(race_order)), race_data.values,
           color=race_colors, alpha=0.85)
    ax.axhline(fair_ratio, color="black", linestyle="--", linewidth=1,
               label=f"Fair ({fair_ratio:.3f})")
    for i, v in enumerate(race_data.values):
        if not np.isnan(v):
            ax.text(i, v + text_offset, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_xticks(range(len(race_order)))
    ax.set_xticklabels(race_order, rotation=15, ha="right")
    ax.set_ylabel("Median assessment ratio")
    ax.set_title("Assessment ratio by racial composition")
    ax.set_ylim(0, y_max)
    ax.legend()

    ax = axes[1]
    ax.bar(range(len(inc_data)), inc_data.values, color=inc_colors, alpha=0.85)
    ax.axhline(fair_ratio, color="black", linestyle="--", linewidth=1,
               label=f"Fair ({fair_ratio:.3f})")
    for i, v in enumerate(inc_data.values):
        if not np.isnan(v):
            ax.text(i, v + text_offset, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_xticks(range(len(inc_data)))
    ax.set_xticklabels(inc_data.index, rotation=15, ha="right")
    ax.set_ylabel("Median assessment ratio")
    ax.set_title("Assessment ratio by income quintile")
    ax.set_ylim(0, y_max)
    ax.legend()

    fig.suptitle(
        f"{city_label} — Assessment ratio by demographic group", fontsize=13)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_intersectional_heatmap(intersect_df, city_label="Cook County",
                                fair_ratio=1.0, save_path=None):
    """Color scale is centered on fair_ratio; cells below fair are green, above are red."""
    pivot = intersect_df.pivot(
        index="Income", columns="Race", values="Median ratio")
    pivot = pivot.reindex(["Low income", "Mid income", "High income"])

    data_vals = pivot.values.flatten()
    data_vals = data_vals[~np.isnan(data_vals)]
    if len(data_vals) > 0:
        vmin = min(float(data_vals.min()), fair_ratio) * 0.85
        vmax = max(float(data_vals.max()), fair_ratio) * 1.15
    else:
        vmin, vmax = fair_ratio * 0.5, fair_ratio * 1.5

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="RdYlGn",
        vmin=vmin, vmax=vmax, center=fair_ratio,
        ax=ax, linewidths=0.5,
    )
    ax.set_title(
        f"{city_label} — Assessment ratio: race \u00d7 income  (fair = {fair_ratio:.3f})")
    ax.set_xlabel("")
    ax.set_ylabel("Income level (within racial group)")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_lvt_benefit_by_group(res_lvt, city_label="Cook County", save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    race_order = ["Majority Black",
                  "Majority Hispanic", "Mixed", "Majority White"]
    race_colors = ["#E24B4A", "#EF9F27", "#888780", "#378ADD"]
    race_data = (
        res_lvt.groupby("majority_group")["lvt_benefits"].mean() * 100
    ).reindex(race_order)

    ax = axes[0]
    ax.bar(range(len(race_order)), race_data.values,
           color=race_colors, alpha=0.85)
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, label="50% line")
    ax.set_xticks(range(len(race_order)))
    ax.set_xticklabels(race_order, rotation=15, ha="right")
    ax.set_ylabel("% parcels with tax decrease")
    ax.set_title("Share benefiting from LVT by racial composition")
    for i, v in enumerate(race_data.values):
        if not np.isnan(v):
            ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)
    ax.legend()

    inc_data = res_lvt.groupby("income_quartile", observed=False)[
        "lvt_benefits"].mean() * 100
    inc_colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(inc_data)))

    ax = axes[1]
    ax.bar(range(len(inc_data)), inc_data.values, color=inc_colors, alpha=0.85)
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, label="50% line")
    ax.set_xticks(range(len(inc_data)))
    ax.set_xticklabels(inc_data.index, rotation=15, ha="right")
    ax.set_ylabel("% parcels with tax decrease")
    ax.set_title("Share benefiting from LVT by income quartile")
    for i, v in enumerate(inc_data.values):
        if not np.isnan(v):
            ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)
    ax.legend()

    fig.suptitle(
        f"{city_label} — Who benefits from a revenue-neutral land-value tax?", fontsize=13)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_effective_rate_comparison(res_lvt, city_label="Cook County", save_path=None):
    valid = res_lvt[
        (res_lvt["predicted_market_value"] > 0) &
        (res_lvt["current_tax_est"] > 0) &
        (res_lvt["lvt_tax_est"] > 0)
    ].copy()
    valid["eff_current"] = valid["current_tax_est"] / \
        valid["predicted_market_value"] * 100
    valid["eff_lvt"] = valid["lvt_tax_est"] / \
        valid["predicted_market_value"] * 100

    groups = ["Majority Black", "Majority Hispanic", "Mixed", "Majority White"]
    curr = [valid[valid["majority_group"] == g]["eff_current"].median()
            for g in groups]
    lvt = [valid[valid["majority_group"] == g]["eff_lvt"].median()
           for g in groups]

    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.2, curr, 0.35, label="Current system",
           color="#E24B4A", alpha=0.75)
    ax.bar(x + 0.2, lvt,  0.35, label="Land-value tax",
           color="#378ADD", alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=15, ha="right")
    ax.set_ylabel("Median effective tax rate (%)")
    ax.set_title(
        f"{city_label} — Effective tax rate: current system vs land-value tax")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_cluster_heatmap(profile_summary, features, n_clusters, city_label="Cook County", save_path=None):
    """Z-scored display with raw median values annotated in each cell."""
    profile_z = profile_summary[features].copy()
    for col in profile_z.columns:
        std = profile_z[col].std()
        if std > 0:
            profile_z[col] = (profile_z[col] - profile_z[col].mean()) / std

    labels_x = [f.replace("median_", "").replace("pct_", "% ").replace("_", " ")
                for f in features]
    labels_y = [f"Cluster {i}" for i in profile_z.index]
    annot = profile_summary[features].round(1).values

    fig, ax = plt.subplots(figsize=(14, max(4, n_clusters * 0.85)))
    sns.heatmap(
        profile_z, annot=annot, fmt="", cmap="RdBu_r", center=0,
        ax=ax, yticklabels=labels_y, xticklabels=labels_x, linewidths=0.3,
    )
    ax.set_title(
        f"{city_label} — Cluster profiles (z-scored; annotated with actual values)")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_cluster_map(map_df, cluster_col="cluster", city_label="Cook County", save_path=None):
    fig, ax = plt.subplots(figsize=(9, 11))
    sc = ax.scatter(
        map_df["lon"], map_df["lat"],
        c=map_df[cluster_col], cmap="tab10", alpha=0.65, s=20,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{city_label} — Neighborhood clusters (GMM)")
    ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, label="Cluster", shrink=0.6)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_roc_curves(trained_models, X_test_scaled, y_test, city_label="Cook County", save_path=None):
    palette = {
        "Logistic Regression": "#378ADD",
        "SVM (Linear)": "#E24B4A",
        "Neural Network":      "#5DCAA5",
    }
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, model in trained_models.items():
        RocCurveDisplay.from_estimator(
            model, X_test_scaled, y_test,
            ax=ax, name=name, color=palette.get(name), alpha=0.85,
        )
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5, label="Random")
    ax.set_title(f"{city_label} — ROC curves: LVT benefit classification")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_permutation_importance(trained_models, X_test_scaled, y_test,
                                feature_names, city_label="Cook County", save_path=None):
    """Averages permutation importance (AUC drop) across all classifiers; shows top 10."""
    avg = pd.DataFrame({"feature": feature_names})
    for name, model in trained_models.items():
        perm = permutation_importance(
            model, X_test_scaled, y_test,
            n_repeats=10, random_state=42, scoring="roc_auc",
        )
        avg[name] = perm.importances_mean

    avg["Average"] = avg[[n for n in trained_models]].mean(axis=1)
    avg = avg.sort_values("Average").tail(10)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(avg["feature"], avg["Average"], color="steelblue", alpha=0.85)
    ax.set_xlabel("Mean permutation importance (AUC drop)")
    ax.set_title(f"{city_label} — Feature importance: LVT benefit classifier")
    ax.axvline(0, color="gray", linewidth=0.5)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
