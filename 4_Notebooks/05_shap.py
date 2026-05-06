"""
Computes and visualizes SHAP values for the final best models:
    - Woningen:    LightGBM  (no zeros, with structural, 64 features)
    - Commercieel: XGBoost   (with zeros, with structural, 64 features)

Outputs (per property type):
    1. Bar plot        — mean |SHAP| per feature (top 20)
    2. Beeswarm plot   — feature importance with direction and distribution
    3. Dependence plot — top 3 most important features
    4. CSV             — full SHAP values for all test observations
"""

from __future__ import annotations

import logging
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Auto-install

def _ensure_installed(package: str, import_name: str) -> None:
    try:
        __import__(import_name)
    except ImportError:
        logging.info("Installing %s ...", package)
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

_ensure_installed("xgboost",  "xgboost")
_ensure_installed("lightgbm", "lightgbm")
_ensure_installed("shap",     "shap")

import shap
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

#Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# shap already installed above

#Paths

PROCESSED       = Path("../3_Data_processed")
PLOTS           = Path("../4_Plots/shap")
PREDICTORS_FILE = PROCESSED / "predictors_list.txt"

#Constants

TARGET = "vacancy_rate_pct"

STRUCTURAL_PREDICTOR_NAMES = [
    "structural_vacancy_count",
    "structural_vacancy_pct",
]

TRAIN_YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
TEST_YEARS  = [2022, 2023, 2024]

# Final model configuration — one per property type
# Based on grid search tuning results (65 features, with lag):
#   Woningen:    Random Forest, no zeros, with structural → R²=0.526
#   Commercieel: XGBoost, with zeros, with structural → R²=0.481
FINAL_MODELS = {
    "Woningen": {
        "model_name": "RandomForest",
        "zeros":      False,   # no zeros
        "structural": True,
        "build": lambda: RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            max_features=0.7,
            min_samples_leaf=1,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1,
        ),
    },
    "commercieel": {
        "model_name": "XGBoost",
        "zeros":      True,    # with zeros
        "structural": True,
        "build": lambda: XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.6,
            min_child_weight=5,
            reg_alpha=0.5,
            reg_lambda=5.0,
            random_state=42,
            verbosity=0,
        ),
    },
}

# Top N features to show in plots
TOP_N = 20

# Clean display names for features (Dutch → readable English)
FEATURE_DISPLAY_NAMES = {
    "structural_vacancy_count":    "Structural vacancy count",
    "structural_vacancy_pct":      "Structural vacancy %",
    "total_population":            "Total population",
    "share_working_age":           "Share working age (20–65)",
    "share_owner_occupied":        "Share owner-occupied",
    "share_social_rental":         "Share social rental",
    "share_private_rental":        "Share private rental",
    "avg_property_value":          "Avg property value (WOZ)",
    "population_growth_per1000":   "Population growth / 1000",
    "grey_pressure_pct":           "Grey pressure ratio",
    "ses_score":                   "SES score",
    "urbanisation_level_enc":      "Urbanisation level",
    "unemployment":                "Unemployment",
    "net_migration":               "Net migration",
    "total_businesses":            "Total businesses",
    "population_density":          "Population density",
    "dist_train_station":          "Distance to train station",
    "avg_household_size":          "Avg household size",
    "corop_unemployment":          "COROP unemployment",
    "corop_population_density":    "COROP population density",
    "corop_net_migration":         "COROP net migration",
    "corop_total_businesses":      "COROP total businesses",
    "corop_dist_train_station":    "COROP dist. train station",
    "corop_avg_household_size":    "COROP avg household size",
    "corop_total_jobs":            "COROP total jobs",
}

# Colors
TEAL   = "#0D9488"
NAVY   = "#1B2A4A"
ORANGE = "#D97706"
RED    = "#C0392B"
GREEN  = "#1A7F5A"

#Helpers

def savefig(name: str) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    path = PLOTS / name
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    log.info("Saved: %s", path)


def load_predictors(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Load predictors from file, split into base and structural."""
    if PREDICTORS_FILE.exists():
        all_predictors = PREDICTORS_FILE.read_text().strip().splitlines()
        log.info("Loaded %d predictors from file.", len(all_predictors))
    else:
        log.warning("predictors_list.txt not found — using hardcoded fallback.")
        all_predictors = [
            "total_population", "share_working_age", "share_owner_occupied",
            "share_social_rental", "share_private_rental", "avg_property_value",
            "population_growth_per1000", "grey_pressure_pct", "ses_score",
            "structural_vacancy_count", "structural_vacancy_pct",
        ]

    available  = [p for p in all_predictors if p in df.columns]
    structural = [p for p in available if p in STRUCTURAL_PREDICTOR_NAMES]
    base       = [p for p in available if p not in STRUCTURAL_PREDICTOR_NAMES]
    return base, structural


def get_display_name(col: str) -> str:
    # COROP dummies: corop_CR01 → COROP: Oost-Groningen (just show as COROP region)
    if col.startswith("corop_CR") or col.startswith("corop_cr"):
        return f"COROP region ({col.split('_')[-1]})"
    return FEATURE_DISPLAY_NAMES.get(col, col.replace("_", " ").title())

#Data preparation

def prepare_data(
    df: pd.DataFrame,
    func: str,
    zeros: bool,
    base_predictors: list[str],
    structural_predictors: list[str],
    with_structural: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Prepare train/test split for a given property type and scenario."""
    subset = df[df["property_type"] == func].copy()
    subset = subset.dropna(subset=[TARGET])

    if not zeros:
        before = len(subset)
        subset = subset[subset[TARGET] > 0].copy()
        log.info("  Dropped %d zero-vacancy rows.", before - len(subset))

    predictors = base_predictors.copy()
    if with_structural:
        predictors = predictors + structural_predictors

    subset = subset.dropna(subset=predictors)

    train = subset[subset["year"].isin(TRAIN_YEARS)].copy()
    test  = subset[subset["year"].isin(TEST_YEARS)].copy()

    log.info(
        "  %s: train=%d rows, test=%d rows, predictors=%d",
        func, len(train), len(test), len(predictors),
    )
    return train, test, predictors

#SHAP computation

def compute_shap(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    model_name: str,
):
    log.info("  Computing SHAP values ...")
    explainer = shap.TreeExplainer(model)

    if model_name == "RandomForest":
        # Exact SHAP for RF is O(T * 2^D) per sample — extremely slow.
        # approximate=True uses the tree-path dependent approximation (~100x faster).
        shap_raw = explainer.shap_values(X_test, check_additivity=False, approximate=True)
        shap_values = shap.Explanation(
            values=shap_raw,
            base_values=np.full(len(X_test), explainer.expected_value),
            data=X_test.values,
            feature_names=list(X_test.columns),
        )
    else:
        shap_values = explainer(X_test, check_additivity=False)

    log.info("  SHAP shape=%s", shap_values.values.shape)
    return shap_values

#Plots

def plot_bar(shap_values, func: str, model_name: str) -> None:
    """Mean absolute SHAP bar chart — top N features."""
    vals         = shap_values.values
    feature_names = [get_display_name(c) for c in shap_values.feature_names]

    mean_abs = np.abs(vals).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:TOP_N]

    top_vals   = mean_abs[top_idx]
    top_names  = [feature_names[i] for i in top_idx]

    color = TEAL if func == "Woningen" else ORANGE

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.suptitle(
        f"Feature Importance — {func} ({model_name})\nMean |SHAP value|",
        fontsize=13, fontweight="bold",
    )
    bars = ax.barh(range(TOP_N), top_vals[::-1], color=color, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(TOP_N))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP value| (impact on vacancy rate prediction, pp)", fontsize=10)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.invert_yaxis()
    plt.tight_layout()
    savefig(f"shap_bar_{func.lower()}.png")


def plot_beeswarm(shap_values, func: str, model_name: str) -> None:
    """Beeswarm plot — feature importance with direction."""
    # Get top N feature indices by mean absolute SHAP
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:TOP_N]

    # Slice to top features — works with new shap Explanation API
    shap_top = shap_values[:, top_idx]

    # Rename features for display
    shap_top.feature_names = [
        get_display_name(shap_values.feature_names[i]) for i in top_idx
    ]

    plt.figure(figsize=(10, 8))
    plt.suptitle(
        f"SHAP Beeswarm — {func} ({model_name})\nFeature impact on predictions",
        fontsize=13, fontweight="bold",
    )
    shap.plots.beeswarm(shap_top, max_display=TOP_N, show=False, plot_size=None)
    plt.tight_layout()
    savefig(f"shap_beeswarm_{func.lower()}.png")


def plot_dependence(
    shap_values,
    X_test: pd.DataFrame,
    func: str,
    model_name: str,
    n_features: int = 3,
) -> None:
    """Dependence plots for the top N most important features."""
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:n_features]

    feature_names = shap_values.feature_names
    color = TEAL if func == "Woningen" else ORANGE

    fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 5))
    fig.suptitle(
        f"SHAP Dependence — Top {n_features} Features — {func} ({model_name})",
        fontsize=13, fontweight="bold",
    )

    for ax, idx in zip(axes, top_idx):
        feat_name    = feature_names[idx]
        display_name = get_display_name(feat_name)
        feat_vals    = X_test[feat_name].values
        shap_vals    = shap_values.values[:, idx]

        ax.scatter(feat_vals, shap_vals, alpha=0.3, s=8, color=color)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel(display_name, fontsize=10)
        ax.set_ylabel("SHAP value", fontsize=10)
        ax.set_title(f"{display_name}", fontsize=10, fontweight="bold")

    plt.tight_layout()
    savefig(f"shap_dependence_{func.lower()}.png")


def save_shap_csv(
    shap_values,
    X_test: pd.DataFrame,
    func: str,
) -> None:
    """Save SHAP values to CSV for further analysis."""
    display_names = [get_display_name(c) for c in shap_values.feature_names]
    df_shap = pd.DataFrame(
        shap_values.values,
        columns=display_names,
        index=X_test.index,
    )
    df_shap["expected_value"] = shap_values.base_values

    out = PROCESSED / f"shap_values_{func.lower()}.csv"
    df_shap.to_csv(out)
    log.info("SHAP values saved → %s", out)

#Main

def main() -> None:
    import matplotlib as mpl
    mpl.rcParams["figure.dpi"] = 130
    plt.rcParams["font.size"]  = 10

    # Load data
    path = PROCESSED / "model_ready.csv"
    df   = pd.read_csv(path)
    log.info("Loaded: %s  shape=%s", path, df.shape)

    base_predictors, structural_predictors = load_predictors(df)

    # Load Stage 1 predictions if available (for Woningen two-stage model)
    stage1_file  = PROCESSED / "stage1_predictions.csv"
    stage1_preds = pd.read_csv(stage1_file) if stage1_file.exists() else None
    if stage1_preds is not None:
        log.info("Loaded Stage 1 predictions: %d rows", len(stage1_preds))
    else:
        log.warning("stage1_predictions.csv not found — using all non-zero test rows for Woningen")

    for func, cfg in FINAL_MODELS.items():
        log.info("=" * 60)
        log.info("SHAP analysis: %s — %s", func, cfg["model_name"])
        log.info("=" * 60)

        # 1. Prepare data
        train, test, predictors = prepare_data(
            df, func,
            zeros=cfg["zeros"],
            base_predictors=base_predictors,
            structural_predictors=structural_predictors,
            with_structural=cfg["structural"],
        )

        X_train = train[predictors]
        y_train = train[TARGET]
        X_test  = test[predictors]
        y_test  = test[TARGET]

        # Apply Stage 1 filter for Woningen — filter by original DataFrame index
        if func == "Woningen" and stage1_preds is not None:
            positive_idx = set(
                stage1_preds[stage1_preds["stage1_label"] == 1]["df_index"].values
            )
            before = len(test)
            test_filtered = test[test.index.isin(positive_idx)]
            log.info("  Stage 1 filter: %d → %d test rows (%.1f%% passed)",
                     before, len(test_filtered),
                     100 * len(test_filtered) / before)
            X_test = test_filtered[predictors]
            y_test = test_filtered[TARGET]

        # 2. Train final model
        log.info("  Training %s ...", cfg["model_name"])
        model = cfg["build"]()
        model.fit(X_train, y_train)

        # Verify holdout performance
        from sklearn.metrics import r2_score as r2
        y_pred = model.predict(X_test)
        r2_val = r2(y_test, y_pred)
        log.info("  Holdout R² = %.4f", r2_val)

        # 3. Compute SHAP values — sample for speed (rankings stable at 500 rows)
        SHAP_SAMPLE = 200
        X_test_shap = X_test.sample(n=min(SHAP_SAMPLE, len(X_test)), random_state=42)
        log.info("  SHAP sample: %d / %d test rows", len(X_test_shap), len(X_test))
        shap_values = compute_shap(model, X_train, X_test_shap, cfg["model_name"])

        # 4. Plots
        log.info("  Generating plots ...")
        plot_bar(shap_values, func, cfg["model_name"])
        plot_beeswarm(shap_values, func, cfg["model_name"])
        plot_dependence(shap_values, X_test_shap, func, cfg["model_name"], n_features=3)

        # 5. Save SHAP CSV
        save_shap_csv(shap_values, X_test_shap, func)

        log.info("  Done: %s", func)

    log.info("All SHAP outputs saved to %s", PLOTS)


if __name__ == "__main__":
    main()