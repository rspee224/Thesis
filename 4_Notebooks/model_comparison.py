"""
Scenarios:
    1. No structural, no imputation  — only demographic/housing predictors
    2. No structural, with imputation — same + NaN target filled with 0
    3. With structural, no imputation — adds structural vacancy features
    4. With structural, with imputation — structural features + imputation

Models:
    - SVR (rbf, log target)
    - Random Forest
    - XGBoost
    - LightGBM
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
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore", category=UserWarning)


# Auto-install optional dependencies

def _ensure_installed(package: str, import_name: str) -> None:
    try:
        __import__(import_name)
    except ImportError:
        logging.info("Installing %s ...", package)
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


_ensure_installed("xgboost",  "xgboost")
_ensure_installed("lightgbm", "lightgbm")

from xgboost import XGBRegressor    # noqa: E402
from lightgbm import LGBMRegressor  # noqa: E402

# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Paths


PROCESSED = Path("../3_Data_processed")
PLOTS     = Path("../4_Plots/comparison")

# Constants


TARGET = "vacancy_rate_pct"

# Predictors are loaded dynamically from the file saved by pre_processing.py
# This ensures model comparison always uses the same features as preprocessing
PREDICTORS_FILE = PROCESSED / "predictors_list.txt"

# These are the structural vacancy feature names — used to split scenarios
STRUCTURAL_PREDICTOR_NAMES = [
    "structural_vacancy_count",
    "structural_vacancy_pct",
]

def load_predictors(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Load full predictor list from file saved by pre_processing.py.
    Returns (base_predictors, structural_predictors) where:
      - base_predictors = all predictors except structural vacancy features
      - structural_predictors = structural vacancy features only
    Falls back to hardcoded minimal list if file not found.
    Filters to columns that exist in the dataframe.
    """
    if PREDICTORS_FILE.exists():
        all_predictors = PREDICTORS_FILE.read_text().strip().splitlines()
        log.info("Loaded %d predictors from %s", len(all_predictors), PREDICTORS_FILE)
    else:
        log.warning(
            "predictors_list.txt not found — falling back to hardcoded list. "
            "Run pre_processing.py first."
        )
        all_predictors = [
            "total_population", "share_working_age", "share_owner_occupied",
            "share_social_rental", "share_private_rental", "avg_property_value",
            "population_growth_per1000", "grey_pressure_pct", "ses_score",
            "structural_vacancy_count", "structural_vacancy_pct",
        ]

    # Filter to columns that exist in the dataframe
    missing = [p for p in all_predictors if p not in df.columns]
    if missing:
        log.warning("  %d predictors not in dataset, skipping: %s", len(missing), missing)

    available = [p for p in all_predictors if p in df.columns]

    # Split into base and structural
    structural = [p for p in available if p in STRUCTURAL_PREDICTOR_NAMES]
    base       = [p for p in available if p not in STRUCTURAL_PREDICTOR_NAMES]

    log.info("  Base predictors: %d  |  Structural predictors: %d", len(base), len(structural))
    return base, structural

TRAIN_YEARS      = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
TEST_YEARS       = [2022, 2023, 2024]
PROPERTY_TYPES   = ["Woningen", "commercieel"]

# Scenarios
# To add a new scenario: add a dict with "name", "structural", "impute" keys


SCENARIOS = [
    # Separate models per property type
    {"name": "No zeros, no structural, separate",    "zeros": False, "structural": False, "combined": False},
    {"name": "No zeros, with structural, separate",  "zeros": False, "structural": True,  "combined": False},
    {"name": "With zeros, no structural, separate",  "zeros": True,  "structural": False, "combined": False},
    {"name": "With zeros, with structural, separate","zeros": True,  "structural": True,  "combined": False},
    # Combined model for both property types
    {"name": "No zeros, no structural, combined",    "zeros": False, "structural": False, "combined": True},
    {"name": "No zeros, with structural, combined",  "zeros": False, "structural": True,  "combined": True},
    {"name": "With zeros, no structural, combined",  "zeros": True,  "structural": False, "combined": True},
    {"name": "With zeros, with structural, combined","zeros": True,  "structural": True,  "combined": True},
]

# Model builders
# To add a new model: add a build function and register it in MODELS below


def build_svr() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svr",    SVR(kernel="rbf", C=1.0, epsilon=0.1)),
    ])


def build_rf() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )


def build_xgb() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )


def build_lgbm() -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )


MODELS = [
    {"name": "SVR",           "build": build_svr,  "log_target": True},
    {"name": "Random Forest", "build": build_rf,   "log_target": False},
    {"name": "XGBoost",       "build": build_xgb,  "log_target": False},
    {"name": "LightGBM",      "build": build_lgbm, "log_target": False},
]

# Helpers


def savefig(name: str) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    path = PLOTS / name
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log.info("Saved plot: %s", path)


def evaluate(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    scenario_name: str,
    func: str,
    split: str = "test",
) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2   = r2_score(y_true, y_pred)
    log.info("    %-15s [%-5s]  MAE=%.3f  RMSE=%.3f  R2=%.3f", model_name, split, mae, rmse, r2)
    return {
        "Scenario": scenario_name,
        "Functie":  func,
        "Model":    model_name,
        "Split":    split,
        "MAE":      round(mae,  4),
        "RMSE":     round(rmse, 4),
        "R2":       round(r2,   4),
    }


# Data preparation


def load_data() -> tuple[pd.DataFrame, list[str], list[str]]:
    path = PROCESSED / "model_ready.csv"
    df   = pd.read_csv(path)
    log.info("Loaded model-ready data: %s  shape=%s", path, df.shape)
    base_predictors, structural_predictors = load_predictors(df)
    return df, base_predictors, structural_predictors


def prepare_scenario(
    df: pd.DataFrame,
    scenario: dict,
    base_predictors: list[str],
    structural_predictors: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    log.info("Preparing scenario: %s", scenario["name"])
    df = df.copy()

    df = df.dropna(subset=[TARGET])

    # Either keep or drop rows where vacancy_rate_pct == 0
    if not scenario["zeros"]:
        before = len(df)
        df = df[df[TARGET] > 0].copy()
        log.info("  Dropped %d zero-vacancy rows.", before - len(df))
    else:
        log.info("  Keeping zero-vacancy rows: %d rows with vacancy = 0.", (df[TARGET] == 0).sum())

    predictors = base_predictors.copy()
    if scenario["structural"]:
        predictors = predictors + structural_predictors

    before = len(df)
    df     = df.dropna(subset=predictors)
    if len(df) < before:
        log.info("  Dropped %d rows with NaN predictors.", before - len(df))

    log.info("  Scenario ready: %d rows, %d predictors.", len(df), len(predictors))
    return df, predictors


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["year"].isin(TRAIN_YEARS)].copy()
    test  = df[df["year"].isin(TEST_YEARS)].copy()
    return train, test


# Training and evaluation

def run_scenario(
    df: pd.DataFrame,
    predictors: list[str],
    scenario: dict,
) -> list[dict]:
    results      = []
    scenario_name = scenario["name"]

    if scenario["combined"]:
        # Train one model on all property types combined
        # Add property_type as a binary feature so the model knows which type it's predicting
        log.info("  -- Combined (Woningen + commercieel) --")
        df = df.copy()
        df["is_commercieel"] = (df["property_type"] == "commercieel").astype(int)
        combined_predictors  = predictors + ["is_commercieel"]

        train, test = split(df)
        if len(train) == 0 or len(test) == 0:
            log.warning("  Skipping combined — empty train or test set.")
            return results

        X_train = train[combined_predictors]
        y_train = train[TARGET]
        X_test  = test[combined_predictors]
        y_test  = test[TARGET]

        log.info(
            "  Train: %d rows | Test: %d rows | Predictors: %d",
            len(train), len(test), len(combined_predictors),
        )

        for model_cfg in MODELS:
            model_name = model_cfg["name"]
            log.info("    Training %s ...", model_name)
            model = model_cfg["build"]()

            if model_cfg["log_target"]:
                model.fit(X_train, np.log1p(y_train))
                y_pred = np.expm1(model.predict(X_test))
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            # Evaluate per property type and combined
            for func in PROPERTY_TYPES + ["combined"]:
                if func == "combined":
                    yt = y_test
                    yp = y_pred
                else:
                    mask = test["property_type"] == func
                    yt   = y_test[mask]
                    yp   = y_pred[mask.values]
                results.append(evaluate(yt, yp, model_name, scenario_name, func))

    else:
        # Train separate models per property type
        for func in PROPERTY_TYPES:
            log.info("  -- %s --", func)
            subset      = df[df["property_type"] == func].copy()
            train, test = split(subset)

            if len(train) == 0 or len(test) == 0:
                log.warning("  Skipping %s -- empty train or test set.", func)
                continue

            X_train = train[predictors]
            y_train = train[TARGET]
            X_test  = test[predictors]
            y_test  = test[TARGET]

            log.info(
                "  Train: %d rows | Test: %d rows | Predictors: %d",
                len(train), len(test), len(predictors),
            )

            for model_cfg in MODELS:
                model_name = model_cfg["name"]
                log.info("    Training %s ...", model_name)
                model = model_cfg["build"]()

                if model_cfg["log_target"]:
                    model.fit(X_train, np.log1p(y_train))
                    y_pred_test  = np.expm1(model.predict(X_test))
                    y_pred_train = np.expm1(model.predict(X_train))
                else:
                    model.fit(X_train, y_train)
                    y_pred_test  = model.predict(X_test)
                    y_pred_train = model.predict(X_train)

                results.append(evaluate(y_test,  y_pred_test,  model_name, scenario_name, func, split="test"))
                results.append(evaluate(y_train, y_pred_train, model_name, scenario_name, func, split="train"))

                # Overfitting check
                r2_test  = r2_score(y_test,  y_pred_test)
                r2_train = r2_score(y_train, y_pred_train)
                gap = r2_train - r2_test
                log.info(
                    "    %-15s  Train R²=%.3f  Test R²=%.3f  Gap=%.3f%s",
                    model_name, r2_train, r2_test, gap,
                    " ⚠ possible overfitting" if gap > 0.1 else " ✓ generalizes well",
                )

    return results


# Plots

def plot_r2_heatmap(results_df: pd.DataFrame) -> None:
    df_test = results_df[results_df["Split"] == "test"]
    for func in PROPERTY_TYPES + ["combined"]:
        subset = df_test[df_test["Functie"] == func]
        if subset.empty:
            continue
        try:
            pivot = subset.pivot(index="Model", columns="Scenario", values="R2")
        except Exception:
            continue
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.suptitle(f"R2 per model per scenario -- {func}", fontsize=13, fontweight="bold")
        sns.heatmap(
            pivot, annot=True, fmt=".3f", cmap="RdYlGn",
            center=0, linewidths=0.4, annot_kws={"size": 9}, ax=ax,
        )
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.tick_params(axis="y", rotation=0,  labelsize=9)
        plt.tight_layout()
        savefig(f"r2_heatmap_{func.lower()}.png")


def plot_mae_heatmap(results_df: pd.DataFrame) -> None:
    df_test = results_df[results_df["Split"] == "test"]
    for func in PROPERTY_TYPES + ["combined"]:
        subset = df_test[df_test["Functie"] == func]
        if subset.empty:
            continue
        try:
            pivot = subset.pivot(index="Model", columns="Scenario", values="MAE")
        except Exception:
            continue
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.suptitle(f"MAE per model per scenario -- {func}", fontsize=13, fontweight="bold")
        sns.heatmap(
            pivot, annot=True, fmt=".3f", cmap="RdYlGn_r",
            linewidths=0.4, annot_kws={"size": 9}, ax=ax,
        )
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.tick_params(axis="y", rotation=0,  labelsize=9)
        plt.tight_layout()
        savefig(f"mae_heatmap_{func.lower()}.png")


def plot_r2_bars(results_df: pd.DataFrame) -> None:
    df_test = results_df[results_df["Split"] == "test"]
    for func in PROPERTY_TYPES + ["combined"]:
        subset = df_test[df_test["Functie"] == func]
        if subset.empty:
            continue
        scenarios = subset["Scenario"].unique()
        models    = subset["Model"].unique()
        x         = np.arange(len(scenarios))
        width     = 0.8 / len(models)
        colors    = ["#7F77DD", "#5DCAA5", "#D85A30", "#F0A500"]

        fig, ax = plt.subplots(figsize=(13, 5))
        fig.suptitle(f"R2 per model per scenario -- {func}", fontsize=13, fontweight="bold")

        for i, (model, color) in enumerate(zip(models, colors)):
            vals = [
                subset[(subset["Model"] == model) & (subset["Scenario"] == sc)]["R2"].values
                for sc in scenarios
            ]
            vals = [v[0] if len(v) > 0 else 0 for v in vals]
            ax.bar(x + i * width, vals, width, label=model, color=color, alpha=0.85)

        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(scenarios, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("R2")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.legend()
        plt.tight_layout()
        savefig(f"r2_bars_{func.lower()}.png")


# Main

def main() -> None:
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["figure.dpi"] = 130

    df_base, base_predictors, structural_predictors = load_data()
    all_results = []

    for scenario in SCENARIOS:
        log.info("=== Scenario: %s ===", scenario["name"])
        df_scenario, predictors = prepare_scenario(
            df_base.copy(), scenario, base_predictors, structural_predictors
        )
        results = run_scenario(df_scenario, predictors, scenario)
        all_results.extend(results)

    results_df = pd.DataFrame(all_results)
    log.info("\nFull results summary:\n%s", results_df.to_string(index=False))

    out = PROCESSED / "comparison_results.csv"
    results_df.to_csv(out, index=False)
    log.info("Results saved -> %s", out)

    log.info("Generating plots ...")
    plot_r2_heatmap(results_df)
    plot_mae_heatmap(results_df)
    plot_r2_bars(results_df)
    log.info("All plots saved to %s", PLOTS)


if __name__ == "__main__":
    main()