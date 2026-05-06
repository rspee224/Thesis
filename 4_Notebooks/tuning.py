"""
Tunes SVR, Random Forest, XGBoost and LightGBM using time-series
cross-validation and RandomizedSearchCV.

Approach:
    - Time-series cross-validation (TimeSeriesSplit, n_splits=5)
      Expanding window: train 2015-2016 → test 2017, etc.
    - RandomizedSearchCV with n_iter=30 per model
    - Separate tuning per property type (Woningen / commercieel)
    - Separate tuning per chosen scenario (no zeros vs with zeros)
    - Best parameters saved to best_params.json
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from scipy.stats import randint, uniform, loguniform

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*delayed.*")
warnings.filterwarnings("ignore", message=".*joblib.*")

#Auto-install optional dependencies

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

#Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

#Paths

PROCESSED       = Path("../3_Data_processed")
PREDICTORS_FILE = PROCESSED / "predictors_list.txt"

#Constants

TARGET = "vacancy_rate_pct"

STRUCTURAL_PREDICTOR_NAMES = [
    "structural_vacancy_count",
    "structural_vacancy_pct",
]

TRAIN_YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
TEST_YEARS  = [2022, 2023, 2024]

PROPERTY_TYPES = ["Woningen", "commercieel"]

# Number of random parameter combinations to try per model
N_ITER = 30

# Number of time-series CV folds
N_SPLITS = 5

#Parameter search spaces

PARAM_SPACES = {
    "SVR": {
        "svr__C":       loguniform(0.1, 100),     # regularization strength
        "svr__epsilon": loguniform(0.01, 1.0),    # epsilon tube width
        "svr__gamma":   ["scale", "auto"],        # RBF kernel bandwidth
    },
    "Random Forest": {
        "n_estimators":    randint(100, 500),
        "max_depth":       [None, 5, 10, 15, 20],
        "min_samples_leaf": randint(2, 20),
        "max_features":    ["sqrt", "log2", 0.3, 0.5, 0.7],
    },
    "XGBoost": {
        "n_estimators":    randint(100, 500),
        "learning_rate":   loguniform(0.01, 0.3),
        "max_depth":       randint(3, 10),
        "subsample":       uniform(0.5, 0.5),     # 0.5 to 1.0
        "colsample_bytree": uniform(0.5, 0.5),
        "reg_alpha":       loguniform(1e-3, 10),  # L1 regularization
        "reg_lambda":      loguniform(1e-3, 10),  # L2 regularization
        "min_child_weight": randint(1, 10),
    },
    "LightGBM": {
        "n_estimators":    randint(100, 500),
        "learning_rate":   loguniform(0.01, 0.3),
        "num_leaves":      randint(15, 100),
        "min_child_samples": randint(5, 50),
        "subsample":       uniform(0.5, 0.5),
        "colsample_bytree": uniform(0.5, 0.5),
        "reg_alpha":       loguniform(1e-3, 10),
        "reg_lambda":      loguniform(1e-3, 10),
    },
}

#Model builders (base, untuned)

def build_svr() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svr",    SVR(kernel="rbf")),
    ])

def build_rf() -> RandomForestRegressor:
    return RandomForestRegressor(n_jobs=-1, random_state=42)

def build_xgb() -> XGBRegressor:
    return XGBRegressor(random_state=42, verbosity=0, tree_method="hist")

def build_lgbm() -> LGBMRegressor:
    return LGBMRegressor(random_state=42, verbosity=-1)

MODELS = [
    {"name": "SVR",           "build": build_svr,  "log_target": True},
    {"name": "Random Forest", "build": build_rf,   "log_target": False},
    {"name": "XGBoost",       "build": build_xgb,  "log_target": False},
    {"name": "LightGBM",      "build": build_lgbm, "log_target": False},
]

# Data loading

def load_data() -> tuple[pd.DataFrame, list[str], list[str]]:
    path = PROCESSED / "model_ready.csv"
    df   = pd.read_csv(path)
    log.info("Loaded: %s  shape=%s", path, df.shape)

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

    available      = [p for p in all_predictors if p in df.columns]
    structural     = [p for p in available if p in STRUCTURAL_PREDICTOR_NAMES]
    base           = [p for p in available if p not in STRUCTURAL_PREDICTOR_NAMES]

    log.info("Base predictors: %d  |  Structural: %d", len(base), len(structural))
    return df, base, structural

# Scenario preparation
def prepare_data(
    df: pd.DataFrame,
    zeros: bool,
    base_predictors: list[str],
    structural_predictors: list[str],
    with_structural: bool,
) -> tuple[pd.DataFrame, list[str]]:
    """Prepare dataset for a given scenario."""
    df = df.dropna(subset=[TARGET]).copy()

    if not zeros:
        before = len(df)
        df = df[df[TARGET] > 0].copy()
        log.info("  Dropped %d zero-vacancy rows.", before - len(df))

    predictors = base_predictors.copy()
    if with_structural:
        predictors = predictors + structural_predictors

    df = df.dropna(subset=predictors)
    return df, predictors

# Time-series cross-validation

def make_tscv_splits(df: pd.DataFrame) -> list[tuple]:
    """
    Folds (on TRAIN_YEARS = 2015-2021):
        Fold 1: train 2015-2016, val 2017
        Fold 2: train 2015-2017, val 2018
        Fold 3: train 2015-2018, val 2019
        Fold 4: train 2015-2019, val 2020
        Fold 5: train 2015-2020, val 2021
    """
    train_df = df[df["year"].isin(TRAIN_YEARS)].copy()
    years    = sorted(train_df["year"].unique())

    splits = []
    for i in range(1, len(years)):
        train_years = years[:i]
        val_year    = years[i]
        train_idx   = train_df[train_df["year"].isin(train_years)].index
        val_idx     = train_df[train_df["year"] == val_year].index
        if len(train_idx) > 0 and len(val_idx) > 0:
            splits.append((train_idx, val_idx))

    # Use only the last N_SPLITS folds to keep runtime manageable
    splits = splits[-N_SPLITS:]
    log.info("  Time-series CV: %d folds.", len(splits))
    return splits, train_df

#Tuning

def tune_model(
    model_cfg: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: list,
    scenario_key: str,
    func: str,
) -> dict:
    """Run RandomizedSearchCV for one model and return best params + CV score."""
    model_name  = model_cfg["name"]
    log_target  = model_cfg["log_target"]
    model       = model_cfg["build"]()
    param_space = PARAM_SPACES[model_name]

    # For SVR we always fit on log1p(y) but score against original y.
    # We pass y_train (original) to the scorer and log-transform inside the model
    # by wrapping SVR in a pipeline that transforms the target.
    # Simplest approach: fit on log1p(y) and use a scorer that back-transforms.
    # We pass y_train_fit to fit() and y_train (original) for the scorer reference.

    if log_target:
        # Fit on log-transformed target, score on back-transformed predictions vs original
        y_train_fit = np.log1p(y_train)

        # Build scorer that back-transforms predictions and scores against original y
        # Note: RandomizedSearchCV passes y as whatever we gave to fit(),
        # so we need to back-transform both y_true (log) and y_pred (log)
        # and compare against expm1(y_true)
        def svr_r2_scorer(estimator, X, y_log):
            y_pred_log = estimator.predict(X)
            y_pred     = np.expm1(y_pred_log)
            y_true     = np.expm1(y_log)
            return r2_score(y_true, y_pred)

        scorer = make_scorer(svr_r2_scorer)
    else:
        y_train_fit = y_train
        scorer      = "r2"

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_space,
        n_iter=N_ITER,
        scoring=scorer,
        cv=cv_splits,
        n_jobs=1,           # n_jobs=1 avoids the sklearn/joblib parallel warning in Python 3.14
        random_state=42,
        verbose=0,
        refit=True,
        error_score=np.nan,
    )

    log.info("    Tuning %s (%s | %s) — %d iterations ...", model_name, func, scenario_key, N_ITER)
    search.fit(X_train, y_train_fit)

    mean_cv     = search.best_score_
    std_cv      = search.cv_results_["std_test_score"][search.best_index_]
    best_params = search.best_params_

    log.info(
        "    %-15s  CV R²=%.3f ± %.3f  best_params=%s",
        model_name, mean_cv if not np.isnan(mean_cv) else -999, std_cv if not np.isnan(std_cv) else 0,
        {k: round(v, 4) if isinstance(v, float) else v for k, v in best_params.items()},
    )

    return {
        "model":         model_name,
        "scenario":      scenario_key,
        "property_type": func,
        "cv_r2_mean":    round(float(mean_cv), 4) if not np.isnan(mean_cv) else None,
        "cv_r2_std":     round(float(std_cv),  4) if not np.isnan(std_cv)  else None,
        "best_params":   best_params,
        "log_target":    log_target,
    }

#Final evaluation on holdout

def evaluate_holdout(
    model_cfg: dict,
    best_params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[float, float, float]:
    """Train tuned model on full train set and evaluate on 2022-2024 holdout."""
    log_target = model_cfg["log_target"]
    model      = model_cfg["build"]()
    model.set_params(**best_params)

    y_fit = np.log1p(y_train) if log_target else y_train
    model.fit(X_train, y_fit)

    y_pred = model.predict(X_test)
    if log_target:
        y_pred = np.expm1(y_pred)

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2   = r2_score(y_test, y_pred)
    return mae, rmse, r2

# Main
def main() -> None:
    log.info("Starting hyperparameter tuning ...")

    df, base_predictors, structural_predictors = load_data()

    # Define which scenarios to tune
    # Based on model comparison results:
    #   Woningen:    no zeros + structural → best results
    #   Commercieel: with zeros + structural → best results
    # We tune both with and without structural to enable full comparison
    scenarios = [
        {"key": "no_zeros_no_structural",   "zeros": False, "structural": False},
        {"key": "no_zeros_with_structural", "zeros": False, "structural": True},
        {"key": "with_zeros_no_structural",   "zeros": True, "structural": False},
        {"key": "with_zeros_with_structural", "zeros": True, "structural": True},
    ]

    all_results  = []
    best_params_store = {}   # {scenario_key: {func: {model_name: params}}}

    for scenario in scenarios:
        scenario_key = scenario["key"]
        log.info("=== Scenario: %s ===", scenario_key)

        df_scenario, predictors = prepare_data(
            df,
            zeros=scenario["zeros"],
            base_predictors=base_predictors,
            structural_predictors=structural_predictors,
            with_structural=scenario["structural"],
        )

        best_params_store[scenario_key] = {}

        for func in PROPERTY_TYPES:
            log.info("  -- %s --", func)
            subset   = df_scenario[df_scenario["property_type"] == func].copy()
            train_df = subset[subset["year"].isin(TRAIN_YEARS)].copy()
            test_df  = subset[subset["year"].isin(TEST_YEARS)].copy()

            if len(train_df) == 0 or len(test_df) == 0:
                log.warning("  Skipping %s — empty train or test set.", func)
                continue

            X_train = train_df[predictors]
            y_train = train_df[TARGET]
            X_test  = test_df[predictors]
            y_test  = test_df[TARGET]

            log.info(
                "  Train: %d rows | Test: %d rows | Predictors: %d",
                len(X_train), len(X_test), len(predictors),
            )

            # Build time-series CV splits on training data
            cv_splits, train_df_indexed = make_tscv_splits(subset)

            # Map indices back to X_train positional indices for sklearn
            train_df_indexed = train_df_indexed.reset_index(drop=True)
            X_train_cv = train_df_indexed[predictors]
            y_train_cv = train_df_indexed[TARGET]

            positional_splits = []
            for tr_idx, val_idx in cv_splits:
                tr_pos  = np.where(train_df_indexed.index.isin(tr_idx))[0]
                val_pos = np.where(train_df_indexed.index.isin(val_idx))[0]
                if len(tr_pos) > 0 and len(val_pos) > 0:
                    positional_splits.append((tr_pos, val_pos))

            if not positional_splits:
                log.warning("  No valid CV splits for %s — skipping.", func)
                continue

            best_params_store[scenario_key][func] = {}

            for model_cfg in MODELS:
                result = tune_model(
                    model_cfg, X_train_cv, y_train_cv,
                    positional_splits, scenario_key, func,
                )

                # Evaluate on holdout test set with tuned params
                mae, rmse, r2 = evaluate_holdout(
                    model_cfg, result["best_params"],
                    X_train, y_train, X_test, y_test,
                )
                result["holdout_mae"]  = round(mae,  4)
                result["holdout_rmse"] = round(rmse, 4)
                result["holdout_r2"]   = round(r2,   4)

                log.info(
                    "    %-15s  Holdout → MAE=%.3f  RMSE=%.3f  R²=%.3f",
                    result["model"], mae, rmse, r2,
                )

                all_results.append(result)

                # Store best params (convert numpy types for JSON serialisation)
                clean_params = {}
                for k, v in result["best_params"].items():
                    if isinstance(v, (np.integer,)):
                        v = int(v)
                    elif isinstance(v, (np.floating,)):
                        v = float(v)
                    clean_params[k] = v

                best_params_store[scenario_key][func][result["model"]] = {
                    "params":     clean_params,
                    "log_target": result["log_target"],
                }

    # Save results

    # CV + holdout results as CSV
    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "best_params"}
        for r in all_results
    ])
    cv_out = PROCESSED / "cv_results.csv"
    results_df.to_csv(cv_out, index=False)
    log.info("CV results saved -> %s", cv_out)

    # Best params as JSON
    params_out = PROCESSED / "best_params.json"
    with open(params_out, "w") as f:
        json.dump(best_params_store, f, indent=2)
    log.info("Best params saved -> %s", params_out)

    # Summary table
    log.info("\nTuning summary:")
    summary = results_df[["scenario", "property_type", "model", "cv_r2_mean", "cv_r2_std", "holdout_r2"]]
    log.info("\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()