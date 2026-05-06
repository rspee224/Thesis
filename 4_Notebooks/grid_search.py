"""
Grid Search Tuning Script — Full
=================================
Tunes LightGBM, XGBoost, and Random Forest across all dataset scenarios:
  - No zeros  / With zeros
  - No structural features / With structural features
  = 4 scenarios × 3 models × 2 property types = 24 grid searches

Uses GridSearchCV with TimeSeriesSplit(n_splits=3):
  Fold 1: train 2015-2017, validate 2018
  Fold 2: train 2015-2018, validate 2019-2020
  Fold 3: train 2015-2020, validate 2021

Holdout test set (2022-2024) is never touched during tuning.
Best parameters and all results saved to CSV and JSON.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

warnings.filterwarnings("ignore")


def _ensure(pkg: str, imp: str) -> None:
    try:
        __import__(imp)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])


_ensure("xgboost", "xgboost")
_ensure("lightgbm", "lightgbm")

from xgboost import XGBRegressor  # noqa: E402
from lightgbm import LGBMRegressor  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED = Path("../3_Data_processed")
PREDICTORS_FILE = PROCESSED / "predictors_list.txt"
OUTPUT_JSON = PROCESSED / "best_params_grid_full.json"
OUTPUT_CSV = PROCESSED / "tuning_grid_full_results.csv"

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET = "vacancy_rate_pct"
TRAIN_YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
TEST_YEARS = [2022, 2023, 2024]
PROPERTY_TYPES = ["Woningen", "commercieel"]

STRUCTURAL_PREDICTOR_NAMES = ["structural_vacancy_count", "structural_vacancy_pct"]

# ── Dataset scenarios ─────────────────────────────────────────────────────────
SCENARIOS = [

    {"name": "no_zeros_with_structural", "zeros": False, "structural": True},

    {"name": "with_zeros_with_structural", "zeros": True, "structural": True},
]

# ── Model configurations with focused grids ────────────────────────────────────
MODELS = {
    "LightGBM": {
        "build": lambda **kw: LGBMRegressor(random_state=42, verbosity=-1, **kw),
        "param_grid": {
            "n_estimators":      [100, 200, 300],
            "learning_rate":     [0.03, 0.05, 0.10],
            "num_leaves":        [20, 31, 50],
            "min_child_samples": [10, 20, 30],
            "subsample":         [0.7, 0.8, 1.0],
            "colsample_bytree":  [0.6, 0.8, 1.0],
            "reg_alpha":         [0.0, 0.5, 1.0],
            "reg_lambda":        [0.0, 1.0, 5.0],
        },
    },
    "XGBoost": {
        "build": lambda **kw: XGBRegressor(random_state=42, verbosity=0, **kw),
        "param_grid": {
            "n_estimators":    [100, 200, 300],
            "learning_rate":   [0.05, 0.10, 0.20],
            "max_depth":       [4, 6, 8],
            "subsample":       [0.7, 0.8, 1.0],
            "colsample_bytree":[0.6, 0.8, 1.0],
            "min_child_weight":[1, 3, 5],
            "reg_alpha":       [0.0, 0.5, 1.0],
            "reg_lambda":      [0.5, 1.0, 5.0],
        },
    },
    "RandomForest": {
        "build": lambda **kw: RandomForestRegressor(random_state=42, n_jobs=1, **kw),
        "param_grid": {
            "n_estimators":      [100, 200, 300],
            "max_depth":         [None, 10, 20],
            "min_samples_leaf":  [1, 5, 10],
            "max_features":      [0.5, 0.7, 1.0],
            "min_samples_split": [2, 5, 10],
        },
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, list[str], list[str]]:
    path = PROCESSED / "model_ready.csv"
    df = pd.read_csv(path)
    log.info("Loaded: %s  shape=%s", path, df.shape)

    if PREDICTORS_FILE.exists():
        all_preds = PREDICTORS_FILE.read_text().strip().splitlines()
        log.info("Loaded %d predictors from file.", len(all_preds))
    else:
        log.warning("predictors_list.txt not found — using fallback list.")
        all_preds = [
            "total_population", "share_working_age", "share_owner_occupied",
            "share_social_rental", "share_private_rental", "avg_property_value",
            "population_growth_per1000", "grey_pressure_pct", "ses_score",
            "structural_vacancy_count", "structural_vacancy_pct",
        ]

    available = [p for p in all_preds if p in df.columns]
    structural = [p for p in available if p in STRUCTURAL_PREDICTOR_NAMES]
    base = [p for p in available if p not in STRUCTURAL_PREDICTOR_NAMES]
    log.info("  Base: %d  |  Structural: %d", len(base), len(structural))
    return df, base, structural


def prepare_subset(
        df: pd.DataFrame,
        func: str,
        scenario: dict,
        base_preds: list[str],
        struct_preds: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    sub = df[df["property_type"] == func].dropna(subset=[TARGET]).copy()

    if not scenario["zeros"]:
        sub = sub[sub[TARGET] > 0]

    preds = base_preds + (struct_preds if scenario["structural"] else [])
    preds = [p for p in preds if p in sub.columns]
    sub = sub.dropna(subset=preds)

    train = sub[sub["year"].isin(TRAIN_YEARS)].sort_values("year").copy()
    test = sub[sub["year"].isin(TEST_YEARS)].copy()
    return train, test, preds


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    df, base_preds, struct_preds = load_data()
    tscv = TimeSeriesSplit(n_splits=3)
    results = []
    best_params_all = {}

    total = len(PROPERTY_TYPES) * len(SCENARIOS) * len(MODELS)
    done = 0

    for func, scenario, (model_name, model_cfg) in product(
            PROPERTY_TYPES, SCENARIOS, MODELS.items()
    ):
        done += 1
        scenario_name = scenario["name"]
        log.info("")
        log.info("─" * 65)
        log.info("[%d/%d]  %s | %s | %s",
                 done, total, func, scenario_name, model_name)
        log.info("─" * 65)

        train, test, preds = prepare_subset(
            df, func, scenario, base_preds, struct_preds
        )

        if len(train) == 0 or len(test) == 0:
            log.warning("  No data — skipping.")
            continue

        log.info("  Train=%d  Test=%d  Predictors=%d",
                 len(train), len(test), len(preds))

        X_train = train[preds].values
        y_train = train[TARGET].values
        X_test = test[preds].values
        y_test = test[TARGET].values

        # ── Default model (baseline for comparison) ───────────────────────────
        default = model_cfg["build"]()
        default.fit(X_train, y_train)
        y_pred_default = default.predict(X_test)
        r2_def = r2_score(y_test, y_pred_default)
        mae_def = mean_absolute_error(y_test, y_pred_default)
        rmse_def = mean_squared_error(y_test, y_pred_default) ** 0.5
        log.info("  Default    R²=%.4f  MAE=%.4f  RMSE=%.4f",
                 r2_def, mae_def, rmse_def)

        # ── Grid search ───────────────────────────────────────────────────────
        n_comb = 1
        for v in model_cfg["param_grid"].values():
            n_comb *= len(v)
        log.info("  GridSearch: %d combos × 3 folds = %d fits ...",
                 n_comb, n_comb * 3)

        gs = GridSearchCV(
            estimator=model_cfg["build"](),
            param_grid=model_cfg["param_grid"],
            cv=tscv,
            scoring="r2",
            n_jobs=1,
            refit=True,
            verbose=0,
        )
        gs.fit(X_train, y_train)

        log.info("  Best CV R² = %.4f", gs.best_score_)
        log.info("  Best params: %s", gs.best_params_)

        y_pred_tuned = gs.best_estimator_.predict(X_test)
        r2_tuned = r2_score(y_test, y_pred_tuned)
        mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
        rmse_tuned = mean_squared_error(y_test, y_pred_tuned) ** 0.5
        delta = r2_tuned - r2_def
        log.info("  Grid-tuned R²=%.4f  MAE=%.4f  RMSE=%.4f  ΔR²=%+.4f",
                 r2_tuned, mae_tuned, rmse_tuned, delta)

        # Save best params
        key = f"{func}|{scenario_name}|{model_name}"
        best_params_all[key] = gs.best_params_

        # Append rows
        for version, r2, mae, rmse, cv_r2 in [
            ("default", r2_def, mae_def, rmse_def, None),
            ("grid_tuned", r2_tuned, mae_tuned, rmse_tuned, gs.best_score_),
        ]:
            results.append({
                "property_type": func,
                "scenario": scenario_name,
                "model": model_name,
                "version": version,
                "cv_r2": cv_r2,
                "holdout_r2": r2,
                "MAE": mae,
                "RMSE": rmse,
                "delta_r2": delta if version == "grid_tuned" else None,
            })

    # ── Save outputs ──────────────────────────────────────────────────────────
    summary = pd.DataFrame(results)
    summary.to_csv(OUTPUT_CSV, index=False)
    log.info("\nResults saved → %s", OUTPUT_CSV)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(best_params_all, f, indent=2)
    log.info("Best params saved → %s", OUTPUT_JSON)

    # ── Print top 5 results per property type (tuned only) ────────────────────
    log.info("\n%s", "=" * 65)
    log.info("TOP TUNED RESULTS PER PROPERTY TYPE")
    log.info("=" * 65)
    tuned = summary[summary["version"] == "grid_tuned"].copy()
    for func in PROPERTY_TYPES:
        sub = tuned[tuned["property_type"] == func].nlargest(5, "holdout_r2")
        log.info("\n%s — top 5:", func)
        for _, row in sub.iterrows():
            log.info("  R²=%.4f  ΔR²=%+.4f  %s | %s",
                     row["holdout_r2"], row["delta_r2"],
                     row["scenario"], row["model"])


if __name__ == "__main__":
    main()