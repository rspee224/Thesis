import json
import logging
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PROCESSED = Path("../3_Data_processed")
OUTPUT_JSON = PROCESSED / "best_params_grid_full.json"
OUTPUT_CSV = PROCESSED / "tuning_grid_full_results.csv"

TARGET = "vacancy_rate_pct"
TRAIN_YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
TEST_YEARS = [2022, 2023, 2024]
PROPERTY_TYPES = ["Woningen", "commercieel"]
STRUCTURAL_COLS = ["structural_vacancy_count", "structural_vacancy_pct"]

SCENARIOS = [
    {"name": "no_zeros_with_structural",   "zeros": False, "structural": True},
    {"name": "with_zeros_with_structural", "zeros": True,  "structural": True},
]

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


def load_data():
    df = pd.read_csv(PROCESSED / "model_ready.csv")
    log.info("Loaded data: shape=%s", df.shape)
    all_preds = (PROCESSED / "predictors_list.txt").read_text().strip().splitlines()
    available = [p for p in all_preds if p in df.columns]
    structural = [p for p in available if p in STRUCTURAL_COLS]
    base = [p for p in available if p not in STRUCTURAL_COLS]
    return df, base, structural


def prepare_subset(df, func, scenario, base_preds, struct_preds):
    sub = df[df["property_type"] == func].dropna(subset=[TARGET]).copy()
    if not scenario["zeros"]:
        sub = sub[sub[TARGET] > 0]
    preds = base_preds + (struct_preds if scenario["structural"] else [])
    preds = [p for p in preds if p in sub.columns]
    sub = sub.dropna(subset=preds)
    #sort_values ensures timeseriessplit receives data in chronological order
    train = sub[sub["year"].isin(TRAIN_YEARS)].sort_values("year").copy()
    test = sub[sub["year"].isin(TEST_YEARS)].copy()
    return train, test, preds


def main():
    df, base_preds, struct_preds = load_data()
    tscv = TimeSeriesSplit(n_splits=3)
    results = []
    best_params_all = {}

    total = len(PROPERTY_TYPES) * len(SCENARIOS) * len(MODELS)
    done = 0
    # itertools.product generates every combination of property type, scenario, and model
    for func, scenario, (model_name, model_cfg) in product(PROPERTY_TYPES, SCENARIOS, MODELS.items()):
        done += 1
        scenario_name = scenario["name"]
        log.info("[%d/%d] %s | %s | %s", done, total, func, scenario_name, model_name)

        train, test, preds = prepare_subset(df, func, scenario, base_preds, struct_preds)

        if len(train) == 0 or len(test) == 0:
            log.warning("  no data, skipping")
            continue

        X_train = train[preds].values
        y_train = train[TARGET].values
        X_test = test[preds].values
        y_test = test[TARGET].values
        #train default model first to measure how much tuning actually improves performance
        default = model_cfg["build"]()
        default.fit(X_train, y_train)
        y_pred_default = default.predict(X_test)
        r2_def   = r2_score(y_test, y_pred_default)
        mae_def  = mean_absolute_error(y_test, y_pred_default)
        rmse_def = mean_squared_error(y_test, y_pred_default) ** 0.5
        log.info("  default   R2=%.4f  MAE=%.4f  RMSE=%.4f", r2_def, mae_def, rmse_def)


        # holdout test set is never passed here, GridSearchCV only sees train data,
        # ensuring the final evaluation on 2022-2024 remains unbiased
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

        y_pred_tuned = gs.best_estimator_.predict(X_test)
        r2_tuned   = r2_score(y_test, y_pred_tuned)
        mae_tuned  = mean_absolute_error(y_test, y_pred_tuned)
        rmse_tuned = mean_squared_error(y_test, y_pred_tuned) ** 0.5
        delta = r2_tuned - r2_def
        log.info("  tuned     R2=%.4f  MAE=%.4f  RMSE=%.4f  dR2=%+.4f", r2_tuned, mae_tuned, rmse_tuned, delta)
        log.info("  best params: %s", gs.best_params_)

        best_params_all[f"{func}|{scenario_name}|{model_name}"] = gs.best_params_

        for version, r2, mae, rmse, cv_r2 in [
            ("default",    r2_def,   mae_def,   rmse_def,   None),
            ("grid_tuned", r2_tuned, mae_tuned, rmse_tuned, gs.best_score_),
        ]:
            results.append({
                "property_type": func,
                "scenario":      scenario_name,
                "model":         model_name,
                "version":       version,
                "cv_r2":         cv_r2,
                "holdout_r2":    r2,
                "MAE":           mae,
                "RMSE":          rmse,
                "delta_r2":      delta if version == "grid_tuned" else None,
            })

    summary = pd.DataFrame(results)
    summary.to_csv(OUTPUT_CSV, index=False)
    log.info("Saved results -> %s", OUTPUT_CSV)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(best_params_all, f, indent=2)
    log.info("Saved best params -> %s", OUTPUT_JSON)

    tuned = summary[summary["version"] == "grid_tuned"].copy()
    for func in PROPERTY_TYPES:
        log.info("\n%s - top 5:", func)
        top = tuned[tuned["property_type"] == func].nlargest(5, "holdout_r2")
        for _, row in top.iterrows():
            log.info("  R2=%.4f  dR2=%+.4f  %s | %s", row["holdout_r2"], row["delta_r2"], row["scenario"], row["model"])


if __name__ == "__main__":
    main()