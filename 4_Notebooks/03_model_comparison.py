import logging
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
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PROCESSED = Path("../3_Data_processed")
PLOTS     = Path("../4_Plots/comparison")

TARGET = "vacancy_rate_pct"
STRUCTURAL_COLS = ["structural_vacancy_count", "structural_vacancy_pct"]
TRAIN_YEARS    = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
TEST_YEARS     = [2022, 2023, 2024]
PROPERTY_TYPES = ["Woningen", "commercieel"]

SCENARIOS = [
    {"name": "No zeros, no structural, separate",     "zeros": False, "structural": False, "combined": False},
    {"name": "No zeros, with structural, separate",   "zeros": False, "structural": True,  "combined": False},
    {"name": "With zeros, no structural, separate",   "zeros": True,  "structural": False, "combined": False},
    {"name": "With zeros, with structural, separate", "zeros": True,  "structural": True,  "combined": False},
    {"name": "No zeros, no structural, combined",     "zeros": False, "structural": False, "combined": True},
    {"name": "No zeros, with structural, combined",   "zeros": False, "structural": True,  "combined": True},
    {"name": "With zeros, no structural, combined",   "zeros": True,  "structural": False, "combined": True},
    {"name": "With zeros, with structural, combined", "zeros": True,  "structural": True,  "combined": True},
]


def build_svr():
    #scalar is inside the pipeline so it is fitted on training data only
    # and applied consistently to the test set without leakage
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svr",    SVR(kernel="rbf", C=1.0, epsilon=0.1)),
    ])


def build_rf():
    return RandomForestRegressor(
        n_estimators=200, max_depth=None, min_samples_leaf=5,
        n_jobs=-1, random_state=42,
    )


def build_xgb():
    return XGBRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
    )


def build_lgbm():
    return LGBMRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=-1, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1,
    )

#SVR is logged as the values are right-skewed, again they are back transformed with expm1
MODELS = [
    {"name": "SVR",           "build": build_svr,  "log_target": True},
    {"name": "Random Forest", "build": build_rf,   "log_target": False},
    {"name": "XGBoost",       "build": build_xgb,  "log_target": False},
    {"name": "LightGBM",      "build": build_lgbm, "log_target": False},
]


def savefig(name):
    PLOTS.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS / name, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", PLOTS / name)


def evaluate(y_true, y_pred, model_name, scenario_name, func, split="test"):
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


def load_predictors(df):
    all_predictors = (PROCESSED / "predictors_list.txt").read_text().strip().splitlines()
    available  = [p for p in all_predictors if p in df.columns]
    structural = [p for p in available if p in STRUCTURAL_COLS]
    base       = [p for p in available if p not in STRUCTURAL_COLS]
    log.info("  Base: %d  |  Structural: %d", len(base), len(structural))
    return base, structural


def load_data():
    df = pd.read_csv(PROCESSED / "model_ready.csv")
    log.info("Loaded data: shape=%s", df.shape)
    base_predictors, structural_predictors = load_predictors(df)
    return df, base_predictors, structural_predictors


def prepare_scenario(df, scenario, base_predictors, structural_predictors):
    log.info("Preparing: %s", scenario["name"])
    df = df.copy()
    df = df.dropna(subset=[TARGET])

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
    df = df.dropna(subset=predictors)
    if len(df) < before:
        log.info("  Dropped %d rows with NaN predictors.", before - len(df))

    log.info("  Ready: %d rows, %d predictors.", len(df), len(predictors))
    return df, predictors


def split(df):
    #Time-based split to prevent future data leaking into training
    train = df[df["year"].isin(TRAIN_YEARS)].copy()
    test  = df[df["year"].isin(TEST_YEARS)].copy()
    return train, test


def run_scenario(df, predictors, scenario):
    results = []
    scenario_name = scenario["name"]

    if scenario["combined"]:
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

        log.info("  Train: %d rows | Test: %d rows | Predictors: %d",
                 len(train), len(test), len(combined_predictors))

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

            log.info("  Train: %d rows | Test: %d rows | Predictors: %d",
                     len(train), len(test), len(predictors))

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

                r2_test  = r2_score(y_test,  y_pred_test)
                r2_train = r2_score(y_train, y_pred_train)
                gap = r2_train - r2_test
                log.info("    %-15s  train R2=%.3f  test R2=%.3f  gap=%.3f",
                         model_name, r2_train, r2_test, gap)

    return results


def plot_r2_heatmap(results_df):
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
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn",
                    center=0, linewidths=0.4, annot_kws={"size": 9}, ax=ax)
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.tick_params(axis="y", rotation=0,  labelsize=9)
        plt.tight_layout()
        savefig(f"r2_heatmap_{func.lower()}.png")


def plot_mae_heatmap(results_df):
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
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn_r",
                    linewidths=0.4, annot_kws={"size": 9}, ax=ax)
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.tick_params(axis="y", rotation=0,  labelsize=9)
        plt.tight_layout()
        savefig(f"mae_heatmap_{func.lower()}.png")


def plot_r2_bars(results_df):
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


def main():
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["figure.dpi"] = 130

    df_base, base_predictors, structural_predictors = load_data()
    all_results = []

    for scenario in SCENARIOS:
        log.info("=== %s ===", scenario["name"])
        df_scenario, predictors = prepare_scenario(
            df_base.copy(), scenario, base_predictors, structural_predictors
        )
        all_results.extend(run_scenario(df_scenario, predictors, scenario))

    results_df = pd.DataFrame(all_results)
    log.info("\n%s", results_df.to_string(index=False))

    out = PROCESSED / "comparison_results.csv"
    results_df.to_csv(out, index=False)
    log.info("Saved -> %s", out)

    plot_r2_heatmap(results_df)
    plot_mae_heatmap(results_df)
    plot_r2_bars(results_df)
    log.info("Plots saved to %s", PLOTS)


if __name__ == "__main__":
    main()