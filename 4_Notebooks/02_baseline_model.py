import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PROCESSED = Path("../3_Data_processed")
PLOTS = Path("../4_Plots/model")

TARGET = "vacancy_rate_pct"
TRAIN_YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
TEST_YEARS = [2022, 2023, 2024]
PROPERTY_TYPES = ["Woningen", "commercieel"]


def load_predictors(df):
    predictors = (PROCESSED / "predictors_list.txt").read_text().strip().splitlines()
    return [p for p in predictors if p in df.columns]


def savefig(name):
    PLOTS.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS / name, bbox_inches="tight")
    plt.close()


def evaluate(y_true, y_pred, label, split="test"):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2   = r2_score(y_true, y_pred)
    log.info("  %-35s [%s]  MAE=%.3f  RMSE=%.3f  R2=%.3f", label, split, mae, rmse, r2)
    return {"label": label, "split": split, "MAE": mae, "RMSE": rmse, "R2": r2}


def load_data():
    df = pd.read_csv(PROCESSED / "model_ready.csv")
    log.info("Loaded data: shape=%s", df.shape)
    return df


def split(df):
    train = df[df["year"].isin(TRAIN_YEARS)].copy()
    test  = df[df["year"].isin(TEST_YEARS)].copy()
    return train, test


def build_svr():
    # scaler is inside the pipeline so it is fitted on training data only
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svr",    SVR(kernel="rbf", C=1.0, epsilon=0.1)),
    ])


def run_models(df):
    results = []
    predictors = load_predictors(df)

    for func in PROPERTY_TYPES:
        log.info("%s", func)
        subset = df[df["property_type"] == func].copy()
        train, test = split(subset)

        X_train = train[predictors]
        y_train = train[TARGET]
        X_test  = test[predictors]
        y_test  = test[TARGET]
        # dummy always predicts the training mean
        dummy = DummyRegressor(strategy="mean")
        dummy.fit(X_train, y_train)
        results.append(evaluate(y_test,  dummy.predict(X_test),  f"{func} - Dummy", split="test"))
        results.append(evaluate(y_train, dummy.predict(X_train), f"{func} - Dummy", split="train"))


        #SVR is trained on logged vacancy rates, as the values are right-skewed;
        # predictions are back-transformed with expm1 before evaluation
        svr = build_svr()
        svr.fit(X_train, np.log1p(y_train))
        y_pred_test  = np.expm1(svr.predict(X_test))
        y_pred_train = np.expm1(svr.predict(X_train))
        results.append(evaluate(y_test,  y_pred_test,  f"{func} - SVR (rbf, log)", split="test"))
        results.append(evaluate(y_train, y_pred_train, f"{func} - SVR (rbf, log)", split="train"))

        r2_train = r2_score(y_train, y_pred_train)
        r2_test  = r2_score(y_test, y_pred_test)
        log.info("  train R2=%.3f  test R2=%.3f  gap=%.3f", r2_train, r2_test, r2_train - r2_test)

        plot_predictions(y_test, y_pred_test, dummy.predict(X_test), func)
        plot_residuals(y_test, y_pred_test, func)

    return pd.DataFrame(results)


def plot_predictions(y_true, y_pred_svr, y_pred_dummy, func):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Actual vs Predicted - {func}", fontsize=13)

    for ax, preds, label, color in zip(
        axes,
        [y_pred_dummy, y_pred_svr],
        ["Dummy (mean)", "SVR (rbf)"],
        ["#AAAAAA", "#7F77DD"],
    ):
        ax.scatter(y_true, preds, alpha=0.3, s=8, color=color)
        lim = [min(y_true.min(), preds.min()), max(y_true.max(), preds.max())]
        ax.plot(lim, lim, color="#D85A30", linewidth=1.5, linestyle="--", label="Perfect fit")
        ax.set(xlabel="Actual", ylabel="Predicted", title=label)
        ax.legend(fontsize=8)

    plt.tight_layout()
    savefig(f"predictions_{func.lower()}.png")


def plot_residuals(y_true, y_pred, func):
    residuals = y_true.values - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Residuals - SVR - {func}", fontsize=13)

    axes[0].scatter(y_pred, residuals, alpha=0.3, s=8, color="#7F77DD")
    axes[0].axhline(0, color="#D85A30", linewidth=1.5, linestyle="--")
    axes[0].set(xlabel="Predicted", ylabel="Residual", title="Residuals vs Predicted")

    axes[1].hist(residuals, bins=50, color="#5DCAA5", edgecolor="white")
    axes[1].axvline(0, color="#D85A30", linewidth=1.5, linestyle="--")
    axes[1].set(xlabel="Residual", ylabel="Count", title="Residual distribution")

    plt.tight_layout()
    savefig(f"residuals_{func.lower()}.png")


def main():
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["figure.dpi"] = 130

    df = load_data()
    summary = run_models(df)

    log.info("\n%s", summary.to_string(index=False))

    out = PROCESSED / "model_results.csv"
    summary.to_csv(out, index=False)
    log.info("Saved -> %s", out)


if __name__ == "__main__":
    main()