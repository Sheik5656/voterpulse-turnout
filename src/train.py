#!/usr/bin/env python3
"""
VoterPulse training script.

Run from the project root:

    python src/train.py

It loads data/voter_turnout.csv, trains six models, writes figures into
reports/figures, metrics into reports/metrics.json, and joblib files into models/.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import (  # noqa: E402
    BASELINE_FEATURES,
    IMPROVED_FEATURES,
    add_base_features,
)

SEED = 42
TEST_SIZE = 0.20


def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100)
    return {
        "mae": round(float(mae), 3),
        "rmse": round(float(rmse), 3),
        "r2": round(float(r2), 4),
        "mape": round(mape, 3),
    }


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=12, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    data_path = ROOT / "data" / "voter_turnout.csv"
    fig_dir = ROOT / "reports" / "figures"
    model_dir = ROOT / "models"
    fig_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(data_path)
    df = add_base_features(raw)

    print(f"Loaded {len(df)} rows, {df['state_name'].nunique()} states.")
    print("Nulls:", int(df.isna().sum().sum()), "duplicates:", int(df.duplicated().sum()))

    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=SEED)
    state_means = train_df.groupby("state_name")["previous_turnout_pct"].mean().to_dict()
    global_prev_mean = float(train_df["previous_turnout_pct"].mean())

    def apply_state(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["state_mean_prev"] = out["state_name"].map(state_means).fillna(global_prev_mean)
        return out

    train_df = apply_state(train_df)
    test_df = apply_state(test_df)

    y_train = train_df["target_turnout_pct"].values
    y_test = test_df["target_turnout_pct"].values
    Xb_train, Xb_test = train_df[BASELINE_FEATURES], test_df[BASELINE_FEATURES]
    Xi_train, Xi_test = train_df[IMPROVED_FEATURES], test_df[IMPROVED_FEATURES]

    results = []

    def record(name, y_hat, extra=None):
        row = {"model": name, **metrics(y_test, y_hat)}
        if extra:
            row.update(extra)
        results.append(row)
        print(f"{name:28s}  MAE {row['mae']:.3f}  RMSE {row['rmse']:.3f}  R2 {row['r2']:.4f}")

    # Naive baseline
    record("Naive (previous = target)", test_df["previous_turnout_pct"].values)

    lin = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    lin.fit(Xb_train, y_train)
    record("Linear Regression", lin.predict(Xb_test))
    joblib.dump(lin, model_dir / "linear_regression.joblib")

    ridge = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
    ridge.fit(Xi_train, y_train)
    record("Ridge Regression", ridge.predict(Xi_test))
    joblib.dump(ridge, model_dir / "ridge.joblib")

    tree = DecisionTreeRegressor(max_depth=6, min_samples_leaf=8, random_state=SEED)
    tree.fit(Xi_train, y_train)
    record("Decision Tree", tree.predict(Xi_test))
    joblib.dump(tree, model_dir / "decision_tree.joblib")

    rf = RandomForestRegressor(
        n_estimators=160, max_depth=10, min_samples_leaf=4, random_state=SEED
    )
    rf.fit(Xi_train, y_train)
    record("Random Forest", rf.predict(Xi_test))
    joblib.dump(rf, model_dir / "random_forest.joblib")

    gb = GradientBoostingRegressor(
        n_estimators=80, max_depth=3, learning_rate=0.08, min_samples_leaf=6, random_state=SEED
    )
    gb.fit(Xi_train, y_train)
    record("Gradient Boosting", gb.predict(Xi_test))
    joblib.dump(gb, model_dir / "gradient_boosting.joblib")

    search = GridSearchCV(
        GradientBoostingRegressor(random_state=SEED),
        param_grid={
            "n_estimators": [60, 80, 120],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.05, 0.08, 0.12],
            "min_samples_leaf": [4, 6, 10],
        },
        scoring="r2",
        cv=KFold(n_splits=4, shuffle=True, random_state=SEED),
        n_jobs=1,
    )
    search.fit(Xi_train, y_train)
    best = search.best_estimator_
    record("Tuned Gradient Boosting", best.predict(Xi_test), extra={"best_params": search.best_params_})
    joblib.dump(best, model_dir / "tuned_gradient_boosting.joblib")
    joblib.dump(
        {
            "features": IMPROVED_FEATURES,
            "state_means": state_means,
            "global_prev_mean": global_prev_mean,
        },
        model_dir / "feature_config.joblib",
    )

    (ROOT / "reports" / "metrics.json").write_text(json.dumps(results, indent=2, default=str))

    # Figures
    plt.rcParams.update({"figure.facecolor": "#f3efe6", "axes.facecolor": "#fbf8f1"})

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(df["previous_turnout_pct"], bins=16, alpha=0.55, color="#1a4d4a", label="Previous")
    ax.hist(df["target_turnout_pct"], bins=16, alpha=0.55, color="#8a5a3b", label="Target")
    style(ax, "Turnout distribution")
    ax.legend(frameon=False)
    fig.savefig(fig_dir / "01_turnout_distribution.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["previous_turnout_pct"], df["target_turnout_pct"], s=14, c="#1a4d4a", alpha=0.55)
    ax.plot([42, 94], [42, 94], color="#8a5a3b", ls="--", lw=1)
    style(ax, "Previous vs target turnout")
    fig.savefig(fig_dir / "02_prev_vs_target.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    state = df.groupby("state_name")["target_turnout_pct"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(state.index, state.values, color="#1a4d4a")
    style(ax, "Average target turnout by state")
    fig.savefig(fig_dir / "03_state_turnout.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    pred = best.predict(Xi_test)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, pred, s=22, c="#1a4d4a", alpha=0.75)
    lo, hi = min(y_test.min(), pred.min()) - 1, max(y_test.max(), pred.max()) + 1
    ax.plot([lo, hi], [lo, hi], color="#8a5a3b", ls="--", lw=1)
    style(ax, "Tuned model — actual vs predicted")
    fig.savefig(fig_dir / "06_actual_vs_pred.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    names = [r["model"] for r in results]
    r2s = [r["r2"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.barh(names, r2s, color="#1a4d4a")
    style(ax, "Test-set R²")
    fig.savefig(fig_dir / "05_model_r2.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print("Wrote models to", model_dir)
    print("Wrote figures to", fig_dir)
    print("Best params", search.best_params_)


if __name__ == "__main__":
    main()
