#!/usr/bin/env python3
"""Predict turnout for one constituency from the saved tuned model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import IMPROVED_FEATURES, add_base_features  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Predict Lok Sabha turnout")
    parser.add_argument("--state", required=True)
    parser.add_argument("--name", required=True, help="Constituency name, e.g. Hyderabad")
    args = parser.parse_args()

    df = add_base_features(pd.read_csv(ROOT / "data" / "voter_turnout.csv"))
    cfg = joblib.load(ROOT / "models" / "feature_config.joblib")
    model = joblib.load(ROOT / "models" / "tuned_gradient_boosting.joblib")

    row = df[(df["state_name"] == args.state) & (df["pc_name"] == args.name)]
    if row.empty:
        # case-insensitive fallback
        row = df[
            (df["state_name"].str.lower() == args.state.lower())
            & (df["pc_name"].str.lower() == args.name.lower())
        ]
    if row.empty:
        print("Constituency not found. Check --state and --name.")
        sys.exit(1)

    row = row.copy()
    row["state_mean_prev"] = row["state_name"].map(cfg["state_means"]).fillna(cfg["global_prev_mean"])
    pred = float(model.predict(row[IMPROVED_FEATURES])[0])
    actual = float(row["target_turnout_pct"].iloc[0])
    prev = float(row["previous_turnout_pct"].iloc[0])
    print(f"{row['pc_name'].iloc[0]} ({row['state_name'].iloc[0]})")
    print(f"  previous : {prev:.2f}%")
    print(f"  predicted: {pred:.2f}%")
    print(f"  actual   : {actual:.2f}%")
    print(f"  error    : {pred - actual:+.2f} pp")


if __name__ == "__main__":
    main()
