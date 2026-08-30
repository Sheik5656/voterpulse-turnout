"""Shared feature engineering for train / predict / Streamlit."""

from __future__ import annotations

import numpy as np
import pandas as pd

URBAN_KEYWORDS = (
    "Mumbai",
    "Delhi",
    "Bangalore",
    "Chennai",
    "Hyderabad",
    "Kolkata",
    "Ahmedabad",
    "Pune",
    "Jaipur",
    "Lucknow",
    "Chandigarh",
    "Secunderabad",
    "Malkajgiri",
    "Chevella",
    "Gurgaon",
    "Faridabad",
    "Thane",
    "Kalyan",
    "Bhiwandi",
    "Gandhinagar",
    "New Delhi",
    "Patna Sahib",
    "Bhopal",
    "Indore",
    "Nagpur",
    "Coimbatore",
    "Visakhapatnam",
    "Vijayawada",
)

BASELINE_FEATURES = [
    "previous_turnout_pct",
    "total_electors",
    "female_share",
    "third_share",
]

IMPROVED_FEATURES = [
    "previous_turnout_pct",
    "log_electors",
    "female_share",
    "third_share",
    "gender_gap",
    "state_mean_prev",
    "is_reserved",
    "is_urban",
]


def is_reserved(name: str) -> int:
    n = str(name).upper()
    return int("(SC)" in n or "(ST)" in n)


def is_urban(name: str) -> int:
    n = str(name).lower()
    return int(any(k.lower() in n for k in URBAN_KEYWORDS))


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["female_share"] = out["female_electors"] / out["total_electors"]
    out["male_share"] = out["male_electors"] / out["total_electors"]
    out["third_share"] = out["third_gender_electors"] / out["total_electors"]
    out["gender_gap"] = (out["male_electors"] - out["female_electors"]) / out[
        "total_electors"
    ]
    out["log_electors"] = np.log1p(out["total_electors"])
    out["is_reserved"] = out["pc_name"].map(is_reserved)
    out["is_urban"] = out["pc_name"].map(is_urban)
    return out
