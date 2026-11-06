"""Build the modelling table from application_train.csv.

Run:  python src/features.py

Three jobs:
  1. Keep the ~25 chosen columns; refuse the ones listed in config.EXCLUDED
  2. Repair the dataset's storage quirks (negative day counts, the 365243 placeholder)
  3. Engineer affordability and stability ratios

Writes data/processed/features.parquet (plus a .csv sample for inspection).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (CATEGORICAL, EXCLUDED, NUMERIC, PROCESSED, RAW, TARGET)

DAYS_EMPLOYED_PLACEHOLDER = 365243  # ~1000 years: Home Credit's code for "not employed"


def load_raw() -> pd.DataFrame:
    if not RAW.exists():
        raise SystemExit(
            f"Missing {RAW}\n"
            "Download application_train.csv from the Kaggle Home Credit Default Risk "
            "competition and place it in data/raw/."
        )
    usecols = sorted(set(NUMERIC + CATEGORICAL + [TARGET]))
    head = pd.read_csv(RAW, nrows=0)
    missing = [c for c in usecols if c not in head.columns]
    if missing:
        raise SystemExit(f"These expected columns are not in the file: {missing}")
    return pd.read_csv(RAW, usecols=usecols, low_memory=False)


def build(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # --- 2. repair storage quirks --------------------------------------------------------
    out["AGE_YEARS"] = (-out["DAYS_BIRTH"] / 365.25).round(2)
    employed = out["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_PLACEHOLDER, np.nan)
    out["EMPLOYED_YEARS"] = (-employed / 365.25).round(2)
    out["IS_UNEMPLOYED"] = (out["DAYS_EMPLOYED"] == DAYS_EMPLOYED_PLACEHOLDER).astype("int8")
    out["YEARS_SINCE_REGISTRATION"] = (-out["DAYS_REGISTRATION"] / 365.25).round(2)
    out["YEARS_SINCE_ID_PUBLISH"] = (-out["DAYS_ID_PUBLISH"] / 365.25).round(2)
    out = out.drop(columns=["DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH"])

    # --- 3. affordability and stability ratios -------------------------------------------
    income = out["AMT_INCOME_TOTAL"].replace(0, np.nan)
    out["CREDIT_TO_INCOME"] = (out["AMT_CREDIT"] / income).round(4)
    out["ANNUITY_TO_INCOME"] = (out["AMT_ANNUITY"] / income).round(4)      # debt-service ratio
    out["CREDIT_TO_GOODS"] = (out["AMT_CREDIT"] / out["AMT_GOODS_PRICE"].replace(0, np.nan)).round(4)
    out["CREDIT_TERM_YEARS"] = (out["AMT_CREDIT"] / out["AMT_ANNUITY"].replace(0, np.nan)).round(2)
    out["INCOME_PER_PERSON"] = (out["AMT_INCOME_TOTAL"] / out["CNT_FAM_MEMBERS"].replace(0, np.nan)).round(0)
    ext = out[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    out["EXT_SOURCE_MEAN"] = ext.mean(axis=1).round(4)
    out["EXT_SOURCE_MIN"] = ext.min(axis=1).round(4)

    return out


def profile(out: pd.DataFrame) -> pd.DataFrame:
    num = out.select_dtypes("number").drop(columns=[TARGET])
    return (pd.DataFrame({
        "missing_pct": (num.isna().mean() * 100).round(1),
        "median": num.median().round(3),
        "corr_with_default": num.corrwith(out[TARGET]).round(4),
    }).sort_values("corr_with_default", key=abs, ascending=False))


if __name__ == "__main__":
    raw = load_raw()
    print(f"Loaded {len(raw):,} applications, default rate {raw[TARGET].mean():.2%}")
    print("\nRefused columns:")
    for col, why in EXCLUDED.items():
        print(f"  - {col}: {why}")

    out = build(raw)
    feats = [c for c in out.columns if c != TARGET]
    print(f"\nModelling table: {len(out):,} rows x {len(feats)} features")

    prof = profile(out)
    print("\nFeature profile (sorted by |correlation with default|):")
    print(prof.to_string())
    prof.to_csv(PROCESSED / "feature_profile.csv")

    out.to_parquet(PROCESSED / "features.parquet", index=False)
    out.head(500).to_csv(PROCESSED / "features_sample.csv", index=False)
    print(f"\nSaved {PROCESSED / 'features.parquet'}")
