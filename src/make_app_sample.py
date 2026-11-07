"""Build the synthetic applicant sample the deployed app ships with.

Run:  python src/make_app_sample.py        (after train.py)

Why this exists
---------------
Kaggle's competition rules do not permit redistributing the Home Credit data, and the
public repository should not contain individual application records. The app still needs
realistic applicants to demonstrate scoring, so this script generates them instead.

Method: stratified column-wise resampling. Held-out applicants are grouped into ten
predicted-risk deciles; within each decile every column is drawn independently, and
numeric values are jittered by a few percent. The result keeps the marginal distributions
and the broad risk structure (a high-risk synthetic applicant still has weak external
scores and tight affordability), while no synthetic row reproduces a real one. The script
checks that explicitly before writing.

What ships: data/processed/app_sample.csv, synthetic, with each row's model score.
What never ships: the raw file and any row-level real outcomes (both gitignored).
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from config import MODELS, PROCESSED, RANDOM_STATE, TARGET
from train import load, split

N_ROWS = 600
N_DECILES = 10
JITTER = 0.03          # +/- 3% on continuous columns
SHIPPED_DROPS_AGE = True


def synthesise(real: pd.DataFrame, proba: np.ndarray, n: int, rng) -> pd.DataFrame:
    """Column-wise resampling within predicted-risk deciles, plus jitter."""
    work = real.copy()
    work["_decile"] = pd.qcut(proba, N_DECILES, labels=False, duplicates="drop")
    per_decile = max(1, n // work["_decile"].nunique())

    continuous = {"AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
                  "EMPLOYED_YEARS", "YEARS_SINCE_REGISTRATION", "YEARS_SINCE_ID_PUBLISH",
                  "REGION_POPULATION_RELATIVE", "EXT_SOURCE_1", "EXT_SOURCE_2",
                  "EXT_SOURCE_3"}
    out = []
    for _, group in work.groupby("_decile"):
        block = {}
        for col in real.columns:
            drawn = group[col].sample(per_decile, replace=True,
                                      random_state=rng.integers(1 << 31)).to_numpy()
            if col in continuous:
                factor = 1 + rng.uniform(-JITTER, JITTER, size=per_decile)
                drawn = np.where(pd.isna(drawn), drawn, drawn * factor)
                if col.startswith("EXT_SOURCE"):
                    drawn = np.clip(drawn, 0, 1).round(4)
                elif col in {"AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_GOODS_PRICE"}:
                    drawn = np.round(drawn, -3)
                else:
                    drawn = np.round(drawn, 2)
            block[col] = drawn
        out.append(pd.DataFrame(block))

    synthetic = pd.concat(out, ignore_index=True)

    # recompute the engineered ratios so the synthetic rows stay internally consistent
    income = synthetic["AMT_INCOME_TOTAL"].replace(0, np.nan)
    synthetic["CREDIT_TO_INCOME"] = (synthetic["AMT_CREDIT"] / income).round(4)
    synthetic["ANNUITY_TO_INCOME"] = (synthetic["AMT_ANNUITY"] / income).round(4)
    synthetic["CREDIT_TO_GOODS"] = (
        synthetic["AMT_CREDIT"] / synthetic["AMT_GOODS_PRICE"].replace(0, np.nan)).round(4)
    synthetic["CREDIT_TERM_YEARS"] = (
        synthetic["AMT_CREDIT"] / synthetic["AMT_ANNUITY"].replace(0, np.nan)).round(2)
    synthetic["INCOME_PER_PERSON"] = (
        synthetic["AMT_INCOME_TOTAL"] / synthetic["CNT_FAM_MEMBERS"].replace(0, np.nan)).round(0)
    ext = synthetic[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    synthetic["EXT_SOURCE_MEAN"] = ext.mean(axis=1).round(4)
    synthetic["EXT_SOURCE_MIN"] = ext.min(axis=1).round(4)
    return synthetic[real.columns]


def assert_no_real_rows(synthetic: pd.DataFrame, real: pd.DataFrame) -> None:
    """No synthetic row may reproduce a real applicant, on any column subset that matters."""
    key = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
           "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    key = [c for c in key if c in real.columns]
    real_keys = set(map(tuple, real[key].round(6).fillna(-1).to_numpy().tolist()))
    syn_keys = list(map(tuple, synthetic[key].round(6).fillna(-1).to_numpy().tolist()))
    collisions = sum(1 for k in syn_keys if k in real_keys)
    if collisions:
        raise SystemExit(f"{collisions} synthetic rows match a real applicant; raise JITTER.")
    print(f"Checked {len(syn_keys)} synthetic rows against {len(real_keys)} real ones: "
          "no matches")


if __name__ == "__main__":
    rng = np.random.default_rng(RANDOM_STATE)
    pipe = joblib.load(MODELS / "model.pkl")

    df = load(quick=False)
    _, X_test, _, _ = split(df, drop_age=SHIPPED_DROPS_AGE)
    proba_real = pipe.predict_proba(X_test)[:, 1]

    synthetic = synthesise(X_test, proba_real, N_ROWS, rng)
    assert_no_real_rows(synthetic, X_test)

    synthetic["pd_hat"] = pipe.predict_proba(synthetic)[:, 1]
    synthetic[TARGET] = np.nan        # synthetic applicants have no true outcome
    synthetic.to_csv(PROCESSED / "app_sample.csv", index=False)

    print(f"Wrote {len(synthetic):,} synthetic applicants to "
          f"{PROCESSED / 'app_sample.csv'}")
    print(f"Score range: {synthetic['pd_hat'].min():.3f} to {synthetic['pd_hat'].max():.3f}, "
          f"median {synthetic['pd_hat'].median():.3f}")
    print(f"Real held-out scores for comparison: {proba_real.min():.3f} to "
          f"{proba_real.max():.3f}, median {np.median(proba_real):.3f}")
