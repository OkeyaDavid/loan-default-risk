"""Train and compare models. No resampling and no class weights: probabilities stay
calibrated so the expected-loss dollars in Step 6 mean what they say. The 8%/92%
imbalance is handled by choosing the decision threshold, not by reweighting the data.

Run:  python src/train.py                 # all variants
      python src/train.py --quick         # 60k-row subsample, for a fast check

Variants trained:
  logistic_age      logistic regression, age included
  lightgbm_age      LightGBM, age included
  lightgbm_noage    LightGBM, age removed  -> the fairness/accuracy trade-off, measured

Outputs: models/model.pkl (the shipped model), models/metrics.json,
         data/processed/test_predictions_<variant>.csv, docs/model_comparison.csv
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (AGE_FEATURES, DOCS, MODELS, PROCESSED, RANDOM_STATE,
                    TARGET, TEST_SIZE)

SHIPPED = "lightgbm_noage"  # chosen on evidence: dropping age costs 0.0019 ROC-AUC
#          (see docs/model_comparison.csv). The app and SHAP both use this variant.


def load(quick: bool) -> pd.DataFrame:
    path = PROCESSED / "features.parquet"
    if not path.exists():
        raise SystemExit("Run `python src/features.py` first.")
    df = pd.read_parquet(path)
    if quick:
        df = df.sample(60_000, random_state=RANDOM_STATE)
    return df


def split(df: pd.DataFrame, drop_age: bool):
    X = df.drop(columns=[TARGET])
    if drop_age:
        X = X.drop(columns=[c for c in AGE_FEATURES if c in X.columns])
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)


def preprocessor(X: pd.DataFrame, scale: bool) -> ColumnTransformer:
    num = X.select_dtypes("number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), num),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01,
                                     sparse_output=False)),
        ]), cat),
    ])


def make_pipeline(kind: str, X: pd.DataFrame) -> Pipeline:
    if kind == "logistic":
        model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        return Pipeline([("prep", preprocessor(X, scale=True)), ("model", model)])
    model = LGBMClassifier(
        n_estimators=500, learning_rate=0.03, num_leaves=31, max_depth=-1,
        min_child_samples=200,       # no split may rest on fewer than 200 applicants
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    return Pipeline([("prep", preprocessor(X, scale=False)), ("model", model)])


def evaluate(name: str, pipe: Pipeline, Xtr, Xte, ytr, yte, cv_folds: int = 3) -> dict:
    t0 = time.time()
    pipe.fit(Xtr, ytr)
    fit_s = time.time() - t0
    proba = pipe.predict_proba(Xte)[:, 1]

    auc = roc_auc_score(yte, proba)
    ap = average_precision_score(yte, proba)
    brier = brier_score_loss(yte, proba)
    cv = cross_val_score(pipe, Xtr, ytr, cv=StratifiedKFold(cv_folds, shuffle=True,
                         random_state=RANDOM_STATE), scoring="roc_auc", n_jobs=1)

    # Calibration: are predicted probabilities close to observed default rates?
    bins = pd.qcut(proba, 10, labels=False, duplicates="drop")
    cal = (pd.DataFrame({"bin": bins, "pred": proba, "actual": yte.values})
             .groupby("bin").agg(pred=("pred", "mean"), actual=("actual", "mean"),
                                 n=("actual", "size")).round(4))
    cal_gap = float((cal["pred"] - cal["actual"]).abs().max())

    print(f"\n=== {name} ===  fit {fit_s:.0f}s")
    print(f"ROC-AUC {auc:.4f} | PR-AUC {ap:.4f} (base rate {yte.mean():.4f}) "
          f"| Brier {brier:.5f}")
    print(f"CV ROC-AUC {cv.mean():.4f} +/- {cv.std():.4f}  (gap to test: {auc - cv.mean():+.4f})")
    print(f"Calibration, worst decile gap: {cal_gap:.4f}")
    print(cal.to_string())

    return {"variant": name, "roc_auc": round(auc, 4), "pr_auc": round(ap, 4),
            "brier": round(brier, 5), "cv_mean": round(cv.mean(), 4),
            "cv_std": round(cv.std(), 4), "calibration_max_gap": round(cal_gap, 4),
            "fit_seconds": round(fit_s, 1), "n_train": len(Xtr), "n_test": len(Xte),
            "_pipe": pipe, "_proba": proba, "_yte": yte, "_Xte": Xte}


def main(quick: bool) -> None:
    df = load(quick)
    print(f"{len(df):,} applications, default rate {df[TARGET].mean():.2%}")

    results = []
    for name, kind, drop_age in [
        ("logistic_age", "logistic", False),
        ("lightgbm_age", "lightgbm", False),
        ("lightgbm_noage", "lightgbm", True),
    ]:
        Xtr, Xte, ytr, yte = split(df, drop_age=drop_age)
        res = evaluate(name, make_pipeline(kind, Xtr), Xtr, Xte, ytr, yte)
        results.append(res)
        pd.DataFrame({
            "y_true": res["_yte"].values,
            "proba": res["_proba"],
            "amt_credit": res["_Xte"]["AMT_CREDIT"].values,
        }).to_csv(PROCESSED / f"test_predictions_{name}.csv", index=False)

    table = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                          for r in results]).set_index("variant")
    print("\n=== Comparison ===")
    print(table.to_string())

    age_cost = table.loc["lightgbm_age", "roc_auc"] - table.loc["lightgbm_noage", "roc_auc"]
    lift = table.loc["lightgbm_age", "roc_auc"] - table.loc["logistic_age", "roc_auc"]
    print(f"\nCost of removing age: {age_cost:+.4f} ROC-AUC")
    print(f"LightGBM over logistic baseline: {lift:+.4f} ROC-AUC")

    table.to_csv(DOCS / "model_comparison.csv")
    shipped = next(r for r in results if r["variant"] == SHIPPED)
    joblib.dump(shipped["_pipe"], MODELS / "model.pkl")

    meta = {k: v for k, v in shipped.items() if not k.startswith("_")}
    meta.update({"shipped_variant": SHIPPED,
                 "age_cost_roc_auc": round(float(age_cost), 4),
                 "lightgbm_minus_logistic_roc_auc": round(float(lift), 4),
                 "trained_on_rows": int(len(df)), "quick_mode": quick})
    (MODELS / "metrics.json").write_text(json.dumps(meta, indent=2))

    # A small, shippable slice for the Streamlit app (free tier has ~1 GB of memory).
    # Built from the SHIPPED variant's own feature set, so the app and the model agree.
    Xtr, Xte, ytr, yte = split(df, drop_age=(SHIPPED == "lightgbm_noage"))
    sample = Xte.copy()
    sample[TARGET] = yte.values
    sample = sample.sample(min(2000, len(sample)), random_state=RANDOM_STATE)
    # Score it here, once, so the deployed app does no model work until a user edits an
    # applicant. Cold starts on the free tier are CPU-throttled; this keeps boot to file reads.
    sample["pd_hat"] = shipped["_pipe"].predict_proba(sample.drop(columns=[TARGET]))[:, 1]
    sample.to_csv(PROCESSED / "app_sample.csv", index=False)
    print(f"\nShipped {SHIPPED} -> {MODELS / 'model.pkl'}; metrics -> models/metrics.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="60k-row subsample for a fast check")
    main(p.parse_args().quick)
