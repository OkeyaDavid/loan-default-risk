"""Render README.md from the artifacts, so the numbers in it cannot drift from the model.

Run:  python src/make_readme.py      (after train.py, cutoff.py and explain.py)
"""
from __future__ import annotations

import json

import pandas as pd

from config import BREAK_EVEN_PD, LGD, MARGIN, DOCS, MODELS, PROCESSED, ROOT
from explain import pretty

VARIANT = "lightgbm_noage"
GH = "https://github.com/OkeyaDavid/loan-default-risk"
APP = "https://loan-default-risk.streamlit.app"   # set after the first deploy


def main() -> None:
    m = json.loads((MODELS / "metrics.json").read_text())
    comp = pd.read_csv(DOCS / "model_comparison.csv").set_index("variant")
    cut = pd.read_csv(PROCESSED / f"cutoff_{VARIANT}.csv")
    best = cut.loc[cut["profit"].idxmax()]
    cut_age = pd.read_csv(PROCESSED / "cutoff_lightgbm_age.csv")
    best_age_profit = float(cut_age["profit"].max())
    all_in = cut[cut["approval_rate"] == 100].iloc[0]
    avoided = all_in["expected_loss"] - best["expected_loss"]
    shap_imp = pd.read_csv(DOCS / "shap_feature_importance.csv", index_col=0).head(5)

    comp_rows = "\n".join(
        f"| `{v}` | {r.roc_auc:.4f} | {r.pr_auc:.4f} | {r.brier:.5f} | "
        f"{r.cv_mean:.4f} ± {r.cv_std:.4f} |"
        for v, r in comp.iterrows())
    policy_rows = "\n".join(
        f"| {int(r.approval_rate)}% | {r.threshold:.3f} | {r.bad_rate_approved:.2f}% | "
        f"${r.expected_loss/1e6:,.1f}m | ${r.profit/1e6:,.1f}m |"
        for _, r in cut[cut["approval_rate"].isin([60, 70, 80, 90, 95, 100])].iterrows())
    top_feats = "\n".join(f"{i+1}. {pretty(name)}" for i, name in enumerate(shap_imp.index))

    text = f"""# Loan Default Risk & Approval Tool

**[Live app]({APP})** · [Code]({GH}) · Python · scikit-learn · LightGBM · SHAP · Streamlit

Predicts the probability that a loan applicant defaults, explains every decision in plain
English, and shows a lender what each approval policy costs.

![Approval cut-off versus expected loss](docs/cutoff_{VARIANT}.png)

## The question

A lender cannot approve everyone and cannot decline everyone. Given {m['n_train'] + m['n_test']:,}
past applications, how well can we rank applicants by risk, and what does each approval
policy actually cost?

## Results

- **Ranking quality:** ROC-AUC **{m['roc_auc']:.3f}** on {m['n_test']:,} held-out applicants
  (cross-validated {m['cv_mean']:.3f} ± {m['cv_std']:.3f}, so no sign of overfitting).
- **The cut-off is worth more than the model.** Approving the safest
  **{best['approval_rate']:.0f}%** of applicants instead of everyone holds defaults among
  approved loans to **{best['bad_rate_approved']:.2f}%** against
  {all_in['bad_rate_approved']:.2f}%, and avoids **${avoided/1e6:,.1f}m** of expected loss on
  the test set.
- **Dropping a protected attribute was nearly free, so it was dropped.** Removing age costs
  **{abs(m['age_cost_roc_auc']):.4f} ROC-AUC** and about
  ${(best_age_profit - best['profit'])/1e6:,.1f}m of contribution on a
  ${best['profit']/1e9:,.2f}bn book — around
  {100*(best_age_profit - best['profit'])/best_age_profit:.1f}% — so the shipped model uses
  neither age nor gender.
- **Probabilities are calibrated.** Worst gap between predicted and observed default rates
  across ten risk deciles: **{m['calibration_max_gap']:.3f}**. That is what makes the dollar
  figures meaningful.

### What drives the score

{top_feats}

(Mean absolute SHAP value; see `docs/shap_summary.png` and `docs/example_reasons.md`.)

## The approval policy

Loss given default {LGD:.0%}, lifetime margin {MARGIN:.0%}, so a marginal applicant is worth
approving up to a default probability of **{BREAK_EVEN_PD:.1%}**. Both numbers are stated
assumptions, not measurements, and they are the first thing to change in `src/config.py`.

| Approve | Cut-off PD | Defaults among approved | Expected loss | Contribution |
|---|---|---|---|---|
{policy_rows}

Losses use realised outcomes on applicants the model never saw, not its own predictions.

## Model comparison

| Variant | ROC-AUC | PR-AUC | Brier | CV ROC-AUC |
|---|---|---|---|---|
{comp_rows}

**Shipped:** `{VARIANT}`. LightGBM beats the logistic baseline by
{m['lightgbm_minus_logistic_roc_auc']:+.4f} ROC-AUC.

**No resampling and no class weights.** With an 8% default rate the temptation is SMOTE or
`class_weight='balanced'`; both distort predicted probabilities, which would make the
expected-loss figures above meaningless. The imbalance is handled where it belongs: in the
choice of threshold.

## Responsible lending

- **`CODE_GENDER` excluded from every variant.** Sex is a prohibited ground under the
  Canadian Human Rights Act and under US ECOA / Regulation B.
- **Age excluded too, on evidence.** Age is also protected and is genuinely predictive, so
  both variants were trained before deciding. Keeping it would have bought
  {abs(m['age_cost_roc_auc']):.4f} ROC-AUC. That is not worth using a protected attribute in
  a lending decision, so the shipped model is `lightgbm_noage`. The age variant stays in
  `train.py` as the evidence.
- **Per-applicant reasons.** SHAP turns one score into the three factors that raised the risk
  and the three that lowered it — the shape an adverse-action notice needs.

## Data

[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data),
`application_train.csv` only: ~307k applications, about 8% of which hit payment difficulties.
The other eight tables are deliberately out of scope; published scores that use them run
higher, around 0.79 ROC-AUC. The raw file is not committed (Kaggle terms, and it is 158 MB).

## Run it

```bash
git clone {GH}.git && cd loan-default-risk
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# place application_train.csv in data/raw/
PYTHONPATH=src python src/features.py     # build the modelling table
PYTHONPATH=src python src/train.py        # three variants, ~10 min on a laptop
PYTHONPATH=src python src/cutoff.py       # policy sweep and chart
PYTHONPATH=src python src/explain.py      # SHAP summary and worked examples
PYTHONPATH=src python src/make_readme.py  # regenerate this file from the artifacts
streamlit run app.py
```

## Limitations

- One lender, one country, one definition of default ("payment difficulties").
- Loss given default and margin are assumptions; the optimal policy moves with them.
- No proxy test yet: excluding gender does not prove the remaining features carry no signal
  about it.
- The app scores one applicant at a time; it is a demonstration of the decision, not a
  production scoring service.

---
Built by David Okeya · [LinkedIn](https://www.linkedin.com/in/okeyadavid)
"""
    (ROOT / "README.md").write_text(text)
    print(f"Wrote {ROOT / 'README.md'}")


if __name__ == "__main__":
    main()
