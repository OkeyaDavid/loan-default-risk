"""Loan Default Risk & Approval Tool - Streamlit app.

Run locally:   streamlit run app.py
Deployed on Streamlit Community Cloud; everything it needs is committed to the repo
(models/model.pkl, data/processed/app_sample.csv, data/processed/cutoff_*.csv).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from config import BREAK_EVEN_PD, LGD, MARGIN, TARGET  # noqa: E402
from labels import pretty  # noqa: E402

VARIANT = "lightgbm_noage"
INK, BLUE, ORANGE, GREEN, RED = "#3d3c39", "#2a78d6", "#eb6834", "#1baf7a", "#e34948"

st.set_page_config(page_title="Loan Default Risk & Approval Tool", page_icon="📊",
                   layout="wide")


@st.cache_resource
def load_model():
    """Loaded on first use, not at boot: nothing on the landing tab needs the model."""
    return joblib.load(ROOT / "models" / "model.pkl")


@st.cache_data
def load_tables():
    cut = pd.read_csv(ROOT / "data" / "processed" / f"cutoff_{VARIANT}.csv")
    sample = pd.read_csv(ROOT / "data" / "processed" / "app_sample.csv")
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text())
    return cut, sample, metrics


st.title("Loan Default Risk & Approval Tool")

try:
    cut, sample, metrics = load_tables()
except Exception as exc:                      # noqa: BLE001
    st.error("The app could not load its committed artifacts (model metrics, cut-off table "
             f"or applicant sample).\n\n**{type(exc).__name__}:** {exc}")
    st.stop()

best = cut.loc[cut["profit"].idxmax()]

# Scores were computed at training time and stored with the sample, so booting the app is
# three file reads: no model, no SHAP, no inference until someone edits an applicant.
scores = sample["pd_hat"]
median_idx = int(scores.sort_values().index[len(scores) // 2])
low_idx, high_idx = int(scores.idxmin()), int(scores.idxmax())
FEATURES = [c for c in sample.columns if c not in (TARGET, "pd_hat")]

st.caption(
    f"Predicts the probability that a loan applicant defaults, explains each decision, and "
    f"shows what each approval policy costs. LightGBM on ~307k Home Credit applications; "
    f"held-out ROC-AUC {metrics['roc_auc']:.3f}. Loss given default {LGD:.0%}, "
    f"margin {MARGIN:.0%}, so the break-even probability is {BREAK_EVEN_PD:.1%}."
)

# ---------------------------------------------------------------- policy control
st.sidebar.header("Approval policy")
st.sidebar.write("Move the line between approve and decline.")
approval_rate = st.sidebar.slider(
    "Share of applicants approved (%)", int(cut["approval_rate"].min()), 100,
    int(best["approval_rate"]), 1,
    help="The most profitable policy under the stated assumptions is preselected.")
row = cut.loc[cut["approval_rate"] == approval_rate].iloc[0]
threshold = float(row["threshold"])
st.sidebar.metric("Cut-off: decline above this PD", f"{threshold:.3f}")
st.sidebar.metric("Defaults among approved", f"{row['bad_rate_approved']:.2f}%")
st.sidebar.metric("Expected loss", f"${row['expected_loss']/1e6:,.1f}m")
st.sidebar.metric("Contribution", f"${row['profit']/1e6:,.1f}m")
if approval_rate != int(best["approval_rate"]):
    delta = row["profit"] - best["profit"]
    st.sidebar.caption(f"{delta/1e6:+,.1f}m against the most profitable policy "
                       f"({best['approval_rate']:.0f}% approved).")

tab_score, tab_policy, tab_card = st.tabs(
    ["Score an applicant", "Approval policy", "Model card"])

# ---------------------------------------------------------------- score one applicant
with tab_score:
    left, right = st.columns([1, 1.15])
    with left:
        st.subheader("Applicant")
        presets = {f"Typical applicant (median risk, {scores[median_idx]:.1%})": median_idx,
                   f"Lowest risk in the sample ({scores[low_idx]:.1%})": low_idx,
                   f"Highest risk in the sample ({scores[high_idx]:.1%})": high_idx}
        presets.update({f"Applicant {i} ({scores[i]:.1%})": int(i)
                        for i in scores.sort_values().index[::40][:20]})
        preset = st.selectbox("Start from an example applicant", list(presets),
                              help="Synthetic applicants drawn from the held-out "
                                   "distribution. Kaggle's rules do not allow republishing "
                                   "the real records, so none are in this repository.")
        base = sample[FEATURES].loc[presets[preset]]
        template = sample[FEATURES].iloc[0].copy()
        for c in template.index:
            if c in base.index and pd.notna(base[c]):
                template[c] = base[c]

        c1, c2 = st.columns(2)
        template["AMT_INCOME_TOTAL"] = c1.number_input(
            "Annual income", 20_000, 2_000_000, int(float(template["AMT_INCOME_TOTAL"])), 5_000)
        template["AMT_CREDIT"] = c2.number_input(
            "Loan amount", 20_000, 4_000_000, int(float(template["AMT_CREDIT"])), 10_000)
        template["AMT_ANNUITY"] = c1.number_input(
            "Annual payment", 2_000, 300_000, int(float(template["AMT_ANNUITY"])), 1_000)
        template["AMT_GOODS_PRICE"] = c2.number_input(
            "Price of the purchase", 20_000, 4_000_000,
            int(float(template["AMT_GOODS_PRICE"])), 10_000)
        emp = c1.slider("Years in current job (0 = unemployed)", 0.0, 40.0,
                        float(np.nan_to_num(template["EMPLOYED_YEARS"], nan=0.0)), 0.1)
        c2.markdown("&nbsp;", unsafe_allow_html=True)
        c2.caption("No age or gender input: both are protected attributes and neither is in "
                   "the shipped model. See the model card.")
        template["EMPLOYED_YEARS"] = np.nan if emp == 0 else emp
        template["IS_UNEMPLOYED"] = int(emp == 0)
        st.markdown("**External credit scores** (0 = weak, 1 = strong)")
        e1, e2, e3 = st.columns(3)
        for i, col in zip((1, 2, 3), (e1, e2, e3)):
            cur = template.get(f"EXT_SOURCE_{i}")
            template[f"EXT_SOURCE_{i}"] = col.slider(
                f"Score {i}", 0.0, 1.0, float(cur) if pd.notna(cur) else 0.5, 0.01)

        # keep engineered ratios consistent with the edited inputs
        inc = max(float(template["AMT_INCOME_TOTAL"]), 1.0)
        template["CREDIT_TO_INCOME"] = float(template["AMT_CREDIT"]) / inc
        template["ANNUITY_TO_INCOME"] = float(template["AMT_ANNUITY"]) / inc
        template["CREDIT_TO_GOODS"] = float(template["AMT_CREDIT"]) / max(
            float(template["AMT_GOODS_PRICE"]), 1.0)
        template["CREDIT_TERM_YEARS"] = float(template["AMT_CREDIT"]) / max(
            float(template["AMT_ANNUITY"]), 1.0)
        fam = float(template.get("CNT_FAM_MEMBERS") or 1) or 1
        template["INCOME_PER_PERSON"] = inc / fam
        exts = [float(template[f"EXT_SOURCE_{i}"]) for i in (1, 2, 3)]
        template["EXT_SOURCE_MEAN"] = float(np.mean(exts))
        template["EXT_SOURCE_MIN"] = float(np.min(exts))

    X_one = pd.DataFrame([template])[FEATURES]
    pipe = load_model()                      # first model touch happens here
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    pd_hat = float(pipe.predict_proba(X_one)[0, 1])
    approve = pd_hat <= threshold

    with right:
        st.subheader("Decision")
        a, b = st.columns(2)
        a.metric("Probability of default", f"{pd_hat:.1%}")
        b.metric("Decision at this cut-off", "Approve" if approve else "Decline")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=pd_hat * 100,
            number={"suffix": "%", "font": {"size": 34, "color": INK}},
            gauge={"axis": {"range": [0, max(40, pd_hat * 120)], "ticksuffix": "%"},
                   "bar": {"color": GREEN if approve else RED, "thickness": 0.7},
                   "threshold": {"line": {"color": INK, "width": 3}, "value": threshold * 100},
                   "steps": [{"range": [0, threshold * 100], "color": "#eaf1fb"}]}))
        gauge.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=0))
        st.plotly_chart(gauge, width="stretch")
        st.caption(f"The dark line is the current cut-off, {threshold:.1%}. "
                   f"Break-even on the stated economics is {BREAK_EVEN_PD:.1%}.")

        st.subheader("Why")
        import shap                          # imported here: ~2s, and only on this tab
        Xt = prep.transform(X_one)
        names = prep.get_feature_names_out()
        sv = shap.TreeExplainer(model).shap_values(pd.DataFrame(Xt, columns=names))
        if isinstance(sv, list):
            sv = sv[1]
        contrib = pd.Series(np.asarray(sv).reshape(-1), index=names)
        ranked = contrib.reindex(contrib.abs().sort_values(ascending=False).index)
        up = [(pretty(i), v) for i, v in ranked.items() if v > 0][:3]
        down = [(pretty(i), v) for i, v in ranked.items() if v < 0][:3]
        u, d = st.columns(2)
        u.markdown("**Raised the risk**")
        for label, v in up:
            u.write(f"- {label}  `+{v:.3f}`")
        d.markdown("**Lowered the risk**")
        for label, v in down:
            d.write(f"- {label}  `{v:.3f}`")
        st.caption("SHAP contributions in log-odds. These are the reasons an adverse-action "
                   "notice would have to give a declined applicant.")

# ---------------------------------------------------------------- policy curve
with tab_policy:
    st.subheader("What each approval policy costs")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Approved", f"{approval_rate}%")
    k2.metric("Defaults among approved", f"{row['bad_rate_approved']:.2f}%")
    k3.metric("Expected loss", f"${row['expected_loss']/1e6:,.1f}m")
    k4.metric("Contribution", f"${row['profit']/1e6:,.1f}m")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cut["approval_rate"], y=cut["profit"] / 1e6, name="Contribution",
                             line=dict(color=INK, width=2.5),
                             hovertemplate="approve %{x}%: $%{y:.1f}m<extra></extra>"))
    fig.add_trace(go.Scatter(x=cut["approval_rate"], y=cut["expected_loss"] / 1e6,
                             name="Expected loss", line=dict(color=ORANGE, width=2),
                             hovertemplate="approve %{x}%: $%{y:.1f}m<extra></extra>"))
    fig.add_vline(x=approval_rate, line_dash="dash", line_color=BLUE)
    fig.update_layout(height=380, hovermode="x unified", yaxis_title="$m (test applicants)",
                      xaxis_title="Share of applicants approved (%)",
                      legend=dict(orientation="h", y=-0.2),
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")

    fig2 = go.Figure(go.Scatter(x=cut["approval_rate"], y=cut["bad_rate_approved"],
                                line=dict(color=BLUE, width=2.5),
                                hovertemplate="approve %{x}%: %{y:.2f}% default<extra></extra>"))
    fig2.add_vline(x=approval_rate, line_dash="dash", line_color=BLUE)
    fig2.update_layout(height=300, yaxis_title="Defaults among approved (%)",
                       xaxis_title="Share of applicants approved (%)",
                       margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig2, width="stretch")
    st.caption(f"Realised outcomes on held-out applicants. Loss = defaulted balance x "
               f"{LGD:.0%}; contribution = performing balance x {MARGIN:.0%} minus loss. "
               "Change those two assumptions in src/config.py and the whole curve moves.")
    with st.expander("See the table"):
        st.dataframe(cut, width="stretch", hide_index=True)

# ---------------------------------------------------------------- model card
with tab_card:
    st.subheader("Model card")
    st.markdown(f"""
**Data.** Home Credit Default Risk, `application_train.csv` only: {metrics['n_train']:,}
training and {metrics['n_test']:,} held-out applications, default rate about 8%.

**Model.** LightGBM, {len(FEATURES)} source features (~25 columns before encoding). No resampling and no class weights, so predicted probabilities stay calibrated and
the dollar figures above mean what they say. The imbalance is handled by choosing the
threshold, which is the slider in the sidebar.

| Measure | Value | How to read it |
|---|---|---|
| ROC-AUC | {metrics['roc_auc']:.4f} | Chance a random defaulter scores above a random non-defaulter |
| PR-AUC | {metrics['pr_auc']:.4f} | Against a base rate of about 0.08 |
| Brier score | {metrics['brier']:.5f} | Squared error of the probabilities; lower is better |
| Cross-validated ROC-AUC | {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f} | Close to the test score, so no sign of overfitting |
| Worst calibration gap | {metrics['calibration_max_gap']:.4f} | Largest gap between predicted and actual in 10 risk deciles |

**Excluded attributes.** `CODE_GENDER` is excluded from every variant: sex is a prohibited
ground under the Canadian Human Rights Act and US ECOA. `SK_ID_CURR` is an identifier.

**Age.** Age is also protected and is genuinely predictive, so both variants were trained
before deciding. Keeping age would have bought {abs(metrics['age_cost_roc_auc']):.4f} ROC-AUC
— about 0.2% of contribution on this book. That is not worth putting a protected attribute
into a lending decision, so the shipped model is `lightgbm_noage` and the age variant stays
in `train.py` as the evidence.

**Data handling.** The model is trained on the Kaggle Home Credit data, but no application
records are published here. Kaggle's competition rules do not permit redistributing them, so
the applicants in this app are synthetic: drawn column-wise from the held-out distribution
within risk deciles and jittered, then checked against every real row for collisions
(`src/make_app_sample.py`). The policy curve and all metrics come from the real held-out
applicants and are aggregates.

**Limitations.** One lender, one country, one definition of default. Only the main
application table is used, so published scores using all nine tables run higher. Loss given
default and margin are assumptions, not measurements. No check yet for proxy variables that
could stand in for the excluded attributes.
""")
    st.caption("Built by David Okeya · [GitHub](https://github.com/OkeyaDavid/loan-default-risk) "
               "· Data: Kaggle Home Credit Default Risk")
