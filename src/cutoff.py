"""The approval cut-off decision: what each policy approves, loses and earns.

Run:  python src/cutoff.py [variant]      # default: lightgbm_age

The model outputs a probability; the lender still has to choose where to cut. This sweeps
the approval rate from 40% to 100% on the HELD-OUT test applicants and reports, for each:

  threshold          the predicted PD of the last applicant approved
  bad_rate_approved  share of approved loans that actually defaulted
  expected_loss      realised: defaulted approved balances x LGD
  revenue            performing approved balances x MARGIN
  profit             revenue - expected_loss

Losses use realised test outcomes, not predicted probabilities, so the model cannot
flatter itself. LGD and MARGIN are stated assumptions (config.py), printed on the chart.
"""
from __future__ import annotations

import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config import BREAK_EVEN_PD, DOCS, LGD, MARGIN, PROCESSED  # noqa: E402

INK, BLUE, ORANGE, GRID = "#3d3c39", "#2a78d6", "#eb6834", "#d8d7d2"


def load(variant: str) -> pd.DataFrame:
    path = PROCESSED / f"test_predictions_{variant}.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found - run `python src/train.py` first.")
    return pd.read_csv(path)


def sweep(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rate in range(40, 101):
        thr = df["proba"].quantile(rate / 100)
        appr = df[df["proba"] <= thr]
        bad = appr[appr["y_true"] == 1]
        good = appr[appr["y_true"] == 0]
        loss = float((bad["amt_credit"] * LGD).sum())
        rev = float((good["amt_credit"] * MARGIN).sum())
        rows.append({
            "approval_rate": rate,
            "threshold": round(float(thr), 4),
            "approved": len(appr),
            "bad_rate_approved": round(100 * len(bad) / len(appr), 3),
            "defaults_approved": len(bad),
            "expected_loss": round(loss),
            "revenue": round(rev),
            "profit": round(rev - loss),
        })
    return pd.DataFrame(rows)


def chart(t: pd.DataFrame, variant: str, best: pd.Series) -> None:
    fig, ax = plt.subplots(3, 1, figsize=(9, 9.5), sharex=True)
    x = t["approval_rate"]

    ax[0].plot(x, t["bad_rate_approved"], color=BLUE, lw=2)
    ax[0].set_ylabel("Defaults among approved (%)")
    ax[0].set_title("Where to draw the approval line", loc="left", fontsize=14, color=INK)

    ax[1].plot(x, t["expected_loss"] / 1e6, color=ORANGE, lw=2)
    ax[1].set_ylabel("Expected loss ($m)")

    ax[2].plot(x, t["profit"] / 1e6, color=INK, lw=2)
    ax[2].set_ylabel("Contribution ($m)")
    ax[2].set_xlabel("Share of applicants approved (%)")

    for a in ax:
        a.axvline(best["approval_rate"], color=BLUE, ls="--", lw=1, alpha=0.7)
        a.grid(alpha=0.25, color=GRID)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    ax[2].annotate(
        f"most profitable: approve {best['approval_rate']:.0f}%\n"
        f"(cut-off PD {best['threshold']:.3f}, bad rate {best['bad_rate_approved']:.2f}%)",
        xy=(best["approval_rate"], best["profit"] / 1e6),
        xytext=(-200, 18), textcoords="offset points", color=INK, fontsize=9,
        arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))

    fig.text(0.01, 0.006,
             f"Held-out test applicants, model {variant}. Assumes loss given default "
             f"{LGD:.0%} and lifetime margin {MARGIN:.0%}; break-even PD {BREAK_EVEN_PD:.1%}.",
             fontsize=8, color="#6b6a66")
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    out = DOCS / f"cutoff_{variant}.png"
    fig.savefig(out, dpi=150)
    print(f"Chart -> {out}")


def main(variant: str) -> None:
    df = load(variant)
    t = sweep(df)
    t.to_csv(PROCESSED / f"cutoff_{variant}.csv", index=False)

    best = t.loc[t["profit"].idxmax()]
    approve_all = t[t["approval_rate"] == 100].iloc[0]

    print(f"\nCut-off sweep ({variant}; LGD {LGD:.0%}, margin {MARGIN:.0%}, "
          f"break-even PD {BREAK_EVEN_PD:.1%})\n")
    print(t[t["approval_rate"].isin([50, 60, 70, 80, 85, 90, 95, 100])].to_string(index=False))

    print(f"\nMost profitable policy: approve {best['approval_rate']:.0f}% "
          f"(cut-off PD {best['threshold']:.3f})")
    print(f"  bad rate among approved {best['bad_rate_approved']:.2f}% "
          f"vs {approve_all['bad_rate_approved']:.2f}% if everyone is approved")
    print(f"  expected loss ${best['expected_loss']:,.0f} vs ${approve_all['expected_loss']:,.0f}"
          f"  (avoided ${approve_all['expected_loss'] - best['expected_loss']:,.0f})")
    print(f"  contribution ${best['profit']:,.0f} vs ${approve_all['profit']:,.0f}")
    chart(t, variant, best)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lightgbm_noage")
