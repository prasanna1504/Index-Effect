"""
08_aum_regressor.py
===================
Tests the passive AUM mechanism directly: regresses CAR[0,+1] and peak VaR
on log(passive AUM) as a continuous variable, replacing the discrete era dummies.

This is methodologically superior to era dummies because:
  1. Uses all 47 events jointly — no arbitrary cutpoint
  2. Directly tests the passive AUM mechanism
  3. Gives an interpretable coefficient: "a 10× increase in AUM is associated
     with X bp change in announcement-day CAR"

Model:
  CAR[0,+1]     = β0 + β1·log(AUM) + β2·Inclusion + β3·log(AUM)×Inclusion + ε
  peak_var_ratio = β0 + β1·log(AUM) + ε

AUM is mapped to each event via the most recent quarter-end before announcement.

Outputs:
  results/aum_regression.csv       — event-level with AUM attached
  results/aum_regression_stats.txt — regression output
  results/figures/fig14_aum_car.png
  results/figures/fig15_aum_var.png
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path
from scipy import stats as sp_stats
import warnings
warnings.filterwarnings("ignore")

ROOT    = Path(__file__).parent.parent
DATA    = ROOT / "data"
RESULTS = ROOT / "results"
FIGS    = RESULTS / "figures"

# ── Load data ─────────────────────────────────────────────────────────────────

aum = pd.read_csv(DATA / "passive_aum.csv", parse_dates=["date"])
aum = aum.sort_values("date").reset_index(drop=True)

cars = pd.read_csv(RESULTS / "event_study_cars.csv")
var_ = pd.read_csv(RESULTS / "var_spikes.csv")

cars["announcement_date"] = pd.to_datetime(cars["announcement_date"])
var_["announcement_date"] = pd.to_datetime(var_["announcement_date"])

df = cars.merge(
    var_[["announcement_date","symbol","peak_var_ratio"]],
    on=["announcement_date","symbol"], how="left"
)
df = df.sort_values("announcement_date").reset_index(drop=True)

# ── Map AUM to each event (most recent quarter-end before announcement) ───────

def get_aum(dt: pd.Timestamp) -> float | None:
    prior = aum[aum["date"] <= dt]
    if len(prior) == 0:
        # Use earliest available AUM for events before first quarter-end
        return float(aum.iloc[0]["passive_aum_cr"])
    return float(prior.iloc[-1]["passive_aum_cr"])

df["aum_cr"]     = df["announcement_date"].apply(get_aum)
df["log_aum"]    = np.log(df["aum_cr"])
df["aum_lakh_cr"] = df["aum_cr"] / 100_000   # in lakh crore for display
df["inclusion"]  = (df["action"] == "Inclusion").astype(int)

df.to_csv(RESULTS / "aum_regression.csv", index=False)
print(f"Saved {len(df)} events with AUM → results/aum_regression.csv\n")

# ── Descriptive: AUM at each event ───────────────────────────────────────────

print("AUM at each announcement date:")
for dt, aum_val in df.groupby("announcement_date")["aum_cr"].first().items():
    n = len(df[df.announcement_date == dt])
    print(f"  {dt.strftime('%Y-%m-%d')}  ₹{aum_val/100000:.2f}L Cr  ({n} events)")

# ── Regression 1: CAR[0,+1] ~ log(AUM) ───────────────────────────────────────

print("\n" + "="*60)
print("REGRESSION 1: CAR[0,+1] ~ log(AUM) × action")
print("="*60)

# Separate regressions by action (cleaner interpretation)
stat_lines = []
for action in ["Inclusion", "Exclusion"]:
    sub = df[df.action == action].dropna(subset=["car_ann_day", "log_aum"])
    mod = smf.ols("car_ann_day ~ log_aum", data=sub).fit()

    beta = mod.params["log_aum"]
    # Economic interpretation: 10× AUM increase → how many bp change in CAR?
    bp_per_10x = beta * np.log(10) * 100 * 100  # in basis points

    print(f"\n  {action}s (N={len(sub)}):")
    print(f"    Intercept: {mod.params['Intercept']:+.4f}  "
          f"t={mod.tvalues['Intercept']:+.2f}  p={mod.pvalues['Intercept']:.3f}")
    print(f"    log(AUM):  {beta:+.6f}  "
          f"t={mod.tvalues['log_aum']:+.2f}  p={mod.pvalues['log_aum']:.3f}"
          + ("***" if mod.pvalues['log_aum']<0.01
             else "**" if mod.pvalues['log_aum']<0.05
             else "*"  if mod.pvalues['log_aum']<0.1 else ""))
    print(f"    R² = {mod.rsquared:.3f}")
    print(f"    Economic magnitude: 10× AUM → {bp_per_10x:+.1f} bp change in CAR[0,+1]")

    stat_lines.append(f"{action}: log(AUM) coef={beta:+.6f}  "
                      f"t={mod.tvalues['log_aum']:+.2f}  p={mod.pvalues['log_aum']:.3f}  "
                      f"R²={mod.rsquared:.3f}  [10× AUM → {bp_per_10x:+.1f}bp]")

# Pooled with interaction
sub_all = df.dropna(subset=["car_ann_day","log_aum"])
mod_pool = smf.ols("car_ann_day ~ log_aum + inclusion + log_aum:inclusion",
                   data=sub_all).fit()
print(f"\n  Pooled (N={len(sub_all)}) with Inclusion interaction:")
for term in mod_pool.params.index:
    p = mod_pool.pvalues[term]
    stars = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.1 else ""
    print(f"    {term:<30} {mod_pool.params[term]:+.6f}  "
          f"t={mod_pool.tvalues[term]:+.2f}  p={p:.3f}{stars}")
print(f"    R² = {mod_pool.rsquared:.3f}")

# ── Regression 2: peak_var_ratio ~ log(AUM) ──────────────────────────────────

print("\n" + "="*60)
print("REGRESSION 2: peak_var_ratio ~ log(AUM)")
print("="*60)

for action in ["Inclusion", "Exclusion"]:
    sub = df[df.action == action].dropna(subset=["peak_var_ratio","log_aum"])
    mod = smf.ols("peak_var_ratio ~ log_aum", data=sub).fit()
    print(f"\n  {action}s (N={len(sub)}):")
    print(f"    log(AUM): {mod.params['log_aum']:+.6f}  "
          f"t={mod.tvalues['log_aum']:+.2f}  p={mod.pvalues['log_aum']:.3f}"
          + ("***" if mod.pvalues['log_aum']<0.01
             else "**" if mod.pvalues['log_aum']<0.05
             else "*"  if mod.pvalues['log_aum']<0.1 else ""))
    print(f"    R² = {mod.rsquared:.3f}")

# ── Figure 14: CAR[0,+1] vs log(AUM) ─────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
fig.suptitle("Announcement-Day CAR vs. Passive AUM\n"
             "Testing the passive mechanism directly (continuous variable)",
             fontsize=12)

action_color = {"Inclusion": "#1f77b4", "Exclusion": "#d62728"}

for ax, action in zip(axes, ["Inclusion", "Exclusion"]):
    sub = df[df.action == action].dropna(subset=["car_ann_day","aum_lakh_cr"])

    # Scatter coloured by era
    era_color_map = {"Era1_2015-2017": "#42a5f5", "Era2_2018-2025": "#ef5350"}
    for era, grp in sub.groupby("era"):
        ax.scatter(grp["aum_lakh_cr"], grp["car_ann_day"] * 100,
                   s=60, alpha=0.75, color=era_color_map.get(era, "grey"),
                   edgecolors="white", linewidth=0.5,
                   label=era.replace("Era1_2015-2017","Era 1 (2015–17)")
                            .replace("Era2_2018-2025","Era 2 (2018–25)"),
                   zorder=3)

    # Regression line on log scale
    sub2 = sub.dropna(subset=["log_aum"])
    if len(sub2) > 3:
        mod = smf.ols("car_ann_day ~ log_aum", data=sub2).fit()
        x_log = np.linspace(sub2["log_aum"].min(), sub2["log_aum"].max(), 100)
        y_fit = (mod.params["Intercept"] + mod.params["log_aum"] * x_log) * 100
        x_lakh = np.exp(x_log) / 100_000
        p = mod.pvalues["log_aum"]
        stars = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.1 else ""
        ax.plot(x_lakh, y_fit, color="black", lw=1.8, ls="--", alpha=0.7,
                label=f"OLS (log AUM): r²={mod.rsquared:.2f}{stars}", zorder=4)

    ax.axhline(0, color="grey", lw=0.5, ls=":")
    ax.set_xlabel("Passive AUM (₹ Lakh Crore)")
    ax.set_ylabel("CAR[0,+1] (%)")
    ax.set_title(f"{action}s", fontsize=11)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25)

    # Secondary x-axis labels for key AUM milestones
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    milestones = [0.5, 1.0, 2.0, 5.0, 10.0]
    ax2.set_xticks(milestones)
    ax2.set_xticklabels([f"₹{m:.0f}L Cr" for m in milestones], fontsize=7)
    ax2.tick_params(axis="x", length=3)

plt.tight_layout()
plt.savefig(FIGS / "fig14_aum_car.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved fig14_aum_car.png")

# ── Figure 15: AUM timeline + CAR trajectory ──────────────────────────────────

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=False)
fig.suptitle("Passive AUM Growth and the Compression of the Index Effect\n"
             "2015–2025", fontsize=12)

# Top: AUM growth
ax1.fill_between(aum["date"], aum["passive_aum_cr"] / 100_000, alpha=0.3,
                 color="#1a237e")
ax1.plot(aum["date"], aum["passive_aum_cr"] / 100_000, color="#1a237e", lw=2)
ax1.set_ylabel("Passive AUM (₹ Lakh Crore)")
ax1.set_title("India Passive Equity AUM (Index Funds + ETFs)", fontsize=10)
ax1.grid(alpha=0.25)
ax1.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"₹{x:.1f}L Cr"))

# Mark event dates on AUM chart
inc_events = df[df.action == "Inclusion"].drop_duplicates("announcement_date")
exc_events = df[df.action == "Exclusion"].drop_duplicates("announcement_date")
for dt in inc_events["announcement_date"]:
    ax1.axvline(dt, color="#1f77b4", lw=0.7, alpha=0.5, ls="-")
for dt in exc_events["announcement_date"]:
    ax1.axvline(dt, color="#d62728", lw=0.7, alpha=0.5, ls="-")
ax1.plot([], [], color="#1f77b4", lw=1.5, label="Inclusion events")
ax1.plot([], [], color="#d62728", lw=1.5, label="Exclusion events")
ax1.legend(fontsize=9)

# Bottom: event-level CAR[0,+1] scatter
by_date = (df.groupby(["announcement_date","action"])["car_ann_day"]
             .mean().reset_index())
for action, col in [("Inclusion","#1f77b4"), ("Exclusion","#d62728")]:
    sub = by_date[by_date.action==action]
    ax2.scatter(sub["announcement_date"], sub["car_ann_day"]*100,
                s=80, color=col, alpha=0.85, label=action,
                edgecolors="white", linewidth=0.5, zorder=3)
    # Rolling mean
    if len(sub) >= 3:
        roll = sub.set_index("announcement_date")["car_ann_day"].rolling(3, center=True).mean()
        ax2.plot(roll.index, roll*100, color=col, lw=1.5, alpha=0.6)

ax2.axhline(0, color="grey", lw=0.5, ls=":")
ax2.set_ylabel("Mean CAR[0,+1] per announcement date (%)")
ax2.set_title("Announcement-Day CAR (rolling 3-event mean)", fontsize=10)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
ax2.legend(fontsize=9)
ax2.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(FIGS / "fig15_aum_timeline.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig15_aum_timeline.png")
print("\nAUM regressor complete.")
