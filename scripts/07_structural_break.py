"""
07_structural_break.py
======================
Data-driven era discovery: let the event sequence determine where structural
breaks occur in the reconstitution effect, rather than imposing arbitrary
calendar cutpoints.

Three complementary methods:

  1. Sup-Wald (Quandt-Andrews) test
     For each candidate break date d (from 15th to 85th percentile of events),
     split the sample into before/after and compute an F-statistic for the null
     of equal mean CAR. The argmax gives the "most likely" break date;
     the test statistic is evaluated against asymptotic critical values.

  2. PELT (Pruned Exact Linear Time) via ruptures
     Minimises a penalised cost function (L2 / mean-shift model) over the ordered
     event sequence to detect the globally optimal number and placement of breaks.
     Run with penalty calibrated by BIC to avoid overfitting.

  3. Rolling window mean (visual)
     3-event and 5-event centred rolling average of car_ann_day, plotted against
     announcement date, with detected breakpoints overlaid.

All three are run on:
  (a) car_ann_day  — the variable most theoretically tied to the passive AUM hypothesis
  (b) peak_var_ratio — for the risk dimension

After identifying the data-driven break date, the script re-runs the
era comparison and outputs a side-by-side table vs. the fixed-era results.
"""

import numpy as np
import pandas as pd
import ruptures as rpt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path
from scipy import stats as sp_stats
import warnings
warnings.filterwarnings("ignore")

ROOT    = Path(__file__).parent.parent
RESULTS = ROOT / "results"
FIGS    = RESULTS / "figures"

# ── Load data ─────────────────────────────────────────────────────────────────
cars = pd.read_csv(RESULTS / "event_study_cars.csv")
var_ = pd.read_csv(RESULTS / "var_spikes.csv")

cars["announcement_date"] = pd.to_datetime(cars["announcement_date"])
var_["announcement_date"] = pd.to_datetime(var_["announcement_date"])

# Merge
df = cars.merge(
    var_[["announcement_date", "symbol", "peak_var_ratio", "ann_day_var_ratio"]],
    on=["announcement_date", "symbol"], how="left"
)
df = df.sort_values("announcement_date").reset_index(drop=True)
df["t"] = np.arange(len(df))   # integer time index (event order)
df["year_frac"] = (df["announcement_date"] - df["announcement_date"].min()).dt.days / 365.25

# For announcement-date level analysis (aggregate events per date)
ann_level = (df.groupby("announcement_date")
               .agg(mean_car_ann=("car_ann_day","mean"),
                    mean_car_post=("car_post_medium","mean"),
                    mean_var=("peak_var_ratio","mean"),
                    n=("symbol","count"))
               .reset_index()
               .sort_values("announcement_date"))
ann_level["t"] = np.arange(len(ann_level))

print(f"Events: {len(df)}  |  Announcement dates: {len(ann_level)}")
print()

# ── Helper ────────────────────────────────────────────────────────────────────

def ttest_two_groups(series, labels, break_idx):
    """t-test between series[:break_idx] and series[break_idx:]."""
    before = series[:break_idx].dropna()
    after  = series[break_idx:].dropna()
    if len(before) < 2 or len(after) < 2:
        return None, None, None, None
    t, p = sp_stats.ttest_ind(before, after, equal_var=False)
    return before.mean(), after.mean(), t, p


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 1: Sup-Wald (Quandt-Andrews) test on individual events
# ══════════════════════════════════════════════════════════════════════════════

def sup_wald(series: pd.Series, trim: float = 0.15) -> dict:
    """
    Quandt-Andrews Sup-Wald test for a single structural break in mean.
    Tests at every candidate break point between [trim, 1-trim] of the sample.
    Returns the break index, break date (if index is available), F-statistic,
    and a 95% critical value (Andrews 1993: ~8.85 for a mean-only model).
    """
    n   = len(series)
    lo  = int(np.ceil(trim * n))
    hi  = int(np.floor((1 - trim) * n))
    vals = series.values

    fstats = []
    indices = range(lo, hi + 1)

    for k in indices:
        b = vals[:k]
        a = vals[k:]
        if len(b) < 3 or len(a) < 3:
            fstats.append(0)
            continue
        # F-statistic = [RSS_restricted - RSS_unrestricted] / [RSS_unrestricted/(n-2)]
        # Restricted: single mean; Unrestricted: separate means
        rss_r = np.sum((vals - vals.mean()) ** 2)
        rss_u = np.sum((b - b.mean()) ** 2) + np.sum((a - a.mean()) ** 2)
        f = ((rss_r - rss_u) / 1) / (rss_u / (n - 2))
        fstats.append(f)

    sup_f   = max(fstats)
    best_k  = list(indices)[np.argmax(fstats)]
    # Andrews (1993) 5% critical value for single-break, single restriction: 8.85
    # (More conservative than standard F-critical because of data-snooping)
    critical_5pct = 8.85
    return {
        "sup_f":      round(sup_f, 3),
        "break_idx":  best_k,
        "significant": sup_f > critical_5pct,
        "crit_5pct":  critical_5pct,
        "f_series":   list(zip(indices, fstats)),
    }

print("=" * 65)
print("METHOD 1: Sup-Wald Test (Quandt-Andrews)")
print("=" * 65)

results_sw = {}
for action in ["Inclusion", "Exclusion", "All"]:
    if action == "All":
        sub = df
    else:
        sub = df[df.action == action]
    sub = sub.sort_values("announcement_date").reset_index(drop=True)
    series = sub["car_ann_day"]

    sw = sup_wald(series)
    break_date = sub.loc[sw["break_idx"], "announcement_date"]
    before_mean, after_mean, t, p = ttest_two_groups(
        series, None, sw["break_idx"])

    results_sw[action] = {**sw, "break_date": break_date,
                          "before_mean": before_mean, "after_mean": after_mean}

    sig_str = "YES (p<0.05 asymptotically)" if sw["significant"] else "No"
    print(f"\n  {action}:")
    print(f"    Sup-F statistic : {sw['sup_f']:.3f}  (5% critical = {sw['crit_5pct']})")
    print(f"    Structural break: {break_date.strftime('%Y-%m-%d')}  (event #{sw['break_idx']})")
    print(f"    Significant?    : {sig_str}")
    if before_mean is not None:
        stars = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.1 else ""
        print(f"    Before break: mean={before_mean*100:+.3f}%  "
              f"After break: mean={after_mean*100:+.3f}%  "
              f"t={t:.2f}  p={p:.3f}{stars}")

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 2: PELT (Bai-Perron) via ruptures
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("METHOD 2: PELT Change-Point Detection (BIC penalty)")
print("=" * 65)

results_pelt = {}
for variable, label in [("car_ann_day", "CAR[0,+1]"),
                         ("peak_var_ratio", "Peak VaR ratio")]:
    for action in ["Inclusion", "Exclusion"]:
        sub = (df[df.action == action]
               .sort_values("announcement_date")
               .reset_index(drop=True))
        signal = sub[variable].fillna(sub[variable].mean()).values.reshape(-1, 1)

        # PELT with rbf kernel (handles mean + variance shifts)
        # Penalty = BIC-style: log(n) * sigma²  where sigma estimated from data
        sigma2 = np.var(signal)
        penalty = np.log(len(signal)) * sigma2

        try:
            algo = rpt.Pelt(model="rbf", min_size=3, jump=1).fit(signal)
            bkps = algo.predict(pen=penalty)
            # bkps always includes n at end; remove it for display
            interior_bkps = [b for b in bkps if b < len(signal)]
        except Exception as e:
            interior_bkps = []

        results_pelt[f"{action}_{variable}"] = interior_bkps
        if interior_bkps:
            break_dates = [sub.loc[b - 1, "announcement_date"].strftime("%Y-%m-%d")
                           for b in interior_bkps if b <= len(sub)]
            print(f"\n  {action} / {label}:")
            print(f"    {len(interior_bkps)} break(s) at event(s) #{interior_bkps}")
            print(f"    → dates: {break_dates}")
        else:
            print(f"\n  {action} / {label}:  No break detected (penalty too high)")

# Also run on the announcement-level aggregates (15 points) — less noisy
print("\n  — Announcement-date level aggregates (N=15) —")
for variable, label in [("mean_car_ann", "Mean CAR[0,+1] per date"),
                         ("mean_var",     "Mean VaR ratio per date")]:
    signal = ann_level[variable].fillna(ann_level[variable].mean()).values.reshape(-1, 1)
    sigma2 = np.var(signal)
    penalty = np.log(len(signal)) * sigma2
    try:
        algo = rpt.Pelt(model="rbf", min_size=2, jump=1).fit(signal)
        bkps = algo.predict(pen=penalty)
        interior_bkps = [b for b in bkps if b < len(signal)]
    except Exception:
        interior_bkps = []

    if interior_bkps:
        break_dates = [ann_level.loc[b - 1, "announcement_date"].strftime("%Y-%m-%d")
                       for b in interior_bkps if b <= len(ann_level)]
        print(f"  {label}: {len(interior_bkps)} break(s) → {break_dates}")
    else:
        print(f"  {label}: No break detected")

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 3: Rolling window mean (visual)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("METHOD 3: Rolling window means → see figures")
print("=" * 65)

# ── Figure 11: Sup-Wald F-statistic trajectory + break date ──────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Structural Break Analysis — Data-Driven Era Discovery\n"
             "Nifty 50 Reconstitution 2015–2025", fontsize=12)

for col_idx, action in enumerate(["Inclusion", "Exclusion"]):
    sub    = df[df.action == action].sort_values("announcement_date").reset_index(drop=True)
    series = sub["car_ann_day"]
    dates  = sub["announcement_date"]
    sw     = results_sw[action]

    # ── Top row: rolling mean + Sup-Wald break ────────────────────────────────
    ax = axes[0][col_idx]
    # Individual events
    ax.scatter(dates, series * 100, s=25, alpha=0.55, color="#90a4ae", zorder=2)
    # Rolling 3-event centred mean
    roll3 = series.rolling(3, center=True).mean()
    roll5 = series.rolling(5, center=True).mean()
    ax.plot(dates, roll3 * 100, color="#1a237e", lw=1.8, label="3-event rolling mean", zorder=3)
    ax.plot(dates, roll5 * 100, color="#e53935", lw=1.5, ls="--",
            label="5-event rolling mean", alpha=0.8, zorder=3)

    # Sup-Wald break
    sw_date = sw["break_date"]
    ax.axvline(sw_date, color="darkorange", lw=1.8, ls="-.",
               label=f"Sup-Wald break: {sw_date.strftime('%Y-%m')}", zorder=4)
    ax.axhline(0, color="grey", lw=0.5, ls=":")
    ax.set_title(f"{action}s — CAR[0,+1] with rolling mean", fontsize=10)
    ax.set_xlabel("Announcement date")
    ax.set_ylabel("CAR (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # ── Bottom row: Sup-Wald F-statistic trajectory ───────────────────────────
    ax2 = axes[1][col_idx]
    f_idx, f_vals = zip(*sw["f_series"])
    f_dates = [dates.iloc[i] for i in f_idx if i < len(dates)]
    f_valid = f_vals[:len(f_dates)]
    ax2.plot(f_dates, f_valid, color="#1a237e", lw=1.8)
    ax2.axhline(sw["crit_5pct"], color="#e53935", lw=1.2, ls="--",
                label=f"5% critical value ({sw['crit_5pct']})")
    ax2.axvline(sw_date, color="darkorange", lw=1.8, ls="-.",
                label=f"argmax: {sw_date.strftime('%Y-%m')}")
    ax2.fill_between(f_dates, f_valid, sw["crit_5pct"],
                     where=[v > sw["crit_5pct"] for v in f_valid],
                     alpha=0.2, color="darkorange")
    ax2.set_title(f"{action}s — Sup-Wald F-statistic trajectory", fontsize=10)
    ax2.set_xlabel("Candidate break date")
    ax2.set_ylabel("F-statistic")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(FIGS / "fig11_structural_break.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig11_structural_break.png")

# ── Figure 12: PELT segmentation on both metrics ─────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("PELT Change-Point Detection on CAR and VaR\n"
             "Segment means shown in shaded bands", fontsize=12)

for row_idx, (variable, label, unit) in enumerate([
        ("car_ann_day",    "CAR[0,+1]",       "%"),
        ("peak_var_ratio", "Peak VaR ratio",   "×")]):
    for col_idx, action in enumerate(["Inclusion", "Exclusion"]):
        ax  = axes[row_idx][col_idx]
        sub = (df[df.action == action]
               .sort_values("announcement_date")
               .reset_index(drop=True))
        vals  = sub[variable].fillna(sub[variable].mean())
        dates = sub["announcement_date"]
        scale = 100 if unit == "%" else 1

        ax.scatter(dates, vals * scale, s=30, alpha=0.55, color="#90a4ae", zorder=2)

        bkps_key = f"{action}_{variable}"
        bkps     = results_pelt.get(bkps_key, [])
        segments = [0] + bkps + [len(vals)]
        seg_colors = ["#1a237e", "#e53935", "#388e3c", "#f57f17"]

        for i in range(len(segments) - 1):
            lo, hi = segments[i], segments[i + 1]
            seg_dates = dates.iloc[lo:hi]
            seg_vals  = vals.iloc[lo:hi]
            seg_mean  = seg_vals.mean() * scale
            c = seg_colors[i % len(seg_colors)]
            ax.hlines(seg_mean, seg_dates.min(), seg_dates.max(),
                      color=c, lw=2.5, zorder=4)
            ax.fill_between(seg_dates, seg_mean, 0 if unit == "%" else 1,
                            alpha=0.1, color=c)
            ax.text(seg_dates.mean(), seg_mean + 0.2 * scale,
                    f"μ={seg_mean:.2f}{unit}", ha="center", fontsize=8,
                    color=c, fontweight="bold")

            if i > 0:  # break line
                ax.axvline(dates.iloc[lo], color="darkorange", lw=1.5,
                           ls="-.", zorder=5,
                           label=f"Break: {dates.iloc[lo].strftime('%Y-%m')}")

        ax.axhline(0 if unit == "%" else 1, color="grey", lw=0.5, ls=":")
        ax.set_title(f"{action}s — {label}", fontsize=10)
        ax.set_xlabel("Announcement date")
        ax.set_ylabel(f"{label} ({unit})")
        if unit == "%":
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(FIGS / "fig12_pelt_segmentation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig12_pelt_segmentation.png")

# ══════════════════════════════════════════════════════════════════════════════
# RE-RUN ERA COMPARISON WITH DATA-DRIVEN BREAK DATE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("ERA COMPARISON: Fixed cutpoint vs. Data-Driven break")
print("=" * 65)

# Use the Inclusion break date as primary (it's the theoretically motivated one)
inc_break_date = results_sw["Inclusion"]["break_date"]
exc_break_date = results_sw["Exclusion"]["break_date"]
all_break_date = results_sw["All"]["break_date"]

print(f"\nDetected break dates:")
print(f"  Inclusions : {inc_break_date.strftime('%Y-%m-%d')}")
print(f"  Exclusions : {exc_break_date.strftime('%Y-%m-%d')}")
print(f"  All events : {all_break_date.strftime('%Y-%m-%d')}")

# Apply data-driven era label using the Inclusion break (most theory-grounded)
df["era_dynamic"] = df["announcement_date"].apply(
    lambda d: f"Before {inc_break_date.strftime('%Y-%m')}"
              if d < inc_break_date
              else f"After {inc_break_date.strftime('%Y-%m')}")

print(f"\n{'─'*65}")
print(f"  Data-Driven Era Comparison (break = {inc_break_date.strftime('%Y-%m')})")
print(f"{'─'*65}")
print(f"\n  {'Group':<40} {'N':>4} {'mean CAR[0,+1]':>14}  {'t':>6}  {'p':>6}")
print(f"  {'-'*70}")

for action in ["Inclusion", "Exclusion"]:
    for era_label in sorted(df["era_dynamic"].unique()):
        sub   = df[(df.action == action) & (df.era_dynamic == era_label)]
        vals  = sub["car_ann_day"].dropna()
        if len(vals) < 2:
            continue
        t, p  = sp_stats.ttest_1samp(vals, 0)
        stars = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.1 else ""
        print(f"  {action:<12} {era_label:<28} {len(vals):>4} "
              f"{vals.mean()*100:>+12.3f}%  {t:>6.2f}  {p:>6.3f}{stars}")

print(f"\n{'─'*65}")
print(f"  Fixed-Era Comparison (break = 2018-01, original specification)")
print(f"{'─'*65}")
print(f"\n  {'Group':<40} {'N':>4} {'mean CAR[0,+1]':>14}  {'t':>6}  {'p':>6}")
print(f"  {'-'*70}")

era_map = {"Era1_2015-2017": "Era 1 (2015-2017)", "Era2_2018-2025": "Era 2 (2018-2025)"}
for action in ["Inclusion", "Exclusion"]:
    for era_key, era_label in era_map.items():
        sub  = df[(df.action == action) & (df.era == era_key)]
        vals = sub["car_ann_day"].dropna()
        if len(vals) < 2:
            continue
        t, p = sp_stats.ttest_1samp(vals, 0)
        stars = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.1 else ""
        print(f"  {action:<12} {era_label:<28} {len(vals):>4} "
              f"{vals.mean()*100:>+12.3f}%  {t:>6.2f}  {p:>6.3f}{stars}")

# ── Figure 13: Side-by-side era comparison ───────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Fixed vs. Data-Driven Era Split — CAR[0,+1] Comparison\n"
             "Does the break date choice change the findings?", fontsize=12)

dynamic_eras  = sorted(df["era_dynamic"].unique())
dynamic_color = ["#2196F3", "#FF5722"]

for col_idx, action in enumerate(["Inclusion", "Exclusion"]):
    ax   = axes[col_idx]
    sub  = df[df.action == action]
    x    = np.arange(2)
    w    = 0.35

    # Fixed eras
    fixed_means = []
    fixed_sems  = []
    for era_key in ["Era1_2015-2017", "Era2_2018-2025"]:
        v = sub[sub.era == era_key]["car_ann_day"].dropna() * 100
        fixed_means.append(v.mean()); fixed_sems.append(v.sem())

    # Dynamic eras
    dyn_means = []
    dyn_sems  = []
    for era_lbl in dynamic_eras:
        v = sub[sub.era_dynamic == era_lbl]["car_ann_day"].dropna() * 100
        dyn_means.append(v.mean()); dyn_sems.append(v.sem())

    b1 = ax.bar(x - w/2, fixed_means, w, label="Fixed eras (2018 cutpoint)",
                color=["#1a237e","#b71c1c"], alpha=0.75,
                yerr=fixed_sems, capsize=4, error_kw={"elinewidth":1.2})
    b2 = ax.bar(x + w/2, dyn_means[:2], w,
                label=f"Data-driven ({inc_break_date.strftime('%Y-%m')} break)",
                color=["#42a5f5","#ef5350"], alpha=0.75,
                yerr=dyn_sems[:2], capsize=4, error_kw={"elinewidth":1.2})

    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["Before break", "After break"], fontsize=9)
    ax.set_title(f"{action}s", fontsize=11)
    ax.set_ylabel("Mean CAR[0,+1] (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=2))
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(FIGS / "fig13_era_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved fig13_era_comparison.png")
print("\nStructural break analysis complete.")
