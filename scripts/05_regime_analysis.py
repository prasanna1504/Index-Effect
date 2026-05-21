"""
05_regime_analysis.py
=====================
Combines the event study CAR results and VaR spike data with market regime labels
(crisis vs calm) and runs the cross-sectional regressions.

Crisis windows (announcement date falls in):
  GFC:        2008-09-01 – 2009-06-30
  COVID:      2020-01-01 – 2020-09-30
  Rates:      2022-01-01 – 2022-12-31

Analyses:
  1. CAR × regime:   Do returns differ in crisis vs calm?
  2. VaR spike × regime × era:
       OLS: peak_var_ratio = β0 + β1*Era2 + β2*Crisis + β3*(Era2×Crisis) + ε
  3. Abnormal volume: avol_ratio = volume / avg_volume[-30,-1] on announcement day
     (serves as the mechanism: is forced turnover elevated?)

Outputs:
  results/regime_analysis.csv
  results/figures/fig8_crisis_var_comparison.png
  results/figures/fig9_abnormal_volume.png
  results/figures/fig10_regression_summary.png
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
PRICES  = ROOT / "data" / "prices"
RESULTS = ROOT / "results"
FIGS    = RESULTS / "figures"

CRISIS_WINDOWS = [
    ("GFC",    pd.Timestamp("2008-09-01"), pd.Timestamp("2009-06-30")),
    ("COVID",  pd.Timestamp("2020-01-01"), pd.Timestamp("2020-09-30")),
    ("Rates",  pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
]

def is_crisis(dt: pd.Timestamp) -> tuple[bool, str]:
    for name, start, end in CRISIS_WINDOWS:
        if start <= dt <= end:
            return True, name
    return False, "Calm"

def load_price_df(symbol: str) -> pd.DataFrame | None:
    path = PRICES / f"{symbol}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df

# ── Load existing results and merge ──────────────────────────────────────────

cars = pd.read_csv(RESULTS / "event_study_cars.csv")
var_ = pd.read_csv(RESULTS / "var_spikes.csv")
corr = pd.read_csv(RESULTS / "correlation_jumps.csv")

# Merge on symbol + announcement_date
cars["announcement_date"] = cars["announcement_date"].astype(str)
var_["announcement_date"] = var_["announcement_date"].astype(str)
corr["announcement_date"] = corr["announcement_date"].astype(str)

merged = (cars
          .merge(var_[["announcement_date","symbol","peak_var_ratio",
                        "ann_day_var_ratio","baseline_var_pct"]],
                 on=["announcement_date","symbol"], how="left")
          .merge(corr[["announcement_date","symbol","corr_jump","corr_pre","corr_post"]],
                 on=["announcement_date","symbol"], how="left"))

merged["announcement_date"] = pd.to_datetime(merged["announcement_date"])

# ── Regime label ──────────────────────────────────────────────────────────────

merged["is_crisis"]    = merged["announcement_date"].apply(lambda d: is_crisis(d)[0])
merged["regime_label"] = merged["announcement_date"].apply(lambda d: is_crisis(d)[1])
merged["era2"]         = (merged["era"] == "Era2_2018-2025").astype(int)
merged["crisis"]       = merged["is_crisis"].astype(int)
merged["era2_x_crisis"] = merged["era2"] * merged["crisis"]
merged["inclusion"]    = (merged["action"] == "Inclusion").astype(int)

print("=== Regime breakdown ===")
print(merged.groupby(["era","is_crisis","action"]).size().to_string())

# ── Abnormal volume ───────────────────────────────────────────────────────────
# avol = volume on announcement day / avg volume [−30, −2]

avol_records = []
for _, row in merged.iterrows():
    sym = row["symbol"]
    ann = row["announcement_date"]
    df  = load_price_df(sym)
    if df is None or "Volume" not in df.columns:
        continue
    vol = df["Volume"].dropna()
    idx = vol.index
    pos = idx.searchsorted(ann, side="left")
    if pos >= len(idx) or abs((idx[pos] - ann).days) > 5:
        continue
    # Baseline volume: [−30, −2]
    lo = max(pos - 30, 0)
    hi = pos - 1
    if hi <= lo:
        continue
    baseline = vol.iloc[lo:hi].mean()
    if baseline < 1:
        continue
    ann_vol  = vol.iloc[pos]
    avol_records.append({
        "announcement_date": ann.date(),
        "symbol":            sym,
        "action":            row["action"],
        "era":               row["era"],
        "is_crisis":         row["is_crisis"],
        "avol_ratio":        round(ann_vol / baseline, 3),
    })

avol_df = pd.DataFrame(avol_records)
if len(avol_df) > 0:
    print(f"\n=== Abnormal volume (N={len(avol_df)}) ===")
    for action in ["Inclusion","Exclusion"]:
        sub = avol_df[avol_df.action==action]
        t, p = sp_stats.ttest_1samp(sub["avol_ratio"].dropna(), 1.0)
        stars = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.1 else ""
        print(f"  {action}: mean={sub['avol_ratio'].mean():.2f}x  "
              f"median={sub['avol_ratio'].median():.2f}x  t={t:.2f}  p={p:.3f}{stars}")
    merged = merged.merge(
        avol_df[["announcement_date","symbol","avol_ratio"]].assign(
            announcement_date=lambda x: pd.to_datetime(x["announcement_date"])),
        on=["announcement_date","symbol"], how="left")

# ── Regression: VaR spike ~ era + crisis + era×crisis ─────────────────────────

print("\n=== OLS: peak_var_ratio ~ era2 + crisis + era2*crisis ===")
for action in ["Inclusion","Exclusion"]:
    sub = merged[merged["action"]==action].dropna(subset=["peak_var_ratio"])
    if len(sub) < 10:
        continue
    mod = smf.ols("peak_var_ratio ~ era2 + crisis + era2_x_crisis", data=sub).fit()
    print(f"\n  {action}s (N={len(sub)}):")
    print(f"  {'Coef':<25} {'Est':>8}  {'t':>6}  {'p':>6}")
    for term in ["Intercept","era2","crisis","era2_x_crisis"]:
        if term in mod.params:
            print(f"  {term:<25} {mod.params[term]:>8.4f}  "
                  f"{mod.tvalues[term]:>6.2f}  {mod.pvalues[term]:>6.3f}"
                  + ("***" if mod.pvalues[term]<0.01
                     else "**" if mod.pvalues[term]<0.05
                     else "*"  if mod.pvalues[term]<0.1 else ""))
    print(f"  R² = {mod.rsquared:.3f}  Adj-R² = {mod.rsquared_adj:.3f}")

# ── CAR × regime ──────────────────────────────────────────────────────────────

print("\n=== CAR by regime (calm vs crisis) ===")
for action in ["Inclusion","Exclusion"]:
    sub = merged[merged["action"]==action]
    for regime in [False, True]:
        group = sub[sub["is_crisis"]==regime]
        label = "Crisis" if regime else "Calm"
        n     = len(group)
        if n < 2:
            continue
        ann   = group["car_ann_day"].dropna()
        post  = group["car_post_medium"].dropna()
        print(f"  {action} | {label} (N={n}): "
              f"ann_day={ann.mean()*100:+.2f}%  post_medium={post.mean()*100:+.2f}%")

merged.to_csv(RESULTS / "regime_analysis.csv", index=False)
print(f"\nSaved merged results → results/regime_analysis.csv")

# ── Figure 8: VaR spike — Crisis vs Calm × Era ───────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
fig.suptitle("VaR Spike by Era and Market Regime\n(peak_var_ratio: higher = larger spike)",
             fontsize=12)

era_color  = {"Era1_2015-2017": "#2196F3", "Era2_2018-2025": "#FF5722"}
era_nice   = {"Era1_2015-2017": "Era 1 (2015–17)", "Era2_2018-2025": "Era 2 (2018–25)"}
calm_alpha = 0.9
crisis_hatch = "//"

for ax, action in zip(axes, ["Inclusion", "Exclusion"]):
    sub    = merged[merged["action"]==action].dropna(subset=["peak_var_ratio"])
    eras   = ["Era1_2015-2017", "Era2_2018-2025"]
    x      = np.arange(len(eras))
    width  = 0.35

    calm_means = []
    calm_errs  = []
    cris_means = []
    cris_errs  = []

    for era in eras:
        calm_vals  = sub[(sub.era==era) & (~sub.is_crisis)]["peak_var_ratio"]
        cris_vals  = sub[(sub.era==era) & (sub.is_crisis)]["peak_var_ratio"]
        calm_means.append(calm_vals.mean() if len(calm_vals) > 0 else 0)
        calm_errs.append(calm_vals.sem()   if len(calm_vals) > 1 else 0)
        cris_means.append(cris_vals.mean() if len(cris_vals) > 0 else 0)
        cris_errs.append(cris_vals.sem()   if len(cris_vals) > 1 else 0)

    ax.bar(x - width/2, calm_means, width, label="Calm market",
           color=["#2196F3","#FF5722"], alpha=0.85,
           yerr=calm_errs, capsize=4, error_kw={"elinewidth":1})
    ax.bar(x + width/2, cris_means, width, label="Crisis market",
           color=["#2196F3","#FF5722"], alpha=0.45, hatch="//",
           yerr=cris_errs, capsize=4, error_kw={"elinewidth":1})

    ax.axhline(1, color="grey", lw=0.6, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([era_nice[e] for e in eras], fontsize=9)
    ax.set_title(f"{action}s", fontsize=11)
    ax.set_ylabel("VaR spike ratio (×baseline)")
    ax.legend(["Calm", "Crisis"], fontsize=9)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(FIGS / "fig8_crisis_var_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig8_crisis_var_comparison.png")

# ── Figure 9: Abnormal volume on announcement day ─────────────────────────────

if len(avol_df) > 0:
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Abnormal Volume on Announcement Day\n"
                 "(ratio vs. 30-day pre-event baseline)", fontsize=12)

    eras   = ["Era1_2015-2017", "Era2_2018-2025"]
    acts   = ["Inclusion", "Exclusion"]
    x      = np.arange(len(eras))
    width  = 0.35
    colors = ["#1f77b4", "#d62728"]

    for i, action in enumerate(acts):
        means = []
        errs  = []
        for era in eras:
            vals = avol_df[(avol_df.action==action) & (avol_df.era==era)]["avol_ratio"]
            means.append(vals.mean() if len(vals) > 0 else 0)
            errs.append(vals.sem()   if len(vals) > 1 else 0)
        ax.bar(x + (i - 0.5) * width, means, width, label=action,
               color=colors[i], alpha=0.8,
               yerr=errs, capsize=4, error_kw={"elinewidth":1})

    ax.axhline(1, color="grey", lw=0.6, ls=":", label="Baseline (1.0x)")
    ax.set_xticks(x)
    ax.set_xticklabels([era_nice.get(e, e) for e in eras])
    ax.set_ylabel("Volume ratio (×baseline)")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "fig9_abnormal_volume.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig9_abnormal_volume.png")

# ── Figure 10: Summary scatter — CAR vs VaR spike ────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Announcement-Day CAR vs. VaR Spike — Cross-sectional\n"
             "Do events with larger price moves also have larger VaR spikes?", fontsize=11)

action_color = {"Inclusion": "#1f77b4", "Exclusion": "#d62728"}

for ax, action in zip(axes, ["Inclusion", "Exclusion"]):
    sub = merged[merged["action"]==action].dropna(subset=["car_ann_day","peak_var_ratio"])
    for era in ["Era1_2015-2017", "Era2_2018-2025"]:
        pts = sub[sub.era==era]
        ax.scatter(pts["car_ann_day"]*100, pts["peak_var_ratio"],
                   s=60, alpha=0.7, label=era_nice[era],
                   color=era_color[era], edgecolors="white", linewidth=0.5)
    # OLS trendline
    if len(sub) > 3:
        m, b, r, p, se = sp_stats.linregress(sub["car_ann_day"]*100,
                                               sub["peak_var_ratio"])
        xl = np.linspace(sub["car_ann_day"].min()*100, sub["car_ann_day"].max()*100, 50)
        ax.plot(xl, m*xl+b, "k--", lw=1, alpha=0.5,
                label=f"OLS (r={r:.2f}, p={p:.2f})")
    ax.axhline(1, color="grey", lw=0.4, ls=":")
    ax.axvline(0, color="grey", lw=0.4, ls=":")
    ax.set_xlabel("Announcement-day CAR (%)")
    ax.set_ylabel("VaR spike ratio")
    ax.set_title(f"{action}s", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(FIGS / "fig10_car_vs_var_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig10_car_vs_var_scatter.png")
print("\nRegime analysis complete.")
