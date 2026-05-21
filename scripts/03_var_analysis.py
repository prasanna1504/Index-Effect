"""
03_var_analysis.py
==================
Rolling historical VaR spike analysis around Nifty 50 reconstitution events.

For each event:
  1. Compute 20-day rolling historical VaR (95th percentile loss) in a [-20, +20]
     window around announcement date.
  2. Normalise: VaR_spike[t] = VaR[t] / mean(VaR[-20:-2])  →  ratio vs baseline
  3. Also compute realised volatility (20-day rolling std) as a parallel metric.

Aggregate across events → average VaR trajectory.
Segment by era and action to test the hypothesis that Era 2 shows larger spikes.

Outputs:
  results/var_spikes.csv
  results/figures/fig4_var_trajectory.png
  results/figures/fig5_var_era_comparison.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

ROOT    = Path(__file__).parent.parent
PRICES  = ROOT / "data" / "prices"
RESULTS = ROOT / "results"
FIGS    = RESULTS / "figures"

VAR_WINDOW = 20    # rolling window for historical VaR
CONF       = 0.05  # 5% → 95% VaR
DAYS_PRE   = 20    # days before announcement to compute
DAYS_POST  = 20    # days after

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_returns(symbol: str) -> pd.Series | None:
    path = PRICES / f"{symbol}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df["Return"].dropna()


def rolling_hist_var(returns: pd.Series, window: int = 20,
                     conf: float = 0.05) -> pd.Series:
    """Historical VaR (as a positive number = loss)."""
    return returns.rolling(window).quantile(conf).abs()


def get_offsets(base_idx: pd.DatetimeIndex, anchor: pd.Timestamp,
                lo: int, hi: int) -> tuple[pd.DatetimeIndex, int]:
    """Return trading-day slice and anchor position within it."""
    all_dates = base_idx.sort_values()
    pos = all_dates.searchsorted(anchor, side="left")
    s = max(pos + lo, 0)
    e = min(pos + hi + 1, len(all_dates))
    return all_dates[s:e], pos - s   # dates, anchor_pos_in_slice


# ── Main ──────────────────────────────────────────────────────────────────────

events = pd.read_csv(ROOT / "nifty50_changes.csv")
events["announcement_date"] = pd.to_datetime(events["announcement_date"])
events["era"] = events["announcement_date"].dt.year.apply(
    lambda y: "Era1_2015-2017" if y <= 2017 else "Era2_2018-2025"
)

available_syms = {f.stem for f in PRICES.glob("*.csv") if f.stem != "^NSEI"}
events = events[events["symbol"].isin(available_syms)].copy()

records    = []
# Collect normalised VaR series: shape (N, DAYS_PRE+DAYS_POST+1)
series_all = {"Inclusion": [], "Exclusion": []}
series_era = {
    "Era1_2015-2017": {"Inclusion": [], "Exclusion": []},
    "Era2_2018-2025": {"Inclusion": [], "Exclusion": []},
}

for _, ev in events.iterrows():
    sym    = ev["symbol"]
    ann    = ev["announcement_date"]
    action = ev["action"]
    era    = ev["era"]

    ret = load_returns(sym)
    if ret is None:
        continue

    # Need enough history before the window for a meaningful rolling VaR
    var_series = rolling_hist_var(ret, VAR_WINDOW, CONF)
    vol_series = ret.rolling(VAR_WINDOW).std()

    window_dates, anchor_pos = get_offsets(var_series.index, ann,
                                           -DAYS_PRE - VAR_WINDOW, DAYS_POST)
    var_slice = var_series.reindex(window_dates).dropna()
    vol_slice = vol_series.reindex(window_dates).dropna()

    if len(var_slice) < DAYS_PRE + DAYS_POST:
        continue

    # Re-align to [-DAYS_PRE, +DAYS_POST]
    ann_in_var = var_slice.index.searchsorted(ann, side="left")
    lo = ann_in_var - DAYS_PRE
    hi = ann_in_var + DAYS_POST + 1
    if lo < 0 or hi > len(var_slice):
        continue

    var_window  = var_slice.iloc[lo:hi].values    # length 41
    vol_window  = vol_slice.iloc[lo:hi].values

    # Baseline: days [-20, -2]  (pre-announcement, excluding day -1 and 0)
    baseline_var = var_window[:DAYS_PRE - 1].mean()
    baseline_vol = vol_window[:DAYS_PRE - 1].mean()
    if baseline_var < 1e-8 or baseline_vol < 1e-8:
        continue

    norm_var = var_window / baseline_var
    norm_vol = vol_window / baseline_vol

    # Peak spike stats
    peak_idx  = np.argmax(norm_var[DAYS_PRE:]) + DAYS_PRE
    peak_val  = norm_var[peak_idx]
    post_mean = norm_var[DAYS_PRE + 5:].mean()

    records.append({
        "announcement_date": ann.date(),
        "symbol":            sym,
        "action":            action,
        "era":               era,
        "baseline_var_pct":  round(baseline_var * 100, 3),
        "peak_var_ratio":    round(peak_val, 3),
        "peak_day_offset":   int(peak_idx - DAYS_PRE),
        "post_mean_ratio":   round(post_mean, 3),
        "ann_day_var_ratio": round(norm_var[DAYS_PRE], 3),
    })

    series_all[action].append(norm_var)
    series_era[era][action].append(norm_var)

results_df = pd.DataFrame(records)
results_df.to_csv(RESULTS / "var_spikes.csv", index=False)
print(f"Saved {len(results_df)} events → results/var_spikes.csv\n")

# ── Summary stats ─────────────────────────────────────────────────────────────

def summarise_var(df, label):
    print(f"\n  {label} (N={len(df)})")
    for col in ["ann_day_var_ratio", "peak_var_ratio", "post_mean_ratio"]:
        vals = df[col].dropna()
        t, p = stats.ttest_1samp(vals, 1.0)  # test H0: ratio = 1 (no spike)
        stars = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.1 else ""
        print(f"    {col:<25} mean={vals.mean():.3f}x  "
              f"median={vals.median():.3f}x  t={t:+.2f}  p={p:.3f}{stars}")

print("=== VaR spike summary (ratio vs baseline, H0: ratio=1) ===")
for action in ["Inclusion", "Exclusion"]:
    sub = results_df[results_df.action==action]
    summarise_var(sub, f"All {action}s")
    for era in ["Era1_2015-2017", "Era2_2018-2025"]:
        summarise_var(sub[sub.era==era], f"  {era}")

# Era 2 vs Era 1 comparison t-test
print("\n=== Era 1 vs Era 2 spike magnitude t-test ===")
for action in ["Inclusion", "Exclusion"]:
    e1 = results_df[(results_df.action==action) & (results_df.era=="Era1_2015-2017")]["peak_var_ratio"]
    e2 = results_df[(results_df.action==action) & (results_df.era=="Era2_2018-2025")]["peak_var_ratio"]
    if len(e1) > 1 and len(e2) > 1:
        t, p = stats.ttest_ind(e1, e2, equal_var=False)
        print(f"  {action}: Era1 mean={e1.mean():.3f}x  Era2 mean={e2.mean():.3f}x  "
              f"t={t:.2f}  p={p:.3f}")

# ── Figure 4: Average normalised VaR trajectory ───────────────────────────────

days = np.arange(-DAYS_PRE, DAYS_POST + 1)

def mean_series(lst):
    if not lst:
        return None
    return np.array(lst).mean(axis=0)

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
fig.suptitle("Normalised VaR Trajectory Around Announcement Date\n"
             "(Ratio vs. pre-event baseline = 1.0)", fontsize=12)

era_color = {"Era1_2015-2017": "#2196F3", "Era2_2018-2025": "#FF5722"}
era_nice  = {"Era1_2015-2017": "Era 1 (2015–17)", "Era2_2018-2025": "Era 2 (2018–25)"}

for ax, action in zip(axes, ["Inclusion", "Exclusion"]):
    # Overall mean
    overall = mean_series(series_all[action])
    if overall is not None:
        ax.plot(days, overall, color="black", lw=2.5,
                label=f"All {action}s (N={len(series_all[action])})", zorder=5)

    # Per-era
    for era in ["Era1_2015-2017", "Era2_2018-2025"]:
        ser = mean_series(series_era[era][action])
        n   = len(series_era[era][action])
        if ser is not None:
            ax.plot(days, ser, color=era_color[era], lw=1.8,
                    ls="--" if "Era1" in era else "-",
                    label=f"{era_nice[era]} (N={n})", alpha=0.9)

    ax.axvline(0, color="black", lw=0.8, ls="--", label="Announcement")
    ax.axhline(1, color="grey", lw=0.6, ls=":", label="Baseline (1.0x)")
    ax.set_title(f"{action}s", fontsize=11)
    ax.set_xlabel("Trading days relative to announcement")
    ax.set_ylabel("VaR ratio (×baseline)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0.5)

plt.tight_layout()
plt.savefig(FIGS / "fig4_var_trajectory.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved fig4_var_trajectory.png")

# ── Figure 5: Era comparison bar chart ───────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
fig.suptitle("VaR Spike Magnitude: Era 1 vs Era 2", fontsize=12)

metrics = ["ann_day_var_ratio", "peak_var_ratio", "post_mean_ratio"]
mlabels = ["Day 0\n(announcement)", "Peak\n(post-ann)", "Post-ann\nmean [+5,+20]"]
x       = np.arange(len(metrics))
width   = 0.35

for ax, action in zip(axes, ["Inclusion", "Exclusion"]):
    sub = results_df[results_df.action==action]
    for i, (era, lbl, col) in enumerate(zip(
            ["Era1_2015-2017", "Era2_2018-2025"],
            ["Era 1 (2015–17)", "Era 2 (2018–25)"],
            ["#2196F3", "#FF5722"])):
        era_sub = sub[sub.era==era]
        means   = [era_sub[m].mean() for m in metrics]
        sems    = [era_sub[m].sem() for m in metrics]
        n       = len(era_sub)
        ax.bar(x + (i - 0.5) * width, means, width, label=f"{lbl} (N={n})",
               color=col, alpha=0.8, yerr=sems, capsize=4,
               error_kw={"elinewidth": 1.2})

    ax.axhline(1, color="grey", lw=0.7, ls=":")
    ax.set_title(f"{action}s", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(mlabels, fontsize=9)
    ax.set_ylabel("VaR ratio (×baseline)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(FIGS / "fig5_var_era_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig5_var_era_comparison.png")
print("\nVaR analysis complete.")
