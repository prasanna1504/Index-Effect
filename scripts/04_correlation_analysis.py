"""
04_correlation_analysis.py
==========================
Measures the structural correlation change between each affected stock and the
Nifty 50 index around the effective (rebalancing) date.

For each event:
  Pre window:  60 trading days ending 2 days before effective date  → correlation_pre
  Post window: 60 trading days starting 2 days after effective date → correlation_post
  Jump:        correlation_post - correlation_pre

The hypothesis for inclusions: passive funds must permanently hold the stock,
increasing co-movement with the index → positive correlation jump.
For exclusions: the reverse → negative jump.

Also tests persistence: does the jump survive to the [+60, +120] window?

Outputs:
  results/correlation_jumps.csv
  results/figures/fig6_corr_jump_distribution.png
  results/figures/fig7_corr_jump_by_era.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

ROOT    = Path(__file__).parent.parent
PRICES  = ROOT / "data" / "prices"
RESULTS = ROOT / "results"
FIGS    = RESULTS / "figures"

CORR_WINDOW = 60   # trading days for rolling correlation
BUFFER      = 2    # days gap around effective date

def load_returns(symbol: str) -> pd.Series | None:
    path = PRICES / f"{symbol}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df["Return"].dropna()

def window_corr(stock: pd.Series, market: pd.Series,
                idx: pd.DatetimeIndex, anchor: pd.Timestamp,
                offset_start: int, offset_end: int) -> float | None:
    """Pearson correlation in a trading-day window around anchor."""
    pos = idx.searchsorted(anchor, side="left")
    lo  = pos + offset_start
    hi  = pos + offset_end
    lo  = max(lo, 0)
    hi  = min(hi, len(idx))
    if hi - lo < 20:
        return None
    dates = idx[lo:hi]
    s = stock.reindex(dates).dropna()
    m = market.reindex(s.index).dropna()
    common = s.index.intersection(m.index)
    if len(common) < 20:
        return None
    return float(np.corrcoef(s[common], m[common])[0, 1])

# ── Main ──────────────────────────────────────────────────────────────────────

mkt_ret = load_returns("^NSEI")
trading_cal = mkt_ret.index

events = pd.read_csv(ROOT / "nifty50_changes.csv")
events["announcement_date"] = pd.to_datetime(events["announcement_date"])
events["effective_date"]    = pd.to_datetime(events["effective_date"])
events["era"] = events["announcement_date"].dt.year.apply(
    lambda y: "Era1_2015-2017" if y <= 2017 else "Era2_2018-2025"
)
available_syms = {f.stem for f in PRICES.glob("*.csv") if f.stem != "^NSEI"}
events = events[events["symbol"].isin(available_syms)].copy()

records = []
# Rolling correlation series per day offset (for trajectory chart)
offsets  = np.arange(-90, 91, 5)   # sample every 5 trading days
traj_all = {"Inclusion": [], "Exclusion": []}
traj_era = {
    "Era1_2015-2017": {"Inclusion": [], "Exclusion": []},
    "Era2_2018-2025": {"Inclusion": [], "Exclusion": []},
}

for _, ev in events.iterrows():
    sym    = ev["symbol"]
    eff    = ev["effective_date"]
    ann    = ev["announcement_date"]
    action = ev["action"]
    era    = ev["era"]

    stk_ret = load_returns(sym)
    if stk_ret is None:
        continue

    # Pre/post windows around effective date
    corr_pre  = window_corr(stk_ret, mkt_ret, trading_cal, eff,
                            -(CORR_WINDOW + BUFFER), -BUFFER)
    corr_post = window_corr(stk_ret, mkt_ret, trading_cal, eff,
                            BUFFER, CORR_WINDOW + BUFFER)
    corr_persist = window_corr(stk_ret, mkt_ret, trading_cal, eff,
                               CORR_WINDOW + BUFFER, 2 * CORR_WINDOW + BUFFER)
    # Pre-announcement correlation (baseline)
    corr_pre_ann = window_corr(stk_ret, mkt_ret, trading_cal, ann,
                               -(CORR_WINDOW + 10), -10)

    if corr_pre is None or corr_post is None:
        continue

    jump       = corr_post - corr_pre
    persist    = (corr_persist - corr_pre) if corr_persist is not None else None
    jump_ann   = (corr_post - corr_pre_ann) if corr_pre_ann is not None else None

    # Build rolling correlation trajectory for this event
    traj_row = []
    for off in offsets:
        c = window_corr(stk_ret, mkt_ret, trading_cal, eff,
                        off - CORR_WINDOW // 2, off + CORR_WINDOW // 2)
        traj_row.append(c if c is not None else np.nan)

    if not all(np.isnan(traj_row)):
        traj_all[action].append(traj_row)
        traj_era[era][action].append(traj_row)

    records.append({
        "announcement_date": ann.date(),
        "effective_date":    eff.date(),
        "symbol":            sym,
        "action":            action,
        "era":               era,
        "corr_pre":          round(corr_pre, 4),
        "corr_post":         round(corr_post, 4),
        "corr_jump":         round(jump, 4),
        "corr_persist":      round(persist, 4) if persist is not None else None,
        "corr_jump_vs_ann":  round(jump_ann, 4) if jump_ann is not None else None,
    })

results_df = pd.DataFrame(records)
results_df.to_csv(RESULTS / "correlation_jumps.csv", index=False)
print(f"Saved {len(results_df)} events → results/correlation_jumps.csv\n")

# ── Summary ───────────────────────────────────────────────────────────────────

def summarise_corr(df, label):
    print(f"\n  {label} (N={len(df)})")
    for col in ["corr_pre", "corr_post", "corr_jump", "corr_persist"]:
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        # For jump: test H0: jump = 0
        ref = 0 if "jump" in col or "persist" in col else None
        if ref is not None:
            t, p = stats.ttest_1samp(vals, ref)
            stars = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.1 else ""
            print(f"    {col:<22} mean={vals.mean():+.4f}  t={t:+.2f}  p={p:.3f}{stars}")
        else:
            print(f"    {col:<22} mean={vals.mean():.4f}")

print("=== Correlation jump summary (H0: jump = 0) ===")
for action in ["Inclusion", "Exclusion"]:
    sub = results_df[results_df.action==action]
    summarise_corr(sub, f"All {action}s")
    for era in ["Era1_2015-2017", "Era2_2018-2025"]:
        summarise_corr(sub[sub.era==era], f"  {era}")

# Era comparison
print("\n=== Era 1 vs Era 2 correlation jump t-test ===")
for action in ["Inclusion", "Exclusion"]:
    e1 = results_df[(results_df.action==action) & (results_df.era=="Era1_2015-2017")]["corr_jump"]
    e2 = results_df[(results_df.action==action) & (results_df.era=="Era2_2018-2025")]["corr_jump"]
    if len(e1) > 1 and len(e2) > 1:
        t, p = stats.ttest_ind(e1, e2, equal_var=False)
        print(f"  {action}: Era1={e1.mean():+.4f}  Era2={e2.mean():+.4f}  "
              f"Δ={e2.mean()-e1.mean():+.4f}  t={t:.2f}  p={p:.3f}")

# ── Figure 6: Jump distribution ───────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Correlation Jump with Nifty 50 Around Effective Date\n"
             "60-day post-minus-pre correlation", fontsize=12)

era_color = {"Era1_2015-2017": "#2196F3", "Era2_2018-2025": "#FF5722"}
era_nice  = {"Era1_2015-2017": "Era 1 (2015–17)", "Era2_2018-2025": "Era 2 (2018–25)"}

for ax, action in zip(axes, ["Inclusion", "Exclusion"]):
    sub = results_df[results_df.action==action]
    for era in ["Era1_2015-2017", "Era2_2018-2025"]:
        vals = sub[sub.era==era]["corr_jump"].dropna()
        ax.hist(vals, bins=8, alpha=0.6, color=era_color[era],
                label=f"{era_nice[era]} (N={len(vals)}, mean={vals.mean():+.3f})",
                edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_title(f"{action}s", fontsize=11)
    ax.set_xlabel("Correlation jump (post − pre)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGS / "fig6_corr_jump_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved fig6_corr_jump_distribution.png")

# ── Figure 7: Rolling correlation trajectory ─────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
fig.suptitle("Rolling 60-day Correlation with Nifty 50 Index\n"
             "Centered on effective (rebalancing) date", fontsize=12)

for ax, action in zip(axes, ["Inclusion", "Exclusion"]):
    for era in ["Era1_2015-2017", "Era2_2018-2025"]:
        lst = traj_era[era][action]
        if not lst:
            continue
        mat = np.array(lst, dtype=float)
        mean_traj = np.nanmean(mat, axis=0)
        n = len(lst)
        ax.plot(offsets, mean_traj, color=era_color[era], lw=2,
                ls="--" if "Era1" in era else "-",
                label=f"{era_nice[era]} (N={n})", alpha=0.9)

    ax.axvline(0, color="black", lw=0.8, ls="--", label="Effective date")
    ax.axhline(0, color="grey", lw=0.4, ls=":")
    ax.set_title(f"{action}s", fontsize=11)
    ax.set_xlabel("Trading days relative to effective date")
    ax.set_ylabel("Correlation with ^NSEI")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGS / "fig7_corr_trajectory.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig7_corr_trajectory.png")
print("\nCorrelation analysis complete.")
