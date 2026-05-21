"""
02_event_study.py
=================
Market-model event study on Nifty 50 reconstitution events (2015–2025).

Methodology:
  - Estimation window: trading days [-260, -11] relative to announcement date
  - Event window: [-10, +60]
  - Market model: OLS regression of stock return on ^NSEI return
  - Abnormal return: AR[t] = R_stock[t] - (alpha + beta * R_market[t])
  - CAR[t1,t2]: cumulative sum of AR

CAR windows reported:
  Pre-announcement drift:  [-10, -1]
  Announcement:            [ 0,  +1]
  Short post:              [+2,  +5]
  Medium post:             [+2, +20]
  Long post:               [+2, +40]
  Effective-date window:   [eff-2, eff+5]   (around the actual rebalancing date)

Outputs:
  results/event_study_cars.csv   — one row per event with all CAR values
  results/figures/               — all charts
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

ROOT    = Path(__file__).parent.parent
PRICES  = ROOT / "data" / "prices"
RESULTS = ROOT / "results"
FIGS    = RESULTS / "figures"
RESULTS.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
EST_START  = -260
EST_END    = -11
EVT_START  = -10
EVT_END    =  60
MIN_EST_OBS = 100   # minimum trading days in estimation window

CAR_WINDOWS = {
    "pre_drift":   (-10, -1),
    "ann_day":     (  0,  1),
    "post_short":  (  2,  5),
    "post_medium": (  2, 20),
    "post_long":   (  2, 40),
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_prices(symbol: str) -> pd.Series | None:
    path = PRICES / f"{symbol}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df["Return"].dropna()


def get_trading_calendar(base: pd.DatetimeIndex, anchor: pd.Timestamp,
                         start_offset: int, end_offset: int) -> pd.DatetimeIndex:
    """Return the slice of `base` trading dates between [anchor+start_offset, anchor+end_offset]
    measured in trading-day offsets."""
    all_dates = base.sort_values()
    anchor_pos = all_dates.searchsorted(anchor, side="left")
    lo = anchor_pos + start_offset
    hi = anchor_pos + end_offset + 1
    lo = max(lo, 0)
    hi = min(hi, len(all_dates))
    return all_dates[lo:hi]


def market_model(stock_ret: pd.Series, mkt_ret: pd.Series,
                 est_dates: pd.DatetimeIndex):
    """OLS market model. Returns (alpha, beta, sigma_e)."""
    s = stock_ret.reindex(est_dates).dropna()
    m = mkt_ret.reindex(s.index).dropna()
    common = s.index.intersection(m.index)
    if len(common) < MIN_EST_OBS:
        return None, None, None
    s, m = s[common], m[common]
    X = sm.add_constant(m)
    res = sm.OLS(s, X).fit()
    alpha, beta = res.params
    sigma_e = np.sqrt(res.mse_resid)
    return alpha, beta, sigma_e


def compute_ars(stock_ret, mkt_ret, event_dates, alpha, beta):
    """AR[t] = R_stock[t] - (alpha + beta * R_market[t])"""
    s = stock_ret.reindex(event_dates).fillna(0)
    m = mkt_ret.reindex(event_dates).fillna(0)
    return (s - (alpha + beta * m)).values


def bmp_test(ar_matrix: np.ndarray) -> dict:
    """
    Boehmer-Musumeci-Poulsen (1991) standardised cross-sectional test.
    ar_matrix: shape (N_events, T_days)
    Returns dict with t-stat and p-value for each column (day).
    """
    from scipy import stats
    N, T = ar_matrix.shape
    # Standardise each event's AR by its std deviation across the event window
    std_per_event = ar_matrix.std(axis=1, ddof=1) + 1e-10
    sar = ar_matrix / std_per_event[:, None]
    # Cross-sectional mean SAR per day
    mean_sar = sar.mean(axis=0)
    se_sar   = sar.std(axis=0, ddof=1) / np.sqrt(N)
    t_stats  = mean_sar / (se_sar + 1e-10)
    p_vals   = 2 * stats.t.sf(np.abs(t_stats), df=N - 1)
    return {"t_stat": t_stats, "p_val": p_vals}


# ── Main event loop ───────────────────────────────────────────────────────────

mkt_ret = load_prices("^NSEI")
if mkt_ret is None:
    raise FileNotFoundError("^NSEI price file missing — run 01_fetch_prices.py first")

events = pd.read_csv(ROOT / "nifty50_changes.csv")
events["announcement_date"] = pd.to_datetime(events["announcement_date"])
events["effective_date"]    = pd.to_datetime(events["effective_date"])

available_syms = {f.stem for f in PRICES.glob("*.csv") if f.stem != "^NSEI"}
events = events[events["symbol"].isin(available_syms)].copy()
events["era"] = events["announcement_date"].dt.year.apply(
    lambda y: "Era1_2015-2017" if y <= 2017 else "Era2_2018-2025"
)

mkt_trading = mkt_ret.index

records     = []
ar_matrices = {"Inclusion": [], "Exclusion": []}  # for aggregate CAR plots
ar_era      = {"Era1_2015-2017": {"Inclusion": [], "Exclusion": []},
               "Era2_2018-2025": {"Inclusion": [], "Exclusion": []}}

for _, ev in events.iterrows():
    sym   = ev["symbol"]
    ann   = ev["announcement_date"]
    eff   = ev["effective_date"]
    action = ev["action"]
    era   = ev["era"]

    stk_ret = load_prices(sym)
    if stk_ret is None:
        continue

    # ── Estimation window ────────────────────────────────────────────────────
    est_dates = get_trading_calendar(mkt_trading, ann, EST_START, EST_END)
    alpha, beta, sigma_e = market_model(stk_ret, mkt_ret, est_dates)
    if alpha is None:
        print(f"  SKIP (insuff. estimation data): {sym} {ann.date()}")
        continue

    # ── Event window ARs ─────────────────────────────────────────────────────
    evt_dates = get_trading_calendar(mkt_trading, ann, EVT_START, EVT_END)
    if len(evt_dates) < (EVT_END - EVT_START) * 0.5:
        continue
    ars = compute_ars(stk_ret, mkt_ret, evt_dates, alpha, beta)

    # ── Effective-date window ─────────────────────────────────────────────────
    eff_dates = get_trading_calendar(mkt_trading, eff, -2, 5)
    eff_ars   = compute_ars(stk_ret, mkt_ret, eff_dates, alpha, beta)
    car_eff   = eff_ars.sum()

    # ── CAR windows ──────────────────────────────────────────────────────────
    t0_pos = EVT_START * -1   # position of day 0 in the array (index 10)
    row = {
        "announcement_date": ann.date(),
        "effective_date":    eff.date(),
        "symbol":            sym,
        "action":            action,
        "era":               era,
        "alpha":             round(alpha, 6),
        "beta":              round(beta, 4),
        "sigma_e":           round(sigma_e, 6),
        "car_eff_window":    round(car_eff, 4),
    }
    for name, (t1, t2) in CAR_WINDOWS.items():
        lo = t0_pos + t1
        hi = t0_pos + t2 + 1
        lo = max(lo, 0)
        hi = min(hi, len(ars))
        row[f"car_{name}"] = round(ars[lo:hi].sum(), 4)
    records.append(row)

    # Store full AR series (padded/trimmed to EVT_END-EVT_START+1 length)
    target_len = EVT_END - EVT_START + 1
    if len(ars) >= target_len:
        ar_matrices[action].append(ars[:target_len])
        ar_era[era][action].append(ars[:target_len])

results_df = pd.DataFrame(records)
results_df.to_csv(RESULTS / "event_study_cars.csv", index=False)
print(f"Saved {len(results_df)} event records → results/event_study_cars.csv\n")

# ── Summary statistics ────────────────────────────────────────────────────────

from scipy import stats as sp_stats

def summarise(df, label=""):
    print(f"\n{'─'*60}")
    print(f"  {label}  (N={len(df)})")
    print(f"{'─'*60}")
    for col in [c for c in df.columns if c.startswith("car_")]:
        vals = df[col].dropna()
        t, p = sp_stats.ttest_1samp(vals, 0)
        stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        print(f"  {col:<25} mean={vals.mean()*100:+.2f}%  t={t:+.2f}  p={p:.3f} {stars}")

print("\n=== OVERALL ===")
summarise(results_df[results_df.action=="Inclusion"], "Inclusions")
summarise(results_df[results_df.action=="Exclusion"], "Exclusions")

print("\n=== BY ERA ===")
for era in ["Era1_2015-2017", "Era2_2018-2025"]:
    sub = results_df[results_df.era == era]
    summarise(sub[sub.action=="Inclusion"], f"{era} — Inclusions")
    summarise(sub[sub.action=="Exclusion"], f"{era} — Exclusions")

# ── Figure 1: Average CAR trajectory, Inclusions vs Exclusions ───────────────

days = np.arange(EVT_START, EVT_END + 1)

def mean_cum_ar(ar_list):
    if not ar_list:
        return None
    mat = np.array(ar_list)
    return np.cumsum(mat, axis=1).mean(axis=0)

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
fig.suptitle("Average Cumulative Abnormal Returns Around Nifty 50 Reconstitution\n"
             "Market Model, 2015–2025  |  47 Events", fontsize=12)

for ax, action, color in zip(axes, ["Inclusion", "Exclusion"], ["#1f77b4", "#d62728"]):
    cum = mean_cum_ar(ar_matrices[action])
    if cum is None:
        continue
    n = len(ar_matrices[action])
    ax.plot(days, cum * 100, color=color, lw=2, label=f"Mean CAR (N={n})")
    ax.axvline(0, color="black", lw=0.8, ls="--", label="Announcement")
    # effective date approx (+30 trading days)
    ax.axvline(30, color="grey", lw=0.8, ls=":", label="~Effective date")
    ax.axhline(0, color="black", lw=0.5, alpha=0.4)
    ax.fill_between(days, cum * 100, 0, alpha=0.1, color=color)
    ax.set_title(f"{action}s", fontsize=11)
    ax.set_xlabel("Trading days relative to announcement")
    ax.set_ylabel("CAR (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGS / "fig1_car_trajectory.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved fig1_car_trajectory.png")

# ── Figure 2: Era comparison — CAR[0,+5] and CAR[+2,+20] ────────────────────

car_cols   = ["car_ann_day", "car_post_short", "car_post_medium", "car_pre_drift"]
labels     = ["[0,+1]", "[+2,+5]", "[+2,+20]", "[-10,-1]"]
era_labels = ["Era 1\n2015–2017", "Era 2\n2018–2025"]
eras       = ["Era1_2015-2017", "Era2_2018-2025"]
colors     = ["#2196F3", "#FF5722"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("CAR Comparison Across Eras — Inclusions vs Exclusions", fontsize=12)

for ax, action in zip(axes, ["Inclusion", "Exclusion"]):
    sub = results_df[results_df.action == action]
    x = np.arange(len(car_cols))
    width = 0.35

    for i, (era, col, lbl) in enumerate(zip(eras, eras, era_labels)):
        era_sub = sub[sub.era == era]
        means   = [era_sub[c].mean() * 100 for c in car_cols]
        sems    = [era_sub[c].sem() * 100 for c in car_cols]
        bars = ax.bar(x + (i - 0.5) * width, means, width,
                      label=era_labels[i], color=colors[i], alpha=0.8,
                      yerr=sems, capsize=4, error_kw={"elinewidth": 1})

    ax.set_title(f"{action}s", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("CAR (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax.axhline(0, color="black", lw=0.6)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(FIGS / "fig2_car_era_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig2_car_era_comparison.png")

# ── Figure 3: Era-split CAR trajectories ─────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("CAR Trajectories by Era and Action\nNifty 50 Reconstitution 2015–2025",
             fontsize=12)

era_color = {"Era1_2015-2017": "#2196F3", "Era2_2018-2025": "#FF5722"}
era_nice  = {"Era1_2015-2017": "Era 1 (2015–17)", "Era2_2018-2025": "Era 2 (2018–25)"}

for col_idx, action in enumerate(["Inclusion", "Exclusion"]):
    for row_idx, era in enumerate(["Era1_2015-2017", "Era2_2018-2025"]):
        ax = axes[row_idx][col_idx]
        ar_list = ar_era[era][action]
        n = len(ar_list)
        if n == 0:
            ax.set_visible(False)
            continue
        cum = mean_cum_ar(ar_list)
        ax.plot(days, cum * 100, color=era_color[era], lw=2)
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.axhline(0, color="black", lw=0.5, alpha=0.4)
        ax.fill_between(days, cum * 100, 0, alpha=0.12, color=era_color[era])
        ax.set_title(f"{era_nice[era]} — {action}s  (N={n})", fontsize=10)
        ax.set_xlabel("Trading days relative to announcement")
        ax.set_ylabel("CAR (%)")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
        ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGS / "fig3_car_by_era.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig3_car_by_era.png")

print("\nEvent study complete.")
