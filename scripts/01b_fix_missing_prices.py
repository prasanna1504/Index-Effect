"""
01b_fix_missing_prices.py
=========================
Retries the 6 failed tickers using yf.download() instead of Ticker.history(),
which handles timezone-missing edge cases. Also tries alternate tickers for
renamed/merged stocks.
"""

import pandas as pd
import yfinance as yf
from pathlib import Path

ROOT   = Path(__file__).parent.parent
PRICES = ROOT / "data" / "prices"

# ticker → list of candidates to try in order
RETRY = {
    "CAIRN":      ["CAIRN.NS", "CAIRNIND.NS"],        # Cairn India, delisted 2016
    "IBULHSGFIN": ["IBULHSGFIN.NS"],                   # Still listed, timezone bug
    "INFRATEL":   ["INFRATEL.NS", "INDUSTOWER.NS"],    # Merged → Indus Towers 2021
    "LTIM":       ["LTIM.NS", "LTIMINDTREE.NS"],       # LTIMindtree, listed 2022
    "TATAMTRDVR": ["TATAMTRDVR.NS"],                   # Tata DVR, delisted Jul 2023
    "ZOMATO":     ["ZOMATO.NS"],                       # Listed Jul 2021
}

START = "2013-01-01"
END   = "2026-01-01"

for sym, candidates in RETRY.items():
    outfile = PRICES / f"{sym}.csv"
    if outfile.exists():
        print(f"  CACHED  {sym}")
        continue

    saved = False
    for cand in candidates:
        try:
            df = yf.download(cand, start=START, end=END,
                             auto_adjust=True, progress=False)
            if df.empty:
                print(f"  EMPTY   {cand}")
                continue

            out = df[["Close"]].copy()
            out.index = pd.to_datetime(out.index).tz_localize(None)
            out.index.name = "Date"
            # Flatten MultiIndex columns if present
            if isinstance(out.columns, pd.MultiIndex):
                out.columns = out.columns.get_level_values(0)
            out["Return"] = out["Close"].pct_change()
            out.to_csv(outfile)
            print(f"  OK      {sym:<18}  via {cand}  {len(out)} rows  "
                  f"({out.index.min().date()} → {out.index.max().date()})")
            saved = True
            break
        except Exception as e:
            print(f"  ERROR   {cand}  {e}")

    if not saved:
        print(f"  FAILED  {sym} — will be excluded from analysis")
