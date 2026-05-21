"""
01_fetch_prices.py
==================
Downloads daily adjusted-close price data for every stock in nifty50_changes.csv
plus the Nifty 50 index (^NSEI) and saves each to data/prices/{SYMBOL}.csv.

Each file has columns: Date, Close, Return
Uses auto_adjust=True so splits and dividends are already baked in.
Skips symbols whose file already exists (safe to re-run).
"""

import pandas as pd
import yfinance as yf
import time
from pathlib import Path

ROOT   = Path(__file__).parent.parent
DATA   = ROOT / "data"
PRICES = DATA / "prices"
PRICES.mkdir(parents=True, exist_ok=True)

events = pd.read_csv(ROOT / "nifty50_changes.csv")

# All unique stock symbols + the market index
symbols = sorted(events["symbol"].dropna().unique().tolist())
all_tickers = symbols + ["^NSEI"]

print(f"Fetching {len(symbols)} stocks + Nifty index = {len(all_tickers)} tickers total\n")

# Need data from 2013 (estimation window starts ~260 days before first 2015 event)
START = "2013-01-01"
END   = "2026-01-01"

failed = []
for ticker in all_tickers:
    yf_sym  = ticker if ticker.startswith("^") else f"{ticker}.NS"
    outfile = PRICES / f"{ticker}.csv"

    if outfile.exists():
        df_check = pd.read_csv(outfile)
        print(f"  CACHED  {ticker:<18}  ({len(df_check)} rows)")
        continue

    try:
        hist = yf.Ticker(yf_sym).history(start=START, end=END, auto_adjust=True)
        if hist.empty:
            print(f"  EMPTY   {ticker:<18}  — no data returned")
            failed.append(ticker)
            continue

        df = hist[["Close"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "Date"
        df["Return"] = df["Close"].pct_change()
        df.to_csv(outfile)
        print(f"  OK      {ticker:<18}  {len(df)} rows  "
              f"({df.index.min().date()} → {df.index.max().date()})")
    except Exception as e:
        print(f"  ERROR   {ticker:<18}  {e}")
        failed.append(ticker)

    time.sleep(0.3)   # polite rate limiting

print(f"\n{'='*50}")
print(f"Done. {len(all_tickers)-len(failed)}/{len(all_tickers)} tickers saved.")
if failed:
    print(f"Failed: {failed}")
    print("These events will be excluded from analysis (documented in paper).")
