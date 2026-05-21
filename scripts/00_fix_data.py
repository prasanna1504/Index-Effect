"""
00_fix_data.py
==============
1. Adds 12 missing reconstitution events (3 announcement dates missed by scraper
   due to the sub-lettered "a) Nifty 50" section format) to both:
     - nifty50_master.csv   (4-column authoritative dataset)
     - nifty50_changes.csv  (6-column scraper output with symbol + source_pdf)
2. Writes the fixed files to data/nifty50_master_clean.csv (master stays 4 cols;
   changes saved in-place after backfill).
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

MISSING = [
    # 2020-08-20: "a) NIFTY 50" format, effective 2020-09-25
    dict(announcement_date="2020-08-20", effective_date="2020-09-25",
         stock_name="Bharti Infratel Ltd.", symbol="INFRATEL", action="Exclusion",
         source_pdf="https://nsearchives.nseindia.com//web/sites/default/files/2020-08/ind_prs20082020.pdf"),
    dict(announcement_date="2020-08-20", effective_date="2020-09-25",
         stock_name="Zee Entertainment Enterprises Ltd.", symbol="ZEEL", action="Exclusion",
         source_pdf="https://nsearchives.nseindia.com//web/sites/default/files/2020-08/ind_prs20082020.pdf"),
    dict(announcement_date="2020-08-20", effective_date="2020-09-25",
         stock_name="Divi's Laboratories Ltd.", symbol="DIVISLAB", action="Inclusion",
         source_pdf="https://nsearchives.nseindia.com//web/sites/default/files/2020-08/ind_prs20082020.pdf"),
    dict(announcement_date="2020-08-20", effective_date="2020-09-25",
         stock_name="SBI Life Insurance Company Ltd.", symbol="SBILIFE", action="Inclusion",
         source_pdf="https://nsearchives.nseindia.com//web/sites/default/files/2020-08/ind_prs20082020.pdf"),
    # 2024-08-23: "a) Nifty 50" sub-lettered format, effective 2024-09-30
    dict(announcement_date="2024-08-23", effective_date="2024-09-30",
         stock_name="Divi's Laboratories Ltd.", symbol="DIVISLAB", action="Exclusion",
         source_pdf="https://nsearchives.nseindia.com//web/sites/default/files/2024-08/ind_prs23082024.pdf"),
    dict(announcement_date="2024-08-23", effective_date="2024-09-30",
         stock_name="LTIMindtree Ltd.", symbol="LTIM", action="Exclusion",
         source_pdf="https://nsearchives.nseindia.com//web/sites/default/files/2024-08/ind_prs23082024.pdf"),
    dict(announcement_date="2024-08-23", effective_date="2024-09-30",
         stock_name="Bharat Electronics Ltd.", symbol="BEL", action="Inclusion",
         source_pdf="https://nsearchives.nseindia.com//web/sites/default/files/2024-08/ind_prs23082024.pdf"),
    dict(announcement_date="2024-08-23", effective_date="2024-09-30",
         stock_name="Trent Ltd.", symbol="TRENT", action="Inclusion",
         source_pdf="https://nsearchives.nseindia.com//web/sites/default/files/2024-08/ind_prs23082024.pdf"),
    # 2025-02-21: "a) Nifty 50" sub-lettered format, effective 2025-03-28
    dict(announcement_date="2025-02-21", effective_date="2025-03-28",
         stock_name="Bharat Petroleum Corporation Ltd.", symbol="BPCL", action="Exclusion",
         source_pdf="https://nsearchives.nseindia.com//web/sites/default/files/2025-02/ind_prs21022025.pdf"),
    dict(announcement_date="2025-02-21", effective_date="2025-03-28",
         stock_name="Britannia Industries Ltd.", symbol="BRITANNIA", action="Exclusion",
         source_pdf="https://nsearchives.nseindia.com//web/sites/default/files/2025-02/ind_prs21022025.pdf"),
    dict(announcement_date="2025-02-21", effective_date="2025-03-28",
         stock_name="Jio Financial Services Ltd.", symbol="JIOFIN", action="Inclusion",
         source_pdf="https://nsearchives.nseindia.com//web/sites/default/files/2025-02/ind_prs21022025.pdf"),
    dict(announcement_date="2025-02-21", effective_date="2025-03-28",
         stock_name="Zomato Ltd.", symbol="ZOMATO", action="Inclusion",
         source_pdf="https://nsearchives.nseindia.com//web/sites/default/files/2025-02/ind_prs21022025.pdf"),
]

# ── nifty50_changes.csv ───────────────────────────────────────────────────────
changes_path = ROOT / "nifty50_changes.csv"
changes = pd.read_csv(changes_path)
new_dates = {r["announcement_date"] for r in MISSING}
existing = set(changes["announcement_date"].unique())
to_add_changes = [r for r in MISSING if r["announcement_date"] not in existing]

if to_add_changes:
    patch = pd.DataFrame(to_add_changes)[changes.columns]
    changes = pd.concat([changes, patch], ignore_index=True)
    changes["announcement_date"] = pd.to_datetime(changes["announcement_date"])
    changes = changes.sort_values("announcement_date").reset_index(drop=True)
    changes["announcement_date"] = changes["announcement_date"].dt.strftime("%Y-%m-%d")
    changes.to_csv(changes_path, index=False)
    print(f"nifty50_changes.csv: added {len(to_add_changes)} rows → {len(changes)} total")
else:
    print("nifty50_changes.csv: already up to date")

# ── nifty50_master_clean.csv ──────────────────────────────────────────────────
master_path = ROOT / "nifty50_master.csv"
master = pd.read_csv(master_path)
existing_m = set(master["announcement_date"].unique())
to_add_master = [r for r in MISSING if r["announcement_date"] not in existing_m]

if to_add_master:
    patch_m = pd.DataFrame(to_add_master)[["announcement_date", "effective_date", "stock_name", "action"]]
    master = pd.concat([master, patch_m], ignore_index=True)
    master["announcement_date"] = pd.to_datetime(master["announcement_date"])
    master = master.sort_values("announcement_date").reset_index(drop=True)
    master["announcement_date"] = master["announcement_date"].dt.strftime("%Y-%m-%d")
else:
    print("nifty50_master.csv: already up to date")

# Add symbol column from changes.csv where available
sym_lookup = (
    pd.read_csv(changes_path)[["stock_name", "symbol"]]
    .drop_duplicates("stock_name")
    .set_index("stock_name")["symbol"]
)
master["symbol"] = master["stock_name"].map(sym_lookup)
out_path = DATA / "nifty50_master_clean.csv"
master.to_csv(out_path, index=False)
print(f"nifty50_master_clean.csv: {len(master)} total rows")
print(f"  symbol coverage: {master['symbol'].notna().sum()}/{len(master)} "
      f"({master['symbol'].notna().mean()*100:.0f}%)")

# Era labels
master["dt"] = pd.to_datetime(master["announcement_date"])
era1 = (master["dt"].dt.year <= 2007).sum()
era2 = ((master["dt"].dt.year > 2007) & (master["dt"].dt.year <= 2017)).sum()
era3 = (master["dt"].dt.year > 2017).sum()
print(f"  Era 1 (1996–2007): {era1}  Era 2 (2008–2017): {era2}  Era 3 (2018–2025): {era3}")
